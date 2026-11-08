# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view feed of one navigate-handover or manual-drive: its target, the circle's state,
and the baton -- who holds the robot, from the gate's own record.

What it writes, in `mapview/0` (CONTRACT.md): a `SessionStatus` whose source is the operation
(``navigate-handover``) and whose text is the action's own message, and an `Overlays` whose
`target` is the action's target with `state` -- the circle's colour:

========== ======== =========================================================================
state      colour   when
========== ======== =========================================================================
robot      yellow   Nav2 drives; or Nav2 declined and the robot is being brought to confirmed
                    rest, with teleop still forwarding nothing
operator   red      the gate has turned to the operator: only after the wait for rest
reached    green    the action Succeeded
done       --       the action Succeeded and was the last target (``last: true``)
stopped    grey     a stop of any scope, a lease that ended, or the handover Failed
========== ======== =========================================================================

THE STATE IS THE JOB'S. It is derived from the job alone -- its drive's phase, whether a stop
path halted it, whether the adapter still holds it as the current action, and whether its gate is
still open for it -- never from the robot's pose. A stop closes the gate under the binding's lock
before anything else, so the feed sees it on its next look (every ``period_s``) and writes
``stopped`` then, whatever the drive's phase says. Once ``stopped`` is written, nothing else is
for this action; a new action has its own feed and its own circle.

A MANUAL-DRIVE has a feed too (source ``guided:manual-drive``: a person drives it, so no viewer
may call it autonomous). Its circle is ``operator`` from the start: the gate opens to the operator
in the lease step, and a manual-drive takes no rest confirmation (rest before the operator's turn
is the operator's own eyes).

THE BATON, on ``target.baton`` (optional, `mapview/0`; CONTRACT.md):

* ``operation``, ``action``: the action's operation and its FleetAction's name, whole;
* ``gate`` and ``gateLog``: every teleop-gate change for this action as the gate recorded it
  (`TeleopGate.changes`, under the binding's lock): ``armed`` (open for the action, forwarding
  nothing), ``open`` (the operator's teleop reaches the base), ``closed``, each with its ``seq``,
  wall-clock ``stamp`` and ``reason``; ``gate`` is the latest, ``closed`` before the first;
* ``phase``: the drive's own phase (``Navigating``: Nav2 drives; ``Declined``: held, waiting for
  rest or the ready; ``Running``: the operator drives; ``Succeeded``; ``Failed``);
* ``turn``: ``startedAt`` (the feed made, in the step that grants the lease), ``endedAt`` (when
  the feed saw the drive end and the gate closed, within ``period_s``), ``outcome``
  (``succeeded``, ``failed`` or ``stopped``) and the drive's ``message``;
* ``circle``: the drive's last in-circle / still judgement (`ManualDrive.circle`): distance to
  the centre, radius, inside, still for how long against ``settle_s``; null until the drive has
  judged a pose (Nav2's half judges none: Nav2 says when it arrives).

A new ``Overlays`` is written whenever the baton changes, so each gate change and each tick the
feed samples (every ``period_s``) is in the feed with its time.

A FEED WATCHES; IT NEVER ACTS. It holds the job and one callable that answers "is the gate still
open for this action" (the binding's `teleop_open`, a read), and nothing else of the binding. It
runs on its own thread; no stop path calls it or waits for it, and nothing it does can raise into
the job or the adapter. ``Overlays`` is written again every ``repeat_s`` while the action runs, so
a viewer that joined late, or another writer's ``Overlays`` (which replaces the whole set), is
corrected within that time.
"""

from __future__ import annotations

import sys
import threading
import time

from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.mapview import contract as c

SOURCE = "navigate-handover"
MANUAL_SOURCE = f"guided:{md.OPERATION}"
ROBOT, OPERATOR, REACHED, STOPPED, DONE = c.TARGET_STATES
FINAL = (REACHED, STOPPED, DONE)
GATE_LOG_MAX = 16          # a turn's gate changes are a handful; the newest are kept
CLOSE_WAIT_S = 1.0         # after the drive ends, how long the feed waits to see the gate close


def circle_state(phase: str, *, halted: bool, current: bool, gate_open: bool,
                 last: bool) -> str:
    """The circle's state for a handover job (see the module docstring). In this order: a stop
    path halted the job, so whatever it would have reported is not said, and nor is its phase
    -- stopped; Succeeded -- reached, or done; Failed, or the gate no longer open for it, or the
    adapter no longer holding it -- stopped (a stop closes the gate before anything else);
    handed over -- operator; otherwise Nav2 drives, or the declined goal waits for rest, with
    nothing forwarded -- robot."""
    if halted:
        return STOPPED
    if phase == md.SUCCEEDED:
        return DONE if last else REACHED
    if phase == md.FAILED or not gate_open or not current:
        return STOPPED
    if phase == md.RUNNING:
        return OPERATOR
    return ROBOT


class HandoverFeed:
    """One navigate-handover's circle. See the module docstring."""

    def __init__(self, *, job, gate_open, sink, robot: str, frame: str, label: str,
                 last: bool = False, simulated: bool = False, clock=time.time,
                 period_s: float = 0.1, repeat_s: float = 5.0, operation: str = SOURCE,
                 action_id: str = "", gate_log=None) -> None:
        self._job, self._gate_open, self._sink = job, gate_open, sink
        self.operation = operation
        self.source = MANUAL_SOURCE if operation == md.OPERATION else operation
        self.action_id = action_id or label
        self.name = label                    # the FleetAction's name, whole (target.label: 32)
        self._gate_log = gate_log            # () -> the binding's gate changes, or None
        self._started = clock()
        self._ended: float | None = None
        self._outcome: str | None = None
        self._gates: list[dict] = []
        self._seen = 0
        self.robot, self.frame, self.label, self.last = robot, frame, label[:32], last
        self.simulated, self._clock = simulated, clock
        self.period_s, self.repeat_s = period_s, repeat_s
        x, y = job.params.target
        self._xy = (float(x), float(y))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.written: list[str] = []
        # A navigate-handover's circle starts at `robot`, and it is true here: the feed is made at
        # assignment, before the job has run, so no decline can have been seen. Written now, not
        # by the sampling thread, which may first run after the gate has turned. A manual-drive's
        # is the operator's from the start.
        self._write(OPERATOR if job.drive.phase == md.RUNNING else ROBOT)

    def start(self, alive) -> None:
        """Run until the circle's state is final. ``alive()``: the adapter still holds the job as
        its current action."""
        self._thread = threading.Thread(target=self._run, args=(alive,), daemon=True,
                                        name=f"mapview-{self.label}"[:60])
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the feed's own thread ---------------------------------------------------------------

    def _put(self, kind: str, **fields) -> None:
        try:
            self._sink.put(c.message(kind, self.robot, self.frame, self._clock(), **fields))
        except Exception as exc:  # noqa: BLE001 - a feed never raises into the adapter
            print(f"mapview: {kind} not emitted: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _read_gates(self) -> None:
        """Take this action's new gate changes from the binding's record."""
        if self._gate_log is None:
            return
        try:
            new = self._gate_log(self._seen)
        except Exception:  # noqa: BLE001 - a record that cannot be read adds nothing
            return
        for e in new:
            self._seen = max(self._seen, int(e.get("seq", 0)))
            if e.get("action") == self.action_id:
                self._gates.append({k: e[k] for k in ("seq", "stamp", "gate", "reason")})
        del self._gates[:-GATE_LOG_MAX]

    def baton(self) -> dict:
        drive = self._job.drive
        return {
            "operation": self.operation, "action": self.name,
            "gate": self._gates[-1]["gate"] if self._gates else "closed",
            "phase": drive.phase,
            "gateLog": [dict(e, reason=e["reason"][:c.MAX_TEXT]) for e in self._gates],
            "turn": {"startedAt": self._started, "endedAt": self._ended,
                     "outcome": self._outcome,
                     "message": (drive.message or "")[:c.MAX_TEXT]},
            "circle": dict(drive.circle) if drive.circle else None,
        }

    def _overlays(self, state: str) -> None:
        x, y = self._xy
        self._put("Overlays", keepouts=[], zones=[],
                  target={"label": self.label, "x": x, "y": y, "state": state,
                          "baton": self.baton()})

    def _status(self, phase: str, text: str) -> None:
        self._put("SessionStatus", source=self.source, phase=phase, text=text[:c.MAX_TEXT],
                  area_free_m2=None, simulated=self.simulated)

    def _state(self, alive) -> str:
        job = self._job
        phase = job.drive.phase                 # read first: a stop after it is seen below
        halted, current = job.halted.is_set(), bool(alive())
        try:
            gate_open = bool(self._gate_open())
        except Exception:  # noqa: BLE001 - a gate that cannot be read is not open
            gate_open = False
        return circle_state(phase, halted=halted, current=current, gate_open=gate_open,
                            last=self.last)

    def _write(self, state: str) -> None:
        if self.written and self.written[-1] == state:
            self._overlays(state)               # the repeat
            return
        if state == DONE and (not self.written or self.written[-1] != REACHED):
            self._write(REACHED)                # green, briefly, before done
        self.written.append(state)
        self._overlays(state)

    def _end_turn(self, state: str) -> None:
        """The turn ended: wait (at most CLOSE_WAIT_S) to see the gate close, then stamp it."""
        until = time.monotonic() + CLOSE_WAIT_S
        while time.monotonic() < until and not self._stop.is_set():
            try:
                if not self._gate_open():
                    break
            except Exception:  # noqa: BLE001 - a gate that cannot be read is not open
                break
            self._stop.wait(self.period_s / 4)
        self._read_gates()
        self._ended = self._clock()
        phase = self._job.drive.phase
        self._outcome = ("succeeded" if state in (REACHED, DONE) else
                         "failed" if phase == md.FAILED and not self._job.halted.is_set()
                         else "stopped")

    def _run(self, alive) -> None:
        try:
            self._status("started", self._job.drive.message)
            said, sent_at, state, shown = None, 0.0, None, None
            while not self._stop.is_set():
                state = self._state(alive)
                self._read_gates()
                if state in FINAL:
                    self._end_turn(state)
                now = time.monotonic()
                baton = self.baton()
                if (not self.written or state != self.written[-1] or baton != shown
                        or now - sent_at >= self.repeat_s):
                    self._write(state)
                    sent_at, shown = now, baton
                text = self._job.drive.message
                if text != said:
                    said = text
                    self._status("running", text)
                if state in FINAL:
                    break
                self._stop.wait(self.period_s)
            self._status("stopped", self._job.drive.message if state != STOPPED else
                         f"stopped: {self._job.drive.message}")
        except Exception as exc:  # noqa: BLE001 - the feed's boundary: never into the adapter
            print(f"mapview: the handover feed for {self.label} ended on "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
