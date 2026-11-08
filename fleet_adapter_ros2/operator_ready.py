# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The operator's ready: the person at the robot says, once, that they see it stopped.

For the navigate-handover with ``--handover-rest-source operator``. After Nav2 declines, the base
is held and the gate stays shut; the operator, standing at the robot as an external witness,
presses an explicit ready key once they see the robot stopped. The ready is one input among
several the binding checks under its stop lock (`hand_over_teleop`): the lidar scan still vetoes
it (any moving, any unknown), a stop of any scope or a lease end still wins, and a timeout never
opens the gate. The ready never answers an estop: `is_stopped` does not read it.

WHAT A READY IS. A press with no content. It carries no goal, no speed, no pose and no target:
the ROS message is ``std_msgs/msg/Empty``, which has no fields, and the subscription reads
nothing from it. It is attributed, on receipt, to the one handover this binding holds in its
Declined phase; it counts only if it arrived after that hold. A ready before the hold, after the
handover, a second one while the first waits to be judged, or one that names another action is
ignored and logged -- never queued for a later handover. A ready that the veto refuses is spent:
the operator presses again once the robot is stopped.

EACH LINE SAYS WHEN. Every line the latch says starts with the wall-clock time it was said, in
UTC to the millisecond, so a field log shows a press's order against the decline. A press ignored
because the handover is not waiting names what is missing -- the drive not Declined, the lease's
deadline passed, the gate closed -- as the binding's gate reads it (`TeleopGate.why_not_declined`).
The wording decides nothing: whether a press counts is still the gate's `declined` alone.

WHY IT ADDS NO AUTHORITY. Whoever can publish on the robot's graph can already publish the teleop
topic (and, past the adapter, the base's own velocity topic). The ready moves nothing by itself:
it only lets the operator's teleop through the same gate, under the same lease and caps, and only
after the hold, a full look of the scan with no veto, and no stop.

Pure: no ROS here except `RosReadySource`, which is handed its node and message type.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from fleet_adapter_ros2 import scan_rest

READY_TOPIC = "/handover_ready"
READY_TYPE = "std_msgs/msg/Empty"
READY_DEPTH = 1          # one press at a time; the latch, not a queue, decides what counts

OnReady = Callable[..., str]


def utc_ms(t: float) -> str:
    """``t`` (seconds since the epoch) as UTC to the millisecond: 2026-10-17T20:43:14.042Z."""
    ms = int(t * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ms // 1000)) + f".{ms % 1000:03d}Z"


class ReadyLatch:
    """The ready presses of one binding, judged on receipt. Not locked itself: the binding calls
    it under its stop lock, so a press, a take and the handover are serialised with every stop.

    ``arm`` when a handover's hold is commanded; ``press`` on receipt; ``take`` by the job, which
    then asks the veto; ``confirm`` inside the binding's `hand_over_teleop`, which spends it."""

    def __init__(self, log: Callable[[str], None] | None = None,
                 wall: Callable[[], float] = time.time) -> None:
        self._armed: tuple[str, float] | None = None      # (action, hold time)
        self._pending: tuple[str, float] | None = None    # a press waiting to be taken
        self._taken: tuple[str, float] | None = None      # taken, waiting for the handover
        self._log = log or (lambda line: print(line, file=sys.stderr))
        self._wall = wall    # for the line's time only; every judgement is on the binding's clock
        self.lines: list[str] = []

    def _say(self, line: str) -> None:
        line = f"{utc_ms(self._wall())} navigate-handover ready: {line}"
        self.lines.append(line)
        self._log(line)

    def arm(self, action_id: str, hold_at: float) -> None:
        """A handover's hold was commanded at ``hold_at``: only a ready after it, for this
        action, counts. Anything still waiting from before is dropped."""
        if self._pending is not None or self._taken is not None:
            self._say(f"dropped a ready for {(self._pending or self._taken)[0]!r}: a new hold "
                      f"for {action_id!r}")
        self._armed, self._pending, self._taken = (action_id, hold_at), None, None

    def press(self, recv: float, waiting: Callable[[str], bool],
              action_id: str | None = None,
              why_not: Callable[[str], str] | None = None) -> str:
        """One press received at ``recv``. "" when it is kept for the held handover; otherwise
        why it is ignored (and logged). ``waiting(action)``: whether that action's handover is in
        Declined now, its gate open and not past its lease (the binding's gate answers).
        ``why_not(action)``, when given, words what ``waiting`` found missing; it decides
        nothing."""
        why = ""
        if self._armed is None:
            why = "no declined handover is held: a ready before the hold is not kept"
        else:
            aid, hold_at = self._armed
            if action_id is not None and action_id != aid:
                why = f"it names another action ({action_id!r}, the held one is {aid!r})"
            elif not waiting(aid):
                why = (f"the handover of {aid!r} is not waiting for one (not declined yet, "
                       f"handed over, ended or stopped)")
                missing = why_not(aid) if why_not is not None else ""
                if missing:
                    why += f": {missing}"
            elif not recv > hold_at:
                why = f"it arrived before the hold of {aid!r}"
            elif self._pending is not None:
                why = f"a ready for {aid!r} is already waiting to be judged; a second is not queued"
        if why:
            self._say(f"ignored: {why}")
            return why
        self._pending = (aid, recv)
        self._say(f"kept for {aid!r}, {recv - hold_at:.1f} s after the hold")
        return ""

    def take(self, action_id: str, waiting: Callable[[str], bool]) -> float | None:
        """The job takes the press kept for ``action_id``: its receipt time, or None. A press for
        another action, or for a handover no longer waiting, is dropped here. Taking replaces
        whatever was taken before: a ready the veto refused is spent by the next take."""
        p, self._pending, self._taken = self._pending, None, None
        if p is None:
            return None
        if p[0] != action_id or self._armed is None or self._armed[0] != action_id \
                or not waiting(action_id):
            self._say(f"dropped the ready for {p[0]!r}: it is not {action_id!r}'s, or that "
                      f"handover no longer waits")
            return None
        self._taken = p
        return p[1]

    def confirm(self, action_id: str) -> bool:
        """Inside the binding's handover, under its lock: whether a ready for ``action_id``,
        received after its hold, was taken. Spent either way."""
        p, self._taken = self._taken, None
        return (p is not None and self._armed is not None and p[0] == action_id
                and self._armed[0] == action_id and p[1] > self._armed[1])

    def hold_at(self, action_id: str) -> float | None:
        return self._armed[1] if self._armed and self._armed[0] == action_id else None


def look_gap(now: float, hold_at: float | None) -> str:
    """Why the scan has not yet had its full look since the hold (LAG_S + WINDOW_S), or ""."""
    look = scan_rest.LAG_S + scan_rest.WINDOW_S
    if hold_at is None:
        return "the base was not commanded to hold"
    if now - hold_at < look:
        return (f"too early: {max(now - hold_at, 0.0):.1f} s since the hold, under the scan's "
                f"full look of {look:g} s")
    return ""


def no_veto(channel) -> str:
    """The scan's answer when it vetoes nothing, in numbers."""
    n, smax = channel.window_stats()
    s = f"S max {smax:.4f} m" if smax is not None else "no S"
    return f"scan: no veto, {s} < T {channel.threshold_m:.4f} m over {n} scans"


class ReadySource:
    """Where the operator's ready comes from. ``attach(on_ready)``: each press calls
    ``on_ready(action_id=None)`` once; None means "the handover this binding holds"."""

    def attach(self, on_ready: OnReady) -> None:
        raise NotImplementedError


class DeskReadySource(ReadySource):
    """For tests and a desk run: ``press()`` is one press."""

    def __init__(self) -> None:
        self._on_ready: OnReady | None = None

    def attach(self, on_ready: OnReady) -> None:
        self._on_ready = on_ready

    def press(self, action_id: str | None = None) -> str:
        if self._on_ready is None:
            raise RuntimeError("the desk ready source is not attached")
        return self._on_ready(action_id) if action_id is not None else self._on_ready()


class RosReadySource(ReadySource):
    """One subscription to ``topic`` (``std_msgs/msg/Empty``, depth 1). The message is not
    read: a press is its arrival."""

    def __init__(self, node, topic: str, msg_type) -> None:
        self._node, self._topic, self._type = node, topic, msg_type

    def attach(self, on_ready: OnReady) -> None:
        self._node.create_subscription(self._type, self._topic, lambda _msg: on_ready(),
                                       READY_DEPTH)
