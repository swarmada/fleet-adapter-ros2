# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`Custom` navigate-handover: Nav2 drives to the target, and when it declines, the operator does,
under the same action and the same lease.

One action of type `Custom` (`customType: navigate-handover`), named in the payload as
``operation: navigate-handover``, with the manual-drive parameters. The adapter sends the target
to Nav2 as a Navigate does. If Nav2 reaches it, the action is Succeeded and the operator was never
involved. If Nav2 declines it -- the server rejects or aborts the goal, or the progress watchdog
stops the robot where it stands -- the base is held and the action stays running: it is now the
operator's turn, and the operator's teleop reaches the base exactly as for a manual-drive. It ends
Succeeded once the robot is inside the target's circle and still, whoever drove it there, so a
FleetTask's next member starts only when the robot is at this one's target.

The FleetTask that runs these is a plain `sequence`. Its select cannot branch on a decline: a robot
action has no `status.result`, and a Failed member fails the select (RFC-0001
#fleettask-structured-control-flow-spec-flow). Only the adapter can tell a decline from a stop,
so the branch is here.

THE HANDOVER IS THE GATE'S. The teleop gate is opened in the step that grants the lease, armed
for this action and forwarding nothing while Nav2 drives. A decline turns it to the operator
through the binding's `hand_over_teleop`, under the lock every stop takes; a stop closed the gate,
so a stop that lands between the decline and the handover wins, and so does a lease deadline that
has passed. Then the action ends Failed. A teleop message never opens or turns the gate.

ONLY AT REST. On a decline the base is held first (`hold_for_handover`), and the gate stays armed
and forwards nothing until the binding confirms the robot at rest: `rest_evidence_gap` judged
first, then `is_stopped`, the estop's own test. The binding checks both again inside
`hand_over_teleop`, under its lock, so the gate never turns on a rest seen a moment before. Not
confirmed within ``rest_s`` (the adapter's bound; by default the estop's confirmed-stop budget):
the action ends Failed and says so; a timeout never opens the gate. A binding that cannot confirm
rest at all -- odometry declared command-derived, or not declared -- never hands over: the action
ends Failed at the decline, with the binding's own sentence.

WHICH REST. A binding may confirm the handover's rest from a channel of its own, the lidar's
(`scan_rest`, ``--handover-rest-source scan``): then it answers `handover_rest_gap` and
`handover_at_rest` from that channel, and the job asks those. A binding without them is asked
`rest_evidence_gap` and `is_stopped`, as before. Either way the estop's own answer is untouched:
`is_stopped` and `rest_evidence_gap` are never answered from the scan.

OPERATOR-CONFIRMED REST (``--handover-rest-source operator``). The binding may instead take the
handover's rest from the operator, an external witness at the robot, with its scan channel as a
veto only (`handover_confirms_by_operator`). Then, after the hold, the job waits for the operator's
ready (`operator_ready`): it takes one ready the binding kept for this action, asks the binding's
`handover_at_rest` -- a full look of the scan since the hold, and no veto: any moving, any unknown,
or motion on any other channel refuses it -- and only then `hand_over_teleop`, which checks the
ready, the look and the veto again under the stop lock. A ready the veto refuses is spent and the
base stays held; the operator presses again. No ready within ``ready_s``: Failed, and the gate
never opens. The scan alone never opens it. The estop's answer reads neither.

THE TURN'S TWO LINES. When the gate turns to the operator the job prints, once,
``handover turn-open action=<action> turn=<n> ts=<UTC ms>``, and when that turn ends, once,
``handover turn-closed action=<action> reason=<completed|timeout|failed|stopped> ts=<UTC ms>``,
for a script that greps the adapter's log for the prefix ``handover turn-``. ``turn`` counts the
operator's turns this adapter has opened since it started: the adapter is never told which
member of a FleetTask an action is. ``completed``: in the circle and still; ``timeout``: the
turn's ``timeout_s``; ``stopped``: a stop closed the gate, or the job was halted (a lease's end,
a cancel, a superseding assignment); ``failed``: any other end. No address is in either line.

Pure: no ROS, no protobuf, as `manual_drive`.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import threading
import time

from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import operator_ready as opr
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.explore import Param
from fleet_adapter_ros2.robot import SimulatedRobot

OPERATION = "navigate-handover"

PARAMS: tuple[Param, ...] = (
    Param("operation", "enum", required=True, allowed=(OPERATION,)), *md.PARAMS[1:])

DESCRIPTION = (
    "navigate-handover: Nav2 drives to (x, y) in map; if it declines, the base is held and the "
    "operator drives by teleop to the circle of radius_m, under the same lease and caps as "
    "manual-drive, once the robot is confirmed at rest. Succeeded once the robot is in the "
    "circle (Nav2) or in it and still for settle_s (operator); Failed on a stop, rest not "
    "confirmed, timeout_s of the operator's turn, or a pose not in map. last: the FleetTask's "
    "last target, so the map view says done once it is reached.")

# The default bound on the wait for rest after a decline: the estop's confirmed-stop budget
# (the adapter's `_ESTOP_TICK_S * _ESTOP_TICKS`), so a handover and a stop on the same robot never
# answer "at rest" on different evidence or in a different time.
REST_S = 2.0

# The default bound on the wait for the operator's ready after a decline, with the operator source.
# Long enough for the person at the robot to see it stopped and press once, with the scan's full
# look (LAG_S + WINDOW_S) inside it many times over; short enough that a held handover nobody
# confirms ends, and says so, instead of holding the robot and its lease. The adapter serves
# READY_S_MIN to READY_S_MAX.
READY_S = 30.0
READY_S_MIN, READY_S_MAX = 5.0, 120.0
CONFIRMED = "operator-confirmed rest"

# The turn's two lines (see the module docstring), and the count of turns opened in this process.
TURN_PREFIX = "handover turn-"
_TURNS = itertools.count(1)


def parse_payload(payload) -> md.Params:
    return md.parse_payload(payload, OPERATION)


def refusal(payload, teleop_topic: str | None, hands_over: bool) -> str:
    """Why a navigate-handover is refused, or "". ``hands_over``: the binding can turn its gate to
    the operator (`hand_over_teleop`); without that, a decline could only end the action."""
    if teleop_topic and not hands_over:
        return (f"{OPERATION} is not served: this binding cannot hand a declined goal to the "
                f"operator")
    return md.refusal(payload, teleop_topic, OPERATION)


def is_navigate_handover(action_type: str, payload) -> bool:
    """Whether an action is a navigate-handover: `Custom` with ``operation: navigate-handover``."""
    if action_type != "Custom":
        return False
    try:
        doc = payload if isinstance(payload, dict) else json.loads(payload or b"{}")
    except (ValueError, TypeError, UnicodeDecodeError):
        return False
    return isinstance(doc, dict) and doc.get("operation") == OPERATION


class NavigateHandoverJob(md.ManualDriveJob):
    """One navigate-handover, on its own thread. A `ManualDriveJob` to the adapter, so every stop
    path, the renewal and the resume treat it as one; it starts Navigating instead of Running.

    The adapter sends the Nav2 goal and opens the armed gate in the lease step. The job watches
    the goal; on a decline it asks the binding to hand over, and from then on it is the
    manual-drive loop. On any end it closes the gate -- one zero Twist -- before it reports."""

    def __init__(self, *, action_id: str, params: md.Params, robot, robot_id: str, emit,
                 clock, rest_s: float = REST_S, ready_s: float = READY_S) -> None:
        super().__init__(action_id=action_id, params=params, robot=robot, robot_id=robot_id,
                         emit=emit, clock=clock)
        self.drive = md.ManualDrive(params, started_at=clock.now(), phase=md.NAVIGATING,
                                    message="navigating: nav2 drives to the target")
        self.rest_s = rest_s
        self.ready_s = ready_s
        self._why = ""
        self._rest_by = 0.0
        self._by_operator = False
        self._ready_by = 0.0
        self._rest_said = ""     # why rest was not confirmed, as the binding last said
        self._turn_open = False  # the operator's turn is open: its closing line is still owed
        self._closing = ""       # why it closes, when the drive's end alone cannot say
        self._wall = time.time   # the lines' time only
        self._log = lambda line: print(line, file=sys.stderr, flush=True)

    def _opened(self) -> None:
        """The gate has just turned to the operator: say so, once."""
        self._turn_open = True
        self._log(f"{TURN_PREFIX}open action={self.action_id} turn={next(_TURNS)} "
                  f"ts={opr.utc_ms(self._wall())}")

    def _closed(self, reason: str) -> None:
        """The operator's turn has ended: say so, once, and only for a turn that opened."""
        if not self._turn_open:
            return
        self._turn_open = False
        self._log(f"{TURN_PREFIX}closed action={self.action_id} reason={reason} "
                  f"ts={opr.utc_ms(self._wall())}")

    def run(self) -> None:
        try:
            self._run()
        finally:
            self._closed("stopped")   # halted with the turn open: a lease's end, cancel, supersession

    def _run(self) -> None:
        r, rid, drive = self._robot, self._rid, self.drive
        while not self.halted.is_set():
            t = self._clock.now()
            if not r.teleop_open(rid, self.action_id):
                self._closing = "stopped"
                drive.abort(t, "a stop closed the gate; the robot was stopped and teleop no "
                               "longer reaches it")
            elif drive.phase == md.NAVIGATING:
                state = r.task_state(rid)
                if state == task_map.TASK_SUCCEEDED:
                    drive.arrived(t, "nav2 reached the target; the operator was not needed")
                elif state == task_map.TASK_FAILED:
                    self._declined(t)
            elif drive.phase == md.DECLINED:
                if self._by_operator:
                    self._wait_for_ready(t)
                else:
                    self._wait_for_rest(t)
            else:
                drive.tick(t)
                if drive.ended:
                    self._closing = "timeout"
                if drive.forwarding:
                    x, y, _ = r.pose(rid)
                    drive.on_pose(t, x, y, r.pose_frame(rid))
                if drive.forwarding:
                    r.teleop_deadman(rid, self.params.deadman_s)
            if drive.ended:
                r.close_teleop(rid, self.action_id)
                ok = drive.phase == md.SUCCEEDED
                self._closed("completed" if ok else self._closing or "failed")
                self._emit(task_map.TASK_SUCCEEDED if ok else task_map.TASK_FAILED,
                           drive.message, 100 if ok else 0)
                return
            self._clock.wait(self.halted, self.POLL_S)

    def _declined(self, t: float) -> None:
        """Nav2 declined: hold the base with the gate still armed, then wait for rest -- or end
        here, when a stop closed the gate first or the binding cannot confirm rest at all."""
        r, rid, drive = self._robot, self._rid, self.drive
        why = (r.task_failure(rid) if hasattr(r, "task_failure") else "") or "declined"
        self._why = why
        if not r.hold_for_handover(rid, self.action_id):
            drive.abort(t, f"nav2 declined ({why}) and the handover was refused: a stop closed "
                           f"the gate first")
            return
        gap = handover_rest_gap(r, rid)
        if gap:
            drive.abort(t, f"nav2 declined ({why}), and the operator's turn cannot open: rest "
                           f"cannot be confirmed on this robot ({gap}). The base is held")
            return
        drive.decline(t, why)
        self._by_operator = handover_by_operator(r, rid)
        if self._by_operator:
            self._ready_by = t + self.ready_s
            self._say(t, f"nav2 declined ({why}); the base is held. {CONFIRMED}: press ready once "
                         f"you see the robot stopped, within {self.ready_s:g} s; the scan only "
                         f"vetoes")
            return
        self._rest_by = t + self.rest_s
        self._emit(task_map.TASK_RUNNING, drive.message, 0)

    def _say(self, t: float, message: str) -> None:
        """A running message while the base is held: the drive's, and the action's status."""
        self.drive.message = message
        self.drive.events.append((t, message))
        self._emit(task_map.TASK_RUNNING, message, 0)

    def _wait_for_ready(self, t: float) -> None:
        """The operator source: turn the gate only on a ready the binding kept for this action,
        with the scan's full look and no veto; end Failed with none within ``ready_s``. A ready
        the veto refuses is spent, and the base stays held."""
        r, rid, drive = self._robot, self._rid, self.drive
        if r.take_ready(rid, self.action_id) is None:
            if t >= self._ready_by:
                drive.abort(t, f"nav2 declined ({self._why}), and no operator confirmation "
                               f"within {self.ready_s:g} s; the operator's turn never opened "
                               f"and the base is held")
            return
        at_rest, said = handover_at_rest(r, rid)
        if not at_rest:
            self._say(t, f"ready refused ({said}); the base is held: press ready again once "
                         f"you see the robot stopped")
            return
        if not r.hand_over_teleop(rid, self.action_id, t, self._why):
            drive.abort(t, f"nav2 declined ({self._why}) and the handover was refused at the "
                           f"ready: a stop, the lease's end, or the scan's veto under the lock "
                           f"closed the gate first")
            return
        self._opened()
        x, y = self.params.target
        self._say(t, f"operator's turn ({CONFIRMED}; {said}): nav2 declined ({self._why}); "
                     f"drive to ({x:.2f}, {y:.2f}) within {self.params.radius_m} m")

    def _wait_for_rest(self, t: float) -> None:
        """Turn the gate once the binding confirms rest; end Failed if it does not within
        ``rest_s``. The binding checks rest again under its lock; a refusal there ends it."""
        r, rid, drive = self._robot, self._rid, self.drive
        at_rest, self._rest_said = handover_at_rest(r, rid)
        if at_rest:
            if r.hand_over_teleop(rid, self.action_id, t, self._why):
                self._opened()
                self._emit(task_map.TASK_RUNNING, drive.message, 0)
            else:
                drive.abort(t, f"nav2 declined ({self._why}) and the handover was refused: a "
                               f"stop, the lease's end, or motion at the handover closed the "
                               f"gate first")
        elif t >= self._rest_by:
            drive.abort(t, f"nav2 declined ({self._why}), and the robot was not confirmed at "
                           f"rest within {self.rest_s:g} s"
                           + (f" ({self._rest_said})" if self._rest_said else "")
                           + "; the operator's turn never opened and the base is held")


def handover_rest_gap(robot, robot_id: str) -> str:
    """Why the binding can never confirm the handover's rest, or "": its own answer when it has a
    handover rest channel, else the estop's (`rest_evidence_gap`)."""
    fn = getattr(robot, "handover_rest_gap", None)
    return fn(robot_id) if fn is not None else robot.rest_evidence_gap(robot_id)


def handover_by_operator(robot, robot_id: str) -> bool:
    """Whether the binding takes the handover's rest from the operator's ready, the scan a veto."""
    fn = getattr(robot, "handover_confirms_by_operator", None)
    return bool(fn(robot_id)) if fn is not None else False


def handover_at_rest(robot, robot_id: str) -> tuple[bool, str]:
    """(confirmed at rest for the handover, why not): the binding's handover rest channel when it
    has one, else the estop's test (`is_stopped`)."""
    fn = getattr(robot, "handover_at_rest", None)
    if fn is not None:
        return fn(robot_id)
    return (True, "") if robot.is_stopped(robot_id) else (False, "")


class SimulatedHandoverRobot(md.SimulatedTeleopRobot):
    """The stand-in for desk and kind runs: the teleop stand-in whose Nav2 can be told to decline.

    ``decline_next(reason)`` makes the next goal end FAILED at once, where the robot stands, with
    ``reason`` as Nav2's; ``on_read`` runs the first time that FAILED is read, so a test can land a
    stop between the decline and the handover. ``stall_next()`` makes the next goal never arrive;
    ``decline_now(reason)`` then ends it FAILED, as Nav2's late answer would.

    For a run with no test driving it (`--binding simulated --teleop-topic`):
    ``decline_beyond_m`` declines every goal farther than that from the robot, and
    ``pulse_s`` makes each teleop command the gate forwards move the pose by its speed times
    ``pulse_s``, so only what the gate lets through moves the stand-in. `follow_operator_file`
    reads the operator's teleop from a file, one ``vx vy wz`` line per key press. ``coast_s``
    keeps the base reporting motion for that long after the handover's hold, as a braking base
    would, so the wait for rest is seen in a run.

    How it proves rest: `is_stopped` reads the stand-in's own plant model (`SimulatedRobot`:
    no travel left and the last command zero), plus ``coast_s``. That is the model's state, not
    a measurement of a base, so a desk or kind run proves the wait's logic -- the gate stays
    armed until the binding says "at rest", and a stop or the lease's end wins in the wait --
    and not that any real base stopped."""

    def __init__(self, *, decline_beyond_m: float | None = None, pulse_s: float = 0.0,
                 coast_s: float = 0.0, rest_channel=None, operator_ready: bool = False,
                 **kw) -> None:
        if operator_ready and rest_channel is None:
            raise ValueError("operator_ready needs a rest_channel: the veto needs the scan")
        super().__init__(**kw)
        # The operator's ready (`operator_ready`), with the scan a veto only; None: no ready.
        self._ready = opr.ReadyLatch() if operator_ready else None
        # The handover's rest from scans (`scan_rest.RestChannel`), fed by `feed_scan`; None: the
        # plant model's `is_stopped`, as the estop. The stand-in has no lidar of its own: a test
        # feeds it scans cast from a plant's true pose.
        self._rest_channel = rest_channel
        self._hold_at: dict[str, float] = {}
        self._decline_beyond_m = decline_beyond_m
        self._pulse_s = pulse_s
        self._coast_s = coast_s
        self._coast_until: dict[str, float] = {}
        self._decline: str | None = None
        self._stall = False
        self._failure = ""
        self._on_read = None

    def decline_next(self, reason: str, on_read=None) -> None:
        self._decline, self._on_read = reason, on_read

    def stall_next(self) -> None:
        self._stall = True

    def decline_now(self, reason: str, robot_id: str = "") -> None:
        rid = robot_id or next(iter(self._moving), "")
        with self._clock_lock:
            self._arrive_at.pop(rid, None)
            self._moving[rid] = False
            self._goal_state[rid] = task_map.TASK_FAILED
            self._failure = reason

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        if self._decline is None and self._decline_beyond_m is not None:
            with self._clock_lock:
                fx, fy = self._pos.get(robot_id, (0.0, 0.0))
            if math.hypot(x - fx, y - fy) > self._decline_beyond_m:
                self._decline = (f"simulated: the goal is more than {self._decline_beyond_m:g} m "
                                 f"away, so nav2 declines it")
        if self._decline is not None:
            reason, self._decline = self._decline, None
            with self._clock_lock:
                self._goal_state[robot_id] = task_map.TASK_FAILED
                self._moving[robot_id] = False
                self._failure = reason
            return
        with self._clock_lock:
            before = self._pos.get(robot_id, (0.0, 0.0))
        super().command_move(robot_id, x, y, yaw)
        if self._stall:
            self._stall = False
            with self._clock_lock:
                self._arrive_at.pop(robot_id, None)
                self._pos[robot_id] = before

    def task_state(self, robot_id: str) -> str:
        state = super().task_state(robot_id)
        if state == task_map.TASK_FAILED and self._on_read is not None:
            cb, self._on_read = self._on_read, None
            cb()
        return state

    def task_failure(self, robot_id: str) -> str:
        return self._failure

    def _publish(self, robot_id: str, cmd) -> None:
        """Caller holds `_teleop_lock`. A forwarded command moves the pose, with ``pulse_s``."""
        super()._publish(robot_id, cmd)
        if self._pulse_s > 0 and tuple(cmd) != md.ZERO:
            with self._clock_lock:
                x, y = self._pos.get(robot_id, (0.0, 0.0))
                self._pos[robot_id] = (x + cmd[0] * self._pulse_s, y + cmd[1] * self._pulse_s)

    def is_stopped(self, robot_id: str) -> bool:
        with self._clock_lock:
            coasting = self._clock() < self._coast_until.get(robot_id, -math.inf)
        return not coasting and super().is_stopped(robot_id)

    def hold_for_handover(self, robot_id: str, action_id: str) -> bool:
        """Nav2 declined: hold the base, with one zero Twist, and keep the gate armed -- or
        refuse, when a stop closed it first. Under the lock the stops close it under."""
        with self._teleop_lock:
            if not self._gate.is_open(action_id):
                return False
            self._publish(robot_id, md.ZERO)
            SimulatedRobot.command_hold(self, robot_id)
            with self._clock_lock:
                self._hold_at[robot_id] = self._clock()
            if self._ready is not None:
                self._ready.arm(action_id, self._hold_at[robot_id])
            if self._coast_s > 0:
                with self._clock_lock:
                    self._coast_until[robot_id] = self._clock() + self._coast_s
        return True

    def hand_over_teleop(self, robot_id: str, action_id: str, t: float, why: str) -> bool:
        """Turn the armed gate to the operator, or refuse (`TeleopGate.hand_over`), under the lock
        the stops close it under, and only with the robot confirmed at rest under that lock.
        Handed over: the base is held, with one zero Twist."""
        with self._teleop_lock:
            if self._ready is not None and not self._ready.confirm(action_id):
                return False
            if self.handover_rest_gap(robot_id) or not self.handover_at_rest(robot_id)[0]:
                return False
            if not self._gate.hand_over(action_id, t, why):
                return False
            self._publish(robot_id, md.ZERO)
            SimulatedRobot.command_hold(self, robot_id)   # the base's hold; the gate stays open
        return True


    def feed_scan(self, scan, recv: float) -> None:
        """One scan for the handover's rest channel, received at ``recv`` on this robot's clock."""
        self._rest_channel.add(scan, recv)

    def handover_rest_gap(self, robot_id: str) -> str:
        return "" if self._rest_channel is not None else self.rest_evidence_gap(robot_id)

    def handover_at_rest(self, robot_id: str) -> tuple[bool, str]:
        """The scan channel's verdict, when there is one, vetoed by the plant model's motion
        (refute wins); else the plant model's `is_stopped`, as the estop's. With the operator's
        ready, the scan must also have had its full look since the hold, and at rest it only
        says it has no veto: the ready is what confirms."""
        if self._rest_channel is None:
            return (True, "") if self.is_stopped(robot_id) else (False, "")
        with self._clock_lock:
            now, hold = self._clock(), self._hold_at.get(robot_id)
        if self._ready is not None:
            early = opr.look_gap(now, hold)
            if early:
                return False, early
        v = self._rest_channel.verdict(now, hold)
        if not v.at_rest:
            return False, f"scan: {v}"
        if not self.is_stopped(robot_id):
            return False, "the stand-in's plant model shows motion"
        if self._ready is not None:
            return True, opr.no_veto(self._rest_channel)
        return True, ""

    # ── the operator's ready ──────────────────────────────────────────────

    def handover_confirms_by_operator(self, robot_id: str) -> bool:
        return self._ready is not None

    def on_ready(self, action_id: str | None = None) -> str:
        """One press of the operator's ready, received now: "" when it is kept for the held
        handover, otherwise why it is ignored (logged)."""
        if self._ready is None:
            return "this robot takes no operator's ready"
        with self._teleop_lock:
            with self._clock_lock:
                now = self._clock()
            return self._ready.press(now, self._gate.declined, action_id,
                                     self._gate.why_not_declined)

    def take_ready(self, robot_id: str, action_id: str) -> float | None:
        if self._ready is None:
            return None
        with self._teleop_lock:
            return self._ready.take(action_id, self._gate.declined)

    def ready_log(self) -> list[str]:
        return list(self._ready.lines) if self._ready is not None else []


def follow_operator_file(robot: SimulatedHandoverRobot, path: str, robot_id: str,
                         stop: threading.Event, poll_s: float = 0.1) -> None:
    """Read the operator's teleop for the stand-in: each line appended to ``path`` is one message
    on the teleop topic, ``vx vy wz``. Lines already in the file when it starts are skipped. The
    gate decides what reaches the base, exactly as for a real teleop source; a malformed line is
    reported and dropped."""
    import sys
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0, 2)
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            try:
                vx, vy, wz = (float(v) for v in line.split())
            except ValueError:
                print(f"operator file: {line.strip()!r} is not 'vx vy wz'; dropped",
                      file=sys.stderr)
                continue
            robot.teleop(vx, vy, wz, robot_id)
