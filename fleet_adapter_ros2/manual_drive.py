# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`Custom` manual-drive: the operator drives the robot by teleop, under the action's lease.

When a Navigate fails and the robot stops where it stands, the operator can take over: a
FleetAction of type `Custom` (`customType: manual-drive`) whose payload names the target the
Navigate did not reach. While it runs, the operator's teleop commands -- published on their own
topic, never on `/cmd_vel` -- are forwarded to the base, clamped to the action's caps. It ends
Succeeded once the robot is inside the target's circle and still; Failed on its time-out, on a
pose that is not in `map`, or when any stop path closed the forwarding.

As for `explore-floor`, `customType` never reaches the adapter: the verb is named in the payload,
as ``operation: manual-drive``.

THE GATE. One object, :class:`TeleopGate`, decides whether a teleop command reaches the base. It
lives in the binding, and the binding calls it under the same lock as its own stop, so a
forwarded command and a stop are never interleaved: once a stop has closed the gate, nothing
forwarded can follow the stop's zero Twist. The gate is opened only by the adapter, in the step
that grants the action's lease, and never by a teleop message. A command is forwarded only while
the gate is open for the running action, the lease's deadline has not passed, and the drive is
Running; otherwise nothing is published -- not a zero, which would fight the next Navigate.

THE GATE'S RECORD. Each change the gate makes -- ``armed`` (open for the action, forwarding
nothing: Nav2 drives, or a declined goal waits for rest), ``open`` (the operator's teleop reaches
the base) and ``closed`` -- is recorded by the gate as it makes it, under the binding's lock, with
the action, the reason and the wall-clock time (`TeleopGate.changes`). A map view reads it
(`mapview/handover_feed.py`); nothing waits for it and nothing acts on it.

Pure: no ROS, no protobuf. The job reads the robot through the binding and reports through a
callback, so everything here runs against the stand-in binding and a fake clock.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.explore import Param
from fleet_adapter_ros2.robot import SimulatedRobot

OPERATION = "manual-drive"
RUNNING, SUCCEEDED, FAILED = "Running", "Succeeded", "Failed"
# A navigate-handover's drive while Nav2 drives: the gate is armed for the action and forwards
# nothing; only the binding's `hand_over_teleop` turns it to Running (see `handover`).
NAVIGATING = "Navigating"
# ... and once Nav2 has declined: the base is held, and the gate still forwards nothing until the
# binding confirms the robot at rest. Only then does `hand_over_teleop` turn it to Running.
DECLINED = "Declined"
ZERO = (0.0, 0.0, 0.0)                     # (vx, vy, wz)
# Why the base's own stops closed the gate, as its record says (`TeleopGate.changes`).
STOP_WHY = "a stop closed it (an estop, a lease's end, a cancel or a shutdown); the base is held"
PAUSE_WHY = "a pause closed it; the base is held"

# Hard ceilings this action never exceeds, whatever the payload asks (the robot's ruled cap is
# 0.20 m/s).
CAP_LINEAR_MPS = 0.20
CAP_ANGULAR_RPS = 1.0
# The one frame a target is given in, and the one a pose must be read in: the frame Navigate
# goals are sent in. A pose in another frame (no map -> odom transform) cannot be judged.
FRAME = "map"
# The topics a teleop source must never be: forwarding is the gate, so teleop on the base's own
# topic would bypass it.
BASE_TOPICS = ("/cmd_vel", "cmd_vel")
REQUIRED = ("operation", "x", "y", "frame", "radius_m")

# What the catalog declares for the operation, and what `refusal` enforces with the same names.
PARAMS: tuple[Param, ...] = (
    Param("operation", "enum", required=True, allowed=(OPERATION,)),
    Param("x", "number", "m", required=True),
    Param("y", "number", "m", required=True),
    Param("frame", "enum", required=True, allowed=(FRAME,)),
    Param("radius_m", "number", "m", required=True, min=0.05, max=2.0),
    Param("settle_s", "number", "s", min=0.0, max=30.0),
    Param("max_speed_mps", "number", "m/s", max=CAP_LINEAR_MPS),
    Param("max_turn_rps", "number", "rad/s", max=CAP_ANGULAR_RPS),
    Param("timeout_s", "number", "s", min=1.0, max=3600.0),
    Param("still_mps", "number", "m/s", min=0.001, max=0.2),
    Param("deadman_s", "number", "s", min=0.1, max=2.0),
    Param("teleop_topic", "string"),
    Param("follows", "string"),
    Param("last", "bool"),
)

DESCRIPTION = (
    "manual-drive: the operator drives the robot by teleop to the circle of radius_m around "
    "(x, y) in map. Teleop is forwarded to the base only while this action holds its lease, "
    "clamped to max_speed_mps and max_turn_rps; any stop ends it. Succeeded once the robot is "
    "in the circle and still for settle_s; Failed on timeout_s, a pose not in map, or a stop.")


class PayloadError(ValueError):
    pass


@dataclass(frozen=True)
class Params:
    target: tuple[float, float]
    frame: str
    radius_m: float
    settle_s: float = 1.0
    max_speed_mps: float = CAP_LINEAR_MPS
    max_turn_rps: float = 0.7
    timeout_s: float = 300.0
    still_mps: float = 0.02                # pose speed below this counts as still
    deadman_s: float = 0.5                 # teleop silence after motion -> one zero Twist
    teleop_topic: str | None = None
    follows: str | None = None
    last: bool = False                     # the FleetTask's last target: the map view says done


def parse_payload(payload, operation: str = OPERATION) -> Params:
    """An AssignAction's ``payload_json`` (bytes, str or an already-decoded dict) -> Params,
    refusing anything unsafe or incomplete. ``operation`` is the verb the payload must name:
    a navigate-handover takes the same parameters under its own name."""
    if isinstance(payload, dict):
        p = payload
    else:
        try:
            p = json.loads(payload or b"{}")
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise PayloadError(f"payload is not JSON: {e}") from None
    if not isinstance(p, dict):
        raise PayloadError("payload must be a JSON object")
    for k in REQUIRED:
        if k not in p:
            raise PayloadError(f"payload: {k} missing; {operation} requires it")
    if p["operation"] != operation:
        raise PayloadError(f"operation {p['operation']!r}, want {operation!r}")
    if p["frame"] != FRAME:
        raise PayloadError(f"frame {p['frame']!r} is not served: the target is given, and the "
                           f"pose read, in the {FRAME!r} frame only")

    def num(k, d):
        v = p.get(k, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise PayloadError(f"payload {k} is not a number")
        return float(v)

    topic = p.get("teleop_topic")
    if topic is not None and not isinstance(topic, str):
        raise PayloadError("payload teleop_topic is not a string")
    follows = p.get("follows")
    if follows is not None and not isinstance(follows, str):
        raise PayloadError("payload follows is not a string")
    last = p.get("last", False)
    if not isinstance(last, bool):
        raise PayloadError("payload last is not true or false")
    prm = Params(target=(num("x", 0), num("y", 0)), frame=FRAME, radius_m=num("radius_m", 0),
                 settle_s=num("settle_s", 1.0), max_speed_mps=num("max_speed_mps", CAP_LINEAR_MPS),
                 max_turn_rps=num("max_turn_rps", 0.7), timeout_s=num("timeout_s", 300.0),
                 still_mps=num("still_mps", 0.02), deadman_s=num("deadman_s", 0.5),
                 teleop_topic=topic, follows=follows, last=last)
    if not all(math.isfinite(v) for v in (*prm.target, prm.radius_m, prm.settle_s,
                                           prm.max_speed_mps, prm.max_turn_rps, prm.timeout_s,
                                           prm.still_mps, prm.deadman_s)):
        raise PayloadError("non-finite number in the payload")
    if not 0.05 <= prm.radius_m <= 2.0:
        raise PayloadError("radius_m must be 0.05..2.0")
    if not 0.0 < prm.max_speed_mps <= CAP_LINEAR_MPS:
        raise PayloadError(f"max_speed_mps must be > 0 and <= {CAP_LINEAR_MPS}")
    if not 0.0 < prm.max_turn_rps <= CAP_ANGULAR_RPS:
        raise PayloadError(f"max_turn_rps must be > 0 and <= {CAP_ANGULAR_RPS}")
    if not (0.0 <= prm.settle_s <= 30.0 and 1.0 <= prm.timeout_s <= 3600.0):
        raise PayloadError("settle_s or timeout_s out of range")
    if not (0.001 <= prm.still_mps <= 0.2 and 0.1 <= prm.deadman_s <= 2.0):
        raise PayloadError("still_mps or deadman_s out of range")
    if prm.teleop_topic in BASE_TOPICS:
        raise PayloadError("teleop_topic must not be /cmd_vel: forwarding is the gate")
    return prm


def refusal(payload, teleop_topic: str | None, operation: str = OPERATION) -> str:
    """Why a manual-drive (or, with ``operation``, a navigate-handover) payload is refused on a
    binding whose teleop topic is ``teleop_topic`` (None: it has none), or "" when it is
    servable."""
    if not teleop_topic:
        return (f"{operation} is not served: this binding forwards no teleop topic, so there is "
                f"nothing the action could gate")
    try:
        prm = parse_payload(payload, operation)
    except PayloadError as e:
        return str(e)
    if prm.teleop_topic is not None and prm.teleop_topic != teleop_topic:
        return (f"teleop_topic {prm.teleop_topic!r} is not served: this binding forwards "
                f"{teleop_topic!r} only")
    return ""


def catalog(actions: list, extra: tuple[tuple[str, str], ...] = ()) -> list:
    """The binding's job actions (``(type, capabilities, params, description, results)`` rows,
    as the adapter's `_STACK_ACTIONS`) with manual-drive declared.

    `SupportedAction` is keyed by the action type alone, so a binding that also serves
    explore-floor declares both operations in its one Custom entry: ``operation`` allows both,
    and a parameter only one operation requires is declared optional, with the wider bounds.
    Each operation's own requirements are enforced when an action is judged (`refusal`).

    ``extra`` is ``(operation, description)`` for each further verb that takes these same
    parameters under its own name (a navigate-handover), declared in the same entry."""
    ops = (OPERATION, *(op for op, _ in extra))
    mine = (Param("operation", "enum", required=True, allowed=ops), *PARAMS[1:])
    described = " ".join((DESCRIPTION, *(d for _, d in extra)))
    out, merged = [], False
    for kind, caps, params, description, results in actions:
        if kind != "Custom":
            out.append((kind, caps, params, description, results))
            continue
        merged = True
        by_name: dict[str, Param] = {}
        allowed: tuple[str, ...] = ()
        for p in (*params, *mine):
            if p.name == "operation":
                allowed += tuple(a for a in p.allowed if a not in allowed)
                continue
            q = by_name.get(p.name)
            if q is None:
                by_name[p.name] = dataclasses.replace(p, required=False)
                continue
            lo = None if q.min is None or p.min is None else min(q.min, p.min)
            hi = None if q.max is None or p.max is None else max(q.max, p.max)
            by_name[p.name] = dataclasses.replace(q, required=False, min=lo, max=hi)
        op = Param("operation", "enum", required=True, allowed=allowed)
        out.append((kind, caps, (op, *by_name.values()), f"{description} {described}", results))
    if not merged:
        out.insert(0, ("Custom", ["navigation.2d"], mine, described, ()))
    return out


def is_manual_drive(action_type: str, payload) -> bool:
    """Whether an action is a manual-drive: `Custom` with ``operation: manual-drive``."""
    if action_type != "Custom":
        return False
    try:
        doc = payload if isinstance(payload, dict) else json.loads(payload or b"{}")
    except (ValueError, TypeError, UnicodeDecodeError):
        return False
    return isinstance(doc, dict) and doc.get("operation") == OPERATION


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v)) if math.isfinite(v) else 0.0


@dataclass
class ManualDrive:
    """The action's state: forwarding while Running, completion, time-out, abort. An ended
    drive never changes again."""

    params: Params
    started_at: float
    phase: str = RUNNING
    message: str = "manual drive: operator in control"
    _in_since: float | None = None
    _last_pose: tuple[float, float, float] | None = None   # (t, x, y)
    events: list[tuple[float, str]] = field(default_factory=list)
    # The last in-circle / still judgement `on_pose` made, for the map view's baton: distance to
    # the centre, the radius, inside or not, still for how long against settle_s. Read only.
    circle: dict | None = None

    def __post_init__(self):
        x, y = self.params.target
        self.events.append(
            (self.started_at, f"started; target ({x:.2f}, {y:.2f}) r={self.params.radius_m} m"))

    @property
    def forwarding(self) -> bool:
        return self.phase == RUNNING

    @property
    def ended(self) -> bool:
        return self.phase in (SUCCEEDED, FAILED)

    def decline(self, t: float, why: str) -> None:
        """Nav2 declined and the base is held: nothing is forwarded until the robot is confirmed
        at rest. Only from Navigating."""
        if self.phase != NAVIGATING:
            return
        self.phase = DECLINED
        self.message = (f"nav2 declined ({why}); the base is held, and teleop opens only once "
                        f"the robot is confirmed at rest")
        self.events.append((t, self.message))

    def hand_over(self, t: float, why: str) -> None:
        """The operator's turn starts now, and its time-out with it. Only from Navigating or
        Declined; the gate calls this, under the binding's lock (`TeleopGate.hand_over`), once
        the binding has confirmed the robot at rest."""
        if self.phase not in (NAVIGATING, DECLINED):
            return
        x, y = self.params.target
        self.phase, self.started_at = RUNNING, t
        self.message = (f"operator's turn: nav2 declined ({why}); drive to ({x:.2f}, {y:.2f}) "
                        f"within {self.params.radius_m} m")
        self.events.append((t, self.message))

    def arrived(self, t: float, msg: str) -> None:
        """Nav2 reached the target: Succeeded, and the operator was never involved."""
        if self.phase == NAVIGATING:
            self._end(t, SUCCEEDED, msg)

    def filter_cmd(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        """The operator's command -> what may reach the robot: clamped while Running, zero
        otherwise."""
        if not self.forwarding:
            return ZERO
        lin = self.params.max_speed_mps
        vx, vy = _clamp(vx, lin), _clamp(vy, lin)
        n = math.hypot(vx, vy)
        if n > lin:                        # the cap is on the speed, not on each axis
            vx, vy = vx * lin / n, vy * lin / n
        return vx, vy, _clamp(wz, self.params.max_turn_rps)

    def on_pose(self, t: float, x: float, y: float, frame: str) -> None:
        if not self.forwarding:
            return
        if frame != self.params.frame:
            self._end(t, FAILED, f"pose in frame {frame!r}, the action's is "
                                 f"{self.params.frame!r}: stopped")
            return
        speed = None
        if self._last_pose and t > self._last_pose[0]:
            speed = (math.hypot(x - self._last_pose[1], y - self._last_pose[2])
                     / (t - self._last_pose[0]))
        self._last_pose = (t, x, y)
        distance = math.hypot(x - self.params.target[0], y - self.params.target[1])
        inside = distance <= self.params.radius_m
        still = speed is not None and speed <= self.params.still_mps
        if inside and still:
            if self._in_since is None:
                self._in_since = t
        else:
            self._in_since = None
        still_s = 0.0 if self._in_since is None else t - self._in_since
        self.circle = {"distance_m": distance, "radius_m": self.params.radius_m,
                       "inside": inside, "still_s": still_s, "settle_s": self.params.settle_s}
        if self._in_since is not None and still_s > 0 and still_s >= self.params.settle_s:
            self._end(t, SUCCEEDED, f"in the circle and still {still_s:.1f} s: operator released")

    def tick(self, t: float) -> None:
        if self.forwarding and t - self.started_at >= self.params.timeout_s:
            self._end(t, FAILED, f"timeout after {self.params.timeout_s:.0f} s: stopped where "
                                 f"it stands")

    def abort(self, t: float, why: str) -> None:
        if self.phase in (RUNNING, NAVIGATING, DECLINED):
            self._end(t, FAILED, f"aborted: {why}")

    def _end(self, t: float, phase: str, msg: str) -> None:
        self.phase, self.message = phase, msg
        self.events.append((t, f"{phase}: {msg}"))


class TeleopGate:
    """The one door from the teleop topic to the base (see the module docstring).

    Not locked itself: the binding holds its own stop lock around every call, so a forwarded
    command and a stop are serialised. ``clock`` is the binding's monotonic clock; the lease
    duration is handed over and the deadline computed on it, so a gate never compares two
    clocks."""

    ARMED, OPEN, CLOSED = "armed", "open", "closed"
    LOG_LEN = 256

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._wall = wall
        self._action = ""
        self._drive: ManualDrive | None = None
        self._deadline = -math.inf
        self._moved_at: float | None = None    # when the last non-zero command was forwarded
        self._log: collections.deque = collections.deque(maxlen=self.LOG_LEN)
        self._seq = 0

    def _note(self, gate: str, reason: str) -> None:
        """Record a change, as it is made (the caller holds the binding's lock)."""
        self._seq += 1
        self._log.append({"seq": self._seq, "stamp": self._wall(), "gate": gate,
                          "action": self._action, "reason": reason})

    def changes(self, after: int = 0) -> list[dict]:
        """The changes recorded after ``after`` (a ``seq``), oldest first: each ``{seq, stamp,
        gate, action, reason}``, ``gate`` one of armed, open, closed. Read only."""
        return [dict(e) for e in self._log if e["seq"] > after]

    def open(self, action_id: str, drive: ManualDrive, lease_s: float) -> None:
        self._action, self._drive = action_id, drive
        self._deadline = self._clock() + lease_s
        self._moved_at = None
        if drive.forwarding:
            self._note(self.OPEN, "the action's lease was granted: the operator drives")
        else:
            self._note(self.ARMED, "the action's lease was granted: nav2 drives, and teleop "
                                   "forwards nothing")

    def extend(self, action_id: str, lease_s: float) -> bool:
        """A renewal of this action's lease moves the deadline; any other action's does not."""
        if not self._action or action_id != self._action:
            return False
        self._deadline = self._clock() + lease_s
        return True

    def close(self, why: str = "") -> bool:
        """Close it; whether it was open. Idempotent: closing a closed gate records nothing.
        ``why`` is recorded; without one, an ended drive's own message, else "closed"."""
        was = bool(self._action)
        if was:
            drive = self._drive
            self._note(self.CLOSED, why or (drive.message if drive is not None and drive.ended
                                            else "closed"))
        self._action, self._drive, self._deadline, self._moved_at = "", None, -math.inf, None
        return was

    def is_open(self, action_id: str = "") -> bool:
        return bool(self._action) and (not action_id or action_id == self._action)

    def navigating(self) -> bool:
        """Whether this gate is armed for a navigate-handover whose Nav2 goal is still driving:
        open, its drive Navigating, forwarding nothing. Read only."""
        drive = self._drive
        return drive is not None and bool(self._action) and drive.phase == NAVIGATING

    def declined(self, action_id: str) -> bool:
        """Whether this gate is armed for ``action_id``, its drive Declined (held, waiting for
        the handover) and its lease's deadline not passed. Read only; it opens nothing."""
        drive = self._drive
        return (drive is not None and bool(self._action) and action_id == self._action
                and drive.phase == DECLINED and self._clock() < self._deadline)

    def why_not_declined(self, action_id: str) -> str:
        """What `declined` finds missing for ``action_id``, each failing half named, or "" when
        it holds. Read only, for the operator's ready's log line: it decides nothing."""
        drive = self._drive
        if drive is None or not self._action:
            return "the gate is closed"
        if action_id != self._action:
            return f"the gate is armed for {self._action!r}"
        missing = []
        if drive.phase != DECLINED:
            missing.append(f"the drive is {drive.phase}, not {DECLINED}")
        late = self._clock() - self._deadline
        if late >= 0:
            missing.append(f"the lease's deadline passed {late:.1f} s ago")
        return "; ".join(missing)

    def hand_over(self, action_id: str, t: float, why: str) -> bool:
        """Turn an armed gate to the operator, or refuse. Only for the action it was opened for,
        only while that action's drive has not been handed over or ended, and only before the
        lease's deadline. A stop closed the gate (`close`), so after one this is refused: a stop
        that landed between Nav2's decline and this call wins. Called under the binding's lock,
        by a binding that has just confirmed the robot at rest under that same lock."""
        drive = self._drive
        if drive is None or not self._action or action_id != self._action:
            return False
        if self._clock() >= self._deadline or drive.phase not in (NAVIGATING, DECLINED):
            return False
        drive.hand_over(t, why)
        self._moved_at = None
        self._note(self.OPEN, f"handed to the operator: {why}")
        return True

    def admit(self, vx: float, vy: float, wz: float) -> tuple[float, float, float] | None:
        """The command to publish for one teleop message, or None to publish nothing: the gate
        is closed, the lease's deadline has passed (a lapse the lease thread has not looked at
        yet is still a lapse), or the drive has ended."""
        drive = self._drive
        if drive is None or not self._action:
            return None
        if self._clock() >= self._deadline:
            return None
        if not drive.forwarding:
            return None
        cmd = drive.filter_cmd(vx, vy, wz)
        if cmd != ZERO:
            self._moved_at = self._clock()
        return cmd

    def idle(self, deadman_s: float) -> bool:
        """True once, when teleop has been silent ``deadman_s`` after a forwarded motion: the
        caller publishes one zero Twist."""
        if self._moved_at is None or not self._action:
            return False
        if self._clock() - self._moved_at < deadman_s:
            return False
        self._moved_at = None
        return True


class ManualDriveJob:
    """One manual drive, run on its own thread by the adapter, as an explore-floor segment is.

    It reads the pose and the gate through the binding, and reports through ``emit`` (the
    adapter's `_job_emit`). On any end it closes the gate -- one zero Twist -- before it
    reports, so the terminal status never precedes the robot's last command."""

    POLL_S = 0.1

    def __init__(self, *, action_id: str, params: Params, robot, robot_id: str, emit, clock) -> None:
        self.action_id = action_id
        self.params = params
        self.halted = threading.Event()
        self.fencing_token: int | None = None
        self._robot, self._rid, self._emit, self._clock = robot, robot_id, emit, clock
        self.drive = ManualDrive(params, started_at=clock.now())

    def run(self) -> None:
        r, rid, drive = self._robot, self._rid, self.drive
        while not self.halted.is_set():
            t = self._clock.now()
            if not r.teleop_open(rid, self.action_id):
                drive.abort(t, "a stop closed the forwarding; the robot was stopped and teleop "
                               "no longer reaches it")
            else:
                drive.tick(t)
                if drive.forwarding:
                    x, y, _ = r.pose(rid)
                    drive.on_pose(t, x, y, r.pose_frame(rid))
                if drive.forwarding:
                    r.teleop_deadman(rid, self.params.deadman_s)
            if not drive.forwarding:
                r.close_teleop(rid, self.action_id)
                state = task_map.TASK_SUCCEEDED if drive.phase == SUCCEEDED else task_map.TASK_FAILED
                self._emit(state, drive.message, 100 if state == task_map.TASK_SUCCEEDED else 0)
                return
            self._clock.wait(self.halted, self.POLL_S)


class SimulatedTeleopRobot(SimulatedRobot):
    """The stand-in robot for desk runs: `SimulatedRobot` with a teleop topic and its gate.

    ``teleop(vx, vy, wz)`` is one message arriving on the teleop topic. ``cmd_vel`` records what
    reached the base, zeros included; ``place`` moves the robot (the SLAM pose) for a test. The
    gate closes in ``command_hold`` and ``command_pause`` -- the base's own stops, which every
    stop path reaches -- under the same lock a forwarded command is published under."""

    def __init__(self, teleop_topic: str = "/cmd_vel_teleop", clock=time.monotonic,
                 **kw) -> None:
        kw.setdefault("explore", None)
        super().__init__(**kw)
        self._teleop_topic = teleop_topic
        self._teleop_lock = threading.Lock()
        self._gate = TeleopGate(clock)
        self._frame = FRAME
        self.cmd_vel: list[tuple[float, float, float]] = []

    # ── what the adapter asks of a binding that forwards teleop ─────────────────

    def teleop_topic(self) -> str | None:
        return self._teleop_topic

    def open_teleop(self, robot_id: str, action_id: str, drive: ManualDrive,
                    lease_s: float) -> None:
        super().command_hold(robot_id)          # one source of motion: no goal under teleop
        with self._teleop_lock:
            self._gate.open(action_id, drive, lease_s)

    def extend_teleop(self, robot_id: str, action_id: str, lease_s: float) -> bool:
        with self._teleop_lock:
            return self._gate.extend(action_id, lease_s)

    def close_teleop(self, robot_id: str, action_id: str = "") -> None:
        with self._teleop_lock:
            if not self._gate.is_open(action_id):
                return
            self._gate.close()
            self._publish(robot_id, ZERO)

    def teleop_open(self, robot_id: str, action_id: str = "") -> bool:
        with self._teleop_lock:
            return self._gate.is_open(action_id)

    def teleop_log(self, robot_id: str, after: int = 0) -> list[dict]:
        """The gate's changes after ``after`` (`TeleopGate.changes`). Read only."""
        with self._teleop_lock:
            return self._gate.changes(after)

    def teleop_deadman(self, robot_id: str, deadman_s: float) -> None:
        with self._teleop_lock:
            if self._gate.idle(deadman_s):
                self._publish(robot_id, ZERO)

    def pose_frame(self, robot_id: str) -> str:
        return self._frame

    # ── the teleop topic and the base ───────────────────────────────────────────

    def teleop(self, vx: float, vy: float, wz: float, robot_id: str = "") -> None:
        rid = robot_id or next(iter(self._moving), "")
        with self._teleop_lock:
            cmd = self._gate.admit(vx, vy, wz)
            if cmd is not None:
                self._publish(rid, cmd)

    def place(self, x: float, y: float, frame: str = FRAME, robot_id: str = "") -> None:
        rid = robot_id or next(iter(self._moving), "")
        with self._clock_lock:
            self._pos[rid] = (x, y)
        self._frame = frame

    def _publish(self, robot_id: str, cmd) -> None:
        """Caller holds `_teleop_lock`."""
        self.cmd_vel.append(tuple(cmd))
        with self._clock_lock:
            self._moving[robot_id] = tuple(cmd) != ZERO

    def command_hold(self, robot_id: str) -> None:
        with self._teleop_lock:
            self._gate.close(STOP_WHY)
            self._publish(robot_id, ZERO)
        super().command_hold(robot_id)

    def command_pause(self, robot_id: str) -> None:
        with self._teleop_lock:
            self._gate.close(PAUSE_WHY)
            self._publish(robot_id, ZERO)
        super().command_pause(robot_id)
