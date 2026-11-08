# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A declined handover's rest confirmed by the operator, with the lidar scan as a veto only.

With ``--handover-rest-source operator``, after Nav2 declines the base is held and the gate stays
shut until ALL of these hold: the operator's ready arrived after the hold and while the job is
Declined (one ready, for this action; a ready before, after, twice or for another action is
ignored and logged, never queued); a whole scan window plus its lag has passed since the hold, so
the veto had a full look; the scan says ``at_rest`` (any ``moving`` and every ``unknown`` keep the
gate shut); and no stop, pause, cancel or lease end. The binding checks all of it again inside
`hand_over_teleop`, under the lock every stop takes. No ready within the bound: Failed, and the
gate never opens. The scan never confirms rest by itself.

The estop does not change: `is_stopped` reads neither the ready nor the scan, so on
command-derived odometry an estop is FAILED, never STOPPED, with the operator source on.

The stand-in is the scan-rest one: its plant model reads at rest once held, and its lidar
ray-casts from the plant's TRUE pose, which drifts while ``braking``.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import queue
import re
import threading
import time

import pytest
import test_handover_waits_for_rest as base
import test_mapview_target_state as mvt
from conftest import mark_transport_live
from scan_room import CORRIDOR, Lidar, Plant
from test_command_derived_odometry import STOP_S, _CommandDerivedBinding, _driving
from test_handover_scan_rest import DRIFT, ScanBrakingRobot, _laser
from test_handover_waits_for_rest import WHY, presses
from test_manual_drive_gate import ROBOT, StepClock, collect, settle, statuses, terminals
from test_manual_drive_nav2 import TOPIC, msg
from test_navigate_handover import ACTION, STOPS, assign, payload, turned
from test_navigate_handover_nav2 import decline
from test_scan_rest import T  # noqa: F401 - the fixture: T from two stationary runs
from test_scan_rest_record import good_record
from test_scan_rest_veto_record import good_veto
from test_stall_reporting import _main
from test_stop_never_waits import (  # noqa: F401 - ros_msgs is a fixture
    _ControlPlane,
    _odom,
    _Publisher,
    _SafetyStub,
    _Server,
    _Twist,
    ros_msgs,
)

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import nav2_binding
from fleet_adapter_ros2 import operator_ready as opr
from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

LOOK_S = sr.LAG_S + sr.WINDOW_S          # the veto's full look after the hold
CONFIRMED = "operator-confirmed rest"


class OperatorRobot(ScanBrakingRobot):
    """The scan-rest stand-in with the operator's ready, fed by a desk source. ``quiet``: the
    lidar publishes nothing (scans fall due and are never received)."""

    def __init__(self, threshold_m: float, **kw) -> None:
        self.quiet = False
        super().__init__(threshold_m, operator_ready=True, **kw)
        self.ready = opr.DeskReadySource()
        self.ready.attach(self.on_ready)

    def pump(self) -> None:
        if self.quiet:
            with self._scan_lock:
                while self._next <= self._clock():
                    self._plant.step(0.1)
                    self._next += 0.1
            return
        super().pump()


def op_rig(threshold_m, ready_s: float = 3600.0, **kw):
    robot = OperatorRobot(threshold_m, **kw)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock())
    a._handover_ready_s = ready_s   # on the job's clock; long unless a test needs it
    return a, robot


def declined(a, robot, **assign_kw) -> list:
    """Assign; Nav2 declines where the robot stands, at rest."""
    robot.decline_next(WHY)
    res, msgs = assign(a, payload(), **assign_kw)
    assert res.accepted
    collect(a, msgs, waiting)
    return msgs


def waiting(msgs) -> list:
    return [s for s in statuses(msgs, ACTION)
            if s.state == pb.ACTION_STATE_RUNNING and "press ready" in s.message]


def refused(msgs) -> list:
    return [s for s in statuses(msgs, ACTION)
            if s.state == pb.ACTION_STATE_RUNNING and "ready refused" in s.message]


def turns(msgs) -> list:
    return [s for s in statuses(msgs, ACTION)
            if s.state == pb.ACTION_STATE_RUNNING and "operator's turn" in s.message]


def look() -> None:
    """Let the veto have its full look after the hold (real time: the scans are)."""
    time.sleep(LOOK_S + 0.2)


def log_has(robot, *words) -> bool:
    return any(all(w in line for w in words) for line in robot.ready_log())


# ── the gate opens on the operator's ready, and only then ─────────────────────

def test_at_rest_the_scan_alone_never_opens_the_gate_the_ready_does(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    look()
    settle(a, msgs, 0.6)
    assert robot.handover_at_rest(ROBOT)[0]                 # the scan has no veto
    assert not turned(msgs) and presses(robot) == []        # and still nothing opens
    assert robot.ready.press() == ""                        # accepted
    collect(a, msgs, turned)
    (c, *_) = presses(robot, 1)
    assert c != md.ZERO and len(turns(msgs)) == 1


def test_the_message_says_operator_confirmed_rest_and_the_scans_no_veto_in_numbers(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    assert "the base is held" in waiting(msgs)[0].message and WHY in waiting(msgs)[0].message
    look()
    robot.ready.press()
    collect(a, msgs, turned)
    text = turns(msgs)[0].message
    assert CONFIRMED in text and "scan: no veto" in text
    m = re.search(r"S max (\d+\.\d{4}) m < T (\d+\.\d{4}) m over (\d+) scans", text)
    assert m and float(m.group(1)) < float(m.group(2)) == round(T, 4) and int(m.group(3)) >= 5
    assert len(text) <= 200


# ── the ready itself: when it counts, and when it is ignored ─────────────────

def test_a_ready_before_the_hold_is_ignored_and_never_queued(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    assert robot.ready.press() != ""                        # no handover held at all
    assert log_has(robot, "ignored", "no declined handover")
    robot.stall_next()                                      # Nav2 drives, and has not declined
    res, msgs = assign(a, payload())
    assert res.accepted
    settle(a, msgs, 0.2)
    assert robot.ready.press() != ""                        # navigating: no hold yet
    assert sum("ignored" in x and "no declined handover" in x for x in robot.ready_log()) == 2
    robot.decline_now(WHY, ROBOT)
    collect(a, msgs, waiting)
    look()
    settle(a, msgs, 0.5)
    assert not turned(msgs) and presses(robot) == []        # neither press was kept
    robot.ready.press()
    collect(a, msgs, turned)


def test_a_ready_while_the_scan_says_moving_is_refused_and_the_circle_stays_held(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = base.declined_moving(a, robot)
    look()
    assert robot.ready.press() == ""                        # taken, then judged by the veto
    collect(a, msgs, refused)
    assert "scan: moving" in refused(msgs)[0].message and "press ready again" in \
        refused(msgs)[0].message
    settle(a, msgs, 0.3)
    assert not turned(msgs) and presses(robot) == [] and robot.teleop_open(ROBOT, ACTION)
    assert not terminals(msgs, ACTION)                      # still held, still the action's
    robot.come_to_rest()
    look()
    settle(a, msgs, 0.3)
    assert not turned(msgs)                                 # the refused ready is not kept
    robot.ready.press()
    collect(a, msgs, turned)


@pytest.mark.parametrize("unknown", ["no scan", "stale", "no structure"])
def test_a_ready_with_an_unknown_scan_is_refused(T, unknown) -> None:  # noqa: F811
    a, robot = op_rig(T)
    if unknown == "no structure":
        robot._lidar = Lidar(walls=list(CORRIDOR), seed=7)
    if unknown == "no scan":
        robot.quiet = True
    msgs = declined(a, robot)
    look()
    if unknown == "stale":
        robot.pump()                                        # it was publishing until now
        robot.quiet = True
        time.sleep(sr.STALE_S + 0.1)
    robot.ready.press()
    collect(a, msgs, refused)
    text = refused(msgs)[0].message
    assert "scan: unknown" in text
    assert {"no scan": "no scan yet", "stale": "old", "no structure": "no structure"}[unknown] \
        in text
    settle(a, msgs, 0.3)
    assert not turned(msgs) and presses(robot) == []


def test_a_ready_before_the_veto_had_its_full_look_is_refused(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    robot.ready.press()                                     # at once: under LAG + WINDOW
    collect(a, msgs, refused)
    assert "full look" in refused(msgs)[0].message
    settle(a, msgs, 0.2)
    assert not turned(msgs) and presses(robot) == []
    look()
    robot.ready.press()
    collect(a, msgs, turned)


def test_a_ready_twice_counts_once(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    look()
    robot.ready.press()
    collect(a, msgs, turned)
    assert robot.ready.press() != ""                        # the second: the turn is open
    settle(a, msgs, 0.3)
    assert len(turns(msgs)) == 1
    assert log_has(robot, "ignored")


def test_a_second_ready_while_one_waits_to_be_judged_is_ignored(T) -> None:  # noqa: F811
    robot, _drive = _held_standin(T)
    look()
    assert robot.on_ready() == ""
    second = robot.on_ready()
    assert "already" in second and log_has(robot, "ignored", "already")
    assert robot.take_ready(ROBOT, ACTION) is not None
    assert robot.take_ready(ROBOT, ACTION) is None          # one press, taken once


def test_a_ready_with_no_decline_changes_nothing(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    res, msgs = assign(a, payload())                        # Nav2 reaches it
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert robot.ready.press() != ""
    settle(a, msgs, 0.2)
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_SUCCEEDED]
    assert not turned(msgs) and log_has(robot, "ignored")


def test_a_ready_naming_another_action_is_ignored(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    look()
    why = robot.ready.press(action_id="handover-b")
    assert "another action" in why and log_has(robot, "ignored", "another action")
    settle(a, msgs, 0.3)
    assert not turned(msgs) and presses(robot) == []
    robot.ready.press(action_id=ACTION)
    collect(a, msgs, turned)


def test_a_ready_for_one_action_never_opens_the_next(T) -> None:  # noqa: F811
    robot, _drive = _held_standin(T)
    look()
    assert robot.on_ready() == ""                           # for ACTION, waiting
    robot.command_hold(ROBOT)                               # a stop ends ACTION
    other = md.ManualDrive(ho.parse_payload(json.loads(payload())), started_at=0.0,
                           phase=md.NAVIGATING)
    robot.open_teleop(ROBOT, "handover-b", other, lease_s=60.0)
    assert robot.hold_for_handover(ROBOT, "handover-b")
    other.decline(0.0, WHY)
    look()
    assert robot.take_ready(ROBOT, "handover-b") is None
    assert not robot.hand_over_teleop(ROBOT, "handover-b", 1.0, WHY)
    assert other.phase == md.DECLINED and presses(robot) == []


# ── a stop of any scope, and the lease's end, win at every point ─────────────

@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_between_the_ready_and_the_handover_wins(T, stop) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    look()
    robot.stop_at_handover = lambda: STOPS[stop](a)
    robot.ready.press()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert robot.stop_at_handover is None                   # it landed there
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)
    end = terminals(msgs, ACTION)
    assert len(end) == 1 and end[0].state != pb.ACTION_STATE_SUCCEEDED


def test_a_lease_end_between_the_ready_and_the_handover_wins(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    lease_s = LOOK_S + 0.9
    t0 = time.monotonic()
    msgs = declined(a, robot, lease_ms=round(lease_s * 1000))
    look()
    robot.stop_at_handover = lambda: time.sleep(max(0.0, t0 + lease_s + 0.1 - time.monotonic()))
    assert robot.ready.press() == ""
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    assert robot.stop_at_handover is None
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_FAILED]
    assert presses(robot) == [] and not turned(msgs)


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_during_the_wait_wins_and_a_later_ready_opens_nothing(T, stop) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot)
    settle(a, msgs, 0.1)
    STOPS[stop](a)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    look()
    assert robot.ready.press() != ""
    settle(a, msgs, 0.3)
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)
    end = terminals(msgs, ACTION)
    assert len(end) == 1 and end[0].state != pb.ACTION_STATE_SUCCEEDED


def test_a_lease_lapse_during_the_wait_wins_and_a_later_ready_opens_nothing(T) -> None:  # noqa: F811
    a, robot = op_rig(T)
    msgs = declined(a, robot, lease_ms=80)
    time.sleep(0.12)
    a._tick_lease()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    look()
    assert robot.ready.press() != ""
    settle(a, msgs, 0.3)
    assert [s.end_cause for s in terminals(msgs, ACTION)] == [pb.ACTION_END_CAUSE_LEASE_EXPIRED]
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)


# ── no ready within the bound ─────────────────────────────────────────────────

def test_no_ready_within_the_bound_is_failed_and_a_timeout_never_opens(T) -> None:  # noqa: F811
    a, robot = op_rig(T, ready_s=5.0)
    msgs = declined(a, robot)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert ("no operator confirmation within 5 s; the operator's turn never opened and the "
            "base is held") in end[0].message
    assert not turned(msgs) and not robot.teleop_open(ROBOT)
    look()
    assert robot.ready.press() != ""                        # after the end: nothing
    settle(a, msgs, 0.3)
    assert presses(robot) == [] and not turned(msgs)


def test_the_default_bound_is_thirty_seconds() -> None:
    robot = ho.SimulatedHandoverRobot()
    robot.spawn(ROBOT)
    assert Adapter(pb, None, robot, ROBOT)._handover_ready_s == ho.READY_S == 30.0


# ── the circle: yellow through the wait, red only after an accepted ready ─────

def test_the_circle_is_yellow_through_the_wait_and_red_only_after_the_ready(T) -> None:  # noqa: F811
    a, robot, sink = mvt.rig(cls=OperatorRobot, threshold_m=T)
    a._handover_ready_s = 3600.0
    msgs = base.declined_moving(a, robot)
    look()
    assert mvt.states(sink) == [mvt.ROBOT_]
    robot.ready.press()                                     # refused: the scan says moving
    collect(a, msgs, refused)
    settle(a, msgs, 0.3)
    assert mvt.states(sink) == [mvt.ROBOT_]
    robot.come_to_rest()
    look()
    settle(a, msgs, 0.3)
    assert mvt.states(sink) == [mvt.ROBOT_]                 # at rest, no ready: still yellow
    robot.ready.press()
    collect(a, msgs, turned)
    assert mvt.wait_states(sink, lambda s: s[-1:] == [mvt.OPERATOR]) == [mvt.ROBOT_,
                                                                          mvt.OPERATOR]


@pytest.mark.parametrize("end", ["estop", "timeout"])
def test_the_circle_turns_grey_on_a_stop_or_failed(T, end) -> None:  # noqa: F811
    a, robot, sink = mvt.rig(cls=OperatorRobot, threshold_m=T)
    a._handover_ready_s = 5.0 if end == "timeout" else 3600.0
    msgs = declined(a, robot)
    if end == "estop":
        STOPS["estop"](a)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert mvt.wait_states(sink, lambda s: s[-1:] == [mvt.STOPPED]) == [mvt.ROBOT_, mvt.STOPPED]


# ── the stand-in and the binding check it all again under the stop lock ──────

def _held_standin(threshold_m, action=ACTION):
    robot = OperatorRobot(threshold_m)
    robot.spawn(ROBOT)
    drive = md.ManualDrive(ho.parse_payload(json.loads(payload())), started_at=0.0,
                           phase=md.NAVIGATING)
    robot.open_teleop(ROBOT, action, drive, lease_s=60.0)
    assert robot.hold_for_handover(ROBOT, action)
    drive.decline(0.0, WHY)
    return robot, drive


def test_the_standin_hands_over_only_on_a_taken_ready_with_no_veto(T) -> None:  # noqa: F811
    robot, drive = _held_standin(T)
    look()
    assert robot.handover_at_rest(ROBOT)[0]
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)      # no ready
    assert robot.on_ready() == ""
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)      # pressed, not yet taken
    robot.on_ready()
    assert robot.take_ready(ROBOT, ACTION) is not None
    robot.braking = True                                            # moving at the handover
    time.sleep(sr.LAG_S + 0.15)
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)
    assert drive.phase == md.DECLINED and presses(robot) == []
    assert robot.take_ready(ROBOT, ACTION) is None                  # that ready was spent


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _held_nav2(T, source="command-derived"):  # noqa: F811
    clock = _Clock()
    b = mark_transport_live(Nav2Binding(
        ROBOT, teleop_topic=TOPIC, odometry_source=source, rest_scan_topic="/scan_raw",
        rest_scan_threshold_m=T, ready_topic=opr.READY_TOPIC))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b._now = clock
    drive = md.ManualDrive(ho.parse_payload(json.loads(payload())), started_at=0.0,
                           phase=md.NAVIGATING)
    b.open_teleop(ROBOT, ACTION, drive, lease_s=1e6)
    with b._lock:
        b._goal = task_map.GoalPose(x=2.0, y=1.0, yaw=0.0)
        b._holding, b._goal_status = False, 0
    decline(b)
    b.on_odom(_odom(0.0))
    assert b.hold_for_handover(ROBOT, ACTION)
    drive.decline(0.0, WHY)
    return b, drive, clock


def _scans(b, clock, plant, lidar, seconds):
    for _ in range(round(seconds * 10)):
        b.on_scan(_laser(lidar.cast(plant, clock.t)))
        plant.step(0.1)
        clock.t += 0.1


def test_nav2_operator_source_needs_the_ready_the_full_look_and_no_veto(T) -> None:  # noqa: F811
    b, drive, clock = _held_nav2(T)
    assert b.handover_confirms_by_operator(ROBOT) and b.handover_rest_gap(ROBOT) == ""
    plant, lidar = Plant(), Lidar(seed=91)
    _scans(b, clock, plant, lidar, 0.8)
    assert b.on_ready() == ""
    assert b.take_ready(ROBOT, ACTION) is not None
    ok, why = b.handover_at_rest(ROBOT)
    assert not ok and "full look" in why
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)
    _scans(b, clock, plant, lidar, 1.5)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)          # the scan alone: no
    b.on_ready()
    b.take_ready(ROBOT, ACTION)
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_nav2_operator_source_moving_scan_and_other_channels_veto_the_ready(T) -> None:  # noqa: F811
    b, drive, clock = _held_nav2(T, source="measured")
    _scans(b, clock, Plant(true=DRIFT), Lidar(seed=92), 2.5)
    b.on_ready()
    b.take_ready(ROBOT, ACTION)
    assert "moving" in b.handover_at_rest(ROBOT)[1]
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)          # measured odom at 0: no
    _scans(b, clock, Plant(), Lidar(seed=93), 2.5)
    b.on_odom(_odom(0.3))
    b.on_ready()
    b.take_ready(ROBOT, ACTION)
    assert b.handover_at_rest(ROBOT) == (False, "odometry shows motion")
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)
    assert drive.phase == md.DECLINED
    n = len(b._cmd_vel.published)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(b._cmd_vel.published) == n


def test_nav2_a_ready_with_no_hold_is_ignored(T) -> None:  # noqa: F811
    b = mark_transport_live(Nav2Binding(
        ROBOT, teleop_topic=TOPIC, rest_scan_topic="/scan_raw", rest_scan_threshold_m=T,
        ready_topic=opr.READY_TOPIC))
    assert "no declined handover" in b.on_ready()
    assert b.take_ready(ROBOT, ACTION) is None


def test_nav2_the_operator_source_needs_the_scan_and_its_own_topic(T) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="veto needs the scan"):
        Nav2Binding(ROBOT, teleop_topic=TOPIC, ready_topic=opr.READY_TOPIC)
    for topic in (TOPIC, "/cmd_vel", "/scan_raw", " "):
        with pytest.raises(ValueError, match="ready"):
            Nav2Binding(ROBOT, teleop_topic=TOPIC, rest_scan_topic="/scan_raw",
                        rest_scan_threshold_m=T, ready_topic=topic)


# ── the honesty row: the estop reads neither the ready nor the scan ──────────

class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"the estop's answer read {name!r}")


def test_with_the_operator_source_an_estop_is_still_failed_on_command_derived(
        T, ros_msgs) -> None:  # noqa: F811
    server = _Server(up=True)
    b = mark_transport_live(_CommandDerivedBinding(
        ROBOT, odometry_source="command-derived", rest_scan_topic="/scan_raw",
        rest_scan_threshold_m=T, teleop_topic=TOPIC, ready_topic=opr.READY_TOPIC))
    from types import SimpleNamespace as NS
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b._cancel_goal_by_id = lambda _id: None
    b.start = lambda: None
    b.on_odom(_odom(0.0))
    with b._lock:
        b._handover_hold_at = time.monotonic() - 10.0
    stop = threading.Event()
    lidar, plant = Lidar(seed=96), Plant()

    def feed() -> None:                                         # a still room, at 10 Hz
        while not stop.is_set():
            b.on_scan(_laser(lidar.cast(plant, time.monotonic())))
            time.sleep(0.1)
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    a = Adapter(pb, None, b, ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    try:
        assert cp.opened.wait(2.0)
        time.sleep(LOOK_S + 0.3)
        assert b.handover_at_rest(ROBOT)[0]                     # the scan: no veto
        _driving(a, b, server, cp)
        b.on_ready()                                            # and a ready on the topic
        safety = threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),),
                                  daemon=True)
        safety.start()
        deadline = time.time() + STOP_S + 1.0
        states, last = [], None
        while time.time() < deadline and (not states or states[-1] == pb.ESTOP_STATE_STOPPING):
            try:
                m = a._safety_outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if m.WhichOneof("payload") == "estop_ack":
                states.append(m.estop_ack.state)
                last = m.estop_ack
        assert states and states[-1] == pb.ESTOP_STATE_FAILED
        assert pb.ESTOP_STATE_STOPPED not in states and "command-derived" in last.message
        assert b.is_stopped(ROBOT) is False and b.rest_evidence_gap(ROBOT) != ""
    finally:
        stop.set()
        a._shutdown.set()
        cp.q.put(None)
        control.join(timeout=5.0)


def test_is_stopped_reads_neither_the_ready_nor_the_scan(T) -> None:  # noqa: F811
    b, _, clock = _held_nav2(T, source="measured")
    _scans(b, clock, Plant(), Lidar(seed=97), 2.5)
    b.on_ready()
    before = (b.is_stopped(ROBOT), b.rest_evidence_gap(ROBOT))
    b._ready, b._scan_rest = _Poison(), _Poison()
    assert (b.is_stopped(ROBOT), b.rest_evidence_gap(ROBOT)) == before
    robot, _ = _held_standin(T)
    before = (robot.is_stopped(ROBOT), robot.rest_evidence_gap(ROBOT))
    robot._ready, robot._rest_channel = _Poison(), _Poison()
    assert (robot.is_stopped(ROBOT), robot.rest_evidence_gap(ROBOT)) == before


@pytest.mark.parametrize("fn", ["is_stopped", "_at_rest_locked", "rest_evidence_gap"])
def test_the_estops_test_names_neither_the_ready_nor_the_scan(fn) -> None:
    src = inspect.getsource(getattr(Nav2Binding, fn))
    names = {n.attr for n in ast.walk(ast.parse(src.strip())) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(ast.parse(src.strip())) if isinstance(n, ast.Name)}
    assert not names & {"_ready", "_scan_rest", "on_ready", "take_ready", "handover_at_rest",
                        "_handover_at_rest_locked", "_handover_hold_at"}


# ── the ready sources: a desk one and a ROS one ───────────────────────────────

class _Node:
    def __init__(self) -> None:
        self.subs = []

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.append((msg_type, topic, cb, qos))


def test_the_ros_source_subscribes_one_empty_topic_and_reads_nothing_from_a_message() -> None:
    node, got = _Node(), []
    empty = type("Empty", (), {})
    opr.RosReadySource(node, opr.READY_TOPIC, empty).attach(lambda aid=None: got.append(aid))
    ((t, topic, cb, qos),) = node.subs
    assert t is empty and topic == "/handover_ready" and qos == 1
    cb(_Poison())                                               # nothing in it is read
    assert got == [None]
    assert opr.READY_TYPE == "std_msgs/msg/Empty"


def test_the_nav2_binding_subscribes_the_ready_through_the_ros_source() -> None:
    src = inspect.getsource(nav2_binding.Nav2Binding.start)
    assert "RosReadySource" in src and "Empty" in src


def test_the_desk_source_presses_once_per_call() -> None:
    got = []
    src = opr.DeskReadySource()
    src.attach(lambda aid=None: got.append(aid) or "")
    assert src.press() == "" and src.press(action_id="x") == ""
    assert got == [None, "x"]


# ── the adapter: the operator source and its flags ────────────────────────────

ROS2 = ("--binding", "ros2", "--stall-window", "5", "--stall-min-progress", "0.1",
        "--goal-response-bound", "1", "--odometry-source", "command-derived",
        "--teleop-topic", "/cmd_vel_teleop")


def _file(tmp_path, doc, name="veto.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


_FILES = iter(range(10**6))


def _operator(tmp_path, doc=None, *extra):
    name = f"record-{next(_FILES)}.json"
    return (*ROS2, "--handover-rest-source", "operator", "--rest-scan-topic", "/scan_raw",
            "--rest-scan-calibration", _file(tmp_path, good_veto() if doc is None else doc, name),
            *extra)


def test_the_operator_source_runs_with_a_veto_only_record_and_says_so(monkeypatch, capsys,
                                                                     tmp_path):
    code, err, robot = _main(monkeypatch, capsys, *_operator(tmp_path))
    assert code is None and isinstance(robot, Nav2Binding)
    assert robot.handover_confirms_by_operator(ROBOT)
    assert robot._scan_rest.threshold_m == 0.0165 and robot._ready_topic == "/handover_ready"
    line = next(x for x in err.splitlines() if x.startswith("navigate-handover rest:"))
    assert CONFIRMED in line and "veto only" in line and "veto-only record md5" in line
    assert "within 30 s" in line and "/handover_ready" in line
    assert robot.is_stopped(ROBOT) is False and robot.rest_evidence_gap(ROBOT) != ""


def test_the_ready_bound_and_topic_can_be_set(monkeypatch, capsys, tmp_path):
    code, err, robot = _main(monkeypatch, capsys, *_operator(
        tmp_path, None, "--handover-ready-s", "45", "--handover-ready-topic", "/hd/ready"))
    assert code is None and robot._ready_topic == "/hd/ready"
    assert "within 45 s" in err


@pytest.mark.parametrize("bound", ["4.9", "120.1", "0", "-1", "nan", "inf"])
def test_a_ready_bound_outside_five_to_120_s_is_refused(monkeypatch, capsys, tmp_path, bound):
    code, err, robot = _main(monkeypatch, capsys, *_operator(
        tmp_path, None, "--handover-ready-s", bound))
    assert code == 2 and robot is None and "--handover-ready-s" in err and "5 to 120 s" in err


@pytest.mark.parametrize("bound", ["5", "120"])
def test_the_ready_bounds_ends_are_served(monkeypatch, capsys, tmp_path, bound):
    code, _, _ = _main(monkeypatch, capsys, *_operator(tmp_path, None, "--handover-ready-s",
                                                       bound))
    assert code is None


def test_the_operator_source_refuses_without_its_evidence(monkeypatch, capsys, tmp_path):
    rec = _file(tmp_path, good_veto())
    cases = [
        ((*ROS2, "--handover-rest-source", "operator", "--rest-scan-calibration", rec),
         "needs --rest-scan-topic"),
        ((*ROS2, "--handover-rest-source", "operator", "--rest-scan-topic", "/scan_raw"),
         "needs --rest-scan-calibration"),
        ((*ROS2[:-2], "--handover-rest-source", "operator", "--rest-scan-topic", "/scan_raw",
          "--rest-scan-calibration", rec), "needs --teleop-topic"),
        (("--binding", "simulated", "--teleop-topic", "/cmd_vel_teleop",
          "--handover-rest-source", "operator", "--rest-scan-topic", "/scan_raw",
          "--rest-scan-calibration", rec), "no lidar"),
        (_operator(tmp_path, good_record()), "not a veto-only record"),
        (_operator(tmp_path, {**good_veto(), "threshold_m": 0.02}), "is not 1.5 x s0"),
        ((*ROS2, "--handover-rest-source", "operator", "--rest-scan-topic", "/scan_raw",
          "--rest-scan-calibration", str(tmp_path / "absent.json")), "no calibration record"),
        (_operator(tmp_path, None, "--handover-ready-topic", "/cmd_vel_teleop"), "ready"),
        (_operator(tmp_path, None, "--handover-ready-topic", "/cmd_vel"), "ready"),
        (_operator(tmp_path, None, "--handover-ready-topic", "/scan_raw"), "ready"),
    ]
    for argv, words in cases:
        code, err, robot = _main(monkeypatch, capsys, *argv)
        assert code == 2 and robot is None and err.startswith("REFUSED: "), (argv, err)
        assert words in err, (argv, err)


@pytest.mark.parametrize("source", ["odometry", "scan"])
def test_the_ready_flags_are_used_only_with_the_operator_source(monkeypatch, capsys, tmp_path,
                                                                source):
    rec = _file(tmp_path, good_record(), "measured.json")
    scan = (("--rest-scan-topic", "/scan_raw", "--rest-scan-calibration", rec)
            if source == "scan" else ())
    for flag, v in (("--handover-ready-s", "30"), ("--handover-ready-topic", "/handover_ready")):
        code, err, _ = _main(monkeypatch, capsys, *ROS2, "--handover-rest-source", source,
                             *scan, flag, v)
        assert code == 2 and f"{flag} is used only with --handover-rest-source operator" in err


def test_the_scan_source_never_takes_a_veto_only_record(monkeypatch, capsys, tmp_path):
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--handover-rest-source", "scan",
                             "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration",
                             _file(tmp_path, good_veto()))
    assert code == 2 and robot is None and "veto-only" in err


def test_the_operator_source_is_listed_in_help(monkeypatch, capsys):
    import contextlib
    import io
    import sys

    from fleet_adapter_ros2 import adapter
    monkeypatch.setattr(sys, "argv", ["adapter", "--help"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        adapter.main()
    text = out.getvalue()
    assert "operator" in text and "--handover-ready-s" in text
    assert "--handover-ready-topic" in text and math.isfinite(ho.READY_S)
