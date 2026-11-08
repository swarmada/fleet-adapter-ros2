# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A navigate-handover turns teleop to the operator only once the robot is confirmed at rest.

When Nav2 declines, the base is told to hold, and the robot may still be moving: braking, or
finishing whatever the navigator last commanded. Until the binding confirms it at rest
(`is_stopped`, the estop's own test, judged after `rest_evidence_gap` as the estop judges it),
the gate stays armed and forwards nothing, so the operator cannot drive while the robot may still
be moving. A stop of any scope wins at every point of that wait, and so does a lease that ends
in it: the action ends, not Succeeded, and teleop never reaches the base. If rest is not
confirmed within the bound, the action ends Failed and says why; a timeout never opens the gate.
A binding that cannot confirm rest at all (odometry declared command-derived, or undeclared)
never hands over: it says so at the decline.

The binding checks rest once more at the handover itself, under the lock every stop takes, so the
gate never turns on a rest the job saw a moment before.

The braking base is the stand-in with one knob: after the decline it reports motion until the
test says it has come to rest.
"""

from __future__ import annotations

import time

import pytest
from test_manual_drive_gate import ROBOT, StepClock, collect, settle, statuses, terminals
from test_manual_drive_nav2 import binding, msg
from test_navigate_handover import ACTION, STOPS, assign, payload, renew_running, turned
from test_navigate_handover_nav2 import decline
from test_stop_never_waits import _odom

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import task as task_map

WHY = "the navigation server aborted the goal"
GAP = "odometry is the velocity command published back, so rest cannot be confirmed"


class BrakingRobot(ho.SimulatedHandoverRobot):
    """The stand-in, whose base is still moving after the decline until `come_to_rest`."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.braking = False
        self.stop_at_handover = None   # a stop to land as the handover is asked for, once

    def hand_over_teleop(self, robot_id: str, action_id: str, t: float, why: str) -> bool:
        stop, self.stop_at_handover = self.stop_at_handover, None
        if stop is not None:
            stop()
        return super().hand_over_teleop(robot_id, action_id, t, why)

    def is_stopped(self, robot_id: str) -> bool:
        return not self.braking and super().is_stopped(robot_id)

    def come_to_rest(self) -> None:
        self.braking = False


class NoRestEvidenceRobot(ho.SimulatedHandoverRobot):
    """The stand-in, declared as a base whose odometry cannot confirm rest (C5.3)."""

    def rest_evidence_gap(self, robot_id: str) -> str:
        return GAP


def rig(rest_s: float = 3600.0, cls=BrakingRobot, **kw):
    from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
    robot = cls(**kw)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock())
    a._handover_rest_s = rest_s     # the bound, on the job's clock; long unless a test needs it
    return a, robot


def declined_moving(a, robot, **assign_kw) -> list:
    """Assign; Nav2 declines, and at that moment the base is still moving."""
    robot.decline_next(WHY, on_read=lambda: setattr(robot, "braking", True))
    res, msgs = assign(a, payload(), **assign_kw)
    assert res.accepted
    collect(a, msgs, lambda _ms: robot.braking)
    return msgs


def presses(robot, n: int = 5) -> list:
    """The operator presses keys; what reached the base."""
    k = len(robot.cmd_vel)
    for _ in range(n):
        robot.teleop(0.15, 0.0, 0.4, ROBOT)
    return robot.cmd_vel[k:]


def waiting(msgs) -> list:
    return [s for s in statuses(msgs, ACTION)
            if s.state == pb.ACTION_STATE_RUNNING and "confirmed at rest" in s.message]


# ── the wait itself ───────────────────────────────────────────────────────

def test_the_operator_cannot_drive_while_the_robot_may_still_be_moving() -> None:
    a, robot = rig()
    msgs = declined_moving(a, robot)
    settle(a, msgs, 0.2)
    assert presses(robot) == []
    assert not turned(msgs)
    assert robot.cmd_vel and robot.cmd_vel[-1] == md.ZERO     # the base was told to hold
    assert waiting(msgs) and WHY in waiting(msgs)[0].message  # and the operator is told why
    assert renew_running(a)                                    # the action and its lease stand
    robot.come_to_rest()
    collect(a, msgs, turned)
    (c, *_) = presses(robot, 1)
    assert c != md.ZERO                                        # now the operator drives


def test_rest_not_confirmed_within_the_bound_ends_it_failed_and_never_opens() -> None:
    a, robot = rig(rest_s=2.0)
    msgs = declined_moving(a, robot)
    assert presses(robot) == []
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert "rest" in end[0].message and "2 s" in end[0].message
    assert not turned(msgs) and not robot.teleop_open(ROBOT)
    robot.come_to_rest()                                       # rest after the bound opens nothing
    settle(a, msgs)
    assert presses(robot) == [] and not turned(msgs)


def test_the_default_bound_is_the_estops_confirmed_stop_budget() -> None:
    from fleet_adapter_ros2.adapter import _ESTOP_TICK_S, _ESTOP_TICKS
    from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
    robot = BrakingRobot()
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    assert a._handover_rest_s == pytest.approx(_ESTOP_TICK_S * _ESTOP_TICKS) == 2.0


# ── a stop of any scope wins at every point of the wait ───────────────────

@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_during_the_rest_wait_wins_and_teleop_never_opens(stop) -> None:
    a, robot = rig()
    msgs = declined_moving(a, robot)
    settle(a, msgs, 0.1)
    assert presses(robot) == []
    STOPS[stop](a)
    robot.come_to_rest()                                       # rest after the stop opens nothing
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert presses(robot) == []
    assert not turned(msgs) and not robot.teleop_open(ROBOT)
    end = terminals(msgs, ACTION)
    assert len(end) == 1 and end[0].state != pb.ACTION_STATE_SUCCEEDED
    assert not renew_running(a)


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_landing_between_the_rest_and_the_handover_wins(stop) -> None:
    """The job has seen the robot at rest, and the stop lands before the binding turns the
    gate: the binding's own check, under its lock, refuses."""
    a, robot = rig()
    msgs = declined_moving(a, robot)
    settle(a, msgs, 0.1)
    assert presses(robot) == []
    robot.stop_at_handover = lambda: STOPS[stop](a)
    robot.come_to_rest()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert robot.stop_at_handover is None                      # the stop did land there
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)
    end = terminals(msgs, ACTION)
    assert len(end) == 1 and end[0].state != pb.ACTION_STATE_SUCCEEDED


def test_a_lease_deadline_passing_during_the_wait_ends_it_failed_with_no_teleop() -> None:
    a, robot = rig()
    msgs = declined_moving(a, robot, lease_ms=80)
    assert presses(robot) == []
    time.sleep(0.12)                                           # past it; `_tick_lease` never ran
    robot.come_to_rest()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_FAILED]
    assert presses(robot) == [] and not turned(msgs)


def test_a_lease_lapse_during_the_wait_ends_it_with_no_teleop() -> None:
    """The lease thread's lapse stops the robot (C4.2). Still braking, rest is not confirmed
    within the stop's budget, so the lapse reports FAILED with LEASE_EXPIRED; the gate never
    turned."""
    a, robot = rig()
    msgs = declined_moving(a, robot, lease_ms=80)
    assert presses(robot) == []
    time.sleep(0.12)
    a._tick_lease()
    robot.come_to_rest()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert [(s.state, s.end_cause) for s in terminals(msgs, ACTION)] == [
        (pb.ACTION_STATE_FAILED, pb.ACTION_END_CAUSE_LEASE_EXPIRED)]
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)


# ── a binding that cannot confirm rest never hands over ───────────────────

def test_a_binding_that_cannot_confirm_rest_ends_it_at_the_decline_and_says_why() -> None:
    a, robot = rig(cls=NoRestEvidenceRobot)
    robot.decline_next(WHY)
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert GAP in end[0].message and WHY in end[0].message
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)


def test_on_such_a_binding_nav2_reaching_the_target_still_succeeds() -> None:
    a, _ = rig(cls=NoRestEvidenceRobot)
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_SUCCEEDED]


# ── the binding checks rest at the handover itself, under its lock ────────

def _armed_standin(cls=BrakingRobot):
    robot = cls()
    robot.spawn(ROBOT)
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    robot.open_teleop(ROBOT, ACTION, drive, lease_s=60.0)
    return robot, drive


def test_the_standin_refuses_a_handover_while_its_base_is_moving() -> None:
    robot, drive = _armed_standin()
    robot.braking = True
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)
    assert drive.phase != md.RUNNING and presses(robot) == []
    robot.come_to_rest()
    assert robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)
    assert drive.phase == md.RUNNING


def test_the_standin_with_no_rest_evidence_refuses_every_handover() -> None:
    robot, drive = _armed_standin(NoRestEvidenceRobot)
    assert robot.is_stopped(ROBOT)
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)
    assert drive.phase != md.RUNNING and presses(robot) == []


def _armed_nav2(odometry_source):
    b, pub = binding()
    b._odometry_source = odometry_source
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    b.open_teleop(ROBOT, ACTION, drive, lease_s=60.0)
    with b._lock:
        b._goal = task_map.GoalPose(x=2.0, y=1.0, yaw=0.0)
        b._holding, b._goal_status = False, 0
    decline(b)
    return b, pub, drive


@pytest.mark.parametrize("source", ["command-derived", None])
def test_nav2_with_odometry_that_cannot_confirm_rest_never_hands_over(source) -> None:
    """Command-derived odometry reads zero the instant the stop is commanded, braking or not;
    undeclared odometry is treated the same (C5.3). Zero on that channel proves nothing."""
    b, pub, drive = _armed_nav2(source)
    b.on_odom(_odom(0.0))
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert drive.phase == md.NAVIGATING
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n


def test_nav2_with_measured_odometry_hands_over_only_at_rest() -> None:
    b, pub, drive = _armed_nav2("measured")
    b.on_odom(_odom(0.3))                                      # still moving
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert drive.phase == md.NAVIGATING
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n
    b.on_odom(_odom(0.0))                                      # at rest, measured
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert drive.phase == md.RUNNING


def test_nav2_with_no_odom_yet_never_hands_over() -> None:
    b, _, drive = _armed_nav2("measured")
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert drive.phase == md.NAVIGATING



# ── the stand-in a run uses, and the bound on the command line ───────────

def test_the_standins_coast_holds_the_gate_armed_until_its_base_stops() -> None:
    """`--sim-coast-s`: after the decline's hold the stand-in's base reports motion for that long,
    on its own clock, so a run shows the wait before the operator's turn."""
    a, robot = rig(cls=ho.SimulatedHandoverRobot, coast_s=0.4)
    robot.decline_next(WHY)
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, waiting)
    assert not robot.is_stopped(ROBOT) and presses(robot) == [] and not turned(msgs)
    collect(a, msgs, turned)
    assert robot.is_stopped(ROBOT)
    (c, *_) = presses(robot, 1)
    assert c != md.ZERO


@pytest.mark.parametrize("bound", ["0", "0.05", "31", "nan"])
def test_a_bound_outside_its_range_is_refused_before_anything_connects(bound) -> None:
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m", "fleet_adapter_ros2.adapter", "--binding", "simulated",
         "--endpoint", "127.0.0.1:1", "--handover-rest-s", bound],
        capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 2 and "REFUSED: --handover-rest-s" in out.stderr
