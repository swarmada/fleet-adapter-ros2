# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A navigate-handover drives to its target with Nav2 and, when Nav2 declines, hands the robot to
the operator under the same action and the same lease. A stop always wins.

The verb is `Custom` with ``operation: navigate-handover``. The teleop gate is opened in the step
that grants the lease, armed but forwarding nothing while Nav2 drives. Only a decline turns it to
the operator, and only through the binding's own lock: if a stop closed the gate, or the lease's
deadline passed, the handover is refused and the action ends Failed. A teleop message never opens
the gate, and nothing but a new assignment reopens it. The adapter sees one Estop message whatever
the stop's scope (zone, namespace or robot), so the estop rows below stand for all three.

Run through the adapter, against the stand-in robot, so the assignment, the lease, the renewal and
every stop path are the shipped ones.
"""

from __future__ import annotations

import json
import time

import pytest
from test_manual_drive_gate import (
    ROBOT,
    StepClock,
    collect,
    command,
    drain,
    settle,
    statuses,
    terminals,
)

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter

ACTION = "handover-a"
TURN = "operator's turn"


def payload(**kw) -> bytes:
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3, **kw}
    return json.dumps(doc).encode()


def rig(**robot_kw):
    robot = ho.SimulatedHandoverRobot(**robot_kw)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock())
    return a, robot


def assign(a: Adapter, body: bytes, *, action: str = ACTION, token: int = 1,
           lease_ms: int = 60_000, action_type: str = "Custom"):
    command(a, token, assign_action=pb.AssignAction(
        action_id=action, action_type=action_type, assignment_id=f"a-{action}",
        fencing_token=token, lease_generation=1, lease_duration_ms=lease_ms,
        payload_json=body))
    out = drain(a)
    # Everything said from the assignment on is kept: the job runs at once, so a decline's turn,
    # or a stop a test lands from the job's thread, can come before this drain.
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"
           and m.command_result.HasField("assign_action")]
    assert len(res) == 1
    return res[0], list(out)


def turned(msgs) -> bool:
    return any(TURN in s.message for s in statuses(msgs, ACTION)
               if s.state == pb.ACTION_STATE_RUNNING)


def renew_running(a: Adapter) -> bool:
    command(a, 99, renew_lease=pb.RenewActionLease(
        action_id=ACTION, lease_generation=1, lease_duration_ms=60_000))
    return [m.command_result.renew_lease for m in drain(a)
            if m.WhichOneof("payload") == "command_result"][-1].running


def handed_over(a: Adapter, robot) -> list:
    """Assign, let Nav2 decline, and wait for the operator's turn."""
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, turned)
    return msgs


# ── Nav2 first: the operator is not involved ─────────────────────────────

def test_nav2_reaches_the_target_and_it_succeeds_without_the_operator() -> None:
    a, robot = rig()
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert "nav2" in end[0].message.lower()
    assert not turned(msgs)
    assert not robot.teleop_open(ROBOT) and a._current_action == ""


def test_while_nav2_drives_teleop_is_not_forwarded() -> None:
    a, robot = rig()
    robot.stall_next()                           # the goal is accepted and never arrives
    res, _ = assign(a, payload())
    assert res.accepted
    n = len(robot.cmd_vel)
    for _ in range(3):
        robot.teleop(0.1, 0.0, 0.3)
    assert len(robot.cmd_vel) == n
    assert robot.teleop_open(ROBOT, ACTION)      # armed for this action, not forwarding


# ── declined: the operator's turn, under the same lease ──────────────────

def test_a_decline_hands_over_and_the_operator_finishes_it() -> None:
    a, robot = rig()
    msgs = handed_over(a, robot)
    assert [s.state for s in terminals(msgs, ACTION)] == []
    turn = [s for s in statuses(msgs, ACTION) if TURN in s.message]
    assert "aborted the goal" in turn[0].message  # the operator is told why
    assert robot.cmd_vel[-1] == md.ZERO            # the base is held at the handover
    n = len(robot.cmd_vel)
    robot.teleop(0.5, 0.5, 5.0)
    (c,) = robot.cmd_vel[n:]
    assert abs((c[0] ** 2 + c[1] ** 2) ** 0.5 - md.CAP_LINEAR_MPS) < 1e-9 and c[2] == 0.7
    assert renew_running(a)
    robot.place(2.1, 1.05)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert "operator" in end[0].message
    assert robot.cmd_vel[-1] == md.ZERO
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n and a._current_action == ""


def test_a_handover_with_no_lease_is_declined_and_nothing_opens() -> None:
    a, robot = rig()
    res, _ = assign(a, payload(), lease_ms=0)
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert "needs a lease" in res.message
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == [] and not robot.teleop_open(ROBOT)


# ── no lease, no operator: the deadline is checked at the handover itself ─

def test_a_lease_deadline_passed_before_the_decline_blocks_the_handover() -> None:
    a, robot = rig()
    robot.stall_next()
    res, msgs = assign(a, payload(), lease_ms=60)
    assert res.accepted
    time.sleep(0.09)                             # past the deadline; `_tick_lease` never ran
    robot.decline_now("the navigation server aborted the goal")
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    n = len(robot.cmd_vel)
    for _ in range(3):
        robot.teleop(0.1, 0.0, 0.2)
    assert len(robot.cmd_vel) == n, robot.cmd_vel[n:]
    assert not turned(msgs)
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_FAILED]


# ── a stop always wins: before the handover, at it, and after it ─────────

def _estop(a: Adapter) -> None:
    a._estop("e-1", lambda _ack: None)


def _pause(a: Adapter) -> None:
    command(a, 7, pause=pb.PauseCapabilities(require_stop_before_ack=True))


def _cancel(a: Adapter) -> None:
    command(a, 8, cancel_action=pb.CancelAction(action_id=ACTION, fencing_token=1,
                                                cause=pb.CANCEL_CAUSE_OPERATOR))


STOPS = {"estop": _estop, "pause": _pause, "cancel": _cancel}


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_that_lands_between_the_decline_and_the_handover_wins(stop) -> None:
    a, robot = rig()
    # Nav2's decline is read, and the stop arrives before the job hands the robot over.
    robot.decline_next("the navigation server aborted the goal",
                       on_read=lambda: STOPS[stop](a))
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert not robot.teleop_open(ROBOT)
    n = len(robot.cmd_vel)
    for _ in range(5):                           # the operator presses anyway
        robot.teleop(0.15, 0.0, 0.4)
    assert len(robot.cmd_vel) == n, robot.cmd_vel[n:]
    assert not turned(msgs)
    end = terminals(msgs, ACTION)
    assert len(end) == 1 and end[0].state != pb.ACTION_STATE_SUCCEEDED
    assert not renew_running(a)


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_while_nav2_drives_ends_it_and_never_hands_over(stop) -> None:
    a, robot = rig()
    robot.stall_next()
    res, msgs = assign(a, payload())
    assert res.accepted
    STOPS[stop](a)
    robot.decline_now("the navigation server aborted the goal")   # a late answer to the cancel
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    n = len(robot.cmd_vel)
    robot.teleop(0.15, 0.0, 0.4)
    assert len(robot.cmd_vel) == n
    assert not turned(msgs)
    assert len(terminals(msgs, ACTION)) == 1


def test_an_estop_mid_handover_closes_the_gate_before_its_zero_and_ends_it_once() -> None:
    a, robot = rig()
    msgs = handed_over(a, robot)
    robot.teleop(0.15, 0.0, 0.4)
    acks: list = []
    a._estop("e-1", acks.append)
    assert acks[-1].state == pb.ESTOP_STATE_STOPPED
    assert robot.cmd_vel[-1] == md.ZERO
    n = len(robot.cmd_vel)
    for _ in range(5):
        robot.teleop(0.15, 0.0, 0.4)
    assert len(robot.cmd_vel) == n
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED] and "stop" in end[0].message
    assert not renew_running(a)                  # the task does not continue on this one


def test_a_lapse_mid_handover_ends_it_stopped_and_closes_the_gate() -> None:
    a, robot = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload(), lease_ms=150)
    assert res.accepted
    collect(a, msgs, turned)
    robot.teleop(0.1, 0.0, 0.0)
    time.sleep(0.2)
    a._tick_lease()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    assert [(s.state, s.end_cause) for s in terminals(msgs, ACTION)] == [
        (pb.ACTION_STATE_STOPPED, pb.ACTION_END_CAUSE_LEASE_EXPIRED)]
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n and robot.cmd_vel[-1] == md.ZERO


def test_a_cancel_mid_handover_ends_it_once_as_cancelled() -> None:
    a, robot = rig()
    msgs = handed_over(a, robot)
    _cancel(a)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_CANCELLED]
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n


def test_a_pause_mid_handover_ends_it_and_a_resume_does_not_reopen_teleop() -> None:
    a, robot = rig()
    msgs = handed_over(a, robot)
    _pause(a)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_FAILED]
    command(a, 9, resume=pb.ResumeCapabilities())
    assert not robot.teleop_open(ROBOT)
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n


def test_a_new_assignment_ends_the_handover_and_its_teleop_with_it() -> None:
    a, robot = rig()
    handed_over(a, robot)
    assert assign(a, b'{"x": 1, "y": 2}', token=2, action="nav-b",
                  action_type="Navigate")[0].accepted
    assert not robot.teleop_open(ROBOT)
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n


def test_after_a_restart_the_reconcile_ends_it_and_teleop_stays_closed() -> None:
    a, robot = rig()
    a._reconcile_action_state(pb.RegisterAck(
        accepted=True, authoritative_action_state=pb.RobotActionState(
            action_id=ACTION, phase=pb.ROBOT_ACTION_PHASE_IN_PROGRESS, fencing_token=1)))
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_CANCELLED]
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == [md.ZERO]


def test_with_the_estop_path_down_a_handover_is_refused() -> None:
    a, robot = rig()
    a._safety_required = True
    res, _ = assign(a, payload())
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert robot.cmd_vel == []


# ── the catalog and the payload ──────────────────────────────────────────

def _custom(a: Adapter):
    snap = a._capabilities_snapshot()
    return [s for s in snap.supported_actions if s.action_type == "Custom"]


def test_a_binding_with_teleop_declares_both_operator_verbs_in_one_custom_entry() -> None:
    a, _ = rig()
    (entry,) = _custom(a)
    op = next(p for p in entry.params if p.name == "operation")
    assert set(op.allowed) == {md.OPERATION, ho.OPERATION}
    b, _ = rig(teleop_topic=None)
    assert _custom(b) == []
    res, _ = assign(b, payload())
    assert not res.accepted and "no teleop topic" in res.message


@pytest.mark.parametrize("bad", [{"frame": "odom"}, {"radius_m": 5.0}, {"max_speed_mps": 0.5},
                                 {"teleop_topic": "/cmd_vel"}, {"x": "here"}])
def test_an_unsafe_payload_is_declined_before_anything_moves(bad) -> None:
    a, robot = rig()
    res, _ = assign(a, payload(**bad))
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert robot.cmd_vel == [] and robot.task_state(ROBOT) == ""


def test_a_manual_drive_payload_is_not_read_as_a_handover() -> None:
    assert not ho.is_navigate_handover("Custom", {"operation": md.OPERATION})
    assert not ho.is_navigate_handover("Navigate", {"operation": ho.OPERATION})
    assert ho.is_navigate_handover("Custom", payload())
    assert not md.is_manual_drive("Custom", payload())


def test_a_pause_while_nav2_drives_ends_it_and_a_resume_does_not_drive_its_goal() -> None:
    """A pause keeps a Navigate's target for its resume; this action ended on the pause, so its
    target must not survive it: a resume would drive the robot with no action and no lease."""
    a, robot = rig()
    res, msgs = assign(a, payload(x=40.0, y=0.0))   # far: the goal is still in flight
    assert res.accepted
    assert robot.task_state(ROBOT) == "ACTION_STATE_RUNNING"
    _pause(a)
    assert robot._left_s                         # the stand-in kept the travel still to go
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs)
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_FAILED]
    command(a, 9, resume=pb.ResumeCapabilities())
    drain(a)
    # No travel was re-armed and no goal runs. (`is_stopped` is not asserted: the stand-in's
    # resume reports motion even with nothing to resume, a defect of the stand-in itself.)
    assert ROBOT not in robot._arrive_at and not robot._left_s
    assert robot.task_state(ROBOT) != "ACTION_STATE_RUNNING"
