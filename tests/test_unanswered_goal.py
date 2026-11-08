# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A goal whose answer never comes, and a cancel that waits for one.

A sent goal stays unanswered until the navigation server answers it. When that answer is lost,
the goal used to stay unanswered for ever: every later stop recorded it again and none settled,
so every later estop ended FAILED, every cancel went unacknowledged, and the goal, if the server
had accepted it, could not be cancelled. Now a sent goal has a bounded life,
`--goal-response-bound`. Past it the goal is abandoned: cancelled by the id it was sent with,
the robot stopped and the goal FAILED with the reason when it was the current goal, and a late
acceptance cancelled on arrival. No later stop waits on it.

And a cancel no longer holds the control thread while it waits for such an answer: the control
thread goes on reading, and the cancel is answered when its stop has settled.

Every test drives the real `_send_nav2_goal` and the adapter's real entry points
(`run_control_stream`, `run_safety_stream`, the lease thread).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from test_stall_reporting import _main
from test_stop_never_waits import (  # noqa: F401 - ros_msgs is a fixture
    BUDGET_S,
    ROBOT,
    _assign,
    _ControlPlane,
    _first,
    _Handle,
    _odom,
    _Publisher,
    _SafetyStub,
    _Server,
    _Twist,
    ros_msgs,
)

from conftest import mark_transport_live
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

BOUND_S = 0.3   # the goal response bound on a shorter clock; the rules are the same
STOP_S = 2.0    # the confirmed-stop budget, 40 looks 0.05 s apart


class _IdServer(_Server):
    """`_Server`, recording the id each goal was sent with."""

    def __init__(self) -> None:
        super().__init__(up=True)
        self.ids: list = []

    def send_goal_async(self, goal, feedback_callback=None, goal_uuid=None):
        self.ids.append(goal_uuid)
        return super().send_goal_async(goal, feedback_callback)


@pytest.fixture
def rig(ros_msgs):  # noqa: F811 - the imported fixture
    """The adapter, its control thread and lease thread running, on a ROS-free Nav2Binding with
    a goal response bound, whose server is up and answers only when a test says so. Goals
    cancelled by id are recorded in `b.cancelled_ids`."""
    server = _IdServer()
    b = mark_transport_live(Nav2Binding(ROBOT, response_bound_s=BOUND_S, odometry_source="measured"))
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b.cancelled_ids = []
    b._cancel_goal_by_id = b.cancelled_ids.append
    b.start = lambda: None
    b.on_odom(_odom(0.0))                                  # odom says rest
    a = Adapter(pb, None, b, ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    assert cp.opened.wait(2.0)
    yield a, b, server, cp
    a._shutdown.set()
    cp.q.put(None)
    control.join(timeout=5.0)


def _answer(server: _Server, n: int, handle) -> None:
    """The server's answer to the n-th goal sent, as the executor delivers it."""
    server.futures[n].cb(NS(result=lambda: handle))


def _result(a: Adapter, kind: str, within: float):
    return _first(a._outbox, lambda m: m.WhichOneof("payload") == "command_result"
                  and m.command_result.WhichOneof("result") == kind, within)


def _goal_lost_then_another_accepted(a, b, server, cp, lease_ms: int = 60_000):
    """t-1's goal is sent and its answer lost; t-2's goal is sent and accepted. Returns t-2's
    handle, once t-1's goal has been abandoned."""
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    cp.send(2, assign_action=_assign("t-2", 2, lease_ms=lease_ms))
    assert _result(a, "assign_action", 2.0)
    h2 = _Handle()
    _answer(server, 1, h2)                                 # t-2 accepted and adopted
    assert b._goal_handle is h2
    time.sleep(BOUND_S + 0.1)                              # t-1's answer never comes
    return h2


# ── the bound, and what happens past it ──────────────────────────────────────────────────

def test_a_lost_answer_to_a_superseded_goal_is_abandoned_without_touching_the_current_goal(rig):
    a, b, server, cp = rig
    h2 = _goal_lost_then_another_accepted(a, b, server, cp)
    assert b.cancelled_ids == [server.ids[0]]              # t-1 cancelled by the id it was sent with
    assert h2.cancels == 0 and b._goal_handle is h2        # t-2 still executing, still held
    assert b._cmd_vel.published == []                      # and the robot was not stopped
    assert b.stop_settled(ROBOT, 0) is True


def test_after_a_lost_answer_a_later_estop_is_a_single_stopped_within_500_ms(rig):
    a, b, server, cp = rig
    h2 = _goal_lost_then_another_accepted(a, b, server, cp)
    t0 = time.monotonic()
    threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),), daemon=True).start()
    first = _first(a._safety_outbox, lambda m: True, BUDGET_S)
    assert first is not None, "no EstopAck within 500 ms"
    assert time.monotonic() - t0 <= BUDGET_S
    assert first.estop_ack.state == pb.ESTOP_STATE_STOPPED   # odom rest, nothing unanswered
    assert h2.cancels == 1 and b._cmd_vel.published          # the current goal stopped


def test_after_a_lost_answer_a_later_cancel_is_acknowledged_within_500_ms(rig):
    a, b, server, cp = rig
    h2 = _goal_lost_then_another_accepted(a, b, server, cp)
    t0 = time.monotonic()
    cp.send(3, cancel_action=pb.CancelAction(
        action_id="t-2", fencing_token=2, cause=pb.CANCEL_CAUSE_OPERATOR))
    res = _result(a, "cancel_action", BUDGET_S)
    assert res is not None, "no CancelActionResult within 500 ms"
    assert time.monotonic() - t0 <= BUDGET_S
    assert res.command_result.cancel_action.acknowledged is True
    assert res.command_result.cancel_action.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert h2.cancels == 1


def test_after_a_lost_answer_a_later_lease_self_stop_reports_stopped_within_500_ms(rig):
    a, b, server, cp = rig
    t_assign = time.monotonic()
    h2 = _goal_lost_then_another_accepted(a, b, server, cp, lease_ms=800)
    lapse = t_assign + 0.8
    assert time.monotonic() < lapse                         # abandoned before t-2's lease lapses
    upd = _first(a._outbox, lambda m: m.WhichOneof("payload") == "action_status"
                 and m.action_status.action_id == "t-2"
                 and m.action_status.state != pb.ACTION_STATE_RUNNING, 2.0)
    assert upd is not None, "no terminal action_status for t-2"
    assert time.monotonic() - lapse <= BUDGET_S
    assert upd.action_status.state == pb.ACTION_STATE_STOPPED
    assert upd.action_status.end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    assert h2.cancels == 1


def test_a_stop_in_the_window_is_confirmed_within_its_budget_once_the_goal_is_abandoned(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)                # sent; its answer never comes
    t0 = time.monotonic()
    threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),), daemon=True).start()
    first = _first(a._safety_outbox, lambda m: True, BUDGET_S)
    assert first is not None and time.monotonic() - t0 <= BUDGET_S
    assert first.estop_ack.state == pb.ESTOP_STATE_STOPPING  # the goal may still be accepted
    last = _first(a._safety_outbox, lambda m: True, STOP_S)
    assert last is not None, "no terminal EstopAck within the confirmed-stop budget"
    assert time.monotonic() - t0 <= STOP_S + 0.1
    assert last.estop_ack.state == pb.ESTOP_STATE_STOPPED    # abandoned, then rest observed
    assert b.cancelled_ids == [server.ids[0]]


def test_a_current_goal_whose_answer_is_lost_is_stopped_and_failed_with_the_reason(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    time.sleep(BOUND_S + 0.1)
    assert b.cancelled_ids == [server.ids[0]]
    assert b._cmd_vel.published and all(
        t.linear.x == 0.0 and t.angular.z == 0.0 for t in b._cmd_vel.published)
    a._maybe_emit_task_progress()                          # the telemetry loop's terminal path
    upd = _first(a._outbox, lambda m: m.WhichOneof("payload") == "action_status"
                 and m.action_status.state != pb.ACTION_STATE_RUNNING, 0.5)
    assert upd is not None
    assert upd.action_status.state == pb.ACTION_STATE_FAILED
    assert upd.action_status.end_cause == pb.ACTION_END_CAUSE_EXECUTION_FAILURE
    assert "did not answer the goal within 0.3 s" in upd.action_status.message
    assert a._current_action == ""                         # over, and its lease released
    assert ROBOT not in a._lease._deadline


def test_a_late_accept_after_abandonment_is_cancelled_on_arrival_and_never_adopted(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    time.sleep(BOUND_S + 0.1)
    twists = len(b._cmd_vel.published)
    late = _Handle()
    late.on_result = None
    late.get_result_async = lambda: pytest.fail("an abandoned goal's result was awaited")
    _answer(server, 0, late)                               # accepted, after all
    assert late.cancels == 1                               # cancelled on arrival
    assert b._goal_handle is None                          # never adopted
    assert len(b._cmd_vel.published) == twists + 1         # and a zero Twist with the cancel
    assert all(t.linear.x == 0.0 and t.angular.z == 0.0 for t in b._cmd_vel.published)
    assert len(server.sent) == 1                           # nothing re-dispatched


def test_a_late_refusal_after_abandonment_changes_nothing(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    time.sleep(BOUND_S + 0.1)
    failure = b.task_failure(ROBOT)
    refused = _Handle()
    refused.accepted = False
    _answer(server, 0, refused)
    assert b.task_failure(ROBOT) == failure                # still the abandonment's reason


# ── the bound is a setting with no default ───────────────────────────────────────────────

def test_ros2_without_a_goal_response_bound_refuses_to_start_naming_it(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2",
                             "--stall-window", "5", "--stall-min-progress", "0.1")
    assert code == 2 and robot is None
    assert err.startswith("REFUSED: --binding ros2 needs --goal-response-bound")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_a_goal_response_bound_that_is_not_a_positive_number_is_refused(monkeypatch, capsys, value):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", value)
    assert code == 2 and robot is None and "--goal-response-bound must be" in err


def test_ros2_with_a_goal_response_bound_arms_it_and_says_so(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", "1.5",
                             "--odometry-source", "measured")
    assert code is None and isinstance(robot, Nav2Binding)
    assert robot._response_bound == 1.5
    assert "--goal-response-bound 1.5" in err


def test_the_simulated_binding_needs_no_goal_response_bound(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "simulated")
    assert code is None and robot is not None


@pytest.mark.parametrize("value", [0, -0.5, float("nan"), float("inf"), True, "1"])
def test_the_binding_refuses_a_bound_that_is_not_a_positive_number(value):
    with pytest.raises(ValueError):
        Nav2Binding(ROBOT, response_bound_s=value)


# ── a cancel waiting for an answer does not hold the control thread ──────────────────────

def test_control_messages_keep_flowing_while_a_cancel_waits_for_an_answer(rig):
    a, b, server, cp = rig
    b._response_bound = None                               # an answer that is slow, not lost
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    t0 = time.monotonic()
    cp.send(3, renew_lease=pb.RenewActionLease(
        action_id="t-1", lease_generation=1, lease_duration_ms=60_000))
    renew = _result(a, "renew_lease", 0.2)
    assert renew is not None, "the renewal waited behind the cancel"
    assert time.monotonic() - t0 <= 0.2
    assert renew.command_result.renew_lease.renewed is True
    assert a._outbox.empty()                               # the cancel is not answered yet
    h = _Handle()
    _answer(server, 0, h)                                  # the answer arrives: cancelled on arrival
    res = _result(a, "cancel_action", 0.5)
    assert res is not None
    assert res.command_result.command_id == 2
    assert res.command_result.cancel_action.acknowledged is True
    assert h.cancels == 1 and b._goal_handle is None


def test_an_assignment_read_while_a_cancel_waits_keeps_its_action_and_lease(rig):
    a, b, server, cp = rig
    b._response_bound = None
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    cp.send(3, assign_action=_assign("t-2", 2))
    assert _result(a, "assign_action", 0.2), "the assignment waited behind the cancel"
    _answer(server, 0, _Handle())                          # t-1 answered: the cancel settles
    res = _result(a, "cancel_action", 0.5)
    assert res is not None and res.command_result.cancel_action.acknowledged is True
    assert a._current_action == "t-2"                      # the cancel did not drop t-2
    assert ROBOT in a._lease._deadline                     # nor its lease


def test_a_cancel_whose_answer_never_comes_in_time_is_answered_unacknowledged_later(rig):
    a, b, server, cp = rig
    b._response_bound = None
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    t0 = time.monotonic()
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    cp.q.put(pb.ControlPlaneMessage(heartbeat=pb.HeartbeatRequest(robot_id=ROBOT)))
    hb = _first(a._outbox, lambda m: m.WhichOneof("payload") == "heartbeat", 0.2)
    assert hb is not None, "the heartbeat waited behind the cancel"
    res = _result(a, "cancel_action", STOP_S + 0.5)
    assert res is not None and time.monotonic() - t0 >= STOP_S - 0.1
    assert res.command_result.cancel_action.acknowledged is False
    assert a._current_action == "t-1" and ROBOT in a._lease._deadline
