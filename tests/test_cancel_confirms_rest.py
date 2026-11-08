# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A cancel says STOPPED_SAFELY only once the robot is observed at rest.

A cancel was acknowledged CANCEL_DISPOSITION_STOPPED_SAFELY as soon as its stop had settled
(no goal sent before it still unanswered), without looking at the robot. STOPPED_SAFELY lets the
control plane requeue the action to another robot, so a cancel acknowledged while this one still
braked, or while a goal whose answer was lost might be driving it, was two executors. The estop
(C5.2/C5.3) and the lease self-stop (C4.2) already confirm rest before they say "stopped"; the
cancel now takes the same rule and budget. Until rest is confirmed it is NOT acknowledged, says
why, and keeps the action and its lease, so the plane can retry and the lease still self-stops.

Driven through `run_control_stream`, as the control plane drives it.
"""

from __future__ import annotations

import threading
import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from test_stop_never_waits import ROBOT, _assign, _first, _Handle, _odom
from test_unanswered_goal import (  # noqa: F401 - rig and ros_msgs are fixtures
    _answer,
    _goal_lost_then_another_accepted,
    _result,
    rig,
    ros_msgs,
)

STOP_S = 2.0   # the confirmed-stop budget, 40 looks 0.05 s apart


def _accepted(a, b, server, cp):
    """t-1 assigned, sent and accepted: the robot is executing it, with a handle."""
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    h = _Handle()
    _answer(server, 0, h)
    assert b._goal_handle is h
    return h


def _cancel(cp, command_id: int = 2, action_id: str = "t-1", token: int = 1) -> None:
    cp.send(command_id, cancel_action=pb.CancelAction(
        action_id=action_id, fencing_token=token, cause=pb.CANCEL_CAUSE_OPERATOR))


def _statuses(a, action_id: str) -> list:
    out = []
    while not a._outbox.empty():
        m = a._outbox.get_nowait()
        if m.WhichOneof("payload") == "action_status" and m.action_status.action_id == action_id:
            out.append(m.action_status.state)
    return out


def test_a_cancel_of_a_moving_robot_is_acknowledged_only_once_rest_is_observed(rig):  # noqa: F811
    a, b, server, cp = rig
    h = _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))                                  # still driving
    rested = {}

    def brake() -> None:
        time.sleep(0.15)
        rested["at"] = time.monotonic()
        b.on_odom(_odom(0.0))                              # the base reaches rest
    threading.Thread(target=brake, daemon=True).start()
    _cancel(cp)
    res = _result(a, "cancel_action", STOP_S + 0.5)
    answered = time.monotonic()
    assert res is not None
    assert "at" in rested and answered >= rested["at"], "STOPPED_SAFELY before rest was observed"
    assert res.command_result.cancel_action.acknowledged is True
    assert res.command_result.cancel_action.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert h.cancels == 1
    assert a._current_action == "" and ROBOT not in a._lease._deadline


def test_a_cancel_of_a_robot_never_observed_at_rest_is_not_acknowledged(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))                                  # and it never stops
    t0 = time.monotonic()
    _cancel(cp)
    res = _result(a, "cancel_action", STOP_S + 0.5)
    assert res is not None and time.monotonic() - t0 >= STOP_S - 0.1
    c = res.command_result.cancel_action
    assert c.acknowledged is False
    assert "rest was not observed within 2 s" in c.message
    assert pb.ACTION_STATE_CANCELLED not in _statuses(a, "t-1")
    assert a._current_action == "t-1" and ROBOT in a._lease._deadline   # the lease still guards


def test_a_cancel_after_a_lost_answer_is_not_acknowledged_while_the_robot_moves(rig):  # noqa: F811
    a, b, server, cp = rig
    _goal_lost_then_another_accepted(a, b, server, cp)     # t-1's goal abandoned, unanswered
    b.on_odom(_odom(0.3))                                  # the robot is moving
    _cancel(cp, command_id=3, action_id="t-2", token=2)
    res = _result(a, "cancel_action", STOP_S + 0.5)
    assert res is not None
    assert res.command_result.cancel_action.acknowledged is False
    assert a._current_action == "t-2"


def test_a_cancel_on_a_dead_transport_is_answered_at_once_and_says_why(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b._record_spin_exit(RuntimeError("executor gone"))     # the transport dies
    t0 = time.monotonic()
    _cancel(cp)
    res = _result(a, "cancel_action", 0.5)
    assert res is not None and time.monotonic() - t0 <= 0.5
    c = res.command_result.cancel_action
    assert c.acknowledged is False
    assert "rest cannot be read from the robot" in c.message and "executor gone" in c.message


def test_a_cancel_of_a_robot_at_rest_is_acknowledged_at_once(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)                            # odom says rest
    t0 = time.monotonic()
    _cancel(cp)
    res = _result(a, "cancel_action", 0.5)
    assert res is not None and time.monotonic() - t0 <= 0.2
    assert res.command_result.cancel_action.acknowledged is True


def test_control_messages_keep_flowing_while_a_cancel_waits_for_rest(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))
    _cancel(cp)
    t0 = time.monotonic()
    cp.send(3, renew_lease=pb.RenewActionLease(
        action_id="t-1", lease_generation=1, lease_duration_ms=60_000))
    seen = {}
    _first(a._outbox, lambda m: m.WhichOneof("payload") == "command_result"
           and seen.setdefault(m.command_result.WhichOneof("result"), m) is not None
           and "renew_lease" in seen, 0.2)
    assert "renew_lease" in seen, "the renewal waited behind the cancel"
    assert time.monotonic() - t0 <= 0.2
    assert "cancel_action" not in seen                     # still braking: not answered yet
    b.on_odom(_odom(0.0))
    res = _result(a, "cancel_action", STOP_S)
    assert res is not None and res.command_result.cancel_action.acknowledged is True


def test_a_renewal_read_while_a_cancel_confirms_rest_reports_the_task_not_running(rig):  # noqa: F811
    # C4.4: once the cancel has commanded the stop the task is no longer executing, even while
    # the robot brakes and the action and its lease are kept until rest is confirmed.
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))
    _cancel(cp)
    cp.send(3, renew_lease=pb.RenewActionLease(
        action_id="t-1", lease_generation=1, lease_duration_ms=60_000))
    renew = _result(a, "renew_lease", 0.2)
    assert renew is not None
    assert a._current_action == "t-1"                      # kept: rest not confirmed yet
    assert renew.command_result.renew_lease.running is False


def test_a_task_whose_cancel_was_not_acknowledged_is_not_reported_running(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))                                  # never at rest
    _cancel(cp)
    res = _result(a, "cancel_action", STOP_S + 0.5)
    assert res is not None and res.command_result.cancel_action.acknowledged is False
    cp.send(3, renew_lease=pb.RenewActionLease(
        action_id="t-1", lease_generation=1, lease_duration_ms=60_000))
    renew = _result(a, "renew_lease", 0.5)
    assert renew is not None and renew.command_result.renew_lease.running is False


def test_a_new_assignment_after_a_cancel_is_reported_running(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)                            # at rest: acknowledged at once
    _cancel(cp)
    assert _result(a, "cancel_action", 0.5).command_result.cancel_action.acknowledged
    cp.send(3, assign_action=_assign("t-2", 2))
    assert _result(a, "assign_action", 0.5)
    cp.send(4, renew_lease=pb.RenewActionLease(
        action_id="t-2", lease_generation=1, lease_duration_ms=60_000))
    renew = _result(a, "renew_lease", 0.5)
    assert renew is not None and renew.command_result.renew_lease.running is True
