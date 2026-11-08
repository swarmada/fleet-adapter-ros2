# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A cancel stops the action it names, and only that one (conformance C3.9).

The cancel stopped whatever action was current and never read `CancelAction.action_id`. The
control plane sends exactly that cancel after every in-place preemption: the robot switches to
the preemptor, then the preempted action's reconcile asks the robot to stop the preempted one,
by name. Fencing tokens are counted per action, so the token does not say it is not about the
running action; only the name does. So every preemptor was stopped the moment it started.

A cancel naming an action this robot is not executing is answered the way the reference
simulation adapter answers it, one of the two answers RFC-0001 accepts for it (the fencing
rule): a terminal acknowledgement for that action, STOPPED_SAFELY, since it is not moving the
robot. The running action, its goal and its lease are left alone.

Driven through `run_control_stream`, as the control plane drives it.
"""

from __future__ import annotations

import queue
import time

from test_cancel_confirms_rest import _statuses
from test_stop_never_waits import ROBOT, _assign, _Handle
from test_unanswered_goal import (  # noqa: F401 - rig and ros_msgs are fixtures
    _answer,
    _result,
    rig,
    ros_msgs,
)

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

_GS_SUCCEEDED = 4   # action_msgs/GoalStatus


def _cancel(cp, command_id: int, action_id: str, token: int) -> None:
    cp.send(command_id, cancel_action=pb.CancelAction(
        action_id=action_id, fencing_token=token, cause=pb.CANCEL_CAUSE_SUPERSEDED))


def _answered(a, kind: str, within: float):
    """The command result of `kind`, and the states of every action status read before it,
    as (result, [(action_id, state), ...]). `_result` drops those statuses, and a cancel's
    terminal status goes out before its answer."""
    seen = []
    deadline = time.monotonic() + within
    while (left := deadline - time.monotonic()) > 0:
        try:
            m = a._outbox.get(timeout=left)
        except queue.Empty:
            break
        which = m.WhichOneof("payload")
        if which == "action_status":
            seen.append((m.action_status.action_id, m.action_status.state))
        elif which == "command_result" and m.command_result.WhichOneof("result") == kind:
            return m.command_result, seen
    return None, seen


def _preempted_in_place(a, b, server, cp):
    """'t-preempted' assigned at token 15 and accepted, then 't-preemptor' in place at token 16
    and accepted. Returns the preemptor's goal handle."""
    cp.send(1, assign_action=_assign("t-preempted", 15))
    assert _result(a, "assign_action", 2.0).command_result.assign_action.accepted
    _answer(server, 0, _Handle())
    cp.send(2, assign_action=_assign("t-preemptor", 16))
    assert _result(a, "assign_action", 2.0).command_result.assign_action.accepted
    h = _Handle()
    _answer(server, 1, h)
    assert b._goal_handle is h and a._current_action == "t-preemptor"
    return h


def test_a_cancel_naming_a_preempted_action_leaves_the_preemptor_running_to_success(rig):  # noqa: F811
    a, b, server, cp = rig
    h = _preempted_in_place(a, b, server, cp)
    ended = []
    for command_id, token in ((3, 15), (4, 16)):
        _cancel(cp, command_id, "t-preempted", token)
        res, seen = _answered(a, "cancel_action", 1.0)
        assert res is not None, f"no answer to the cancel at token {token}"
        ended += [state for action, state in seen if action == "t-preemptor"]
        c = res.cancel_action
        assert c.acknowledged is True and c.stale is False
        assert c.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert h.cancels == 0, "the preemptor's goal was cancelled"
    assert a._current_action == "t-preemptor" and ROBOT in a._lease._deadline
    cp.send(5, renew_lease=pb.RenewActionLease(
        action_id="t-preemptor", lease_generation=1, lease_duration_ms=60_000))
    renew = _result(a, "renew_lease", 1.0)
    assert renew is not None and renew.command_result.renew_lease.running is True
    ended += _statuses(a, "t-preemptor")
    assert pb.ACTION_STATE_CANCELLED not in ended and pb.ACTION_STATE_FAILED not in ended
    b.on_goal_result(_GS_SUCCEEDED)
    a._maybe_emit_task_progress()
    assert _statuses(a, "t-preemptor") == [pb.ACTION_STATE_SUCCEEDED]


def test_a_cancel_naming_the_running_action_still_stops_it(rig):  # noqa: F811
    a, b, server, cp = rig
    h = _preempted_in_place(a, b, server, cp)
    _cancel(cp, 3, "t-preemptor", 16)
    res, seen = _answered(a, "cancel_action", 1.0)
    assert res is not None and res.cancel_action.acknowledged is True
    assert h.cancels == 1
    assert a._current_action == "" and ROBOT not in a._lease._deadline
    assert ("t-preemptor", pb.ACTION_STATE_CANCELLED) in seen
