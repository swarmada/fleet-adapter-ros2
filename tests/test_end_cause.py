# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Every terminal action report says how the action ended, or why it cannot.

`ActionStatusUpdate.end_cause` is REQUIRED on a terminal state other than SUCCEEDED, and an
adapter that cannot determine the cause MUST leave it unset rather than guess. So each terminal
path sets the cause it knows:

* lease self-stop: LEASE_EXPIRED;
* a goal that ends in failure (aborted or refused by the server, or a stall): EXECUTION_FAILURE;
* cancel: CANCELLED_BY_OPERATOR or CAPABILITY_LOSS from the cancel's own cause.

Where the cause is not determinable it is left unset, and the message says why:

* a SUPERSEDED cancel names no superseding action, which that cause requires;
* a REVOCATION or untyped cancel does not name one;
* reconcile gives the plane's phase, not its reason.

SUCCEEDED carries none.
"""

from __future__ import annotations

import time
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _statuses(out):
    return [m.action_status for m in out if m.WhichOneof("payload") == "action_status"]


def _assign(a: Adapter, lease_ms: int = 60_000) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="t-1", assignment_id="a-1", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=1, lease_generation=1,
            lease_duration_ms=lease_ms))), None)
    _drain(a)


def _sim() -> Adapter:
    a = Adapter(pb, None, SimulatedRobot(), ROBOT)
    a.bring_up_robot()
    return a


class _Handle:
    accepted = True

    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.on_result = None

    def cancel_goal_async(self) -> None:
        pass

    def get_result_async(self):
        handle = self

        class _F:
            def add_done_callback(self, cb) -> None:
                handle.on_result = cb
        return _F()


class _Done:
    def __init__(self, v) -> None:
        self._v = v

    def result(self):
        return self._v


def _nav2():
    b = mark_transport_live(Nav2Binding(ROBOT))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(epoch)
    b._server_ready = lambda: True                        # the server is up: goals go at once
    a = Adapter(pb, None, b, ROBOT)
    return a, b, sent


# ── cancel ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cause, end_cause, says_why", [
    (pb.CANCEL_CAUSE_OPERATOR, pb.ACTION_END_CAUSE_CANCELLED_BY_OPERATOR, False),
    (pb.CANCEL_CAUSE_CAPABILITY_LOSS, pb.ACTION_END_CAUSE_CAPABILITY_LOSS, False),
    (pb.CANCEL_CAUSE_SUPERSEDED, pb.ACTION_END_CAUSE_UNSPECIFIED, True),
    (pb.CANCEL_CAUSE_REVOCATION, pb.ACTION_END_CAUSE_UNSPECIFIED, True),
    (pb.CANCEL_CAUSE_UNSPECIFIED, pb.ACTION_END_CAUSE_UNSPECIFIED, True),
])
def test_cancel_sets_the_cause_its_cancel_names_or_says_why_not(cause, end_cause, says_why):
    a = _sim()
    _assign(a)
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=2, robot_id=ROBOT, cancel_action=pb.CancelAction(
            action_id="t-1", fencing_token=1, cause=cause))), None)
    out = _drain(a)
    st = _statuses(out)
    assert [s.state for s in st] == [pb.ACTION_STATE_CANCELLED]
    assert st[0].end_cause == end_cause
    assert bool(st[0].message) is says_why
    if says_why:
        assert pb.CancelCause.Name(cause) in st[0].message
    res = [m.command_result.cancel_action for m in out
           if m.WhichOneof("payload") == "command_result"][0]
    assert res.cause == cause                              # echoes the cause it received


# ── lease self-stop ─────────────────────────────────────────────────────────

def test_lease_self_stop_is_lease_expired():
    a = _sim()
    _assign(a, lease_ms=1)
    time.sleep(0.01)
    a._tick_lease()
    st = _statuses(_drain(a))
    assert [s.state for s in st] == [pb.ACTION_STATE_STOPPED]
    assert st[0].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED


# ── reconcile ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase, state", [
    (pb.ROBOT_ACTION_PHASE_CANCELLED, pb.ACTION_STATE_CANCELLED),
    (pb.ROBOT_ACTION_PHASE_UNKNOWN, pb.ACTION_STATE_UNKNOWN),
    (pb.ROBOT_ACTION_PHASE_IDLE, pb.ACTION_STATE_STOPPED),
])
def test_reconcile_leaves_the_cause_unset_and_says_what_it_knows(phase, state):
    a = _sim()
    _assign(a)
    a._handle_control_plane(pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        accepted=True, authoritative_action_state=pb.RobotActionState(
            phase=phase, action_id="t-1", fencing_token=1))), None)
    st = _statuses(_drain(a))
    assert [s.state for s in st] == [state]
    assert st[0].end_cause == pb.ACTION_END_CAUSE_UNSPECIFIED
    assert pb.RobotActionPhase.Name(phase) in st[0].message


# ── a goal that ends in failure, or succeeds ────────────────────────────────

def test_an_aborted_goal_is_execution_failure_with_the_servers_reason():
    a, b, sent = _nav2()
    _assign(a)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    h.on_result(_Done(NS(status=6)))
    a._maybe_emit_task_progress()
    st = _statuses(_drain(a))
    assert st[0].state == pb.ACTION_STATE_FAILED
    assert st[0].end_cause == pb.ACTION_END_CAUSE_EXECUTION_FAILURE
    assert st[0].message == "the navigation server aborted the goal"


def test_a_refused_goal_is_execution_failure_with_the_servers_reason():
    a, b, sent = _nav2()
    _assign(a)
    b._on_goal_response(_Done(_Handle(accepted=False)), sent[0])
    a._maybe_emit_task_progress()
    st = _statuses(_drain(a))
    assert st[0].state == pb.ACTION_STATE_FAILED
    assert st[0].end_cause == pb.ACTION_END_CAUSE_EXECUTION_FAILURE
    assert st[0].message == "the navigation server rejected the goal"


def test_a_succeeded_goal_carries_no_end_cause():
    a, b, sent = _nav2()
    _assign(a)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    h.on_result(_Done(NS(status=4)))
    a._maybe_emit_task_progress()
    st = _statuses(_drain(a))
    assert st[0].state == pb.ACTION_STATE_SUCCEEDED
    assert st[0].end_cause == pb.ACTION_END_CAUSE_UNSPECIFIED and st[0].message == ""
