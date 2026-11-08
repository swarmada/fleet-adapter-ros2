# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C3.8: the adapter adopts `RegisterAck.authoritative_action_state.fencing_token` as its
fencing high-water mark, so a later AssignAction below the control plane's stated token is
refused STALE.

Driven through `_handle_control_plane` with the real protobuf messages, because the
behaviour turns on field presence (`HasField`), which a hand-rolled fake would have to
re-implement. Semantics mirror the reference simulation adapter: adopt on every ACCEPTED
RegisterAck, ignore a zero or absent token, never lower the mark.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


def _adapter() -> Adapter:
    a = Adapter(pb, None, SimulatedRobot(), ROBOT)
    a.bring_up_robot()
    return a


def _register_ack(token: int | None, accepted: bool = True) -> pb.ControlPlaneMessage:
    ack = pb.RegisterAck(accepted=accepted)
    if token is not None:
        ack.authoritative_action_state.CopyFrom(pb.RobotActionState(
            phase=pb.ROBOT_ACTION_PHASE_IDLE, fencing_token=token))
    return pb.ControlPlaneMessage(register_ack=ack)


# A served action with a target: the adapter declines one it cannot serve.
_TARGET = b'{"x": 1.0, "y": 2.0}'


def _assign(a: Adapter, token: int, assignment: str) -> pb.AssignActionResult:
    """Deliver an AssignAction at `token` and return the adapter's AssignActionResult."""
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=token, robot_id=ROBOT,
        assign_action=pb.AssignAction(action_id=f"t-{assignment}", assignment_id=assignment,
                                      action_type="Navigate", payload_json=_TARGET,
                                      fencing_token=token))), None)
    while True:
        msg = a._outbox.get_nowait()
        if msg.WhichOneof("payload") == "command_result":
            return msg.command_result.assign_action


def _stale(res: pb.AssignActionResult) -> bool:
    return not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_STALE_FENCING_TOKEN


def test_reconnect_with_higher_token_refuses_a_lower_assignment() -> None:
    # The harness's C3.8 shape: the adapter reached 14 on its own, the plane then says 20.
    a = _adapter()
    assert _assign(a, 14, "a-status").accepted
    a._handle_control_plane(_register_ack(20), None)
    assert _stale(_assign(a, 18, "a-adopt"))       # between the two: the dead plane's token
    assert _assign(a, 21, "a-next").accepted        # above it: the new plane's next command


def test_first_registration_adopts_too() -> None:
    a = _adapter()
    a._handle_control_plane(_register_ack(20), None)
    assert _stale(_assign(a, 18, "a-1"))


def test_an_assignment_at_the_adopted_token_itself_is_stale() -> None:
    # adopt() drops the assignment held at the old mark, so nothing at N is a re-delivery.
    a = _adapter()
    a._handle_control_plane(_register_ack(20), None)
    assert _stale(_assign(a, 20, "a-1"))


def test_equal_token_changes_nothing() -> None:
    # The same statement of the mark the adapter already holds: the C3.4 re-ack survives.
    a = _adapter()
    assert _assign(a, 14, "a-1").accepted
    a._handle_control_plane(_register_ack(14), None)
    assert _assign(a, 14, "a-1").accepted           # identical re-delivery, re-acked
    assert _stale(_assign(a, 13, "a-0"))


def test_lower_token_never_lowers_the_mark() -> None:
    # A replayed or reordered RegisterAck must not re-open a token already refused.
    a = _adapter()
    a._handle_control_plane(_register_ack(20), None)
    a._handle_control_plane(_register_ack(10), None)
    assert _stale(_assign(a, 15, "a-1"))


def test_absent_state_leaves_the_mark() -> None:
    a = _adapter()
    assert _assign(a, 14, "a-1").accepted
    a._handle_control_plane(_register_ack(None), None)
    assert _assign(a, 15, "a-2").accepted


def test_zero_token_is_not_a_statement() -> None:
    a = _adapter()
    assert _assign(a, 14, "a-1").accepted
    a._handle_control_plane(_register_ack(0), None)
    assert _assign(a, 15, "a-2").accepted


def test_refused_registration_is_not_adopted() -> None:
    a = _adapter()
    a._handle_control_plane(_register_ack(20, accepted=False), None)
    assert _assign(a, 18, "a-1").accepted
