# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C17.1 / C17.2: an assignment carrying an operating constraint this adapter cannot honour is
DECLINED, never accepted and ignored.

This adapter honours no operating subject (nothing here can switch a camera off, stop a pump or
cap a speed while a robot transits a zone), so every rule that reaches it is one it must refuse:
`accepted=False`, `ASSIGN_ACTION_REJECTION_OPERATING_CONSTRAINT_UNMET`, and
`unhonourable_subjects` naming every subject, including a namespaced one it does not recognise.
An unrecognised subject is not an absent subject.

Semantics mirror the reference simulation adapter: one predicate over both obligations, placed
after the two fencing branches so a superseded command is still refused STALE. Driven through
`_handle_control_plane` with the real protobuf messages.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"
CORE = ("camera", "microphone", "water", "chemical", "uv", "speed", "noise")
EXTENSION = "example.com/leaf-blower"


class _RecordingRobot(SimulatedRobot):
    """A SimulatedRobot that records every motion and stop command it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def command_move(self, robot_id, x, y, yaw=0.0):
        self.calls.append("move")
        super().command_move(robot_id, x, y, yaw)

    def command_stop(self, robot_id):
        self.calls.append("stop")
        super().command_stop(robot_id)


def _adapter() -> Adapter:
    a = Adapter(pb, None, _RecordingRobot(), ROBOT)
    a.bring_up_robot()
    return a


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _rule(subject: str) -> pb.EffectiveOperatingConstraint:
    """One resolved rule per subject, across all three modes, as the harness sends them."""
    mode, cap, allowed = {
        "chemical": (pb.OPERATING_CONSTRAINT_MODE_LIMITED, "", ["quat-ammonium-7"]),
        "uv": (pb.OPERATING_CONSTRAINT_MODE_REQUIRED, "", []),
        "speed": (pb.OPERATING_CONSTRAINT_MODE_LIMITED, "0.4", []),
        "noise": (pb.OPERATING_CONSTRAINT_MODE_LIMITED, "65", []),
    }.get(subject, (pb.OPERATING_CONSTRAINT_MODE_DENIED, "", []))
    return pb.EffectiveOperatingConstraint(subject=subject, mode=mode, max=cap, allowed=allowed,
                                           reason="test", declaring_zone="ward")


def _assign(a: Adapter, action: str, token: int | None, subjects=(),
            assignment: str = "") -> pb.AssignActionResult:
    task = pb.AssignAction(action_id=action, action_type="Navigate",
                           assignment_id=assignment or f"a-{action}", lease_generation=1,
                           lease_duration_ms=5000,
                           payload_json=b'{"x": 1.0, "y": 2.0}',
                           operating_constraints=[_rule(s) for s in subjects])
    if token is not None:
        task.fencing_token = token
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=token or 0, robot_id=ROBOT, assign_action=task)), None)
    results = [m.command_result for m in _drain(a) if m.WhichOneof("payload") == "command_result"]
    assert len(results) == 1
    assert not results[0].unsupported, "assign_action is not an optional command"
    return results[0].assign_action


def _assert_declined_naming(res: pb.AssignActionResult, subjects) -> None:
    assert res.accepted is False
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_OPERATING_CONSTRAINT_UNMET
    assert list(res.unhonourable_subjects) == list(subjects)


def test_every_core_subject_is_declined_and_each_is_named() -> None:
    a = _adapter()
    _assert_declined_naming(_assign(a, "t-core", 1, CORE), CORE)


def test_an_unrecognised_namespaced_subject_is_declined_and_named() -> None:
    a = _adapter()
    _assert_declined_naming(_assign(a, "t-ext", 1, [EXTENSION]), [EXTENSION])


def test_a_single_subject_is_declined_and_named() -> None:
    a = _adapter()
    _assert_declined_naming(_assign(a, "t-cam", 1, ["camera"]), ["camera"])


def test_a_mixed_set_names_every_subject_not_only_the_first() -> None:
    a = _adapter()
    subjects = ["speed", EXTENSION, "camera"]
    _assert_declined_naming(_assign(a, "t-mixed", 1, subjects), subjects)


def test_a_declined_assignment_commands_no_motion_takes_no_lease_and_reports_nothing() -> None:
    a = _adapter()
    res = _assign(a, "t-cam", 1, ["camera"])
    assert res.accepted is False
    assert a._robot.calls == []
    assert a._current_action == ""
    assert ROBOT not in a._lease._deadline
    assert _drain(a) == []  # no RUNNING status for an action that never started


def test_a_decline_leaves_the_action_already_running_untouched() -> None:
    a = _adapter()
    assert _assign(a, "t-1", 1).accepted is True
    a._robot.calls.clear()
    res = _assign(a, "t-2", 2, ["microphone"])
    assert res.accepted is False
    assert a._current_action == "t-1"
    assert a._robot.calls == []  # neither moved toward t-2 nor stopped t-1


def test_no_constraint_is_permission_and_is_accepted() -> None:
    # The decline is a function of what arrived, not a reflex: an empty set is no rule in force.
    a = _adapter()
    res = _assign(a, "t-free", 1)
    assert res.accepted is True
    assert res.accepted_fencing_token == 1
    assert a._robot.calls == ["move"]


def test_a_stale_token_is_refused_stale_before_the_rule_is_read() -> None:
    a = _adapter()
    assert _assign(a, "t-1", 5).accepted is True
    res = _assign(a, "t-old", 3, ["camera"])
    assert res.accepted is False
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_STALE_FENCING_TOKEN
    assert list(res.unhonourable_subjects) == []


def test_a_missing_token_is_refused_missing_before_the_rule_is_read() -> None:
    a = _adapter()
    res = _assign(a, "t-none", None, ["camera"])
    assert res.accepted is False
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_MISSING_FENCING_TOKEN
    assert list(res.unhonourable_subjects) == []


def test_a_redelivered_assignment_with_a_rule_is_declined_again() -> None:
    a = _adapter()
    first = _assign(a, "t-cam", 4, ["camera"], assignment="a-same")
    again = _assign(a, "t-cam", 4, ["camera"], assignment="a-same")
    _assert_declined_naming(first, ["camera"])
    _assert_declined_naming(again, ["camera"])
    assert a._robot.calls == []
