# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C2.2: on a RegisterAck the adapter adopts `authoritative_action_state` as the truth about
what the robot should be doing. The control plane is the source of truth across a connectivity
gap, so a task it cancelled while the robot was unreachable must not resume on reconnect.

Semantics mirror the reference simulation adapter's `_reconcile_action_state`:

* IN_PROGRESS for the action being executed: keep executing.
* IN_PROGRESS for any other action (or none named): treated as CANCELLED.
* CANCELLED / UNKNOWN / IDLE: stop the robot, release the lease, clear the current action,
  and report CANCELLED / UNKNOWN / STOPPED for the plane's action id (else the local one).
* REVOKING, UNSPECIFIED, an absent state or a refused registration: nothing here.

Driven through `_handle_control_plane` with the real protobuf messages.
"""

from __future__ import annotations

import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


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

    def command_resume(self, robot_id):
        self.calls.append("resume")
        super().command_resume(robot_id)


def _adapter(robot=None) -> Adapter:
    a = Adapter(pb, None, robot or _RecordingRobot(), ROBOT)
    if robot is None:
        a.bring_up_robot()
    return a


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _statuses(a: Adapter) -> list[tuple[str, int]]:
    return [(m.action_status.action_id, m.action_status.state)
            for m in _drain(a) if m.WhichOneof("payload") == "action_status"]


# A served action with a target: the adapter declines one it cannot serve.
_TARGET = b'{"x": 1.0, "y": 2.0}'


def _assign(a: Adapter, action: str, token: int, lease_ms: int = 5000) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=token, robot_id=ROBOT,
        assign_action=pb.AssignAction(action_id=action, assignment_id=f"a-{action}",
                                      action_type="Navigate", payload_json=_TARGET,
                                      fencing_token=token, lease_generation=1,
                                      lease_duration_ms=lease_ms))), None)
    _drain(a)


def _register_ack(phase, action: str = "", token: int = 0, accepted: bool = True,
                  with_state: bool = True) -> pb.ControlPlaneMessage:
    ack = pb.RegisterAck(accepted=accepted)
    if with_state:
        ack.authoritative_action_state.CopyFrom(pb.RobotActionState(
            phase=phase, action_id=action, fencing_token=token))
    return pb.ControlPlaneMessage(register_ack=ack)


def _running(a: Adapter, action: str = "t-1") -> Adapter:
    _assign(a, action, 5)
    assert a._current_action == action and ROBOT in a._lease._deadline
    a._robot.calls.clear()
    return a


def _assert_reconciled(a: Adapter, want: list[tuple[str, int]]) -> None:
    assert _statuses(a) == want
    assert a._robot.calls == ["stop"]
    assert a._robot.is_stopped(ROBOT)
    assert a._current_action == ""
    assert ROBOT not in a._lease._deadline       # the lease is released, not left to lapse


# ── each phase ──────────────────────────────────────────────────────────────


def test_in_progress_for_the_same_action_keeps_executing() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_IN_PROGRESS, "t-1", 5), None)
    assert _statuses(a) == []
    assert a._robot.calls == []
    assert a._current_action == "t-1" and ROBOT in a._lease._deadline


def test_in_progress_for_another_action_is_treated_as_cancelled() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_IN_PROGRESS, "t-other", 5), None)
    _assert_reconciled(a, [("t-other", pb.ACTION_STATE_CANCELLED)])


def test_in_progress_naming_no_action_is_treated_as_cancelled() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_IN_PROGRESS, "", 5), None)
    _assert_reconciled(a, [("t-1", pb.ACTION_STATE_CANCELLED)])


def test_cancelled_stops_and_reports_cancelled() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_CANCELLED, "t-1", 5), None)
    _assert_reconciled(a, [("t-1", pb.ACTION_STATE_CANCELLED)])


def test_unknown_stops_and_reports_unknown() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_UNKNOWN, "", 5), None)
    _assert_reconciled(a, [("t-1", pb.ACTION_STATE_UNKNOWN)])


def test_idle_stops_and_reports_stopped() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_IDLE, "", 5), None)
    _assert_reconciled(a, [("t-1", pb.ACTION_STATE_STOPPED)])


def test_revoking_is_left_to_the_lease() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_REVOKING, "t-1", 5), None)
    assert _statuses(a) == [] and a._robot.calls == [] and a._current_action == "t-1"


def test_no_state_a_refused_ack_and_an_unspecified_phase_change_nothing() -> None:
    for msg in (_register_ack(None, with_state=False),
                _register_ack(pb.ROBOT_ACTION_PHASE_CANCELLED, "t-1", 5, accepted=False),
                _register_ack(pb.ROBOT_ACTION_PHASE_UNSPECIFIED, "t-1", 5)):
        a = _running(_adapter())
        a._handle_control_plane(msg, None)
        assert _statuses(a) == [] and a._robot.calls == [] and a._current_action == "t-1"


def test_the_token_is_still_adopted_when_the_phase_is_reconciled() -> None:
    a = _running(_adapter())
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_CANCELLED, "t-1", 20), None)
    assert a._fence.highest(ROBOT) == 20


# ── the hazard, end to end on the Nav2 binding ──────────────────────────────


class _Accepted:
    """An accepted NavigateToPose goal handle whose result never arrives."""

    accepted = True

    def get_result_async(self):
        class _F:
            def add_done_callback(self, cb) -> None:
                pass
        return _F()


class _Done:
    def __init__(self, value) -> None:
        self._v = value

    def result(self):
        return self._v


def test_cancelled_during_outage_cancels_the_nav2_goal_and_commands_no_further_motion():
    """A task runs; the stream drops; the plane cancels the task while the robot is
    unreachable; the adapter re-registers. The Nav2 goal must be cancelled on the RegisterAck,
    and nothing the adapter does afterwards may send the robot moving again."""
    b = mark_transport_live(Nav2Binding(robot_id=ROBOT))
    sent, cancelled = [], []
    b._send_nav2_goal = lambda goal, epoch: sent.append(goal)            # the ROS 2 dispatch
    b._server_ready = lambda: True                                       # the server is up
    b._cancel_nav2_goal = lambda handle: cancelled.append(handle)        # the ROS 2 cancel
    a = _adapter(b)

    _assign(a, "t-1", 5)
    assert len(sent) == 1
    handle = _Accepted()                        # Nav2 accepted the goal, through the
    b._on_goal_response(_Done(handle), b._goal_epoch)   # binding's own goal-response path

    # the stream drops and the adapter redials: Hello → HelloAck → RegisterRobot → RegisterAck
    a._telemetry_started = True                 # hello_ack started it on the first stream
    a._handle_control_plane(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)), None)
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_CANCELLED, "t-1", 5), None)

    assert cancelled == [handle]                # the active Nav2 goal was cancelled
    assert b._goal is None and b._goal_handle is None
    assert ("t-1", pb.ACTION_STATE_CANCELLED) in _statuses(a)

    # Everything the adapter does next on its own: stream progress, answer a stale renewal,
    # let the lease horizon pass. None of it may dispatch a goal.
    a._maybe_emit_task_progress()
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=99, robot_id=ROBOT, renew_lease=pb.RenewActionLease(
            action_id="t-1", lease_generation=1, lease_duration_ms=1))), None)
    time.sleep(0.01)
    a._tick_lease()
    assert len(sent) == 1                       # no further motion commanded
    renew = [m.command_result.renew_lease for m in _drain(a)
             if m.WhichOneof("payload") == "command_result"]
    assert renew and not renew[0].running       # the plane is told t-1 is not running


def test_idle_with_nothing_executing_reports_nothing() -> None:
    # The sim's rule: no action id on either side, no status. The stop is still commanded.
    a = _adapter()
    a._handle_control_plane(_register_ack(pb.ROBOT_ACTION_PHASE_IDLE, "", 5), None)
    assert _statuses(a) == [] and a._robot.calls == ["stop"]
