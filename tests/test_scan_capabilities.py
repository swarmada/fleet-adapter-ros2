# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C12.1: a ScanCapabilities request is answered with a full CapabilitiesSnapshot, never
declined, and the snapshot's `supported_actions` catalog names only what this adapter serves.

What it serves is one action: every assignment becomes a Nav2 NavigateToPose goal built from
the payload's `x`, `y` and `yaw` (task.goal_from_assign), sent in the `map` frame. The
catalog says exactly that. A payload `frame` and a named destination are parsed but never
reach the goal (RobotBinding.command_move takes x, y and yaw only), so the catalog does not
offer them.

The same builder serves the scan reply and the snapshot pushed after registration, as in the
reference simulation adapter, so the robot does not describe itself two ways.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.scenarios import ScenarioEngine, load_scenario

ROBOT = "amr-1"


def _scan(a: Adapter) -> pb.CommandResult:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=7, robot_id=ROBOT, scan=pb.ScanCapabilities())), None)
    results = [m.command_result for m in _drain(a) if m.WhichOneof("payload") == "command_result"]
    assert len(results) == 1
    return results[0]


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _scenario_adapter() -> Adapter:
    robot = SimulatedRobot(ScenarioEngine(load_scenario("healthy-fleet")))
    robot.spawn(ROBOT)
    return Adapter(pb, None, robot, ROBOT)


def test_scan_is_answered_with_a_snapshot_not_declined() -> None:
    res = _scan(Adapter(pb, None, SimulatedRobot(), ROBOT))
    assert res.unsupported is False
    assert res.WhichOneof("result") == "scan"
    assert res.command_id == 7
    assert res.scan.robot_id == ROBOT
    assert res.scan.snapshot_ms > 0


def test_the_catalog_is_navigate_via_nav2_and_nothing_more() -> None:
    # Without an exploration stack, Navigate is all there is.
    snap = _scan(Adapter(pb, None, SimulatedRobot(explore=None), ROBOT)).scan
    assert [a.action_type for a in snap.supported_actions] == ["Navigate"]
    nav = snap.supported_actions[0]
    assert list(nav.required_capabilities) == ["navigation.2d"]
    params = {p.name: (p.kind, p.unit, p.required) for p in nav.params}
    assert params == {"x": ("number", "m", True), "y": ("number", "m", True),
                      "yaw": ("number", "rad", False)}


def test_the_snapshot_declares_no_honoured_operating_subject() -> None:
    # Declaring none is conformant (C17.3); claiming one this adapter cannot police is not.
    snap = _scan(Adapter(pb, None, SimulatedRobot(), ROBOT)).scan
    assert list(snap.honoured_operating_constraints) == []


def test_the_snapshot_claims_no_model_stream_or_firmware_it_does_not_have() -> None:
    snap = _scan(Adapter(pb, None, SimulatedRobot(), ROBOT)).scan
    assert list(snap.installed_models) == []
    assert list(snap.sensor_streams) == []
    assert not snap.HasField("firmware")


def test_scan_reports_the_whole_hardware_manifest_under_a_scenario() -> None:
    snap = _scan(_scenario_adapter()).scan
    assert {c.name for c in snap.hardware} == {"lidar_top", "camera_front"}
    assert all(c.status == pb.HARDWARE_STATUS_HEALTHY for c in snap.hardware)


def test_scan_on_the_nav2_binding_answers_with_the_catalog() -> None:
    # No ROS runtime is needed to describe what the adapter serves, and none is started here.
    res = _scan(Adapter(pb, None, Nav2Binding(), ROBOT))
    assert res.WhichOneof("result") == "scan"
    assert [a.action_type for a in res.scan.supported_actions] == ["Navigate"]
    assert list(res.scan.hardware) == []


def test_the_registration_push_and_the_scan_reply_describe_the_robot_the_same_way() -> None:
    a = _scenario_adapter()
    a._send_capabilities_snapshot()
    pushed = [m.capabilities for m in _drain(a) if m.WhichOneof("payload") == "capabilities"]
    assert len(pushed) == 1
    scanned = _scan(a).scan
    pushed[0].snapshot_ms = scanned.snapshot_ms = 0
    assert pushed[0] == scanned
