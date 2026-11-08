# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The action catalog reaches the control plane on every binding, not only under a scenario.

A v0.4 control plane issues no ScanCapabilities, so the CapabilitiesSnapshot pushed right after
RegisterRobot is the only way it learns this adapter's `supported_actions`. That push used to be
gated on a scenario with hardware, so under `--binding ros2` it was never sent: admission saw no
catalog and failed open, letting every action type through to an adapter that serves one.

The push is now the same snapshot the scan reply carries, sent after every RegisterRobot, as the
reference simulation adapter sends its own at the same point. Driven through
`_handle_control_plane` with the real protobuf messages; no ROS runtime is started.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.scenarios import ScenarioEngine, load_scenario

ROBOT = "amr-1"


def _hello(a: Adapter) -> list:
    a._telemetry_started = True  # no telemetry thread in a unit test
    a._handle_control_plane(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)), None)
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _scan(a: Adapter) -> pb.CapabilitiesSnapshot:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, scan=pb.ScanCapabilities())), None)
    return a._outbox.get_nowait().command_result.scan


def test_the_ros2_binding_pushes_its_catalog_right_after_registering() -> None:
    a = Adapter(pb, None, Nav2Binding(), ROBOT)  # ROS-free: start() is never called
    out = _hello(a)
    assert [m.WhichOneof("payload") for m in out] == ["register", "capabilities"]
    snap = out[1].capabilities
    assert snap.robot_id == ROBOT
    assert [s.action_type for s in snap.supported_actions] == ["Navigate"]
    assert list(snap.supported_actions[0].required_capabilities) == ["navigation.2d"]
    assert list(snap.hardware) == []
    assert list(snap.honoured_operating_constraints) == []


def test_the_push_is_the_scan_reply() -> None:
    a = Adapter(pb, None, Nav2Binding(), ROBOT)
    pushed = _hello(a)[1].capabilities
    scanned = _scan(a)
    pushed.snapshot_ms = scanned.snapshot_ms = 0
    assert pushed == scanned


def test_the_push_is_sent_again_on_every_reconnect() -> None:
    a = Adapter(pb, None, Nav2Binding(), ROBOT)
    assert [m.WhichOneof("payload") for m in _hello(a)] == ["register", "capabilities"]
    assert [m.WhichOneof("payload") for m in _hello(a)] == ["register", "capabilities"]


def test_the_simulated_binding_without_a_scenario_pushes_its_catalog_too() -> None:
    out = _hello(Adapter(pb, None, SimulatedRobot(), ROBOT))
    assert [m.WhichOneof("payload") for m in out] == ["register", "capabilities"]
    # The simulated binding carries a simulated exploration stack, so it also serves
    # explore-floor (Custom) and Charge; the Nav2 binding above has none and does not.
    assert [s.action_type for s in out[1].capabilities.supported_actions] == [
        "Navigate", "Custom", "Charge"]


def test_a_scenario_push_still_carries_its_hardware() -> None:
    robot = SimulatedRobot(ScenarioEngine(load_scenario("healthy-fleet")))
    robot.spawn(ROBOT)
    out = _hello(Adapter(pb, None, robot, ROBOT))
    assert {c.name for c in out[1].capabilities.hardware} == {"lidar_top", "camera_front"}
