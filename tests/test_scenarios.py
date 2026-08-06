# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Scenario-preset tests for the ROS 2 / Nav2 SimulatedRobot binding.

Three layers, in the style of the existing suite:
  * the pure loader/engine (proto-free) — same format as the in-tree simulator;
  * the SimulatedRobot binding, which the scenario drives (battery/hardware/comms/estop);
  * the adapter's proto mapping (skipped where grpc/swarmada_sdk are absent).

The safety contract is never weakened by a scenario — `confirm_estop` still gates on
the binding's real `is_stopped`.
"""

from __future__ import annotations

import pytest

from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.scenarios import (
    HW_DEGRADED,
    HW_HEALTHY,
    HardwareFaultOverrides,
    ScenarioEngine,
    load_scenario,
)
from fleet_adapter_ros2.scenarios.loader import Scenario


def _engine(name, overrides=None):
    return ScenarioEngine(load_scenario(name, overrides))


def _status_at(engine, t, component):
    return next(h.status for h in engine.hardware_at(t) if h.name == component)


# ── loader / engine (proto-free) ────────────────────────────────────────────────

def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenario("does-not-exist")


def test_healthy_fleet_has_no_faults() -> None:
    engine = _engine("healthy-fleet")
    for t in (0.0, 30.0, 10_000.0):
        assert all(h.status == HW_HEALTHY for h in engine.hardware_at(t))


def test_hardware_fault_degrades_then_recovers() -> None:
    engine = _engine("hardware-fault")
    assert _status_at(engine, 29.9, "camera_front") == HW_HEALTHY
    assert _status_at(engine, 30.0, "camera_front") == HW_DEGRADED
    assert _status_at(engine, 89.9, "camera_front") == HW_DEGRADED
    assert _status_at(engine, 90.0, "camera_front") == HW_HEALTHY
    assert _status_at(engine, 45.0, "lidar_top") == HW_HEALTHY


def test_hardware_fault_overrides_retarget_and_retime() -> None:
    engine = _engine("hardware-fault",
                     HardwareFaultOverrides(component="lidar_top", fault_at=10, recover_at=20))
    assert _status_at(engine, 9.9, "lidar_top") == HW_HEALTHY
    assert _status_at(engine, 10.0, "lidar_top") == HW_DEGRADED
    assert _status_at(engine, 20.0, "lidar_top") == HW_HEALTHY
    assert _status_at(engine, 15.0, "camera_front") == HW_HEALTHY


def test_override_unknown_component_raises() -> None:
    with pytest.raises(ValueError, match="not in the scenario hardware manifest"):
        load_scenario("hardware-fault", HardwareFaultOverrides(component="nonesuch"))


def test_hardware_delta_full_then_changed_only() -> None:
    engine = _engine("hardware-fault")
    first = engine.hardware_delta(0.0, prev=None)
    assert {h.name for h in first} == {"lidar_top", "camera_front"}
    prev = {h.name: h.status for h in first}
    assert engine.hardware_delta(10.0, prev) == []
    delta = engine.hardware_delta(30.0, prev)
    assert [(h.name, h.status) for h in delta] == [("camera_front", HW_DEGRADED)]


def test_battery_edge_drains_fast_below_min() -> None:
    engine = _engine("battery-edge")
    assert engine.battery_at(0.0) == 22
    assert engine.battery_at(3.0) == 19
    assert engine.battery_at(1000.0) == 0


def test_battery_at_is_none_without_a_curve() -> None:
    assert ScenarioEngine(Scenario(name="bare")).battery_at(5.0) is None


def test_comms_flaky_stream_drop_window() -> None:
    engine = _engine("comms-flaky")
    assert engine.stream_drop_window() == (20.0, 35.0)
    assert not engine.stream_down_at(19.9)
    assert engine.stream_down_at(20.0)
    assert not engine.stream_down_at(35.0)
    assert engine.telemetry_gap_at(25.0)  # suppressed while down


def test_estop_drill_due_after_timer() -> None:
    engine = _engine("estop-drill")
    assert not engine.estop_due(19.9)
    assert engine.estop_due(20.0)


def test_only_comms_flaky_declares_a_stream_drop() -> None:
    for name in ("healthy-fleet", "hardware-fault", "battery-edge", "estop-drill"):
        assert _engine(name).stream_drop_window() is None


# ── SimulatedRobot binding (scenario-driven state) ──────────────────────────────

def test_binding_battery_follows_the_curve() -> None:
    robot = SimulatedRobot(_engine("battery-edge"))
    robot.spawn("r1")
    robot._elapsed = 3.0
    assert robot.battery_percent("r1") == 19


def test_binding_without_engine_is_constant_battery() -> None:
    robot = SimulatedRobot()
    robot.spawn("r1")
    robot._elapsed = 1000.0
    assert robot.battery_percent("r1") == 100
    assert robot.hardware_delta() == []
    assert robot.manifest_hardware() == []
    assert not robot.telemetry_suppressed()
    assert robot.stream_drop_window() is None


def test_binding_hardware_delta_tracks_the_fault() -> None:
    robot = SimulatedRobot(_engine("hardware-fault"))
    robot.spawn("r1")
    robot._elapsed = 10.0
    first = {h.name: h.status for h in robot.hardware_delta()}
    assert first == {"lidar_top": HW_HEALTHY, "camera_front": HW_HEALTHY}
    robot._elapsed = 45.0
    delta = robot.hardware_delta()
    assert [(h.name, h.status) for h in delta] == [("camera_front", HW_DEGRADED)]


def test_binding_comms_and_estop_schedule() -> None:
    robot = SimulatedRobot(_engine("comms-flaky"))
    robot.spawn("r1")
    robot._elapsed = 25.0
    assert robot.stream_down() and robot.telemetry_suppressed()
    robot._elapsed = 40.0
    assert not robot.stream_down()

    drill = SimulatedRobot(_engine("estop-drill"))
    drill.spawn("r1")
    drill._elapsed = 21.0
    assert drill.estop_due()


def test_scenario_never_weakens_the_confirmed_stop() -> None:
    # C5 contract holds with an engine attached (swarmada_sdk is a hard dep, so it is
    # imported unguarded like the rest of the suite).
    from swarmada_sdk.safety import ESTOP_STOPPED, confirm_estop
    robot = SimulatedRobot(_engine("hardware-fault"))
    robot.spawn("r1")
    robot.command_move("r1", 5.0, 0.0)
    assert confirm_estop(robot, "r1") == ESTOP_STOPPED
    assert robot.is_stopped("r1")
