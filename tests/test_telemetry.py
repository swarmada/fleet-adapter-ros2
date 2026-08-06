# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ROS 2 → Swarmada telemetry mapping (targets C6).

Fake ROS messages are plain namespaces (attribute access), so no ROS install and
no generated protobufs are needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

from fleet_adapter_ros2 import telemetry
from fleet_adapter_ros2.nav2_binding import Nav2Binding


def _odom(x, y, qz, qw):
    return NS(pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=qz, w=qw))))


def test_position_from_odom_and_yaw() -> None:
    # A quaternion for +90° about z: qz = qw = sin/cos(45°).
    q = math.sqrt(2) / 2
    pos = telemetry.position_from_odom(_odom(3.0, -4.0, q, q), floor=2)
    assert (pos.x, pos.y, pos.floor) == (3.0, -4.0, 2)
    assert math.isclose(pos.yaw, math.pi / 2, abs_tol=1e-9)


def test_yaw_zero_for_identity_quaternion() -> None:
    assert telemetry.yaw_from_quaternion(0, 0, 0, 1) == 0.0


def test_battery_fraction_to_percent_and_charging() -> None:
    b = telemetry.battery_from_state(NS(percentage=0.5, power_supply_status=1, voltage=48.2))
    assert b.percent == 50
    assert b.charging is True
    assert b.voltage == 48.2


def test_battery_unknown_status_is_none_not_false() -> None:
    # power_supply_status UNKNOWN → charging absent (None), never defaulted to False.
    b = telemetry.battery_from_state(NS(percentage=0.9, power_supply_status=0, voltage=50.0))
    assert b.charging is None


def test_battery_nan_percentage_is_absent() -> None:
    b = telemetry.battery_from_state(NS(percentage=float("nan"), power_supply_status=2, voltage=47.0))
    assert b.percent is None
    assert b.charging is False  # DISCHARGING


def test_hardware_level_maps_to_status_and_reason() -> None:
    diag = NS(status=[
        NS(name="lidar", level=0, message="ok", values=[NS(key="rpm", value="600")]),
        NS(name="camera", level=1, message="low fps", values=[]),
        NS(name="drive", level=2, message="motor fault", values=[]),
        NS(name="imu", level=3, message="stale", values=[]),
    ])
    hw = {h.component_name: h for h in telemetry.hardware_from_diagnostics(diag)}
    assert hw["lidar"].status == telemetry.STATUS_HEALTHY and hw["lidar"].reason == ""
    assert hw["camera"].status == telemetry.STATUS_DEGRADED and hw["camera"].reason == "low fps"
    assert hw["drive"].status == telemetry.STATUS_FAILED
    assert hw["imu"].status == telemetry.STATUS_FAILED  # STALE → Failed


def test_hardware_does_not_leak_arbitrary_keyvalues() -> None:
    # Numeric-only rule: the arbitrary diagnostic key/value pairs must not appear in
    # the mapped output (the protocol has no opaque field for them).
    diag = NS(status=[NS(name="lidar", level=1, message="warn",
                         values=[NS(key="secret", value="leak-me")])])
    hw = telemetry.hardware_from_diagnostics(diag)[0]
    dumped = repr(hw)
    assert "secret" not in dumped and "leak-me" not in dumped


def test_binding_caches_and_maps_telemetry() -> None:
    b = Nav2Binding(robot_id="amr-1", floor=1)
    q = math.sqrt(2) / 2
    b.on_odom(_odom(1.0, 2.0, q, q))
    b.on_battery(NS(percentage=0.42, power_supply_status=2, voltage=49.0))
    b.on_diagnostics(NS(status=[NS(name="lidar", level=2, message="fault", values=[])]))

    assert b.pose("amr-1") == (1.0, 2.0, 1)
    assert math.isclose(b.yaw("amr-1"), math.pi / 2, abs_tol=1e-9)
    assert b.battery_percent("amr-1") == 42
    assert b.latest_battery("amr-1").charging is False
    assert b.latest_hardware("amr-1")[0].status == telemetry.STATUS_FAILED


def test_binding_defaults_before_any_message() -> None:
    b = Nav2Binding(robot_id="amr-1", floor=3)
    assert b.pose("amr-1") == (0.0, 0.0, 3)   # floor default, no odom yet
    assert b.battery_percent("amr-1") == 0
    assert b.latest_hardware("amr-1") == []
