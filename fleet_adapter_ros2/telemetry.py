# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 → Swarmada telemetry mapping (pure, proto-free, rclpy-free).

Maps the standard Nav2/ROS 2 telemetry topics to the protocol's telemetry shapes:

* ``nav_msgs/Odometry``           → position (x, y, yaw, floor)
* ``sensor_msgs/BatteryState``    → battery (percent, charging, voltage)
* ``diagnostic_msgs/DiagnosticArray`` → per-component hardware status

These functions take duck-typed ROS message objects (attribute access only) and
return plain dataclasses, so they are unit-testable with no ROS install and no
generated protobufs. The adapter converts the dataclasses to protobuf.

Numeric-only (api-principles opaque-data rule): only typed numeric readings and a
bounded status enum + reason cross the boundary. Arbitrary ``DiagnosticStatus``
key/value pairs are NOT smuggled into any free-form field — the protocol has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# diagnostic_msgs/DiagnosticStatus.level and sensor_msgs/BatteryState.power_supply_status.
_DIAG_OK, _DIAG_WARN, _DIAG_ERROR, _DIAG_STALE = 0, 1, 2, 3
_PS_UNKNOWN, _PS_CHARGING, _PS_DISCHARGING, _PS_NOT_CHARGING = 0, 1, 2, 3

# Control-plane HardwareStatus names (bound to the proto enum by the adapter).
STATUS_HEALTHY, STATUS_DEGRADED, STATUS_FAILED = "Healthy", "Degraded", "Failed"


@dataclass
class Position:
    x: float
    y: float
    yaw: float
    floor: int = 0


@dataclass
class Battery:
    percent: int | None = None
    charging: bool | None = None   # None = unknown (never defaulted to False)
    voltage: float | None = None


@dataclass
class Hardware:
    component_name: str
    status: str                    # STATUS_HEALTHY / _DEGRADED / _FAILED
    reason: str = ""               # bounded human-readable reason (not opaque data)


@dataclass
class Snapshot:
    position: Position | None = None
    battery: Battery = field(default_factory=Battery)
    hardware: list[Hardware] = field(default_factory=list)


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Z-axis (yaw) rotation in radians from a quaternion (REP-103)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def position_from_odom(odom, floor: int = 0) -> Position:
    """``nav_msgs/Odometry`` → Position. odom is a planar frame, so floor is not
    encoded there — the caller supplies it (default ground)."""
    p = odom.pose.pose.position
    o = odom.pose.pose.orientation
    return Position(x=float(p.x), y=float(p.y),
                    yaw=yaw_from_quaternion(float(o.x), float(o.y), float(o.z), float(o.w)),
                    floor=floor)


def speed_from_odom(odom) -> tuple[float, float]:
    """``nav_msgs/Odometry`` → (linear_speed, angular_speed) magnitudes, in m/s and
    rad/s. This is the robot's measured motion — the ground truth a confirmed stop
    is verified against (C5.3: never infer rest from a command or a timer). Missing
    twist fields read as 0.0 so a message without a twist does not fake motion."""
    t = getattr(getattr(odom, "twist", None), "twist", None)
    lin = getattr(t, "linear", None)
    ang = getattr(t, "angular", None)
    vx = float(getattr(lin, "x", 0.0) or 0.0)
    vy = float(getattr(lin, "y", 0.0) or 0.0)
    wz = float(getattr(ang, "z", 0.0) or 0.0)
    return math.hypot(vx, vy), abs(wz)


def battery_from_state(state) -> Battery:
    """``sensor_msgs/BatteryState`` → Battery. percentage is a 0.0–1.0 fraction
    (NaN = unknown); power_supply_status gives charging as a tri-state."""
    pct = getattr(state, "percentage", float("nan"))
    percent = None
    if pct == pct:  # not NaN
        percent = max(0, min(100, round(float(pct) * 100)))

    status = getattr(state, "power_supply_status", _PS_UNKNOWN)
    if status == _PS_CHARGING:
        charging: bool | None = True
    elif status in (_PS_DISCHARGING, _PS_NOT_CHARGING):
        charging = False
    else:
        charging = None  # UNKNOWN → absent, NOT False

    voltage = getattr(state, "voltage", None)
    voltage = float(voltage) if voltage is not None else None
    return Battery(percent=percent, charging=charging, voltage=voltage)


def _status_from_level(level: int) -> str:
    if level == _DIAG_OK:
        return STATUS_HEALTHY
    if level == _DIAG_WARN:
        return STATUS_DEGRADED
    return STATUS_FAILED  # ERROR or STALE


def hardware_from_diagnostics(diag) -> list[Hardware]:
    """``diagnostic_msgs/DiagnosticArray`` → per-component Hardware. Only the level
    (→ status enum) and the bounded message (→ reason) cross the boundary; the
    arbitrary key/value pairs are intentionally dropped (numeric-only rule)."""
    out: list[Hardware] = []
    for s in getattr(diag, "status", []):
        status = _status_from_level(int(getattr(s, "level", _DIAG_OK)))
        reason = "" if status == STATUS_HEALTHY else str(getattr(s, "message", ""))
        out.append(Hardware(component_name=str(s.name), status=status, reason=reason))
    return out
