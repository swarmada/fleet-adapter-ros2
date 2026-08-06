# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Adapter-level proto mapping for the scenario presets.

The adapter imports grpc at module load, so this file skips where grpc is absent;
the pure engine/binding coverage lives in test_scenarios.py and always runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("grpc")
pytest.importorskip("swarmada_sdk")

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter  # noqa: E402
from fleet_adapter_ros2.robot import SimulatedRobot  # noqa: E402
from fleet_adapter_ros2.scenarios import ScenarioEngine, load_scenario  # noqa: E402


class _HW:
    def __init__(self, component_name="", status=0, degradation_reason=""):
        self.component_name, self.status, self.degradation_reason = (
            component_name, status, degradation_reason)


class _HWComponent:
    def __init__(self, name="", type="", model="", status=0, degradation_reason=""):
        self.name, self.type, self.model, self.status, self.degradation_reason = (
            name, type, model, status, degradation_reason)


class _CapsSnapshot:
    def __init__(self, robot_id="", hardware=None, snapshot_ms=0):
        self.robot_id, self.hardware, self.snapshot_ms = robot_id, hardware or [], snapshot_ms


class _EstopAck:
    # Mirrors fleet_adapter.v1.EstopAck. stop_initiated_at is required by C5.3 —
    # an ACK reporting STOPPED with nothing attesting a hardware stop is the
    # "inferred, not confirmed" case the estop protocol forbids. Keep this fake in
    # step with the real message: it silently diverged once already, which is how
    # the ActionState rename passed these tests while failing the real harness.
    def __init__(self, estop_id="", state=0, message="", stop_initiated_at=None):
        self.estop_id, self.state, self.message = estop_id, state, message
        self.stop_initiated_at = stop_initiated_at


class _SafetyMsg:
    def __init__(self, robot_id="", estop_ack=None):
        self.robot_id, self.estop_ack = robot_id, estop_ack


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakePB:
    HARDWARE_STATUS_UNSPECIFIED, HARDWARE_STATUS_HEALTHY = 0, 1
    HARDWARE_STATUS_DEGRADED, HARDWARE_STATUS_FAILED = 2, 3
    ESTOP_STATE_UNSPECIFIED, ESTOP_STATE_STOPPED, ESTOP_STATE_FAILED = 0, 1, 2
    HardwareStatusUpdate = _HW
    HardwareComponent = _HWComponent
    CapabilitiesSnapshot = _CapsSnapshot
    EstopAck = _EstopAck
    AdapterSafetyMessage = _SafetyMsg
    AdapterMessage = _Msg


class _FakeCall:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def _adapter(scenario_name):
    engine = ScenarioEngine(load_scenario(scenario_name))
    robot = SimulatedRobot(engine)
    robot.spawn("robot-sim-1")
    return fleet_adapter_ros2Adapter(_FakePB(), None, robot, "robot-sim-1")


def test_hardware_updates_map_to_proto_within_the_fault_window() -> None:
    adapter = _adapter("hardware-fault")
    adapter._robot._elapsed = 45.0
    updates = {u.component_name: u for u in adapter._hardware_updates()}
    assert updates["camera_front"].status == _FakePB.HARDWARE_STATUS_DEGRADED
    assert updates["camera_front"].degradation_reason == "simulated camera degradation"


def test_capabilities_snapshot_carries_the_manifest() -> None:
    adapter = _adapter("hardware-fault")
    adapter._send_capabilities_snapshot()
    msg = adapter._outbox.get_nowait()
    assert {c.name for c in msg.capabilities.hardware} == {"lidar_top", "camera_front"}
    assert all(c.status == _FakePB.HARDWARE_STATUS_HEALTHY for c in msg.capabilities.hardware)


def test_estop_drill_emits_confirmed_ack_once() -> None:
    adapter = _adapter("estop-drill")
    adapter._robot.command_move("robot-sim-1", 5.0, 0.0)
    adapter._robot._elapsed = 10.0
    adapter._maybe_estop_drill()
    assert adapter._safety_outbox.empty()

    adapter._robot._elapsed = 21.0
    adapter._maybe_estop_drill()
    msg = adapter._safety_outbox.get_nowait()
    assert msg.estop_ack.state == _FakePB.ESTOP_STATE_STOPPED
    assert adapter._robot.is_stopped("robot-sim-1")  # ground truth: actually at rest

    adapter._maybe_estop_drill()
    assert adapter._safety_outbox.empty()  # fires exactly once


def test_stream_drop_cancels_the_call_and_requests_reconnect() -> None:
    adapter = _adapter("comms-flaky")
    call = _FakeCall()
    adapter._active_call = call

    adapter._robot._elapsed = 10.0
    adapter._maybe_stream_drop()
    assert not adapter._reconnect_pending and not call.cancelled

    adapter._robot._elapsed = 25.0
    adapter._maybe_stream_drop()
    assert adapter._reconnect_pending and call.cancelled
    assert adapter._outbox.get_nowait() is None
