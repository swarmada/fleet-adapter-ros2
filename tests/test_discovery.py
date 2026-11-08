# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C2.1: a robot the control plane has not admitted is announced, and registered again later.

The control plane answers a RegisterRobot for a robot it has no Robot object for with
NOT_ADMITTED. An adapter that does nothing then stays unmanaged for ever: nothing appears for
an operator to admit, because admission promotes a DiscoveredRobot and only a DiscoverRobot
creates one. So the adapter announces the robot (identity and claimed capabilities, nothing it
does not know), registers again after the provisioning interval, and once admitted sends the
catalog the unadmitted session's snapshot could not deliver. Any other refusal is final and is
not answered by announcing.
"""

from __future__ import annotations

import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot


def _adapter(interval=0.05):
    return Adapter(pb, pb_grpc, SimulatedRobot(None), "amr-1", namespace="warehouse-a",
                   provisioning_interval_s=interval)


def _drain(a) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _ack(accepted, rejection=0, message=""):
    return pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        accepted=accepted, rejection=rejection, message=message))


def _kinds(msgs) -> list[str]:
    return [m.WhichOneof("payload") for m in msgs]


def test_not_admitted_is_announced_then_registered_again() -> None:
    a = _adapter()
    a._handle_control_plane(_ack(False, pb.REGISTRATION_REJECTION_NOT_ADMITTED), None)
    first = _drain(a)
    assert _kinds(first) == ["discover"]
    d = first[0].discover
    assert d.robot_id == "amr-1" and "navigation.2d" in d.reported_capabilities
    assert not d.firmware_version and not d.HasField("mac")      # nothing it does not know
    time.sleep(0.2)
    again = _drain(a)
    assert _kinds(again) == ["register"] and again[0].register.robot_id == "amr-1"


def test_once_admitted_the_catalog_is_sent_again_and_no_more_registers_follow() -> None:
    a = _adapter()
    a._handle_control_plane(_ack(False, pb.REGISTRATION_REJECTION_NOT_ADMITTED), None)
    _drain(a)
    a._handle_control_plane(_ack(True), None)
    assert "capabilities" in _kinds(_drain(a))
    time.sleep(0.2)
    assert "register" not in _kinds(_drain(a))


def test_any_other_refusal_is_not_answered_by_announcing(capsys) -> None:
    a = _adapter()
    a._handle_control_plane(_ack(False, pb.REGISTRATION_REJECTION_NAMESPACE_MISMATCH,
                                 "not this adapter's"), None)
    time.sleep(0.2)
    assert _drain(a) == []
    assert "register refused robot=amr-1: not this adapter's" in capsys.readouterr().err


def test_the_discover_acks_ttl_sets_the_interval() -> None:
    assert adapter_mod._provisioning_interval(0, None) == 60.0
    assert adapter_mod._provisioning_interval(120, None) == 60.0
    assert adapter_mod._provisioning_interval(10_000, None) == 300.0
    assert adapter_mod._provisioning_interval(120, 5.0) == 5.0
    a = _adapter(interval=None)
    a._handle_control_plane(pb.ControlPlaneMessage(discover_ack=pb.DiscoverAck(
        accepted=True, ttl_seconds=40, message="staged")), None)
    assert a._provisioning_s == 20.0
