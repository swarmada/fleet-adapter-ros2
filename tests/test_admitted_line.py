# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter says, in one line, when its robot is admitted.

Until now the adapter printed NOT_ADMITTED while it waited for an operator and nothing once the
operator admitted the robot, so the admission showed only in the cluster. The line names the robot
and nothing else: no address, not even an edge endpoint the same RegisterAck carries.
"""

from __future__ import annotations

import re

import pytest

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ADDRESS = re.compile(r"\d{1,3}(\.\d{1,3}){3}|:\d{2,5}\b|\[[0-9a-fA-F:]+\]")


@pytest.fixture(autouse=True)
def _own_state(tmp_path, monkeypatch):
    """Nothing here may read or write a state file in the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _adapter():
    return Adapter(pb, pb_grpc, SimulatedRobot(None), "amr-1", namespace="warehouse-a",
                   provisioning_interval_s=3600.0)


def _accepted():
    return pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        # A documentation address (RFC 5737), so a line that printed it would be caught.
        accepted=True, edge_endpoints=[pb.EdgeEndpoint(zone="zone-a", address="192.0.2.10:7443")]))


def _not_admitted():
    return pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        accepted=False, rejection=pb.REGISTRATION_REJECTION_NOT_ADMITTED))


def _admitted_lines(err: str) -> list[str]:
    return [x for x in err.splitlines() if x.startswith("ADMITTED")]


def test_an_operators_admission_is_said_in_one_line_with_no_address(capsys) -> None:
    a = _adapter()
    a._handle_control_plane(_not_admitted(), None)
    assert _admitted_lines(capsys.readouterr().err) == []
    a._handle_control_plane(_accepted(), None)
    (line,) = _admitted_lines(capsys.readouterr().err)
    assert "robot=amr-1" in line and "operator" in line
    assert not ADDRESS.search(line), line


def test_a_robot_admitted_from_the_start_is_said_once_and_a_reregistration_is_not(capsys) -> None:
    a = _adapter()
    a._handle_control_plane(_accepted(), None)
    (line,) = _admitted_lines(capsys.readouterr().err)
    assert "robot=amr-1" in line and not ADDRESS.search(line), line
    a._handle_control_plane(_accepted(), None)      # a reconnect's registration: no new admission
    assert _admitted_lines(capsys.readouterr().err) == []


def test_a_refusal_other_than_not_admitted_says_no_admission(capsys) -> None:
    a = _adapter()
    a._handle_control_plane(pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        accepted=False, rejection=pb.REGISTRATION_REJECTION_NAMESPACE_MISMATCH)), None)
    assert _admitted_lines(capsys.readouterr().err) == []
