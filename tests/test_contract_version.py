# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""C13.1: AdapterHello reports a contract_version inside the range the control plane accepts.

The window is N and N-1 minor within the current major (ADR-0032). The control plane
implements contract 1.3.0, so it accepts ">=1.2.0 <1.4.0" and refuses registration for
anything outside it. This adapter claims 1.3.0: it meets 1.2.0's obligations (the
FA-R5/FA-R6 decline, and a scan answered with its action catalog) and answers 1.3.0's one
addition, the zone_constraints push, rather than declining it. The claim is written out
beside the constant in adapter.py.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

# The control plane's contract version, and the window derived from it.
PLANE = (1, 3)
WINDOW = [(PLANE[0], PLANE[1] - 1), PLANE]


def _major_minor(v: str) -> tuple[int, int]:
    parts = v.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), v
    return int(parts[0]), int(parts[1])


def test_the_declared_contract_version_is_the_one_claimed() -> None:
    assert adapter_mod.CONTRACT_VERSION == "1.3.0"


def test_the_declared_contract_version_is_inside_the_window() -> None:
    assert _major_minor(adapter_mod.CONTRACT_VERSION) in WINDOW


def test_hello_reports_the_declared_contract_version() -> None:
    hello = next(Adapter(pb, None, SimulatedRobot(), "amr-1")._control_requests()).hello
    assert hello.contract_version == adapter_mod.CONTRACT_VERSION
    assert _major_minor(hello.contract_version) in WINDOW


def test_the_zone_constraints_push_the_claim_rests_on_is_answered() -> None:
    # 1.3.0's only addition is optional, and claiming 1.3.0 rests on answering it rather than
    # declining it (test_zone_constraints.py covers what the answer says).
    a = Adapter(pb, None, SimulatedRobot(), "amr-1")
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=9, robot_id="amr-1", zone_constraints=pb.ZoneConstraints())), None)
    res = a._outbox.get_nowait().command_result
    assert res.unsupported is False
    assert res.zone_constraints.acknowledged
