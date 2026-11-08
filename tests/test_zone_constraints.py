# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The zone_constraints push (FA-R7): the operating rules in force where the robot now is.

This adapter honours no operating subject, so its answer is an honest receipt: an empty set is
permission and is acknowledged with nothing named; a set carrying rules is acknowledged with
every subject named as one it cannot honour -- the robot is already in the zone, and naming
them is the whole of what it can do. A set with unresolved zones is incomplete, and the
adapter fails closed on the claim: it does not acknowledge having applied rules nobody could
tell it, and names the zones.
"""

from __future__ import annotations

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "robot-1"


def _push(constraints=(), unresolved=(), zones=("ward-a",)):
    robot = SimulatedRobot()
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._handle_command(pb.Command(command_id=3, robot_id=ROBOT, zone_constraints=pb.ZoneConstraints(
        containing_zones=list(zones),
        operating_constraints=[pb.EffectiveOperatingConstraint(subject=s) for s in constraints],
        unresolved_zones=list(unresolved))))
    result = a._outbox.get_nowait().command_result
    assert not result.unsupported
    assert result.HasField("zone_constraints")
    return result.zone_constraints


def test_an_empty_set_is_permission_and_is_acknowledged() -> None:
    ack = _push()
    assert ack.acknowledged
    assert not ack.unhonourable_subjects


def test_every_subject_in_force_is_named_as_unhonourable() -> None:
    ack = _push(constraints=("camera", "example.com/leaf-blower"))
    assert ack.acknowledged  # received: the robot is in a zone it cannot comply with
    assert list(ack.unhonourable_subjects) == ["camera", "example.com/leaf-blower"]


def test_unresolved_zones_fail_closed_on_the_claim() -> None:
    ack = _push(constraints=("camera",), unresolved=("bay-7",))
    assert not ack.acknowledged
    assert list(ack.unhonourable_subjects) == ["camera"]
    assert "bay-7" in ack.message
