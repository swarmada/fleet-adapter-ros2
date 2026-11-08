# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The Nav2 binding's teleop gate, without ROS: a fake publisher stands in for ``cmd_vel``.

The binding subscribes to the teleop topic always and publishes a message to ``cmd_vel`` only as
the gate admits it, under the lock its own stops take. `_halt_locked` closes the gate, so every
stop path -- hold, stop, pause, the estop and the lease self-stop through them -- closes it before
the stop's zero Twist goes out.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace as NS

import pytest
from conftest import mark_transport_live

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
TOPIC = "/cmd_vel_teleop"


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Twist:
    def __init__(self) -> None:
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class Publisher:
    def __init__(self) -> None:
        self.sent: list[tuple[float, float, float]] = []

    def publish(self, t) -> None:
        self.sent.append((t.linear.x, t.linear.y, t.angular.z))


def msg(vx, vy, wz):
    t = Twist()
    t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
    return t


def binding(clock=None):
    b = mark_transport_live(Nav2Binding(ROBOT, clock=clock, teleop_topic=TOPIC))
    b._cmd_vel, b._twist_type = Publisher(), Twist
    return b, b._cmd_vel


def drive(**kw) -> md.ManualDrive:
    doc = {"operation": md.OPERATION, "x": 1.0, "y": 0.0, "frame": "map", "radius_m": 0.25, **kw}
    return md.ManualDrive(md.parse_payload(doc), started_at=0.0)


@pytest.mark.parametrize("topic", ["/cmd_vel", "cmd_vel", " "])
def test_teleop_on_the_base_topic_is_refused_at_construction(topic) -> None:
    with pytest.raises(ValueError, match="forwarding is the gate"):
        Nav2Binding(ROBOT, teleop_topic=topic)


def test_closed_the_teleop_topic_reaches_nothing() -> None:
    b, pub = binding()
    b.on_teleop(msg(0.1, 0.0, 0.3))
    assert pub.sent == []
    assert Nav2Binding(ROBOT).teleop_topic() is None


def test_open_it_forwards_clamped_and_every_stop_closes_it_before_its_zero() -> None:
    for stop in ("command_stop", "command_hold", "command_pause"):
        b, pub = binding()
        b.open_teleop(ROBOT, "manual-a", drive(), lease_s=60)
        sent = len(pub.sent)                    # opening holds the base: its zero
        b.on_teleop(msg(0.5, 0.5, 5.0))
        vx, vy, wz = pub.sent[sent]
        assert abs((vx * vx + vy * vy) ** 0.5 - 0.2) < 1e-9 and wz == 0.7
        getattr(b, stop)(ROBOT)
        assert pub.sent[-1] == md.ZERO, stop
        n = len(pub.sent)
        b.on_teleop(msg(0.1, 0.0, 0.0))
        assert len(pub.sent) == n and not b.teleop_open(ROBOT), stop


def test_the_lease_deadline_closes_it_on_the_binding_clock() -> None:
    clock = Clock()
    b, pub = binding(clock)
    b.open_teleop(ROBOT, "manual-a", drive(), lease_s=1.0)
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    clock.t += 1.0
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert pub.sent[n:] == [(0.1, 0.0, 0.0)]
    assert not b.extend_teleop(ROBOT, "another-action", 10.0)
    assert b.extend_teleop(ROBOT, "manual-a", 10.0)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert pub.sent[n:] == [(0.1, 0.0, 0.0)] * 2


def test_opening_drops_a_retained_goal_so_a_resume_has_nothing_to_drive() -> None:
    b, _ = binding()
    with b._lock:
        b._goal = object()                      # a Navigate target a pause would keep
    b.open_teleop(ROBOT, "manual-a", drive(), lease_s=60)
    assert b._goal is None


def test_without_a_map_transform_the_drive_ends_failed_and_the_gate_closes() -> None:
    """On the robot, a manual-drive started with no map -> odom transform has no pose it may
    judge: it ends Failed at once, with the robot held."""
    b, pub = binding()
    b.spawn(ROBOT)
    a = Adapter(pb, None, b, ROBOT)
    body = json.dumps({"operation": md.OPERATION, "x": 1.0, "y": 0.0, "frame": "map",
                       "radius_m": 0.25}).encode()
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="manual-a", action_type="Custom", assignment_id="a-1", fencing_token=1,
            lease_generation=1, lease_duration_ms=60_000, payload_json=body))), None)
    deadline = time.monotonic() + 5
    ends: list = []
    while not ends:
        assert time.monotonic() < deadline
        ends += [m.action_status for m in list(a._outbox.queue)
                 if m.WhichOneof("payload") == "action_status"
                 and m.action_status.state == pb.ACTION_STATE_FAILED]
        time.sleep(0.01)
    assert "frame 'odom'" in ends[0].message
    assert not b.teleop_open(ROBOT) and pub.sent[-1] == md.ZERO
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n
