# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The guided-session bridge never drives.

An operator drives the robot in a guided session; the bridge only watches. It may subscribe,
listen to TF and run timers. These tests fail if its source names anything that could move a
robot -- a publisher, a service or action client, a goal, a velocity command -- and if, wired
to a node, it asks the node for anything but a subscription and a timer.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

from fleet_adapter_ros2.mapview import bridge as b
from fleet_adapter_ros2.mapview import contract as c

SOURCE = Path(b.__file__)

# Names whose presence in the bridge's source means it could command a robot.
FORBIDDEN = {
    "create_publisher", "create_client", "create_service", "ActionClient", "ActionServer",
    "send_goal", "send_goal_async", "call", "call_async", "publish", "Twist", "TwistStamped",
    "NavigateToPose", "NavigateThroughPoses", "FollowWaypoints", "Goal", "cancel_goal_async",
}
FORBIDDEN_TEXT = ("cmd_vel", "navigate_to_pose", "/goal_pose", "teleop")


def drive_constructs(source: str) -> list[str]:
    """Every forbidden name the source calls, imports or reads as an attribute, and every
    forbidden topic it spells."""
    found = []
    for node in ast.walk(ast.parse(source)):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.alias):
            name = node.name.split(".")[-1]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found += [f"text {t!r}" for t in FORBIDDEN_TEXT if t in node.value.lower()]
        if name in FORBIDDEN:
            found.append(name)
    return found


def test_the_bridge_source_constructs_nothing_that_could_drive():
    assert drive_constructs(SOURCE.read_text()) == []


@pytest.mark.parametrize("line", [
    "node.create_publisher(Twist, 'cmd_vel', 10)",
    "client = ActionClient(node, NavigateToPose, 'navigate_to_pose')",
    "client.send_goal_async(goal)",
    "from geometry_msgs.msg import Twist",
    "node.create_client(Empty, '/reset')",
    "pub.publish(msg)",
])
def test_the_check_would_catch_a_bridge_that_drove(line):
    """The check above is not vacuous: each of these, added to the bridge, would fail it."""
    assert drive_constructs(SOURCE.read_text() + "\n" + line + "\n") != []


class RecordingNode:
    """A node that answers a subscription and a timer, and fails the test on anything else."""

    def __init__(self) -> None:
        self.subscriptions, self.timers = [], []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))

    def create_timer(self, period, callback):
        self.timers.append((period, callback))

    def __getattr__(self, name):
        raise AssertionError(f"the bridge asked its node for {name}")


class ListSink:
    def __init__(self) -> None:
        self.msgs = []

    def put(self, m):
        self.msgs.append(c.check(m))


def grid_msg(data, w, h, res=0.05, ox=-1.0, oy=-2.0, qz=0.0, qw=1.0):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw)
    origin = types.SimpleNamespace(position=types.SimpleNamespace(x=ox, y=oy), orientation=q)
    info = types.SimpleNamespace(resolution=res, width=w, height=h, origin=origin)
    return types.SimpleNamespace(info=info, data=data)


def test_wired_to_a_node_it_asks_for_one_subscription_and_two_timers_only():
    node, sink = RecordingNode(), ListSink()
    bridge = b.GuidedBridge(sink, "sim-a", "s1")
    b.attach(node, bridge, lambda: (1.0, 2.0, 30.0), map_type="OccupancyGrid", map_qos="latched")
    assert [(t, q) for _, t, _, q in node.subscriptions] == [("/map", "latched")]
    assert sorted(p for p, _ in node.timers) == [c.POSE_PERIOD_S, 1.0]


def test_a_guided_session_is_written_as_operator_driven_and_read_back(tmp_path):
    now = [100.0]
    node, sink = RecordingNode(), ListSink()
    run = tmp_path / "run"
    run.mkdir()
    (run / "targets.txt").write_text("A 1.0 2.0\n")
    bridge = b.GuidedBridge(sink, "sim-a", "s1", run_dir=run, clock=lambda: now[0])
    b.attach(node, bridge, lambda: (1.0, 2.0, 30.0), map_type=None, map_qos=None)
    on_map = node.subscriptions[0][2]
    pose_timer = next(cb for p, cb in node.timers if p == c.POSE_PERIOD_S)
    bridge.start()
    on_map(grid_msg([-1, 0, 100, 0, 0, -1], 3, 2))
    pose_timer()
    now[0] += 1.0
    on_map(grid_msg([0, 0, 0, 0, 0, 0], 3, 2))                  # within MAP_CHECK_S: not sent
    (run / "targets.txt").write_text("A 1.0 2.0\nB 3.0 1.0\n")
    bridge.tick()
    now[0] += c.MAP_REPEAT_S
    bridge.tick()                                               # the repeat: same sha, new seq
    bridge.stop()
    st = c.ViewState()
    for m in sink.msgs:
        assert st.accept(m)
    kinds = [m["kind"] for m in sink.msgs]
    assert kinds.count("MapSnapshot") == 2 and st.map[3] == 1
    assert st.map[0] == [[-1, 0, 100], [0, 0, -1]]
    assert st.last_pose[1:] == (1.0, 2.0, 30.0)
    assert st.overlays["target"]["label"] == "B"
    assert st.status["phase"] == "stopped" and st.status["source"] == "guided:s1"
    assert c.driver_of(st.status["source"]) == "operator"
    assert "autonom" not in " ".join(m.get("text", "") for m in sink.msgs).lower()


def test_a_map_turned_in_its_frame_is_refused_not_drawn():
    sink = ListSink()
    bridge = b.GuidedBridge(sink, "sim-a", "s1")
    bridge.on_map(grid_msg([0] * 4, 2, 2, qz=0.38268343, qw=0.92387953))   # 45 degrees
    assert bridge.refused_maps == 1 and not [m for m in sink.msgs if m["kind"] == "MapSnapshot"]


def test_a_session_label_the_contract_cannot_carry_is_refused_at_start():
    with pytest.raises(c.ContractError):
        b.GuidedBridge(ListSink(), "sim-a", "my session")
