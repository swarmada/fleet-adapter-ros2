# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The live facade keeps what arrives cheaply, and converts only what is read.

The map and the costmap are whole occupancy grids, published about once a second while a segment
runs, and only the latest of each is ever read. The facade used to turn every message into a
Python dict as it arrived and queue up to 500 of each topic; on the stand-in the adapter's
memory grew by about 100 MiB in five minutes of two short segments. Now a message is kept as it
arrived, converted once when it is read, and only the topics read as a sequence -- the
navigation feedback, drained for recoveries, and the IMU, read over a window -- are queued.

The facade needs a ROS 2 runtime, so its imports are stood in for here with modules of the same
shape; what runs is the facade's own code.
"""

from __future__ import annotations

import sys
import types

import pytest
from test_feedback_type import _generated_packages

from fleet_adapter_ros2 import ros2_explore


class _Node:
    def __init__(self):
        self.callbacks = {}

    def create_subscription(self, cls, topic, cb, qos):
        self.callbacks[topic] = cb


@pytest.fixture()
def facade(monkeypatch):
    _generated_packages(monkeypatch)
    conversions = []

    def convert(msg):
        conversions.append(msg)
        return {"converted": msg}
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = lambda **kw: kw
    qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="tl", VOLATILE="v")
    qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="r")
    conv = types.ModuleType("rosidl_runtime_py.convert")
    conv.message_to_ordereddict = convert
    for name, mod in {"rclpy": types.ModuleType("rclpy"), "rclpy.qos": qos,
                      "rosidl_runtime_py": types.ModuleType("rosidl_runtime_py"),
                      "rosidl_runtime_py.convert": conv}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    node = _Node()
    f = ros2_explore.RclpyFacade(types.SimpleNamespace(_node=node), imu_topic="imu")
    f._ensure()
    return f, node, conversions


def test_a_grid_is_not_converted_when_it_arrives_and_only_the_latest_is_kept(facade):
    f, node, conversions = facade
    for i in range(20):
        node.callbacks["/global_costmap/costmap"](f"grid-{i}")
    assert conversions == []
    assert "/global_costmap/costmap" not in f._queues
    d, _ = f.latest("/global_costmap/costmap")
    assert d == {"converted": "grid-19"} and conversions == ["grid-19"]


def test_the_latest_is_converted_once_however_often_it_is_read(facade):
    f, node, conversions = facade
    node.callbacks["/map"]("map-1")
    for _ in range(5):
        f.latest("/map")
    assert conversions == ["map-1"]
    node.callbacks["/map"]("map-2")
    assert f.latest("/map")[0] == {"converted": "map-2"} and conversions == ["map-1", "map-2"]


def test_the_feedback_and_the_imu_are_queued_and_converted_as_read(facade):
    f, node, conversions = facade
    for i in range(3):
        node.callbacks["/navigate_to_pose/_action/feedback"](f"fb-{i}")
        node.callbacks["imu"](f"imu-{i}")
    assert conversions == []
    assert [d["converted"] for d in f.drain("/navigate_to_pose/_action/feedback")] == \
        ["fb-0", "fb-1", "fb-2"]
    assert f.drain("/navigate_to_pose/_action/feedback") == []
    assert [d["converted"] for _, d in f.recent("imu", 10.0)] == ["imu-0", "imu-1", "imu-2"]


def test_nothing_read_is_none(facade):
    f, _, _ = facade
    assert f.latest("explore/status") is None
    assert f.drain("/navigate_to_pose/_action/feedback") == []
