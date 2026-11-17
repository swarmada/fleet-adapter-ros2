# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`Nav2Binding` on a real rclpy: odom messages published on the graph are counted, and a stop is
confirmed only from one received after it.

Runs where ROS 2 is sourced (the humble CI job); skipped everywhere else. Every other test drives
the binding through its ROS-free callbacks; this one goes through `start()`, a real node, a real
subscription and the executor thread, as `--binding ros2` does.
"""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav2_msgs")

from nav_msgs.msg import Odometry
from swarmada_sdk.safety import ESTOP_FAILED, ESTOP_STOPPED, confirm_estop

from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"


class _OdomTopic:
    """A second node publishing nav_msgs/Odometry on `odom` at 20 Hz while `on` is set."""

    def __init__(self) -> None:
        self.node = rclpy.create_node("odom_publisher")
        self.pub = self.node.create_publisher(Odometry, "odom", 10)
        self.vx = 0.0
        self.on = threading.Event()
        self._end = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._end.is_set():
            if self.on.is_set():
                msg = Odometry()
                msg.header.frame_id = "odom"
                msg.pose.pose.orientation.w = 1.0
                msg.twist.twist.linear.x = self.vx
                self.pub.publish(msg)
            time.sleep(0.05)

    def close(self) -> None:
        self._end.set()
        self._thread.join(timeout=2.0)
        self.node.destroy_node()


def _wait(pred, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def live():
    b = Nav2Binding(ROBOT, odometry_source="measured")
    b.start()
    b.spawn(ROBOT)
    topic = _OdomTopic()
    try:
        yield b, topic
    finally:
        topic.close()
        b.stop()


def test_odom_on_the_graph_is_counted(live):
    b, topic = live
    topic.on.set()
    assert _wait(lambda: b.state_seq(ROBOT) >= 5), "no odom reached the binding's subscription"
    before = b.state_seq(ROBOT)
    assert _wait(lambda: b.state_seq(ROBOT) > before)
    assert b.is_stopped(ROBOT)


def test_a_stop_is_confirmed_from_odom_received_after_it(live):
    b, topic = live
    topic.on.set()
    assert _wait(lambda: b.state_seq(ROBOT) >= 5)
    assert confirm_estop(b, ROBOT) == ESTOP_STOPPED


def test_a_stop_is_not_confirmed_once_the_odom_topic_goes_quiet(live):
    b, topic = live
    topic.on.set()
    assert _wait(lambda: b.state_seq(ROBOT) >= 5)
    topic.on.clear()
    time.sleep(0.3)                                    # whatever was in flight has landed
    assert b.is_stopped(ROBOT)                         # the last reading says at rest...
    assert confirm_estop(b, ROBOT) == ESTOP_FAILED     # ...but nothing since the stop does
