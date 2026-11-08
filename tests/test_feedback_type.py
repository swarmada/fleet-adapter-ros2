# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Every topic the live facade subscribes to names a class the ROS 2 Python packages have.

rclpy generates an action's feedback message INSIDE the action class
(``nav2_msgs.action.NavigateToPose.Impl.FeedbackMessage``); the module has no
``NavigateToPose_FeedbackMessage`` name. A table that names one crashes on the first read of a
live segment, before anything moves, and nothing here would see it: the facade needs a ROS 2
runtime and every other test uses a fake.

So the resolver runs here against modules shaped as rclpy's generated packages are, and --
where ROS 2 is installed (the stand-in) -- against the real packages.
"""

from __future__ import annotations

import sys
import types

import pytest

from fleet_adapter_ros2 import ros2_explore


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def _generated_packages(monkeypatch) -> None:
    """Modules with the shapes rosidl gives Python: messages at module level, an action's
    Goal/Result/Feedback and its FeedbackMessage nested in the action class."""
    class FeedbackMessage:
        pass

    class NavigateToPose:
        class Impl:
            pass
    NavigateToPose.Impl.FeedbackMessage = FeedbackMessage
    mods = {
        "nav_msgs": _module("nav_msgs"),
        "nav_msgs.msg": _module("nav_msgs.msg", OccupancyGrid=type("OccupancyGrid", (), {}),
                                Path=type("Path", (), {})),
        "explore_lite_msgs": _module("explore_lite_msgs"),
        "explore_lite_msgs.msg": _module("explore_lite_msgs.msg",
                                         ExploreStatus=type("ExploreStatus", (), {})),
        "nav2_msgs": _module("nav2_msgs"),
        "nav2_msgs.action": _module("nav2_msgs.action", NavigateToPose=NavigateToPose),
        "sensor_msgs": _module("sensor_msgs"),
        "sensor_msgs.msg": _module("sensor_msgs.msg", Imu=type("Imu", (), {})),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_every_subscribed_type_resolves_to_a_class_the_packages_have(monkeypatch) -> None:
    _generated_packages(monkeypatch)
    facade = ros2_explore.RclpyFacade(binding=None)
    for topic, (type_name, _latched) in facade.topics.items():
        assert isinstance(ros2_explore.RclpyFacade._type(type_name), type), (topic, type_name)


def test_the_navigation_feedback_is_the_actions_nested_feedback_message(monkeypatch) -> None:
    _generated_packages(monkeypatch)
    facade = ros2_explore.RclpyFacade(binding=None)
    type_name, latched = facade.topics["/navigate_to_pose/_action/feedback"]
    cls = ros2_explore.RclpyFacade._type(type_name)
    assert cls is sys.modules["nav2_msgs.action"].NavigateToPose.Impl.FeedbackMessage
    assert latched is False


def test_the_real_packages_have_every_subscribed_type() -> None:
    """Runs where ROS 2 Humble is installed (the stand-in); skipped elsewhere."""
    pytest.importorskip("nav2_msgs.action")
    pytest.importorskip("explore_lite_msgs.msg")
    facade = ros2_explore.RclpyFacade(binding=None)
    for topic, (type_name, _latched) in facade.topics.items():
        assert ros2_explore.RclpyFacade._type(type_name) is not None, (topic, type_name)
