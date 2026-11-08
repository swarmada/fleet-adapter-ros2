# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`pose()` is a `map`-frame position, composed through TF, and says when it is not.

Goals leave in `map`. Odometry is in `odom`, and `map -> odom` is whatever localisation has
absorbed so far: on hardware, after one 1 m move under SLAM, 0.088 m and -9.8 degrees. A pose
read straight from odometry is that far from the frame every goal is sent in, and nothing
about the number says so. So the binding composes the latest `map -> odom` with the odom pose,
and when `map` is absent it returns the odom pose with `pose_frame() == "odom"` and says so,
once, on each transition -- never a silent fall-back.
"""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "robot-conformance-1"


def _quat(yaw: float) -> NS:
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def _odom(x: float, y: float, yaw: float = 0.0, frame: str = "odom") -> NS:
    return NS(header=NS(frame_id=frame),
              pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0), orientation=_quat(yaw))))


def _map_to_odom(x: float, y: float, yaw: float) -> NS:
    """The shape tf2_ros.Buffer.lookup_transform returns: a TransformStamped."""
    return NS(transform=NS(translation=NS(x=x, y=y, z=0.0), rotation=_quat(yaw)))


def _binding() -> Nav2Binding:
    b = Nav2Binding(ROBOT)
    b.on_odom(_odom(1.0, 0.0))
    return b


def test_a_non_identity_map_to_odom_changes_what_pose_returns() -> None:
    b = _binding()
    odom_pose = b.pose(ROBOT)
    # map -> odom: 90 degrees, then (1, -2). The odom point (1, 0) is (0, 1) after the
    # rotation and (1, -1) after the translation.
    b.on_map_to_odom(_map_to_odom(1.0, -2.0, math.pi / 2))
    x, y, floor = b.pose(ROBOT)
    assert (x, y) != odom_pose[:2]
    assert math.isclose(x, 1.0, abs_tol=1e-9) and math.isclose(y, -1.0, abs_tol=1e-9)
    assert floor == odom_pose[2]
    assert math.isclose(b.yaw(ROBOT), math.pi / 2, abs_tol=1e-9)
    assert b.pose_frame(ROBOT) == "map"


def test_the_measured_hardware_divergence_is_applied() -> None:
    # 2026-09-17, one commanded 1 m move under SLAM: map -> odom was 0.088 m, -9.798 deg.
    b = Nav2Binding(ROBOT)
    b.on_odom(_odom(3.0, 0.0))
    b.on_map_to_odom(_map_to_odom(0.088, 0.0, math.radians(-9.798)))
    x, y, _ = b.pose(ROBOT)
    assert math.isclose(y, -3.0 * math.sin(math.radians(9.798)), abs_tol=1e-9)  # ~0.51 m
    assert math.isclose(x, 0.088 + 3.0 * math.cos(math.radians(9.798)), abs_tol=1e-9)


def test_a_localisation_correction_moves_pose_without_a_new_odom_message() -> None:
    # AMCL jumps map -> odom on a particle-filter correction; the robot has not moved in
    # odom, and pose() must still follow the correction.
    b = _binding()
    b.on_map_to_odom(_map_to_odom(0.0, 0.0, 0.0))
    before = b.pose(ROBOT)
    b.on_map_to_odom(_map_to_odom(0.5, 0.0, 0.0))
    assert math.isclose(b.pose(ROBOT)[0], before[0] + 0.5, abs_tol=1e-9)


def test_map_absent_is_a_declared_degradation(capsys) -> None:
    b = _binding()
    assert b.pose(ROBOT)[:2] == (1.0, 0.0)          # the odom pose, unconverted
    assert b.pose_frame(ROBOT) == "odom"            # ...and it says so
    b.on_map_to_odom(None)                          # a TF lookup that found no map
    b.pose(ROBOT)
    b.pose(ROBOT)
    err = capsys.readouterr().err
    assert err.count("no map -> odom transform") == 1  # announced once, not per call


def test_losing_and_regaining_map_are_both_announced(capsys) -> None:
    b = _binding()
    b.on_map_to_odom(_map_to_odom(0.0, 0.0, 0.0))
    assert b.pose_frame(ROBOT) == "map"
    b.on_map_to_odom(None)
    assert b.pose_frame(ROBOT) == "odom"
    b.on_map_to_odom(_map_to_odom(0.0, 0.0, 0.0))
    assert b.pose_frame(ROBOT) == "map"
    err = capsys.readouterr().err
    assert err.count("no map -> odom transform") == 1
    assert err.count("map -> odom transform available") == 1


def test_the_transform_is_looked_up_from_the_odometry_frame() -> None:
    b = Nav2Binding(ROBOT)
    b.on_odom(_odom(0.0, 0.0, frame="robot1/odom"))
    assert b.odom_frame(ROBOT) == "robot1/odom"
    b2 = Nav2Binding(ROBOT)
    b2.on_odom(NS(pose=_odom(0.0, 0.0).pose))       # a message with no header
    assert b2.odom_frame(ROBOT) == "odom"


def test_stale_feedback_guard_still_reads_navs_own_pose() -> None:
    # The feedback guard compares Nav2's feedback pose -- already in the goal's frame --
    # with the goal. It must not start reading pose(): with a map -> odom offset applied
    # or not, the feedback's own position is the one in the goal's frame.
    b = _binding()
    b.on_map_to_odom(_map_to_odom(100.0, 100.0, 0.0))  # a pose() nowhere near the goal
    b.command_move(ROBOT, 3.0, 0.0)
    b.on_feedback(3.0, pose=(0.0, 0.0))                # straight line 3.0: a real distance
    assert b.task_progress(ROBOT) == 0                 # accepted as the 100% baseline
    b.on_feedback(1.5, pose=(1.5, 0.0))
    assert b.task_progress(ROBOT) == 50
