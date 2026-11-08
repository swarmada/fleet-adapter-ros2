#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A fake ROS 2 robot for exercising fleet-adapter-ros2's LIVE runtime paths.

Why this exists
---------------
`fleet_adapter_ros2/nav2_binding.py` is fully written but four of its blocks carry
``# pragma: no cover - requires the ROS 2 runtime``: ``start()`` (node, subscriptions,
action client, cmd_vel publisher), live goal dispatch, live goal cancel, and the
zero-Twist halt. Two of those four are on the emergency-stop path. No unit test calls
``start()``.

This node is the smallest thing that makes them execute. It is a ROS 2 runtime with a
robot's *interface* and no robot: to the adapter it is indistinguishable from hardware,
which is the whole premise of a ROS 2 *class* adapter (ADR-0005, no vendor branches).

What it does
------------
* publishes ``nav_msgs/Odometry`` on ``/odom``
* publishes ``sensor_msgs/BatteryState`` on ``/battery_state``
* publishes ``diagnostic_msgs/DiagnosticArray`` on ``/diagnostics``
* serves ``nav2_msgs/action/NavigateToPose`` with real feedback and real cancellation
* subscribes ``geometry_msgs/Twist`` on ``/cmd_vel`` and INTEGRATES it into the odometry
  it publishes

That last property is the load-bearing one. A commanded halt produces a real
deceleration in the odometry the adapter reads back, so ``is_stopped`` is confirmed
from ground truth arriving over DDS -- never inferred from the command, a timer, or
silence (CONFORMANCE.md C5.3). Deceleration is deliberately NOT instantaneous, so the
adapter observes a non-zero twist and then a zero one, exactly as it would on a robot.

What it is NOT
--------------
Not Nav2. There is no planner, no costmap and no recovery behaviour, so it cannot
surface goal rejection on an unreachable pose, ABORTED-after-recovery, or a replanning
controller whose ``distance_remaining`` is non-monotonic. That is rung R2 -- run this
against the real Nav2 stack once the transport paths here are green.

Run
---
    docker run --rm -it --net=host -v "$PWD":/ws ros:jazzy bash
    # in the container:
    apt-get update && apt-get install -y ros-jazzy-nav2-msgs
    source /opt/ros/jazzy/setup.bash
    python3 /ws/fake_robot_node.py

    # second shell, same DDS domain:
    python3 fleet_adapter_ros2/adapter.py --binding ros2 --endpoint <control-plane> \
        --stall-window 5.0 --stall-min-progress 0.10

    The --stall-* values are THIS plant's, derived from its constants below: a 180-degree
    turn in place at TURN_SPEED, with the ramps, takes about 4.1 s and gains no distance,
    so the window must outlast it; 0.10 m is GOAL_TOLERANCE, the least approach this node
    distinguishes. They are not a default for any other robot.

Fault injection
---------------
    --goal-outcome abort     terminate every goal ABORTED instead of SUCCEEDED
    --reject-goals           reject every goal at acceptance (adapter must map to FAILED)
    --decel 0.5              slow deceleration, widens the confirmed-stop window
    --stall-after 3.0        stop making progress mid-goal without terminating it
    --cruise 0.2             drive at 0.2 m/s: a platform capped below the cruise floor

Anything else outside a ``--ros-args`` section is REFUSED with exit 2, not handed to rclpy
to be ignored (see ``fake_robot_args.py``).
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from fake_robot_args import ArgsError, build_parser, parse

# The exception rclpy raises on an invalid goal-state transition. It lives in the
# pybind11 extension, which is not a stable public API path, so fall back to a type
# that still lets the except clause compile if it ever moves.
try:  # pragma: no cover - import shape depends on the rclpy build
    from rclpy._rclpy_pybind11 import RCLError as _RCLError
except ImportError:  # pragma: no cover
    _RCLError = RuntimeError
from sensor_msgs.msg import BatteryState

TICK_HZ = 20.0
TICK_DT = 1.0 / TICK_HZ
TURN_SPEED = 1.0        # rad/s
GOAL_TOLERANCE = 0.10   # m
HEADING_TOLERANCE = 0.15  # rad
CMD_VEL_TIMEOUT = 0.5   # s -- an external Twist overrides the controller for this long


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Planar yaw -> quaternion, without pulling in tf_transformations."""
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def _yaw_from_quaternion(q: Quaternion) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class FakeRobot(Node):
    """A differential-drive robot with an interface and no hardware."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("fake_robot")
        self._args = args
        self._lock = threading.Lock()

        # Ground-truth state. Everything published is derived from these.
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = 0.0        # commanded linear, m/s
        self._w = 0.0        # commanded angular, rad/s
        self._battery = 100.0

        # Active NavigateToPose goal, if any. ONE goal at a time, as Nav2 serves them: the goal
        # that started last owns the controller (`_goal_seq`), and an older one still executing
        # is PREEMPTED -- it ends ABORTED and never touches the target. Cancellation is read
        # from each goal's own handle. These used to be shared by every execute loop, so
        # cancelling one goal ended another, far from its target, SUCCEEDED or ABORTED.
        self._goal_xy: tuple[float, float] | None = None
        self._goal_yaw = 0.0
        self._goal_started_at = 0.0
        self._initial_distance = 0.0
        self._goal_seq = 0

        # External /cmd_vel override -- this is how the adapter's halt reaches us.
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._cmd_at = 0.0

        cb = ReentrantCallbackGroup()

        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._batt_pub = self.create_publisher(BatteryState, "battery_state", 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, "diagnostics", 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10, callback_group=cb)

        # PASSIVE MODE (R2). Nav2 owns `navigate_to_pose` and drives the robot over
        # /cmd_vel. Advertising our own action server here would collide with Nav2's,
        # and the built-in controller would fight Nav2's for the same wheels -- so
        # both are switched off and this node becomes a pure plant: integrate the
        # commanded twist, report odometry.
        self._action = None
        if not args.passive:
            self._action = ActionServer(
                self,
                NavigateToPose,
                "navigate_to_pose",
                execute_callback=self._execute,
                goal_callback=self._on_goal_request,
                cancel_callback=self._on_cancel_request,
                callback_group=cb,
            )

        self.create_timer(TICK_DT, self._tick, callback_group=cb)
        self.create_timer(1.0, self._publish_diagnostics, callback_group=cb)

        mode = "PASSIVE (R2: Nav2 drives via /cmd_vel)" if args.passive \
            else "ACTIVE (R1: own NavigateToPose server + controller)"
        self.get_logger().info(
            f"fake_robot up [{mode}]: cruise={args.cruise} m/s decel={args.decel} m/s^2 "
            f"outcome={args.goal_outcome} reject={args.reject_goals}"
        )

    # -- /cmd_vel: the adapter's halt path ------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        """An external Twist preempts the built-in controller for CMD_VEL_TIMEOUT.

        A zero Twist is how fleet-adapter-ros2 commands a halt alongside cancelling the
        Nav2 goal. Honouring it here -- and letting it show up in the odometry we
        publish -- is what makes the adapter's confirmed-stop check meaningful.
        """
        with self._lock:
            self._cmd_v = float(msg.linear.x)
            self._cmd_w = float(msg.angular.z)
            self._cmd_at = time.monotonic()
        if abs(msg.linear.x) < 1e-9 and abs(msg.angular.z) < 1e-9:
            self.get_logger().info("cmd_vel: HALT commanded -- decelerating to rest")

    # -- NavigateToPose -------------------------------------------------------

    def _on_goal_request(self, goal_request) -> GoalResponse:
        if self._args.reject_goals:
            self.get_logger().warn("rejecting goal (--reject-goals)")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel_request(self, goal_handle) -> CancelResponse:
        # Accepting is all: rclpy then marks THIS goal's handle cancel-requested, and only its
        # own execute loop reads that.
        self.get_logger().info("cancel requested -- accepting")
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        pose = goal_handle.request.pose.pose
        gx, gy = float(pose.position.x), float(pose.position.y)
        gyaw = _yaw_from_quaternion(pose.orientation)

        with self._lock:
            self._goal_seq += 1
            mine = self._goal_seq              # this goal owns the controller from here
            self._goal_xy = (gx, gy)
            self._goal_yaw = gyaw
            self._goal_started_at = time.monotonic()
            self._initial_distance = math.hypot(gx - self._x, gy - self._y)

        self.get_logger().info(f"goal accepted: ({gx:.2f}, {gy:.2f}) d={self._initial_distance:.2f} m")

        feedback = NavigateToPose.Feedback()
        while rclpy.ok():
            time.sleep(TICK_DT)
            with self._lock:
                current = mine == self._goal_seq
                if current:
                    dist = math.hypot(gx - self._x, gy - self._y)
                    elapsed = time.monotonic() - self._goal_started_at
                    x, y, yaw = self._x, self._y, self._yaw

            if goal_handle.is_cancel_requested:  # a cancel for THIS goal, and no other
                # Checked first: a goal cancelled and then overtaken by a newer one before this
                # loop looked was still cancelled, and says so.
                self._release(mine)             # stops steering, if it still steers
                self._end(goal_handle, "canceled", "goal CANCELED")
                return NavigateToPose.Result()

            if not current:
                # A newer goal started: this one is PREEMPTED, as Nav2 preempts it -- ended
                # ABORTED, without touching the target, which is the newer goal's now.
                self._end(goal_handle, "abort", "goal ABORTED (preempted by a newer goal)")
                return NavigateToPose.Result()

            feedback.distance_remaining = float(dist)
            feedback.navigation_time.sec = int(elapsed)
            feedback.current_pose.header.frame_id = "map"
            feedback.current_pose.pose.position.x = x
            feedback.current_pose.pose.position.y = y
            feedback.current_pose.pose.orientation = _yaw_to_quaternion(yaw)
            try:
                goal_handle.publish_feedback(feedback)
            except _RCLError as exc:
                # The feedback publisher is destroyed when the handle goes terminal (teardown,
                # or a cancel racing this loop). There is no check that closes that window,
                # only a catch.
                self.get_logger().debug(f"feedback publisher gone: {exc}")
                return NavigateToPose.Result()

            if dist <= GOAL_TOLERANCE:
                self._release(mine)
                if self._args.goal_outcome == "abort":
                    self._end(goal_handle, "abort", "goal ABORTED (--goal-outcome abort)")
                else:
                    self._end(goal_handle, "succeed", "goal SUCCEEDED")
                return NavigateToPose.Result()

        self._end(goal_handle, "abort", "goal ABORTED (shutting down)")
        return NavigateToPose.Result()

    def _release(self, mine: int) -> None:
        """Stop steering -- only if goal `mine` still owns the controller."""
        with self._lock:
            if mine == self._goal_seq:
                self._goal_xy = None

    def _end(self, goal_handle, how: str, log: str) -> None:
        """Move the goal to its terminal state once. A handle can still reach a terminal state
        outside this loop (a cancel racing it, teardown), and rclpy then raises RCLError on the
        second transition; that is caught, since a run whose normal output carries a stack trace
        is one where a real fault hides in plain sight."""
        if not goal_handle.is_active:
            return
        try:
            getattr(goal_handle, how)()
        except _RCLError as exc:
            self.get_logger().debug(f"goal already terminal ({how}): {exc}")
            return
        (self.get_logger().warn if how == "abort" else self.get_logger().info)(log)

    # -- physics --------------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            override_active = (now - self._cmd_at) < CMD_VEL_TIMEOUT
            stalled = (
                self._args.stall_after > 0.0
                and self._goal_xy is not None
                and (now - self._goal_started_at) > self._args.stall_after
            )

            if override_active:
                # External /cmd_vel wins. Approach it through a real deceleration so a
                # halt is observable as motion -> rest, never as an instantaneous jump.
                target_v, target_w = self._cmd_v, self._cmd_w
            elif not self._args.passive and self._goal_xy is not None and not stalled:
                target_v, target_w = self._controller()
            else:
                target_v, target_w = 0.0, 0.0

            self._v = self._approach(self._v, target_v,
                                     self._rate(self._v, target_v) * TICK_DT)
            self._w = self._approach(self._w, target_w,
                                     self._rate(self._w, target_w) * TICK_DT)

            # Integrate. This is the ground truth the adapter reads back.
            self._yaw = math.atan2(math.sin(self._yaw + self._w * TICK_DT),
                                   math.cos(self._yaw + self._w * TICK_DT))
            self._x += self._v * math.cos(self._yaw) * TICK_DT
            self._y += self._v * math.sin(self._yaw) * TICK_DT
            self._battery = max(0.0, self._battery - 0.02 * TICK_DT)

            x, y, yaw, v, w, batt = self._x, self._y, self._yaw, self._v, self._w, self._battery

        self._publish_odom(x, y, yaw, v, w)
        self._publish_battery(batt)

    def _controller(self) -> tuple[float, float]:
        """Turn-then-drive. Crude on purpose -- Nav2 is rung R2, not this node's job."""
        gx, gy = self._goal_xy
        dx, dy = gx - self._x, gy - self._y
        heading = math.atan2(dy, dx)
        err = math.atan2(math.sin(heading - self._yaw), math.cos(heading - self._yaw))
        if abs(err) > HEADING_TOLERANCE:
            return 0.0, math.copysign(TURN_SPEED, err)
        return self._args.cruise, 0.0

    def _rate(self, current: float, target: float) -> float:
        """Acceleration limit when speeding up, deceleration limit when slowing down.

        These used to be ONE number, which made --decel useless as a stop-time control:
        lowering it also capped the speed the robot could ever reach, so the stop stayed
        short and a decel sweep proved nothing about whether a reported stop was
        CONFIRMED or merely announced. A robot that is FAST and SLOW TO HALT is the only
        configuration that tells those two apart, and it needs two independent rates.

        An exact reversal (current +v, target -v) has |target| == |current| and takes the
        DECEL branch -- the conservative choice, since the robot must pass through zero.
        """
        return self._args.accel if abs(target) > abs(current) else self._args.decel

    @staticmethod
    def _approach(current: float, target: float, step: float) -> float:
        if abs(target - current) <= step:
            return target
        return current + math.copysign(step, target - current)

    # -- publishers -----------------------------------------------------------

    def _publish_odom(self, x, y, yaw, v, w) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation = _yaw_to_quaternion(yaw)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w
        self._odom_pub.publish(msg)

    def _publish_battery(self, percent: float) -> None:
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = float(percent) / 100.0   # sensor_msgs uses 0.0-1.0
        msg.voltage = 24.0 * (percent / 100.0)
        msg.present = True
        msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        self._batt_pub.publish(msg)

    def _publish_diagnostics(self) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, level in (("lidar", DiagnosticStatus.OK), ("camera", DiagnosticStatus.OK)):
            st = DiagnosticStatus()
            st.name = name
            st.level = level
            st.hardware_id = "fake-robot-1"
            st.message = "OK"
            st.values = [KeyValue(key="simulated", value="true")]
            msg.status.append(st)
        self._diag_pub.publish(msg)


def main(argv=None) -> int:
    # Refused here, before rclpy.init(): an argument this node does not implement used to
    # go to rclpy, which ignored it, and the run proceeded under a condition that never
    # applied. Only a --ros-args section reaches rclpy now.
    try:
        args, ros_args = parse(argv if argv is not None else sys.argv[1:])
    except ArgsError as exc:
        build_parser().print_usage(sys.stderr)
        print(f"fake_robot_node.py: REFUSED: {exc}", file=sys.stderr)
        return 2

    # `docker compose down` / --abort-on-container-exit send SIGTERM, which Python
    # does NOT turn into KeyboardInterrupt. Without this the process is killed mid
    # `executor.spin()` and rclpy raises
    #   RCLError: failed to initialize wait set: the given context is not valid
    # on the way out -- a non-zero exit for an otherwise clean shutdown.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    rclpy.init(args=ros_args)
    node = FakeRobot(args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:            # noqa: BLE001 - teardown races are not failures
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
