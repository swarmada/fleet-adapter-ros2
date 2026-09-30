# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Nav2 / ROS 2 robot binding — the ROS 2 CLASS seam (no vendor-specific branches).

This binding wraps ``rclpy`` + the Nav2 stack behind the ``RobotBinding`` contract
(see robot.py). The telemetry side (sub-item 2) is implemented as a cache of the
latest odom/battery/diagnostics messages, mapped by ``telemetry.py``. The ROS
node + subscriptions are created in :meth:`start` (which requires a ROS 2 runtime);
the cache + mappings are exercised without ROS by feeding the ``on_*`` callbacks.

Task execution (sub-item 3): ``command_move`` translates a target into a
``nav2_msgs/action/NavigateToPose`` goal (via ``task.GoalPose``) and caches it;
the live action-client dispatch happens once :meth:`start` has a ROS 2 node.

Safety (sub-item 4): ``command_stop`` cancels the active Nav2 goal and holds
(shared by confirmed estop C5 and lease-expiry self-stop C4); ``is_stopped`` is
CONFIRMED from the odom twist being ~zero (never inferred from the command, a
timer, or silence — C5.3). The live goal dispatch/cancel + zero-``cmd_vel`` publish
are the only pieces that need a ROS 2 runtime; they are guarded behind a live node.
"""

from __future__ import annotations

import threading

from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2 import telemetry

# A robot is CONFIRMED at rest when its measured odom twist is within sensor noise.
# Ground truth only — never a timer or the absence of a message (C5.3).
_EPS_LINEAR = 0.01   # m/s
_EPS_ANGULAR = 0.01  # rad/s


class Nav2Binding:
    """ROS 2 / Nav2 implementation of ``RobotBinding`` (see robot.py).

    Constructible without ROS (for telemetry mapping/tests). Call :meth:`start` to
    create the rclpy node and subscribe to the topics — that requires a ROS 2 +
    Nav2 environment. Select it on the CLI with ``--binding ros2``.
    """

    def __init__(self, robot_id: str = "robot-conformance-1", floor: int = 0) -> None:
        self._robot_id = robot_id
        self._floor = floor
        self._lock = threading.Lock()
        self._position: telemetry.Position | None = None
        self._speed: tuple[float, float] | None = None  # (linear, angular); None until first odom
        self._battery = telemetry.Battery()
        self._hardware: list[telemetry.Hardware] = []
        self._goal: task_map.GoalPose | None = None  # pending/active NavigateToPose target
        self._goal_handle = None  # rclpy action goal handle, set once start() sends it
        self._goal_status = 0  # action_msgs/GoalStatus of the active goal; 0 = in-flight
        self._nav_initial: float | None = None  # first distance_remaining (100% baseline)
        self._nav_distance: float | None = None  # latest distance_remaining from feedback
        self._holding = False  # a stop has been commanded; hold until a new goal
        self._node = None  # rclpy node, created in start()
        self._action_client = None   # NavigateToPose ActionClient (start())
        self._cmd_vel = None         # geometry_msgs/Twist publisher (halt on cancel)
        self._nav_action = None      # cached nav2_msgs/action/NavigateToPose type
        self._twist_type = None      # cached geometry_msgs/Twist type
        self._spin_thread = None     # rclpy executor thread
        # Increments on every dispatch. A result callback carries the epoch it was
        # registered under, so a SUPERSEDED goal's result can be told apart from the
        # current goal's -- see on_goal_result.
        self._goal_epoch = 0

    # ── ROS 2 subscription callbacks (also the test entry points) ─────────────

    def on_odom(self, odom) -> None:
        pos = telemetry.position_from_odom(odom, floor=self._floor)
        speed = telemetry.speed_from_odom(odom)  # ground truth for is_stopped (C5.3)
        with self._lock:
            self._position = pos
            self._speed = speed

    def on_battery(self, state) -> None:
        bat = telemetry.battery_from_state(state)
        with self._lock:
            self._battery = bat

    def on_diagnostics(self, diag) -> None:
        hw = telemetry.hardware_from_diagnostics(diag)
        with self._lock:
            self._hardware = hw

    # ── RobotBinding: telemetry accessors ─────────────────────────────────────

    def spawn(self, robot_id: str) -> None:
        self._robot_id = robot_id

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        with self._lock:
            p = self._position
        return (p.x, p.y, p.floor) if p is not None else (0.0, 0.0, self._floor)

    def yaw(self, robot_id: str) -> float:
        with self._lock:
            return self._position.yaw if self._position is not None else 0.0

    def battery_percent(self, robot_id: str) -> int:
        with self._lock:
            return self._battery.percent if self._battery.percent is not None else 0

    def latest_battery(self, robot_id: str) -> telemetry.Battery:
        with self._lock:
            return self._battery

    def latest_hardware(self, robot_id: str) -> list[telemetry.Hardware]:
        with self._lock:
            return list(self._hardware)

    # ── RobotBinding: motion ──────────────────────────────────────────────────

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Record the NavigateToPose target. When a ROS 2 node is live (start()),
        the target is dispatched to the Nav2 action server; without one, it is only
        cached so the translation is unit-testable at $0."""
        with self._lock:
            self._goal = task_map.GoalPose(x=x, y=y, yaw=yaw)
            self._holding = False  # a new goal releases a prior hold
            self._goal_status = 0  # new goal in-flight → RUNNING
            self._nav_initial = None  # progress baseline reset for the new goal
            self._nav_distance = None
            self._goal_epoch += 1
            epoch = self._goal_epoch
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._send_nav2_goal(self._goal, epoch)

    def on_feedback(self, distance_remaining: float) -> None:
        """A NavigateToPose action FeedbackMessage: the remaining distance to the
        goal. The first (largest) value is the 100% baseline; later values give
        progress. Ignores a non-positive baseline (no usable distance)."""
        with self._lock:
            if self._nav_initial is None and distance_remaining > 0:
                self._nav_initial = distance_remaining
            self._nav_distance = distance_remaining

    def task_progress(self, robot_id: str) -> int:
        """Advisory task progress (0-100) from the NavigateToPose feedback distance:
        ``round((1 - remaining/initial) * 100)``, clamped. 0 until feedback arrives."""
        with self._lock:
            initial, dist = self._nav_initial, self._nav_distance
        if initial is None or initial <= 0 or dist is None:
            return 0
        pct = int(round((1.0 - max(0.0, dist) / initial) * 100))
        return max(0, min(100, pct))

    def pending_goal(self, robot_id: str) -> task_map.GoalPose | None:
        """The last NavigateToPose target handed to :meth:`command_move`, or None
        after a stop cleared it (test seam)."""
        with self._lock:
            return self._goal

    def command_stop(self, robot_id: str) -> None:
        """Cancel the active Nav2 goal and hold. This is a REQUEST — confirmation
        that the robot is at rest comes only from :meth:`is_stopped` (odom). Shared
        by confirmed estop (C5) and lease-expiry self-stop (C4)."""
        with self._lock:
            self._goal = None
            self._holding = True
            handle, self._goal_handle = self._goal_handle, None
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)

    def command_pause(self, robot_id: str) -> None:
        """Pause navigation: cancel the active NavigateToPose goal so the base halts,
        but KEEP the target so :meth:`command_resume` can re-dispatch it (unlike
        command_stop, which drops the target). Confirmation the base is at rest is
        odom (:meth:`is_stopped`), never inferred."""
        with self._lock:
            self._holding = True
            handle, self._goal_handle = self._goal_handle, None
            # self._goal (target) intentionally KEPT for resume.
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)

    def command_resume(self, robot_id: str) -> None:
        """Resume a paused navigation: re-dispatch the retained NavigateToPose target."""
        with self._lock:
            goal = self._goal
            self._holding = False
            self._goal_status = 0  # re-dispatched goal is in-flight again
            # The goal cancelled by command_pause will deliver a CANCELED result; it must
            # not be attributed to the goal being resumed here.
            self._goal_epoch += 1
            epoch = self._goal_epoch
        if goal is not None and self._node is not None:  # pragma: no cover — ROS 2 runtime
            self._send_nav2_goal(goal, epoch)

    def on_goal_result(self, status: int, *, epoch: int | None = None) -> None:
        """The NavigateToPose action's terminal ``action_msgs/GoalStatus`` (SUCCEEDED /
        ABORTED / CANCELED), delivered by the action client's result callback.

        ``epoch`` identifies the DISPATCH this result belongs to, and a result carrying a
        stale one is DROPPED. Without it, a superseded goal's result overwrites the status
        of the goal that replaced it, and the adapter then reports that terminal state
        against whatever action is currently assigned — observed as ``SUCCEEDED`` at
        ``progress=100`` for an action assigned 192 ms earlier whose target was 6 m away.
        On the wire that is indistinguishable from real completion, so the control plane
        marks the action done and is free to dispatch the next one while the robot is still
        driving toward the abandoned goal.

        ``epoch=None`` means "unidentified" and is accepted as-is, which keeps this usable
        as a direct test seam.
        """
        with self._lock:
            if epoch is not None and epoch != self._goal_epoch:
                return
            self._goal_status = status

    def task_state(self, robot_id: str) -> str:
        """Protocol ``TaskState`` for the active goal, derived from its Nav2 action
        ``GoalStatus`` — the task-status stream source (C6.4)."""
        with self._lock:
            status = self._goal_status
        return task_map.task_state_from_goal_status(status)

    def _send_nav2_goal(self, goal: task_map.GoalPose, epoch: int) -> None:  # pragma: no cover — ROS 2 runtime
        # Build a NavigateToPose goal (PoseStamped, yaw→quaternion) and dispatch it
        # via the action client; feedback → on_feedback (progress), result → on_goal_result.
        import math

        from geometry_msgs.msg import PoseStamped

        nav_goal = self._nav_action.Goal()
        pose = PoseStamped()
        pose.header.frame_id = getattr(goal, "frame", "") or "map"
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(goal.x)
        pose.pose.position.y = float(goal.y)
        half = float(goal.yaw) / 2.0  # yaw about +Z → quaternion (z, w)
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)
        nav_goal.pose = pose

        self._action_client.wait_for_server()
        send_future = self._action_client.send_goal_async(
            nav_goal, feedback_callback=self._on_nav_feedback)
        send_future.add_done_callback(lambda f: self._on_goal_response(f, epoch))

    def _on_goal_response(self, future, epoch: int) -> None:  # pragma: no cover — ROS 2 runtime
        handle = future.result()
        if not handle.accepted:
            self.on_goal_result(6, epoch=epoch)  # ABORTED — the Nav2 server rejected it
            return
        with self._lock:
            if epoch != self._goal_epoch:
                # Superseded before the server answered. Do NOT adopt this handle: a later
                # command_stop() would then cancel the WRONG goal, leaving the current one
                # running. Its result is dropped too — it says nothing about the goal now
                # in flight.
                return
            self._goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self.on_goal_result(f.result().status, epoch=epoch))

    def _on_nav_feedback(self, feedback_msg) -> None:  # pragma: no cover — ROS 2 runtime
        # NavigateToPose feedback carries distance_remaining → progress baseline.
        self.on_feedback(float(feedback_msg.feedback.distance_remaining))

    def _cancel_nav2_goal(self, handle) -> None:  # pragma: no cover — ROS 2 runtime
        # Cancel the NavigateToPose goal and publish a zero Twist to cmd_vel so the
        # base halts even before Nav2 tears the controller down.
        if handle is not None:
            handle.cancel_goal_async()
        if self._cmd_vel is not None and self._twist_type is not None:
            self._cmd_vel.publish(self._twist_type())

    def tick(self, dt: float) -> None:
        pass  # rclpy spins its own executor; nothing to advance here.

    def is_stopped(self, robot_id: str) -> bool:
        """CONFIRMED at rest: the measured odom twist is within sensor noise. Returns
        False until odom has been observed — rest is never inferred from the stop
        command, a timer, or the absence of a message (C5.3)."""
        with self._lock:
            speed = self._speed
        if speed is None:
            return False  # no odom yet → no evidence of rest
        linear, angular = speed
        return linear <= _EPS_LINEAR and angular <= _EPS_ANGULAR

    # ── ROS runtime ───────────────────────────────────────────────────────────

    def start(self) -> None:  # pragma: no cover — requires the ROS 2 runtime
        """Create the rclpy node, subscribe to odom/battery/diagnostics, open the
        NavigateToPose action client + cmd_vel publisher, and spin rclpy on a
        background thread. Requires a ROS 2 + Nav2 environment.

        NOTE: written to the rclpy / nav2_msgs API; validate against a live ROS 2 +
        Nav2 stack (it cannot be exercised without one). The `$0` path never reaches
        here — every dispatch is guarded by `self._node is not None`."""
        try:
            import atexit
            import sys
            import threading

            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from rclpy.executors import ExternalShutdownException
            from rclpy.node import Node
            from sensor_msgs.msg import BatteryState
        except ImportError as exc:
            raise RuntimeError(
                "Nav2Binding.start() requires ROS 2 (rclpy) and the Nav2 stack. "
                "Use `--binding simulated` for a $0 run, or launch inside a sourced "
                "ROS 2 workspace."
            ) from exc

        if not rclpy.ok():
            rclpy.init()
        node = Node(f"swarmada_nav2_{self._robot_id}".replace("-", "_"))
        # Route each subscription to the existing (unit-tested) parse callbacks.
        node.create_subscription(Odometry, "odom", lambda m: self.on_odom(m), 10)
        node.create_subscription(BatteryState, "battery_state", lambda m: self.on_battery(m), 10)
        node.create_subscription(DiagnosticArray, "diagnostics", lambda m: self.on_diagnostics(m), 10)

        self._cmd_vel = node.create_publisher(Twist, "cmd_vel", 10)
        self._action_client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self._nav_action = NavigateToPose
        self._twist_type = Twist
        self._node = node

        def _spin() -> None:
            try:
                rclpy.spin(node)
            except ExternalShutdownException:
                pass                     # stop() shut the context down: the clean path
            except Exception as exc:     # noqa: BLE001 - a boundary, not a handler
                # NOT a fault-reporting design: this only stops a
                # dying spin thread from printing a traceback; the adapter still does not
                # notice it has gone blind. That remains open.
                print(f"rclpy spin thread exited: {exc!r}", file=sys.stderr)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()
        # Every caller -- the adapter's main(), r1/r2 smoke, the latency script -- creates a
        # binding and never tears it down. Registering here fixes all of them at once with
        # no Protocol change; stop() is idempotent so an explicit call remains fine.
        atexit.register(self.stop)

    def stop(self) -> None:  # pragma: no cover — requires the ROS 2 runtime
        """Shut the executor down and join the spin thread BEFORE the interpreter tears
        the C++ side out from under it.

        Without this, ``rclpy.spin`` runs on a daemon thread that the interpreter kills at
        exit while it is inside a C++ frame. The C++ runtime then unwinds a thread with no
        active exception, calls ``std::terminate``, and the process dies on SIGABRT --
        **exit 134 on a run whose work completed successfully**. Every smoke test, the
        latency script and both conformance containers ended that way.

        Idempotent, and safe to call when :meth:`start` was never called.

        Deliberately a method on the CONCRETE binding and NOT added to the ``RobotBinding``
        Protocol: whether the Protocol should declare a lifecycle is an open
        question, and this does not answer it.
        """
        node, thread = self._node, self._spin_thread
        self._node, self._spin_thread = None, None
        if node is None:
            return
        import rclpy
        try:
            if rclpy.ok():
                rclpy.shutdown()   # makes rclpy.spin() return in the spin thread
        except Exception:          # noqa: BLE001 - teardown must not raise
            pass
        if thread is not None:
            thread.join(timeout=5.0)
        try:
            node.destroy_node()
        except Exception:          # noqa: BLE001 - teardown must not raise
            pass
