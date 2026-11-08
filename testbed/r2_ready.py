#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Nav2 READINESS gate -- "can this stack plan and see?", not "is the lifecycle active?".

WHY THIS EXISTS
----------------
`Makefile.r2 r2-wait` used to block until `ros2 lifecycle get /bt_navigator` reported
`active`, and every measurement script then slept a fixed number of seconds. Neither
signal means the stack can do anything.

  * `active` is a lifecycle transition. It fires when bt_navigator has configured and
    activated ITS OWN interfaces. It reports nothing about the planner, the costmaps, or
    the TF history the costmaps need before they will accept a scan -- and until they
    have it, both costmaps drop every scan with "the timestamp on the message is earlier
    than all the data in the transform cache" and bt_navigator times out on its OWN
    compute_path_to_pose server.

    Measured here 2026-09-02, one process, one clock (`--timings`): on three cold stacks
    the planner was answering BEFORE bt_navigator reported active, by 1-3 s. A separate
    run measured active at t+10 s against a stack that could not plan until past t+60 s. The
    two signals are independent and their ORDER IS NOT FIXED, which is the point: gating
    on `active` is not a conservative approximation of readiness, it is an unrelated
    event that can land on either side of it.

  * A fixed sleep is a guess about how long the rest takes on the machine it was tuned
    on. Host load alone was measured swinging the TF publish rate by 7.7x, so the guess
    is not stable even on ONE machine. A 20 s sleep that works is the
    same defect as a 5 s sleep that does not; it is only a luckier number.

THE PREDICATE
-------------
Two conditions, both read from Nav2's and tf2's own interfaces, both chosen because they
are the exact things observed failing:

  1. `compute_path_to_pose` accepts a goal and returns a NON-EMPTY path from the robot's
     current pose, twice in a row. One round trip exercises planner_server, the global
     costmap and the map -> base_link TF chain -- and it is precisely the server
     bt_navigator was seen timing out on. Twice, because a single plan can land in the
     gap between a costmap update and the next dropped scan.

  2. The newest `/scan` is TRANSFORMABLE into the local costmap's frame AT ITS OWN STAMP.
     This is the literal condition behind "the timestamp on the message is earlier than
     all the data in the transform cache": until the TF buffer holds history covering the
     scan, every observation is discarded, the costmaps stay blind, collision_monitor
     vetoes motion on a stale source, and goals abort with an empty error_msg.

REJECTED, and why it is worth recording
---------------------------------------
The obvious second condition -- "the local costmap has a non-zero cell" -- CANNOT be used
here. `always_send_full_costmap` is false (Nav2's default), so `/local_costmap/costmap`
carries the full grid only when the grid is (re)sized; thereafter changes go out on
`/local_costmap/costmap_updates` as deltas. A TRANSIENT_LOCAL subscriber therefore
latches the grid as it was at activation -- all zeroes -- and never sees another. Tried
on three cold stacks: one reported cells, two sat at zero for 60+ s while Nav2 was
demonstrably healthy and planning. Gating on it would have hung the harness on a working
stack. Recorded rather than silently dropped, because it looks correct.

NOT DONE HERE
-------------
This gate does not widen CRUISE_FLOOR, lengthen a sleep, or retry a measurement until it
passes. It only decides WHEN the measurement may begin. A run that then produces no
samples still produces no samples.

USE
---
As a gate, from the Makefile (exit 0 = ready, 1 = not ready before the deadline):

    python3 /ws/testbed/r2_ready.py --timeout 180

In-process, from a measurement script, INSTEAD of `time.sleep(settle)`:

    from r2_ready import wait_until_ready
    report = wait_until_ready(timeout=180.0)
    print(report.summary())

The probe node is created and destroyed BEFORE the measurement window opens, so it holds
no DDS participant while anything is being measured. Participant churn is the leading
untested hypothesis for the TF degradation that invalidates R2 runs, so an
instrument that churns participants mid-measurement would share a mechanism with the
thing it measures.

To measure the gate rather than use it -- what each signal costs on THIS host:

    python3 /ws/testbed/r2_ready.py --timings
"""

from __future__ import annotations

import argparse
import time

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

# An open-floor pose the planner can always reach. From make_map.py the racking occupies
# x in [-6,-2] and [2,6], y in [-4,4]; the corridor at x=0 is clear. This is the SAME
# goal r2_cancel_latency drives to, deliberately: the gate should prove the planner can
# solve the problem the measurement is about to pose it, not an easier one.
PROBE_GOAL = (0.0, 6.0)
PLANNER_ID = "GridBased"          # nav2_params.yaml -> planner_server.planner_plugins
# nav2_params.yaml -> local_costmap.global_frame. The frame the obstacle layer must
# transform every scan into before it can mark or clear anything.
COSTMAP_FRAME = "odom"
CONSECUTIVE_PLANS = 2
PLAN_TIMEOUT = 10.0               # per attempt; a slow plan is "not ready", not an error
RETRY_INTERVAL = 1.0


class NotReady(RuntimeError):
    """The stack did not become able to plan and see before the deadline."""


class ReadyReport:
    """What the gate observed, with timings. Printed beside every measurement, so a
    number is never quoted without the state of the stack that produced it."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.plan_s: float | None = None      # first non-empty path returned
        self.scan_tf_s: float | None = None   # first scan transformable at its own stamp
        self.ready_s: float | None = None     # both, and the plan repeated
        self.plan_attempts = 0
        self.plan_failures: list[str] = []
        self.path_poses = 0

    def summary(self) -> str:
        def fmt(v: float | None) -> str:
            return "----" if v is None else f"{v:.1f}s"
        return (
            f"nav2 ready after {fmt(self.ready_s)} "
            f"(first path {fmt(self.plan_s)} [{self.path_poses} poses], "
            f"scan transformable into {COSTMAP_FRAME} {fmt(self.scan_tf_s)}, "
            f"{self.plan_attempts} plan attempt(s), {len(self.plan_failures)} refused)"
        )


class _Probe(Node):
    """Its OWN executor, deliberately.

    Callers reach `wait_until_ready()` AFTER the binding's transport is up -- whether
    through `Nav2Binding.start()` directly (a measurement instrument) or through
    `adapter.bring_up_robot()` (a harness) -- and that spins rclpy's default
    global executor on a daemon thread. `rclpy.spin_once(node)`
    and `rclpy.spin_until_future_complete(node, fut)` both reach for that same global
    executor, so calling them here raised `RuntimeError: Executor is already spinning`
    and the gate died before the first measurement -- while working perfectly when run
    standalone from the Makefile, which is the sort of difference that gets shipped.
    A dedicated SingleThreadedExecutor keeps the probe's spinning entirely its own.
    """

    def __init__(self) -> None:
        super().__init__("r2_ready_probe")
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self)
        self._client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")

        # Sensor-data QoS, matching r2_sensors.py's publisher. A RELIABLE subscriber here
        # would never match a BEST_EFFORT publisher and the gate would wait forever on a
        # scan that is being published perfectly well.
        self._scan: LaserScan | None = None
        self.create_subscription(
            LaserScan, "/scan", self._on_scan,
            QoSProfile(
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
            ),
        )
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Diagnostic only (--timings). The GATE never reads this: it is the `active`
        # signal this file exists to replace, and reading it here is how that is shown.
        self._state_client = self.create_client(GetState, "/bt_navigator/get_state")

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    # -- condition 2 -------------------------------------------------------
    def scan_transformable(self) -> tuple[bool, str]:
        """Can the newest scan be put into the costmap frame at its own stamp?

        This is the costmap's own precondition, asked the same way the costmap asks it.
        A False here is exactly the state in which both costmaps discard every
        observation while reporting nothing but an INFO-level message filter warning.
        """
        scan = self._scan
        if scan is None:
            return False, "no /scan received yet"
        frame = scan.header.frame_id or "laser"
        stamp = Time.from_msg(scan.header.stamp)
        try:
            ok = self._tf_buffer.can_transform(COSTMAP_FRAME, frame, stamp)
        except tf2_ros.TransformException as exc:       # pragma: no cover - defensive
            return False, f"{frame} -> {COSTMAP_FRAME}: {exc}"
        if ok:
            return True, ""
        return False, (f"{frame} -> {COSTMAP_FRAME} not yet available at the scan's own "
                       f"stamp (TF history does not cover it)")

    # -- condition 1 -------------------------------------------------------
    def try_plan(self, goal_xy: tuple[float, float], timeout: float) -> tuple[int, str]:
        """Returns (pose count, reason). A pose count of 0 means NOT READY, never an
        error: the caller retries until its own deadline. That is what stops a Nav2
        round-trip inside a wait target from turning an unrelated hiccup into a hard
        failure -- the stated cost of gating on a real round-trip, paid off here."""
        if not self._client.wait_for_server(timeout_sec=1.0):
            return 0, "compute_path_to_pose server not discovered"

        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = "map"
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x = float(goal_xy[0])
        goal.goal.pose.position.y = float(goal_xy[1])
        goal.goal.pose.orientation.w = 1.0
        goal.planner_id = PLANNER_ID
        goal.use_start = False          # plan from where the robot actually is, via TF

        send = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, self._exec, timeout_sec=timeout)
        if not send.done():
            return 0, "planner did not respond to the goal"
        handle = send.result()
        if handle is None or not handle.accepted:
            return 0, "planner rejected the goal"

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, self._exec, timeout_sec=timeout)
        if not result_future.done():
            return 0, "planner accepted but never returned a result"
        wrapped = result_future.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            return 0, f"planner did not succeed (action status {wrapped.status})"
        result = wrapped.result
        code = getattr(result, "error_code", 0)
        if code:
            msg = getattr(result, "error_msg", "")
            return 0, f"planner error_code {code} {msg}".strip()
        n = len(result.path.poses)
        if n < 2:
            return 0, f"planner returned a path of {n} pose(s)"
        return n, ""

    def spin_once(self, timeout_sec: float) -> None:
        self._exec.spin_once(timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        self._exec.remove_node(self)
        self._exec.shutdown()
        self.destroy_node()

    # -- diagnostic --------------------------------------------------------
    def lifecycle_active(self, timeout: float = 1.0) -> bool:
        """What `r2-wait` used to block on, read on the SAME clock as everything else so
        the two can be compared without crossing a container boundary."""
        if not self._state_client.service_is_ready():
            return False
        future = self._state_client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, future, self._exec, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            return False
        return future.result().current_state.label == "active"


def _spin(probe: _Probe, seconds: float) -> None:
    """Let the scan and TF subscriptions deliver. A real wait on callbacks, not a sleep
    that happens to sit next to a node."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        probe.spin_once(0.1)


def wait_until_ready(timeout: float = 180.0, *, verbose: bool = True) -> ReadyReport:
    """Block until Nav2 can plan and its costmaps can consume a scan.

    Raises NotReady if the deadline passes. Safe to call after rclpy is already
    initialised -- the measurement scripts start their binding first.
    """
    owned_context = not rclpy.ok()
    if owned_context:
        rclpy.init()
    probe = _Probe()
    report = ReadyReport()
    deadline = report.t0 + timeout
    consecutive = 0
    last_reason = "no attempt made"
    try:
        # One spin first: the scan and TF listeners need a moment to match before the
        # first plan attempt, or the first iteration reports a failure that is only
        # discovery latency.
        _spin(probe, RETRY_INTERVAL)
        while time.monotonic() < deadline:
            scan_ok, scan_reason = probe.scan_transformable()
            if scan_ok and report.scan_tf_s is None:
                report.scan_tf_s = time.monotonic() - report.t0

            report.plan_attempts += 1
            poses, plan_reason = probe.try_plan(PROBE_GOAL, PLAN_TIMEOUT)
            elapsed = time.monotonic() - report.t0
            if poses:
                if report.plan_s is None:
                    report.plan_s = elapsed
                    report.path_poses = poses
                consecutive += 1
            else:
                consecutive = 0
                last_reason = plan_reason
                report.plan_failures.append(f"t+{elapsed:.1f}s {plan_reason}")
                if verbose:
                    print(f"  not ready at t+{elapsed:.1f}s: {plan_reason}")

            scan_ok, scan_reason = probe.scan_transformable()
            if scan_ok and report.scan_tf_s is None:
                report.scan_tf_s = time.monotonic() - report.t0

            if consecutive >= CONSECUTIVE_PLANS and scan_ok:
                report.ready_s = time.monotonic() - report.t0
                if verbose:
                    print(f"  {report.summary()}")
                return report
            if consecutive >= CONSECUTIVE_PLANS and not scan_ok:
                last_reason = scan_reason
                if verbose:
                    print(f"  not ready at t+{elapsed:.1f}s: planner OK, but {scan_reason}")

            _spin(probe, RETRY_INTERVAL)

        raise NotReady(
            f"Nav2 was not ready within {timeout:.0f}s -- last reason: {last_reason}. "
            f"{report.summary()}")
    finally:
        probe.shutdown()
        if owned_context:
            rclpy.shutdown()


def _timings(timeout: float) -> int:
    """Measure the gap this gate exists to close: lifecycle-active (what r2-wait used to
    return on) against can-actually-plan-and-see. Prints, it does not gate.

    Both signals are read from ONE process on ONE clock. A host-side `ros2 lifecycle get`
    loop compared against a container-side probe would be comparing two different
    CLOCK_MONOTONICs across the Docker Desktop VM boundary, and the SIGN of the gap is
    the whole finding.
    """
    rclpy.init()
    probe = _Probe()
    report = ReadyReport()
    active_s: float | None = None
    ready_s: float | None = None
    try:
        print(f"probing from t+0 (deadline {timeout:.0f}s) ...")
        deadline = report.t0 + timeout
        while time.monotonic() < deadline:
            if active_s is None and probe.lifecycle_active():
                active_s = time.monotonic() - report.t0
                print(f"  t+{active_s:6.1f}s  bt_navigator LIFECYCLE ACTIVE"
                      f"   <- what r2-wait used to return on")

            scan_ok, scan_reason = probe.scan_transformable()
            if scan_ok and report.scan_tf_s is None:
                report.scan_tf_s = time.monotonic() - report.t0
                print(f"  t+{report.scan_tf_s:6.1f}s  /scan transformable into "
                      f"{COSTMAP_FRAME} at its own stamp")

            poses, plan_reason = probe.try_plan(PROBE_GOAL, PLAN_TIMEOUT)
            elapsed = time.monotonic() - report.t0
            if poses and report.plan_s is None:
                report.plan_s = elapsed
                report.path_poses = poses
                print(f"  t+{elapsed:6.1f}s  planner returns a NON-EMPTY path "
                      f"({poses} poses)")
            elif not poses:
                print(f"  t+{elapsed:6.1f}s  no plan: {plan_reason}")

            if ready_s is None and report.plan_s is not None and report.scan_tf_s is not None:
                ready_s = elapsed
                print(f"  t+{ready_s:6.1f}s  READY signals up (planner + scan into "
                      f"costmap frame). The GATE also requires the plan to repeat.")

            if ready_s is not None and active_s is not None:
                print(f"\nlifecycle active t+{active_s:.1f}s | ready t+{ready_s:.1f}s | "
                      f"ready - active = {ready_s - active_s:+.1f}s")
                return 0

            _spin(probe, RETRY_INTERVAL)

        def _f(v):
            return "NEVER" if v is None else f"t+{v:.1f}s"
        print(f"deadline passed. lifecycle active {_f(active_s)} | "
              f"planner {_f(report.plan_s)} | scan-into-{COSTMAP_FRAME} "
              f"{_f(report.scan_tf_s)}")
        return 1
    finally:
        probe.shutdown()
        rclpy.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="deadline for readiness, seconds (default 180)")
    ap.add_argument("--timings", action="store_true",
                    help="report when each signal became true instead of gating")
    args = ap.parse_args()

    if args.timings:
        return _timings(args.timeout)
    try:
        report = wait_until_ready(args.timeout)
    except NotReady as exc:
        print(f"NOT READY: {exc}")
        return 1
    print(f"READY: {report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
