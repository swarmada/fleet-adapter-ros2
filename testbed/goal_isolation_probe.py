#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Does the fake robot end a goal only when it was asked to?

An instrument pointed at ``fake_robot_node.py``, not at the adapter. The live conformance
harness depends on the plant ending a goal only when it is cancelled, when it arrives, or when a
newer goal preempts it, as Nav2 does. When two goals overlapped, the fake robot used to end the
one nobody touched: a goal 6 m away was reported SUCCEEDED or ABORTED about 60 ms after it
started. The adapter faithfully reported what the server said, and the harness's C4.2 failed on
the plant's say-so.

Three cases, each from a fresh pair of goals. Every goal is placed 6 m from where odometry says
the robot is at that moment, on the side nearer the origin, so a goal really is far away however
far earlier cases moved the robot:

1. **cancel, then a new goal.** Cancel A; without waiting, send B. That is a stop followed by a
   new assignment. A must end CANCELED, and B must still be executing 2 s later.
2. **a new goal, then cancel the old one.** Send A, then B, then cancel A. A must end CANCELED or
   ABORTED (preempted), and B must still be executing 2 s later.
3. **preemption.** Send A, then B. A must end ABORTED (preempted), and B must still be executing
   2 s later.

B is cancelled at the end of each case. It must then end CANCELED, and nothing else.

Run inside the testbed, against the ACTIVE fake robot:

    docker compose up -d fake-robot
    docker compose run --rm --no-deps fake-robot bash -lc \\
      'source /opt/ros/jazzy/setup.bash && python3 /ws/testbed/goal_isolation_probe.py'

Exit 0 when every case holds; 1 otherwise, with the offending case printed.
"""

from __future__ import annotations

import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node

_NAMES = {GoalStatus.STATUS_ACCEPTED: "ACCEPTED", GoalStatus.STATUS_EXECUTING: "EXECUTING",
          GoalStatus.STATUS_CANCELING: "CANCELING", GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
          GoalStatus.STATUS_CANCELED: "CANCELED", GoalStatus.STATUS_ABORTED: "ABORTED"}
_STILL_RUNNING_S = 2.0


class _Probe:
    def __init__(self) -> None:
        self.node = Node("goal_isolation_probe")
        self.client = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        if not self.client.wait_for_server(timeout_sec=10.0):
            raise SystemExit("no navigate_to_pose server within 10 s")
        self.pose = None
        self.node.create_subscription(Odometry, "odom", self._on_odom, 10)

    def _on_odom(self, msg) -> None:
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def send(self):
        """A goal 6 m from where the robot is now, along y, on the side nearer the origin."""
        self.pose = None
        deadline = time.monotonic() + 5.0
        while self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.pose is None:
            raise SystemExit("no odometry within 5 s")
        x, y = self.pose
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = "map"
        g.pose.pose.position.x = x
        g.pose.pose.position.y = y + (-6.0 if y > 0 else 6.0)
        g.pose.pose.orientation.w = 1.0
        f = self.client.send_goal_async(g)
        rclpy.spin_until_future_complete(self.node, f, timeout_sec=5.0)
        handle = f.result()
        if handle is None or not handle.accepted:
            raise SystemExit("goal not accepted")
        return handle, handle.get_result_async()

    def cancel(self, handle) -> None:
        f = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self.node, f, timeout_sec=5.0)

    def outcome(self, result_future, within: float) -> str | None:
        """The goal's terminal status within `within` s, or None while it is still running."""
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=within)
        if not result_future.done():
            return None
        return _NAMES.get(result_future.result().status, str(result_future.result().status))


def _case(p: _Probe, name: str, steps, a_may_end: set[str]) -> list[str]:
    a, ra, b, rb = steps(p)
    faults = []
    a_end = p.outcome(ra, 3.0)
    if a_end not in a_may_end:
        faults.append(f"A ended {a_end}, expected one of {sorted(a_may_end)}")
    b_end = p.outcome(rb, _STILL_RUNNING_S)
    if b_end is not None:
        faults.append(f"B, 6 m away and never cancelled, ended {b_end} within {_STILL_RUNNING_S:g} s")
    else:
        p.cancel(b)
        b_end = p.outcome(rb, 3.0)
        if b_end != "CANCELED":
            faults.append(f"B, cancelled at the end, ended {b_end}")
    print(f"{'PASS' if not faults else 'FAIL'}  {name}: A {a_end}, B {b_end}"
          + "".join(f"\n      {f}" for f in faults), flush=True)
    time.sleep(1.0)                              # the robot comes to rest between cases
    return faults


def _cancel_then_new(p: _Probe):
    a, ra = p.send()
    time.sleep(0.2)
    a.cancel_goal_async()                        # cancel A, and without waiting ...
    b, rb = p.send()                             # ... send B
    return a, ra, b, rb


def _new_then_cancel_old(p: _Probe):
    a, ra = p.send()
    b, rb = p.send()
    p.cancel(a)
    return a, ra, b, rb


def _preempt(p: _Probe):
    a, ra = p.send()
    time.sleep(0.2)
    b, rb = p.send()
    return a, ra, b, rb


def main() -> int:
    rclpy.init()
    p = _Probe()
    faults: list[str] = []
    try:
        faults += _case(p, "cancel A, then send B", _cancel_then_new, {"CANCELED"})
        faults += _case(p, "send A and B, then cancel A", _new_then_cancel_old,
                        {"CANCELED", "ABORTED"})
        faults += _case(p, "send A, then B (preemption)", _preempt, {"ABORTED"})
    finally:
        p.node.destroy_node()
        rclpy.shutdown()
    print("RESULT:", "every goal ended only when asked" if not faults
          else f"{len(faults)} fault(s): the plant ended a goal nobody asked it to end")
    return 0 if not faults else 1


if __name__ == "__main__":
    sys.exit(main())
