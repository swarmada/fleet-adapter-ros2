#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Evidence that the adapter's own bring-up produces a real ROS 2 graph.

OBSERVATION ONLY. Asserts nothing about navigation and scores no fault case; its whole
job is to answer one question with a node graph rather than an argument:

    does `--binding ros2` now have an rclpy node, three LIVE subscriptions and a
    CONNECTED action client, created by the ADAPTER rather than by a harness?

It does NOT call ``Nav2Binding.start()``. That is the entire discipline of this file. The
defect being evidenced is that nothing in the shipped package ever called it, and a probe
that starts the binding itself would reproduce the concealment it exists to end -- that
is precisely what ``r1_smoke.py`` did, which is why R1 was green for months while
``--binding ros2`` talked to nothing. Bring-up goes through
``adapter.bring_up_robot()``, the same call ``run_control_stream()`` makes on every run.

After bring-up it HOLDS, so the graph can be captured from outside the process while the
adapter's node is live:

    ros2 node list
    ros2 node info /swarmada_nav2_robot_conformance_1
    ros2 topic list
    ros2 action list

A graph read from inside this process could be produced by any node; read from a separate
process against the same DDS domain, it is the graph as the rest of ROS 2 sees it.

Run inside the testbed container, with fake_robot_node.py already running:

    python3 testbed/bringup_evidence.py --hold 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter
    from fleet_adapter_ros2.nav2_binding import Nav2Binding
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: cannot import the adapter - is it on PYTHONPATH? ({exc})")
    raise SystemExit(2) from exc

ROBOT = "robot-conformance-1"

#: The three subscriptions `Nav2Binding.start()` is documented to create, and the
#: accessor each one feeds. A subscription that exists but feeds nothing is not evidence.
EXPECTED_SUBS = [
    ("/odom", "nav_msgs/msg/Odometry", "pose() / is_stopped()"),
    ("/battery_state", "sensor_msgs/msg/BatteryState", "battery_percent()"),
    ("/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", "latest_hardware()"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold", type=float, default=60.0,
                    help="seconds to hold the node up for external capture (default 60)")
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds for DDS discovery before reading the graph (default 5)")
    args = ap.parse_args()

    print("adapter bring-up evidence")
    print(f"  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')}  "
          f"RMW={os.environ.get('RMW_IMPLEMENTATION')}")
    print()

    binding = Nav2Binding()
    print("1. BEFORE bring-up - the state adapter.py:544 left the binding in")
    print(f"     _node            = {binding._node}")
    print(f"     _action_client   = {binding._action_client}")
    print(f"     _cmd_vel         = {binding._cmd_vel}")
    print(f"     transport_alive()= {binding.transport_alive()}")
    if binding._node is not None:
        print("   UNEXPECTED: a freshly constructed binding already has a node")
        return 2

    print("\n2. BRING-UP THROUGH THE ADAPTER - no binding.start() in this file")
    adapter = fleet_adapter_ros2Adapter(None, None, binding, ROBOT)
    try:
        adapter.bring_up_robot()
    except Exception as exc:
        print(f"   FAILED: adapter.bring_up_robot() raised {exc!r}")
        return 1
    print("     adapter.bring_up_robot() returned")

    print("\n3. AFTER bring-up")
    node = binding._node
    if node is None:
        print("   FAILED: the adapter returned from bring-up without creating a node.")
        print("   That is the defect exactly: a quiet return and an inert binding.")
        return 1
    print(f"     rclpy node       = /{node.get_name()}")
    print(f"     transport_alive()= {binding.transport_alive()}")
    print(f"     _action_client   = {type(binding._action_client).__name__}")
    print(f"     _cmd_vel         = {type(binding._cmd_vel).__name__} -> /cmd_vel")

    print(f"\n   waiting {args.settle:.0f}s for DDS discovery ...")
    time.sleep(args.settle)

    print("\n4. THE THREE SUBSCRIPTIONS - matched against live PUBLISHER counts")
    # get_publishers_info_by_topic is the graph's answer, not this node's bookkeeping:
    # it reports who is publishing the topic this node subscribed to. A subscription with
    # no publisher is a subscription to silence, which is what an unstarted binding is
    # indistinguishable from.
    ok = True
    for topic, msg_type, feeds in EXPECTED_SUBS:
        subs = node.get_subscriptions_info_by_topic(topic)
        mine = [s for s in subs if s.node_name == node.get_name()]
        pubs = node.get_publishers_info_by_topic(topic)
        state = "SUBSCRIBED" if mine else "MISSING"
        if not mine:
            ok = False
        print(f"     {topic:16} {state:11} publishers={len(pubs)}  "
              f"type={msg_type}  feeds {feeds}")

    print("\n5. THE ACTION CLIENT - asked the graph, not the constructor")
    # wait_for_server returns True only when a NavigateToPose action server is discovered.
    # `_action_client is not None` would be true on a binding whose server never appeared.
    connected = binding._action_client.wait_for_server(timeout_sec=10.0)
    print(f"     /navigate_to_pose server discovered = {connected}")
    if not connected:
        ok = False
        print("     (no NavigateToPose server on this domain - is fake_robot_node.py up?)")

    print("\n6. tick() ON A LIVE TRANSPORT IS SILENT")
    # The detection half of it: tick() now raises on a transport that was never
    # started, not only on one that started and died. The risk a live check covers, which
    # a unit test with a marker node cannot, is the opposite error -- a detector that
    # raises on a HEALTHY transport would take the telemetry thread down on every run.
    ticks, raised = 300, 0
    for _ in range(ticks):
        try:
            binding.tick(0.2)
        except RuntimeError:
            raised += 1
    print(f"     {ticks} ticks at telemetry cadence, raises = {raised}  (must be 0)")
    if raised:
        ok = False
        print("     FAILED: tick() raised on a LIVE transport")

    print("\n7. LIVE TELEMETRY THROUGH THE ADAPTER'S BINDING")
    print(f"     pose()            = {binding.pose(ROBOT)}")
    print(f"     battery_percent() = {binding.battery_percent(ROBOT)}")
    print(f"     is_stopped()      = {binding.is_stopped(ROBOT)}")

    print(f"\nVERDICT: {'GRAPH ESTABLISHED' if ok else 'INCOMPLETE - see MISSING above'}")
    print(f"\nholding {args.hold:.0f}s - capture the graph now from another container:")
    print(f"  ros2 node info /{node.get_name()}")
    print("  ros2 topic list && ros2 action list")
    sys.stdout.flush()
    time.sleep(args.hold)

    binding.stop()   # the teardown half of the lifecycle the Protocol now declares
    print("\nbinding.stop() returned; transport_alive() = "
          f"{binding.transport_alive()}")
    # A clean shutdown must NOT be reported as a broken transport, or every orderly exit
    # would raise out of whichever tick landed next -- and the teardown path does tick on
    # its way out.
    try:
        binding.tick(0.2)
        print("tick() after stop() is silent, as a clean shutdown must be")
    except RuntimeError as exc:
        ok = False
        print(f"tick() RAISED after a deliberate stop() - wrong: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
