#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Is the racking LETHAL in the global costmap, or merely UNKNOWN?

This is the question r2_smoke's "unreachable goal" case depends on. If the rack cells are
UNKNOWN rather than lethal, the lidar has never swept that block from anywhere the robot
has been, Nav2 believes those cells are free, and R2's obstacle-avoidance claim is
unproven rather than merely noisy.

WHY THIS FILE EXISTS SEPARATELY FROM costmap_probe.py

costmap_probe.py subscribes with the DEFAULT QoS:

    node.create_subscription(Costmap, TOPIC, cb, 1)          # <- VOLATILE

Nav2 publishes /global_costmap/costmap_raw as RELIABLE + TRANSIENT_LOCAL and sends the
full grid once, at activation. A VOLATILE subscriber is QoS-COMPATIBLE with a
TRANSIENT_LOCAL publisher -- so nothing errors, the topic shows a publisher, and
`ros2 topic info` looks perfect -- but it does NOT receive samples published before it
subscribed. Any late subscriber therefore waits forever on a topic that is working
correctly. That is why the probe reported "NO COSTMAP within 25s" while the data was
sitting there, retrievable with:

    ros2 topic echo /global_costmap/costmap_raw --qos-durability transient_local --once

costmap_probe.py's own docstring names that workaround; its subscription does not use it.
The fix there is one line (durability=TRANSIENT_LOCAL); this file exists so the question
could be answered without editing it.

Run:
    python3 /ws/testbed/costmap_probe_tl.py
"""

from __future__ import annotations

import sys
import time

import rclpy
from nav2_msgs.msg import Costmap
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from tf2_msgs.msg import TFMessage

TOPIC = "/global_costmap/costmap_raw"
# From make_map.py: racking occupies roughly x in [-6,-2] and [2,6], y in [-4,4].
CELLS = [
    # A transect along +x from the robot's start pose, through the right-hand rack.
    (1.5, 0.0),   # open floor before the rack
    (2.0, 0.0),   # the rack's FRONT FACE -- the first cell a ray from the origin can hit
    (2.5, 0.0),   # just inside
    (3.0, 0.0), (4.0, 0.0), (5.0, 0.0),   # the rack INTERIOR (r2_smoke's "unreachable" goal)
    (6.0, 0.0),   # far face
    (0.0, 0.0), (0.0, -6.0),              # poses the robot actually occupies
]
LETHAL, INSCRIBED, NO_INFO = 254, 253, 255


def main() -> int:
    rclpy.init()
    node = Node("costmap_probe_tl")
    state: dict = {}

    node.create_subscription(TFMessage, "/tf",
                             lambda _m: state.__setitem__("tf", state.get("tf", 0) + 1), 50)

    # MUST match the publisher: RELIABLE + TRANSIENT_LOCAL, or the latched full grid is
    # never delivered to a subscriber that joined after activation.
    qos = QoSProfile(depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                     history=QoSHistoryPolicy.KEEP_LAST)
    node.create_subscription(Costmap, TOPIC, lambda m: state.setdefault("m", m), qos)

    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    tf_hz = state.get("tf", 0) / 3.0
    print(f"TF liveness: {state.get('tf', 0)} msgs in 3s ({tf_hz:.1f} Hz) -> "
          f"{'ALIVE' if tf_hz > 20 else 'DEGRADED'}")
    if tf_hz <= 20:
        print("REFUSING to read the costmap through a degraded instrument.")
        return 2

    t0 = time.monotonic()
    while "m" not in state and time.monotonic() - t0 < 25.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    m = state.get("m")
    if m is None:
        print(f"NO COSTMAP on {TOPIC} within 25s even with TRANSIENT_LOCAL.")
        return 1

    md = m.metadata
    res, ox, oy = md.resolution, md.origin.position.x, md.origin.position.y
    print(f"costmap {md.size_x}x{md.size_y} @ {res} m/cell, origin ({ox:.2f}, {oy:.2f})\n")

    def cost_at(x: float, y: float):
        cx, cy = int((x - ox) / res), int((y - oy) / res)
        if not (0 <= cx < md.size_x and 0 <= cy < md.size_y):
            return None
        return m.data[cy * md.size_x + cx]

    def label(c):
        if c is None:
            return "OUT OF BOUNDS"
        if c == NO_INFO:
            return "UNKNOWN (255) -- never swept"
        if c >= LETHAL:
            return "LETHAL (254)"
        if c >= INSCRIBED:
            return "INSCRIBED (253)"
        if c == 0:
            return "FREE (0)"
        return f"cost {c}"

    verdict_cells = []
    for x, y in CELLS:
        c = cost_at(x, y)
        print(f"  ({x:>5.1f}, {y:>5.1f})  {label(c)}")
        if (x, y) in [(5.0, 0.0), (4.0, 0.0), (3.0, 0.0)]:
            verdict_cells.append(c)

    lethal = sum(1 for c in verdict_cells if c is not None and c >= INSCRIBED)
    unknown = sum(1 for c in verdict_cells if c == NO_INFO)
    print(f"\nRACK CELLS: {lethal}/3 lethal-or-inscribed, {unknown}/3 unknown")
    if lethal == 3:
        print("VERDICT: the racking IS an obstacle in the global costmap. r2_smoke's")
        print("         unreachable-goal case is aimed at cells Nav2 knows are blocked.")
    elif unknown:
        print("VERDICT: the racking is NOT established as an obstacle. Nav2 believes those")
        print("         cells are free/unknown, so 'the robot did not drive into the racking'")
        print("         proves nothing about obstacle avoidance -- R2's claim is UNPROVEN.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
