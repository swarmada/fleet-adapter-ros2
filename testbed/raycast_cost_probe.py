#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""How expensive is one r2_sensors scan, and does it depend on WHERE the robot is?

Hypothesis under test: the observed TF-rate decay was never a DDS fault. TF
and the 360-beam raycast shared ONE 10 Hz timer, and `_cast` marches a ray until it hits
an occupied cell or runs out of range -- so a ray that hits a rack early returns in ~40
steps while a ray into open floor runs the full RANGE_MAX/RAY_STEP = 240. As the robot
drives away from the racking into open space, every scan gets several times more
expensive, the shared timer falls behind, and TF -- hostage to the same callback -- decays
with it.

This measures the raycast ALONE: no ROS, no DDS, no publishers. Pure Python against the
same OccupancyMap the node uses. If the cost varies strongly with position, the mechanism
is arithmetic, not networking.

Run inside the testbed:
    python3 /ws/testbed/raycast_cost_probe.py --map /ws/testbed/map/warehouse.yaml
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

from r2_sensors import ANGLE_MAX, ANGLE_MIN, N_BEAMS, RANGE_MAX, RANGE_MIN, RAY_STEP, OccupancyMap


def cast(occ, x0: float, y0: float, world_angle: float) -> float:
    """Verbatim copy of Sensors._cast, decoupled from the node."""
    dx, dy = math.cos(world_angle) * RAY_STEP, math.sin(world_angle) * RAY_STEP
    x, y, dist = x0, y0, 0.0
    steps = 0
    while dist < RANGE_MAX:
        x += dx
        y += dy
        dist += RAY_STEP
        steps += 1
        if dist >= RANGE_MIN and occ.is_occupied(x, y):
            return steps
    return steps


def one_scan(occ, x: float, y: float, yaw: float = 0.0):
    inc = (ANGLE_MAX - ANGLE_MIN) / N_BEAMS
    t0 = time.perf_counter()
    steps = [cast(occ, x, y, yaw + ANGLE_MIN + i * inc) for i in range(N_BEAMS)]
    return (time.perf_counter() - t0), sum(steps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    occ = OccupancyMap(args.map)

    # Positions the robot actually occupies during an R2 run.
    places = [
        ("origin (between the racks)", 0.0, 0.0),
        ("r2_smoke goal (0,-6) open floor", 0.0, -6.0),
        ("cancel-latency goal (0,+6) open floor", 0.0, 6.0),
        ("deep open corner (-9,-9)", -9.0, -9.0),
    ]
    print(f"{N_BEAMS} beams, RAY_STEP={RAY_STEP} m, RANGE_MAX={RANGE_MAX} m "
          f"-> {int(RANGE_MAX / RAY_STEP)} steps is a full-length ray\n")
    print(f"{'position':<40} {'ms/scan':>9} {'steps/scan':>12} {'% of 10 Hz budget':>19}")
    results = []
    for label, x, y in places:
        samples = [one_scan(occ, x, y) for _ in range(args.reps)]
        ms = statistics.median(s[0] for s in samples) * 1000.0
        steps = int(statistics.median(s[1] for s in samples))
        results.append((label, ms))
        print(f"{label:<40} {ms:>9.1f} {steps:>12,} {ms / 100.0 * 100:>18.1f}%")

    worst = max(r[1] for r in results)
    best = min(r[1] for r in results)
    print(f"\nspread: {worst / best:.1f}x between the cheapest and most expensive position")
    print("A single 10 Hz timer has a 100 ms budget for the scan AND the TF publish.")
    if worst > 100.0:
        print("VERDICT: at the worst position one scan EXCEEDS the whole 10 Hz budget, so a")
        print("         shared timer cannot hold its rate -- and anything sharing it decays too.")
    else:
        print(f"VERDICT: worst case {worst:.0f} ms is inside the 100 ms budget; cost alone does")
        print("         not explain a rate collapse, though it narrows the margin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
