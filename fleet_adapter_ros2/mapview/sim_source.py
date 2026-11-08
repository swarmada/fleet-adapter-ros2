# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A simulated data source for the map-view contract: this repository's `SimulatedExploreStack`
(a synthetic 8 m x 6 m room whose map grows with explore time) on a fast clock, written as
contract JSON Lines. No robot, no ROS, no cluster.

    python -m fleet_adapter_ros2.mapview.sim_source --out sim.jsonl [--speed 10] [--seconds 60]

It stands in for a guided session: a simulated operator walks the robot round the room towards
two targets, one point a second, and the room's border plus one synthetic box are the keep-out
overlay. Every message says it is simulated. It drives the stack directly and runs no
FleetAction; `hello_world` runs the adapter's own explore-floor segment instead.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2 import sim_explore as sim
from fleet_adapter_ros2.mapview import contract as c


class FastClock:
    """Simulated seconds, `speed` times real time."""

    def __init__(self, speed: float):
        self.speed, self.w0 = speed, time.time()
        self.t0 = self.w0

    def now(self) -> float:
        return self.t0 + (time.time() - self.w0) * self.speed

    def wait(self, event, seconds: float) -> bool:
        return event.wait(seconds / self.speed)


def route(seconds: float):
    """A loop around the synthetic room from the mark (0, 0): (explore time, x, y), one a second."""
    corners = [(0.0, 0.0), (3.5, 0.5), (4.5, 3.0), (1.0, 3.8), (-1.2, 2.5), (-1.0, 0.3), (0.0, 0.0)]
    legs = list(itertools.pairwise(corners))
    total = sum(math.dist(a, b) for a, b in legs)
    out = []
    for t in range(1, int(seconds)):
        d = total * t / seconds
        for a, b in legs:
            seg = math.dist(a, b)
            if d <= seg:
                f = d / seg
                out.append((float(t), a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
                break
            d -= seg
    return out


def grid_fields(g: steps.Grid) -> dict:
    return c.encode_flat(g.data, g.width, g.height, g.res, (g.ox, g.oy))


def keepout_fields(room: steps.Mask) -> dict:
    """The room's border and a synthetic 1 m x 0.6 m box, as one keep-out grid."""
    data = []
    for iy in range(room.height):
        cy = room.oy + (iy + 0.5) * room.res
        for ix in range(room.width):
            cx = room.ox + (ix + 0.5) * room.res
            box = 2.0 < cx < 3.0 and 1.6 < cy < 2.2
            data.append(100 if room.keep[iy * room.width + ix] or box else 0)
    return c.encode_flat(data, room.width, room.height, room.res, (room.ox, room.oy))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--speed", type=float, default=10.0, help="simulated seconds per real second")
    ap.add_argument("--seconds", type=float, default=60.0, help="explore time")
    ap.add_argument("--map-every", type=float, default=5.0,
                    help="simulated seconds between map snapshots")
    ap.add_argument("--boot-yaw", type=float, default=30.0)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit(f"{a.out} exists: choose a new file")

    clock = FastClock(a.speed)
    pts = route(a.seconds)
    stack = sim.SimulatedExploreStack(clock, explore_seconds=a.seconds, boot_yaw_deg=a.boot_yaw,
                                      recoveries=pts, start_mv=8200.0)
    robot, frame, label = "sim-a", "map@sim", "guided:sim"
    goal_a, goal_b = (4.5, 3.0), (-1.2, 2.5)

    stack.start_slam(None)
    room = steps.rotate_mask(sim._sim_mask(), stack.boot_yaw_deg - stack.bound_theta0_deg)
    keepouts = [{"name": "keep-out", "grid": keepout_fields(room)}]
    with open(a.out, "w", buffering=1) as out:
        def put(m):
            out.write(c.dumps(m) + "\n")

        put(c.message("SessionStatus", robot, frame, clock.now(), source=label, phase="started",
                      text="simulated operator-driven session in a synthetic room",
                      area_free_m2=None, simulated=True))
        put(c.message("Overlays", robot, frame, clock.now(), keepouts=keepouts, zones=[],
                      target={"label": "A", "x": goal_a[0], "y": goal_a[1]}))
        stack.start_explorer()
        seq, next_map, last_target = 0, 0.0, "A"
        t_start = clock.now()
        while True:
            now = clock.now()
            stack.recoveries()                         # moves the robot along its route
            x, y, yaw = stack.robot_pose()
            put(c.message("PoseSample", robot, frame, now, x=x, y=y, yaw_deg=yaw, source="sim"))
            if now - t_start >= next_map:
                put(c.message("MapSnapshot", robot, frame, now, seq=seq,
                              **grid_fields(stack.read_map())))
                seq, next_map = seq + 1, next_map + a.map_every
            if last_target == "A" and math.dist((x, y), goal_a) < 0.6:
                put(c.message("Overlays", robot, frame, now, keepouts=keepouts, zones=[],
                              target={"label": "B", "x": goal_b[0], "y": goal_b[1]}))
                last_target = "B"
            if stack.explorer_status() == "exploration_complete":
                break
            time.sleep(1.0 / a.speed)
        g = stack.read_map()
        put(c.message("MapSnapshot", robot, frame, clock.now(), seq=seq, **grid_fields(g)))
        area = sum(v == 0 for v in g.data) * g.res * g.res
        put(c.message("SessionStatus", robot, frame, clock.now(), source=label, phase="stopped",
                      text="simulated session complete", area_free_m2=float(area),
                      simulated=True))
    print(f"wrote {a.out}: {seq + 1} map snapshots; final free area {area:.1f} m2")


if __name__ == "__main__":
    main()
