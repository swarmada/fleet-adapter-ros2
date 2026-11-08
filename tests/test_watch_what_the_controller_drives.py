# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""While a segment moves, the watch covers what the controller will drive, and the costmap as
it grows.

The planner's global path is held against the loaded mask, but the robot follows the
controller's local trajectory, and TEB's can leave the global path: on the robot its reversing
arcs swept ~0.29 m off a straight line. So the controller's local plan (TEB's ``/local_plan``,
published in ``odom``) is placed in the map and held against the mask too: a point a cell deep
in a keep-out stops the robot where it stands. A local plan that cannot be placed in the map
cannot be judged, and stops it as well.

At a fresh start the costmap does not reach the whole of a hand-measured keep-out. As it grows,
each keep-out cell that comes into it is read: one that is not lethal means the bound is not
in force there, and stops the robot where it stands.
"""

from __future__ import annotations

import math

from test_explore_job import FakeClock, _job

from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SIM_THETA0_DEG, SimulatedExploreStack

SLAB = steps.Keepout("hearth", "a raised slab below the lidar plane",
                     ((0.6, 0.8), (1.6, 0.8), (1.6, 2.0), (0.6, 2.0)))
FRESH = (-1.9, -1.4, 5.9, 1.5)           # the costmap at a fresh start: short of the slab's far edge
GROWN = (-1.9, -1.4, 5.9, 4.4)           # the costmap once exploring has mapped past the slab


def _line(a, b, step=0.05):
    n = max(1, math.ceil(math.dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def _run(tmp_path, **kw):
    clock = FakeClock()
    kw.setdefault("hand_keepouts", (SLAB,))
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=SIM_THETA0_DEG, **kw)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock, doc={"time_box_s": 30})
    job.run()
    return stack, emits


def _held(stack, emits) -> str:
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "go_to_mark" not in stack.log, stack.log
    assert "hold" in stack.log and "stop_explorer" in stack.log
    assert not stack.moving()
    return message


def _to_odom(points, map_to_odom):
    """Map-frame points as the controller publishes them, in odom."""
    tx, ty, deg = map_to_odom
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(c * (x - tx) + s * (y - ty), -s * (x - tx) + c * (y - ty)) for x, y in points]


# ── the controller's local plan ─────────────────────────────────────────────

def test_a_local_plan_that_swings_into_the_slab_stops_the_robot_where_it_stands(tmp_path):
    # The global path runs beside the slab; the controller's arc swings 0.3 m into it, as TEB's
    # reversing arcs swung off the line on the robot.
    beside = _line((0.0, 0.5), (2.0, 0.5))
    arc = [(x, 0.5 + 0.6 * math.sin(math.pi * x / 2.0)) for x, _ in beside]
    assert any(steps.depth_in_polygon(x, y, SLAB.polygon) > 0.25 for x, y in arc)
    odom = (0.4, -0.3, 7.0)
    stack, emits = _run(tmp_path, plans=[(10.0, beside)], map_to_odom=odom,
                        local_plans=[(10.0, _to_odom(arc, odom))])
    message = _held(stack, emits)
    assert "controller's local plan" in message and "hearth" in message
    assert "stopped where it stands" in message


def test_a_local_plan_is_judged_in_the_map_not_in_odom(tmp_path):
    # Read as map coordinates these odom points lie deep in the slab; placed with this
    # map -> odom they lie on the line ahead of the mark, clear of it.
    odom = (-1.0, -1.3, 0.0)
    clear = _line((-0.1, 0.0), (0.3, 0.0))
    in_odom = _to_odom(clear, odom)
    assert all(steps.depth_in_polygon(x, y, SLAB.polygon) == 0 for x, y in clear)
    assert all(steps.depth_in_polygon(x, y, SLAB.polygon) >= 0.25 for x, y in in_odom)
    _, emits = _run(tmp_path, map_to_odom=odom, local_plans=[(10.0, in_odom)])
    assert emits.last[0] == "SUCCEEDED", emits.updates


def test_a_local_plan_along_the_slabs_edge_is_taken(tmp_path):
    beside = _line((0.0, 0.77), (2.0, 0.77)) + [(1.0, 0.82)]
    _, emits = _run(tmp_path, local_plans=[(10.0, beside)])
    assert emits.last[0] == "SUCCEEDED", emits.updates


def test_a_local_plan_that_cannot_be_placed_in_the_map_stops_the_robot(tmp_path):
    stack, emits = _run(tmp_path, map_to_odom=None,
                        local_plans=[(10.0, _line((0.0, 0.0), (0.5, 0.0)))])
    message = _held(stack, emits)
    assert "controller's local plan" in message and "could not be placed" in message


# ── the keep-out as the costmap grows ───────────────────────────────────────

def test_a_keep_out_cell_that_comes_into_the_costmap_not_lethal_stops_the_robot(tmp_path):
    far = steps.Keepout("far", "r", ((0.5, 1.6), (1.7, 1.6), (1.7, 2.1), (0.5, 2.1)))
    stack, emits = _run(tmp_path, costmap_extents=((0.0, FRESH), (10.0, GROWN)),
                        costmap_misses=(far,))
    message = _held(stack, emits)
    assert "keep-out 'hearth'" in message and "newly in the costmap" in message
    assert "not lethal" in message and "stopped where it stands" in message


def test_keep_out_cells_that_come_into_the_costmap_lethal_are_said_and_exploring_goes_on(
        tmp_path):
    _, emits = _run(tmp_path, costmap_extents=((0.0, FRESH), (10.0, GROWN)))
    assert emits.last[0] == "SUCCEEDED", emits.updates
    grew = [m for m in emits.messages() if "more than when the bound was proven" in m]
    assert grew and "keep-out 'hearth'" in grew[0] and "not lethal: 0" in grew[0], emits.updates
