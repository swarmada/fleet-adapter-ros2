# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A hand-measured keep-out is proven at a fresh start, though the costmap cannot see all of it.

A fresh segment's global costmap covers only what SLAM has mapped from the mark. On the robot it
reached x 2.12 m, while the hearth's keep-out (its margin included) reached x 2.67 m: its far
corner lies behind the fireplace wall, which a fresh map never covers. Requiring the whole
polygon inside the costmap refused every segment before motion. So the proof is split:

* the mask the keep-out server SERVES (read back from its topic, not from the file) holds the
  whole polygon, every cell two cells in keep-out;
* every such cell inside the costmap's current extent is lethal, with the edge cells counted;
* the part beyond the extent is said in "bound proven" (cells, metres) and watched while the
  costmap grows;
* anything short of that refuses the segment before motion.
"""

from __future__ import annotations

import math

from test_explore_job import FakeClock, _assert_no_motion, _job

from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SIM_THETA0_DEG, SimulatedExploreStack

# The hearth with its 0.50 m margin in the bound's frame (the bound's keepouts.json on the robot,
# 2026-10-01), and the bound's room grid.
HEARTH_BOUND = ((0.214, 0.257), (1.253, -1.589), (3.651, -0.239), (2.613, 1.607))
R1G_DELTA = -61.192
# The costmap's extent at hws4-r1g's refusal, this boot's frame.
R1G_EXTENT = (-4.78, -5.26, 2.12, 2.04)


def _turn(poly, deg):
    t = math.radians(deg)
    return tuple((math.cos(t) * x - math.sin(t) * y, math.sin(t) * x + math.cos(t) * y)
                 for x, y in poly)


def _r1g():
    room = steps.Mask(222, 228, 0.05, -3.44, -6.87, [False] * (222 * 228))
    hearth = steps.Keepout("hearth", "marble hearth below the lidar plane", HEARTH_BOUND)
    mask = steps.rotate_mask(steps.apply_keepouts(room, [hearth]), R1G_DELTA)
    turned = steps.Keepout(hearth.name, hearth.reason, _turn(HEARTH_BOUND, R1G_DELTA))
    return mask, turned


def _served(mask, misses=()):
    """The mask as map_server serves it on its topic: keep-out 100, the rest unknown."""
    data = []
    for iy in range(mask.height):
        y = mask.oy + (iy + 0.5) * mask.res
        for ix in range(mask.width):
            x = mask.ox + (ix + 0.5) * mask.res
            keep = mask.keep[iy * mask.width + ix] and not any(
                steps.point_in_polygon(x, y, p) for p in misses)
            data.append(100 if keep else -1)
    return steps.Grid(mask.width, mask.height, mask.res, mask.ox, mask.oy, data)


def _costmap_over(mask, extent, res=0.05):
    """A faithful keep-out filter over a costmap of ``extent`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = extent
    w, h = round((x1 - x0) / res), round((y1 - y0) / res)
    data = []
    for iy in range(h):
        for ix in range(w):
            x, y = x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res
            data.append(100 if steps._mask_keep_at(mask, x, y) else 0)
    return steps.Grid(w, h, res, x0, y0, data)


# ── the steps, on hws4-r1g's numbers ────────────────────────────────────────

def test_r1g_the_hearth_reaching_past_a_fresh_costmap_is_proven_by_the_served_mask() -> None:
    mask, hearth = _r1g()
    xs = [p[0] for p in hearth.polygon]
    assert math.isclose(max(xs), 2.67, abs_tol=0.01) and max(xs) > R1G_EXTENT[2]
    ok, lines = steps.grade_keepouts(_costmap_over(mask, R1G_EXTENT), [hearth],
                                     served=_served(mask))
    assert ok, lines
    line = lines[0]
    assert "keep-out 'hearth'" in line and "whole in the served mask" in line
    assert "not lethal: 0" in line and "edge cells free:" in line
    # the part beyond the costmap: said in cells and metres, and watched
    beyond = line.split("beyond the costmap's extent: ")[1]
    cells = int(beyond.split(" cells")[0])
    metres = float(beyond.split("the polygon to ")[1].split(" m")[0])
    assert cells > 0 and math.isclose(metres, 2.667 - 2.12, abs_tol=0.01), line
    assert "watched as the costmap grows" in line


def test_r1g_a_served_mask_missing_a_corner_refuses() -> None:
    mask, hearth = _r1g()
    corner = hearth.polygon[3]                      # the far corner, behind the wall
    cut = ((corner[0] - 0.3, corner[1] - 0.3), (corner[0] + 0.3, corner[1] - 0.3),
           (corner[0] + 0.3, corner[1] + 0.3), (corner[0] - 0.3, corner[1] + 0.3))
    ok, lines = steps.grade_keepouts(_costmap_over(mask, R1G_EXTENT), [hearth],
                                     served=_served(mask, misses=[cut]))
    assert not ok, lines
    assert "served mask" in lines[0] and "not keep-out:" in lines[0]


def test_a_served_mask_missing_a_corner_inside_the_costmap_refuses_too() -> None:
    mask, hearth = _r1g()
    near = hearth.polygon[1]
    cut = ((near[0] - 0.1, near[1] - 0.1), (near[0] + 0.4, near[1] - 0.1),
           (near[0] + 0.4, near[1] + 0.4), (near[0] - 0.1, near[1] + 0.4))
    ok, lines = steps.grade_keepouts(_costmap_over(mask, R1G_EXTENT), [hearth],
                                     served=_served(mask, misses=[cut]))
    assert not ok and "not keep-out:" in lines[0], lines


def test_a_served_mask_that_does_not_reach_the_whole_polygon_refuses() -> None:
    mask, hearth = _r1g()
    served = _served(mask)
    w = round((2.4 - served.ox) / served.res)       # the served grid ends at x 2.4
    small = steps.Grid(w, served.height, served.res, served.ox, served.oy,
                       [served.data[iy * served.width + ix]
                        for iy in range(served.height) for ix in range(w)])
    ok, lines = steps.grade_keepouts(_costmap_over(mask, R1G_EXTENT), [hearth], served=small)
    assert not ok and "not inside the served mask's extent" in lines[0], lines


def test_an_in_extent_cell_not_lethal_in_the_costmap_still_refuses() -> None:
    mask, hearth = _r1g()
    cm = _costmap_over(mask, R1G_EXTENT)
    x, y = 0.9, -1.5                                # deep inside, inside the extent
    assert steps.depth_in_polygon(x, y, hearth.polygon) > 0.2
    ix, iy = steps._cell(x, y, cm.ox, cm.oy, cm.res)
    cm.data[iy * cm.width + ix] = 0
    ok, lines = steps.grade_keepouts(cm, [hearth], served=_served(mask))
    assert not ok and "not lethal: 1" in lines[0], lines


# ── the segment: a fresh start the costmap does not yet cover whole ─────────

SLAB = steps.Keepout("hearth", "a raised slab below the lidar plane",
                     ((0.6, 0.8), (1.6, 0.8), (1.6, 2.0), (0.6, 2.0)))
# The simulated costmap at a fresh start: it reaches y 1.5 m, the slab y 2.0 m.
FRESH = (-1.9, -1.4, 5.9, 1.5)


def _run(tmp_path, **kw):
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=SIM_THETA0_DEG,
                                  hand_keepouts=(SLAB,), **kw)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock, doc={"time_box_s": 30})
    job.run()
    return stack, emits


def test_a_fresh_segment_whose_costmap_does_not_reach_the_whole_slab_explores(tmp_path) -> None:
    stack, emits = _run(tmp_path, costmap_extents=((0.0, FRESH),))
    assert emits.last[0] == "SUCCEEDED", emits.updates
    proven = next(m for m in emits.messages() if "bound proven" in m)
    assert "whole in the served mask" in proven and "beyond the costmap's extent: " in proven
    assert "read_keepout_mask" in stack.log


def test_a_served_mask_without_the_slabs_far_part_refuses_before_motion(tmp_path) -> None:
    far = ((0.5, 1.6), (1.7, 1.6), (1.7, 2.1), (0.5, 2.1))
    stack, emits = _run(tmp_path, costmap_extents=((0.0, FRESH),), mask_misses=(far,))
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "bound not in force" in message and "served mask" in message
    assert "no motion" in message
    _assert_no_motion(stack)
