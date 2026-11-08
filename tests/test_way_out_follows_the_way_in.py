# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The way out of an exhausted area is the way the robot came in, not only a straight line.

While the robot stands in an area it has just exhausted, the area is drawn into the mask but
for a way out, so the planner has a start and the robot can drive back to the mark. That way
was the straight line from where the robot stands to the mark, 0.35 m either side. On a
stand-in (Humble Nav2, real TEB, the robot's footprint, the keep-out filter on both costmaps) a
retreat from a spot whose straight line to the mark passes a post inside the area ABORTED:
TEB found no feasible trajectory through the strip left beside the post, and the navigator
aborted the goal with the robot still in the area. From the same spot with nothing kept out it
reached the mark. So ``keep_out_area`` also leaves open the cells within 0.35 m of the robot's
own trail into the area (``trail_into``). Nothing already kept out -- the bound, the keep-out
set, a keep-out measured by hand -- is ever opened by it.

The explore job no longer draws a way out at all (maintainer's ruling, 2026-10-02): it leaves an
exhausted area for the mark with nothing of the area drawn in, and draws it in whole on the
mark. These tests keep the two library functions honest; the job's own order is pinned in
test_exhausted_area_left_before_it_is_kept_out.py.
"""

from __future__ import annotations

from fleet_adapter_ros2 import explore_steps as steps

SQUARE = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))
# The robot drove in round a post at (1.5, 1.58), which lies on the straight line from where it
# stopped, (1.85, 1.95), to the mark: in by the area's lower edge and up its right side.
WAY_IN = [(0.0, 0.0), (1.5, 0.5), (1.95, 1.2), (1.85, 1.95)]


def _free(n: int = 60, res: float = 0.05, o: float = -0.5) -> steps.Mask:
    return steps.Mask(n, n, res, o, o, [False] * (n * n))


def test_the_cells_along_the_way_in_are_left_open() -> None:
    straight = steps.keep_out_area(_free(), SQUARE, (1.85, 1.95), (0.0, 0.0), 0.35)
    way = steps.keep_out_area(_free(), SQUARE, (1.85, 1.95), (0.0, 0.0), 0.35,
                              trail=steps.trail_into(WAY_IN, SQUARE))
    # Beside the post, on the side the robot drove: closed by the line alone, open now.
    for x, y in ((1.9, 1.3), (1.7, 1.05), (1.95, 1.5)):
        assert steps._mask_keep_at(straight, x, y) is True, (x, y)
        assert steps._mask_keep_at(way, x, y) is False, (x, y)
    # The straight line stays open, and the rest of the area stays kept out.
    assert steps._mask_keep_at(way, 1.3, 1.37) is False
    for x, y in ((1.05, 1.95), (1.25, 1.95)):
        assert steps._mask_keep_at(way, x, y) is True, (x, y)


def test_the_way_in_never_opens_a_cell_already_kept_out() -> None:
    walled = steps.Mask(60, 60, 0.05, -0.5, -0.5, [True] * 3600)
    out = steps.keep_out_area(walled, SQUARE, (1.85, 1.95), (0.0, 0.0), 0.35,
                              trail=steps.trail_into(WAY_IN, SQUARE))
    assert all(out.keep)


def test_no_trail_is_the_straight_line_alone() -> None:
    a = steps.keep_out_area(_free(), SQUARE, (1.85, 1.95), (0.0, 0.0), 0.35)
    b = steps.keep_out_area(_free(), SQUARE, (1.85, 1.95), (0.0, 0.0), 0.35, trail=[])
    assert a.keep == b.keep


def test_the_way_in_is_the_last_entry_into_the_area() -> None:
    # In, out, and in again: the way in is from the last point outside to the end.
    trail = [(0.0, 0.0), (1.2, 1.2), (0.8, 0.8), (1.4, 1.1), (1.6, 1.5)]
    assert steps.trail_into(trail, SQUARE) == [(0.8, 0.8), (1.4, 1.1), (1.6, 1.5)]
    assert steps.trail_into([(1.2, 1.2), (1.5, 1.5)], SQUARE) == [(1.2, 1.2), (1.5, 1.5)]
    assert steps.trail_into([(1.2, 1.2), (0.5, 0.5)], SQUARE) == []      # ends outside
    assert steps.trail_into([], SQUARE) == []
