# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A keep-out measured by hand is in force for every exploring segment.

A raised slab below the lidar's plane is in no map and no costmap: the explorer finds it by
driving onto it. So an operator measures it from the mark, and the bound declares it
(``bound.json`` names a ``keepouts.json`` in the bound's frame). Every segment of that bound
draws it into the keep-out mask before the bound is proven, reads it back in the live costmap --
every cell inside it lethal -- and refuses to move when it cannot. While the robot explores and
returns, every global path the planner publishes is held against the loaded mask: a goal in a
keep-out, or a path across one, means the bound is not in force, and the robot is stopped where
it stands.

The polygons here are in the simulated bound's frame; a boot yaw equal to the bound's theta0
makes this boot's frame the bound's, so the coordinates read as written.
"""

from __future__ import annotations

import json
import math

import pytest
from test_explore_job import FakeClock, _assert_no_motion, _job

from fleet_adapter_ros2 import explore, hand_keepouts
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SIM_THETA0_DEG, SimulatedExploreStack

SLAB = steps.Keepout("hearth", "a raised slab below the lidar plane",
                     ((0.6, 0.8), (1.6, 0.8), (1.6, 2.0), (0.6, 2.0)))
INSIDE = (1.1, 1.4)


def _turned(x: float, y: float, boot_yaw: float) -> tuple[float, float]:
    d = math.radians(boot_yaw - SIM_THETA0_DEG)
    return math.cos(d) * x - math.sin(d) * y, math.sin(d) * x + math.cos(d) * y


def _stack(clock, boot_yaw=SIM_THETA0_DEG, **kw):
    kw.setdefault("hand_keepouts", (SLAB,))
    return SimulatedExploreStack(clock=clock, boot_yaw_deg=boot_yaw, **kw)


def _run(tmp_path, stack, clock, doc=None):
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"time_box_s": 30, **(doc or {})})
    job.run()
    return stack, emits


def _line(a, b, step=0.05):
    n = max(1, math.ceil(math.dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


# ── before motion: drawn in, read back, or no motion ────────────────────────

@pytest.mark.parametrize("boot_yaw", [SIM_THETA0_DEG, 40.0, -95.0])
def test_no_frontier_can_lie_in_the_slab_every_cell_is_lethal_before_the_explorer_starts(
        tmp_path, boot_yaw) -> None:
    # explore_lite makes a frontier of an unknown cell beside a free one, on the global costmap;
    # a lethal cell is neither. So every cell inside the slab is lethal when the explorer starts.
    clock = FakeClock()
    stack = _stack(clock, boot_yaw)
    at_start = {}
    stack.on_explorer_started = lambda: at_start.setdefault("costmap", stack._grid(True))
    stack, emits = _run(tmp_path, stack, clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    costmap = at_start["costmap"]
    turned = [_turned(x, y, boot_yaw) for x, y in SLAB.polygon]
    inside = [(costmap.ox + (ix + 0.5) * costmap.res, costmap.oy + (iy + 0.5) * costmap.res, v)
              for iy in range(costmap.height) for ix in range(costmap.width)
              for v in [costmap.data[iy * costmap.width + ix]]]
    inside = [(x, y, v) for x, y, v in inside
              if steps.depth_in_polygon(x, y, turned) >= costmap.res]
    assert len(inside) >= 60 and all(v == 100 for _, _, v in inside)
    assert stack.loaded_mask_keep_at(*_turned(*INSIDE, boot_yaw)) is True
    assert stack.loaded_mask_keep_at(0.0, 0.0) is False
    proven = next(m for m in emits.messages() if "bound proven" in m)
    assert "keep-out 'hearth'" in proven and "not lethal: 0" in proven
    loading = next(m for m in emits.messages() if "loading the keep-out mask" in m)
    assert "hearth" in loading and "measured by hand" in loading


def test_the_mask_without_the_slab_fails_bound_proven(tmp_path) -> None:
    # The keep-out servers serve a mask the slab is not in: the room's cells are lethal, the
    # slab's are not. Nothing moves.
    clock = FakeClock()
    stack = _stack(clock, costmap_misses=(SLAB,))
    stack, emits = _run(tmp_path, stack, clock)
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "bound not in force" in message and "hearth" in message
    assert "not lethal" in message and "no motion" in message
    _assert_no_motion(stack)


def test_a_mark_inside_a_hand_keep_out_is_refused_naming_it(tmp_path) -> None:
    around = steps.Keepout("hearth", SLAB.reason, ((-0.3, -0.3), (0.3, -0.3), (0.3, 0.3),
                                                   (-0.3, 0.3)))
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock, hand_keepouts=(around,)), clock)
    state, message, _, _ = emits.last
    assert state == "FAILED" and "start mark" in message and "hearth" in message
    assert "no motion" in message
    _assert_no_motion(stack)


# ── while moving: every published path held against the loaded mask ────────

def _held(stack, emits) -> str:
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "go_to_mark" not in stack.log, stack.log
    assert "hold" in stack.log and "stop_explorer" in stack.log
    assert not stack.moving()
    return message


@pytest.mark.parametrize("boot_yaw", [SIM_THETA0_DEG, 40.0])
def test_a_navigation_goal_inside_the_slab_is_refused_and_the_robot_held(
        tmp_path, boot_yaw) -> None:
    clock = FakeClock()
    path = [_turned(x, y, boot_yaw) for x, y in _line((0.0, 0.0), INSIDE)]
    stack = _stack(clock, boot_yaw, plans=[(10.0, path)])
    stack, emits = _run(tmp_path, stack, clock)
    message = _held(stack, emits)
    assert "hearth" in message and "stopped where it stands" in message
    assert "not in force" in message


def test_a_frontier_whose_path_crosses_the_slab_is_not_taken(tmp_path) -> None:
    clock = FakeClock()
    frontier = (2.5, 2.5)                       # outside the slab; the straight path is not
    assert steps.depth_in_polygon(*frontier, SLAB.polygon) == 0
    stack = _stack(clock, plans=[(10.0, _line((0.0, 0.0), frontier))])
    stack, emits = _run(tmp_path, stack, clock)
    message = _held(stack, emits)
    assert "hearth" in message and "path" in message


def test_a_path_along_the_slabs_edge_is_taken(tmp_path) -> None:
    # A path the planner drew beside the slab, one sample just over its edge: the mask is
    # sampled onto the costmap's grid, so only a point a cell inside a keep-out is a violation.
    clock = FakeClock()
    beside = _line((0.0, 0.77), (2.0, 0.77)) + [(1.0, 0.82)] + _line((2.0, 0.77), (2.0, 0.0))
    stack = _stack(clock, plans=[(10.0, beside)])
    stack, emits = _run(tmp_path, stack, clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates


def test_a_path_out_of_the_room_is_a_bound_not_in_force_too(tmp_path) -> None:
    clock = FakeClock()
    stack = _stack(clock, hand_keepouts=(), plans=[(10.0, _line((0.0, 0.0), (0.0, -3.0)))])
    stack, emits = _run(tmp_path, stack, clock)
    message = _held(stack, emits)
    assert "the room's bound" in message


# ── the steps: read back where the costmap is, and only a cell in ───────────

def _costmap(w=40, h=40, res=0.05, ox=0.0, oy=0.0, lethal=()):
    data = []
    for iy in range(h):
        for ix in range(w):
            x, y = ox + (ix + 0.5) * res, oy + (iy + 0.5) * res
            data.append(100 if any(steps.point_in_polygon(x, y, p) for p in lethal) else 0)
    return steps.Grid(width=w, height=h, res=res, ox=ox, oy=oy, data=data)


BOX = steps.Keepout("box", "r", ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)))


def test_a_keep_out_read_back_lethal_is_in_force() -> None:
    ok, lines = steps.grade_keepouts(_costmap(lethal=[BOX.polygon]), [BOX])
    assert ok and "keep-out 'box'" in lines[0] and "not lethal: 0" in lines[0]


def test_a_keep_out_the_costmap_does_not_cover_is_not_proven() -> None:
    # The bound's own grading counts only the keep-out cells inside the costmap: this one would
    # pass it with nothing checked.
    far = steps.Keepout("far", "r", ((3.0, 3.0), (3.5, 3.0), (3.5, 3.5), (3.0, 3.5)))
    ok, lines = steps.grade_keepouts(_costmap(lethal=[BOX.polygon]), [far])
    assert not ok and "not inside the costmap" in lines[0]


def test_one_free_cell_inside_a_keep_out_is_not_in_force() -> None:
    hole = ((0.95, 0.95), (1.05, 0.95), (1.05, 1.05), (0.95, 1.05))
    cm = _costmap(lethal=[BOX.polygon])
    cm.data[20 * cm.width + 20] = 0
    assert steps.point_in_polygon(1.025, 1.025, hole)
    ok, lines = steps.grade_keepouts(cm, [BOX])
    assert not ok and "not lethal: 1" in lines[0]


def test_only_a_point_a_cell_inside_the_mask_is_deep() -> None:
    m = steps.apply_keepouts(steps.Mask(40, 40, 0.05, 0.0, 0.0, [False] * 1600), [BOX])
    assert steps.deep_in_mask(m, 1.0, 1.0)
    assert not steps.deep_in_mask(m, 0.52, 1.0)          # an edge cell
    assert not steps.deep_in_mask(m, 0.3, 1.0)           # outside the keep-out
    assert steps.deep_in_mask(m, -1.0, 1.0)              # beyond the mask: outside the room


# ── the file the bound declares ──────────────────────────────────────────────

def _bound_dir(tmp_path, keepouts_doc=None, declare="keepouts.json", theta0=-60.0):
    bound_doc = {"theta0_deg": theta0}
    if declare is not None:
        bound_doc[explore.HAND_KEEPOUTS_KEY] = declare
    if keepouts_doc is not None:
        (tmp_path / "keepouts.json").write_text(json.dumps(keepouts_doc))
    return tmp_path, bound_doc


def _doc(theta0=-60.0, **over):
    k = {"name": "hearth", "reason": "a raised slab", "polygon": [
        {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}
    return {"theta0_deg": theta0, "keepouts": [{**k, **over}]}


def test_a_bound_that_declares_no_hand_keep_out_has_none(tmp_path) -> None:
    assert explore.read_hand_keepouts(*_bound_dir(tmp_path, declare=None)) == []


def test_a_declared_file_is_read(tmp_path) -> None:
    ks = explore.read_hand_keepouts(*_bound_dir(tmp_path, _doc()))
    assert [k.name for k in ks] == ["hearth"] and ks[0].polygon[1] == (1.0, 0.0)


@pytest.mark.parametrize("doc, declare, words", [
    (None, "keepouts.json", ["keepouts.json", "cannot be read"]),
    (_doc(theta0=12.0), "keepouts.json", ["12", "-60", "another heading line"]),
    (_doc(polygon=[{"x": 0, "y": 0}, {"x": 1, "y": 0}]), "keepouts.json", ["three points"]),
    (_doc(reason=""), "keepouts.json", ["reason"]),
    ({"theta0_deg": -60.0, "keepouts": []}, "keepouts.json", ["no keep-out"]),
    (_doc(), "../keepouts.json", ["not a file name"]),
])
def test_a_declared_file_that_cannot_be_trusted_is_refused_by_name(
        tmp_path, doc, declare, words) -> None:
    with pytest.raises(LookupError) as exc:
        explore.read_hand_keepouts(*_bound_dir(tmp_path, doc, declare=declare))
    for w in words:
        assert w in str(exc.value)


# ── from the mark to the bound's frame ──────────────────────────────────────

def test_corners_measured_from_the_mark_turn_onto_the_heading_line() -> None:
    corners = hand_keepouts.read_corners("# a slab\nC1 1 0\nC2 2 0\nC3 2 1\nC4 1 1\n")
    # Heading line at +90 degrees: ahead is +y, left is -x.
    assert hand_keepouts.to_bound_frame(corners, 90.0, 0.0) == [
        (0.0, 1.0), (0.0, 2.0), (-1.0, 2.0), (-1.0, 1.0)]
    grown = hand_keepouts.to_bound_frame(corners, 0.0, 0.1)
    assert grown == [(0.9, -0.1), (2.1, -0.1), (2.1, 1.1), (0.9, 1.1)]


def test_the_converters_file_is_what_the_bound_reads(tmp_path) -> None:
    corners = hand_keepouts.read_corners("C1 0.4 0.8\nC2 1.5 0.8\nC3 1.5 2.6\nC4 0.4 2.6\n")
    doc = hand_keepouts.keepouts_doc(corners, theta0_deg=-60.0, margin_m=0.25, name="hearth",
                                     reason="a raised slab")
    ks = explore.read_hand_keepouts(*_bound_dir(tmp_path, doc))
    # The mark-frame point 1.0 ahead, 1.7 left is inside; 1.0 ahead on the line is not.
    t = math.radians(-60.0)
    inside = (1.0 * math.cos(t) - 1.7 * math.sin(t), 1.0 * math.sin(t) + 1.7 * math.cos(t))
    on_line = (math.cos(t), math.sin(t))
    assert steps.point_in_polygon(*inside, ks[0].polygon)
    assert not steps.point_in_polygon(*on_line, ks[0].polygon)


@pytest.mark.parametrize("text, words", [
    ("C1 1 0\nC2 2 0\n", ["three corners"]),
    ("C1 1 0\nC2 2 x\nC3 1 1\n", ["line 2"]),
    ("1 0\n2 0\n2 1\n", ["line 1"]),
])
def test_a_corners_file_that_is_not_one_is_refused(text, words) -> None:
    with pytest.raises(ValueError) as exc:
        hand_keepouts.read_corners(text)
    for w in words:
        assert w in str(exc.value)


def _filtered_costmap(mask, w=260, h=260, res=0.05, ox=-3.0543, oy=-6.1095):
    # A faithful keep-out filter: each costmap cell lethal where the loaded mask keeps out its
    # centre. The costmap's grid is not the mask's (its origin is the map's), as on a robot; this
    # one is a stand-in segment's.
    data = []
    for iy in range(h):
        for ix in range(w):
            x, y = ox + (ix + 0.5) * res, oy + (iy + 0.5) * res
            data.append(100 if steps._mask_keep_at(mask, x, y) else 0)
    return steps.Grid(width=w, height=h, res=res, ox=ox, oy=oy, data=data)


@pytest.mark.parametrize("delta", [3.8, -6.1, 0.0, 17.3, 45.0, 128.6, -95.0])
def test_a_keep_out_the_filter_applied_faithfully_reads_back_in_force_at_any_turn(delta) -> None:
    # The polygon is drawn at cell centres in the bound's frame, then the mask is turned and
    # sampled again: each step moves the edge by up to half a cell's diagonal, so a cell one cell
    # inside the polygon may be free in the very mask the filter applied. That is not a filter
    # out of force, and must not refuse a segment. The grids and the slab are a stand-in
    # segment's that was refused so, at a turn of 3.8 degrees: one cell of 1784.
    room = steps.Mask(222, 228, 0.05, -3.44, -6.87, [False] * (222 * 228))
    slab = steps.Keepout("hearth", "r", ((0.351, 0.219), (1.291, -1.453), (3.515, -0.201),
                                         (2.575, 1.47)))
    mask = steps.rotate_mask(steps.apply_keepouts(room, [slab]), delta)
    t = math.radians(delta)
    turned = steps.Keepout("hearth", "r", tuple((math.cos(t) * x - math.sin(t) * y,
                                                 math.sin(t) * x + math.cos(t) * y)
                                                for x, y in slab.polygon))
    ok, lines = steps.grade_keepouts(_filtered_costmap(mask), [turned])
    assert ok, lines


def test_the_read_back_says_how_many_edge_cells_are_free() -> None:
    # The band within two cells of the edge is not judged (sampling can leave a cell free there),
    # so it is counted and said: a shortfall larger than sampling explains is visible in the line.
    cm = _costmap(lethal=[BOX.polygon])
    ok, lines = steps.grade_keepouts(cm, [BOX])
    assert ok and "edge cells free: 0 of " in lines[0]
    freed = 0
    for iy in range(cm.height):
        for ix in range(cm.width):
            x, y = cm.ox + (ix + 0.5) * cm.res, cm.oy + (iy + 0.5) * cm.res
            if 0 < steps.depth_in_polygon(x, y, BOX.polygon) < 2 * cm.res and freed < 7:
                cm.data[iy * cm.width + ix] = 0
                freed += 1
    ok, lines = steps.grade_keepouts(cm, [BOX])
    assert ok and "edge cells free: 7 of " in lines[0], lines
