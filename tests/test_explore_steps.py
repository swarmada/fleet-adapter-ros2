# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The steps a floor-exploration segment is built from, ported from the tools that ran the
first supervised explorations on a robot (2026-09-27/28). Each was proven there; these tests
pin the behaviour that made it hold, so the port cannot drift from it:

* a bounded read and a bounded call: a dropped service reply is read again, a few times and a
  few seconds apart, and the verdict is exact -- a wrong answer is never read again;
* the keep-out mask turned into this boot's frame, with operator keep-outs drawn in;
* the bound proven in the live costmap, not assumed from the mask file;
* the supervisor: the battery window, and the stop -> wait paused + settle -> return order;
* an area that keeps needing recoveries is marked exhausted once, and named.
"""

from __future__ import annotations

import math

import pytest

from fleet_adapter_ros2 import explore_steps as st

# ── bounded read / bounded call ─────────────────────────────────────────────


class _Waits:
    """A wait that records each gap and never sleeps; `halt_after` makes the nth wait report a
    halt, as a stop arriving during the gap would."""

    def __init__(self, halt_after: int | None = None) -> None:
        self.gaps: list[float] = []
        self.halt_after = halt_after

    def __call__(self, seconds: float) -> bool:
        self.gaps.append(seconds)
        return self.halt_after is not None and len(self.gaps) >= self.halt_after


def _answers(*values):
    it = iter(values)
    return lambda: next(it)


def test_the_ruled_fallback_is_three_more_reads_five_seconds_apart() -> None:
    assert (st.RETRIES, st.RETRY_GAP_S) == (3, 5.0)


def test_a_read_answered_exactly_at_once_is_not_read_again() -> None:
    waits = _Waits()
    v = st.bounded_read(_answers("0.2"), lambda a: a == "0.2", what="cap", wait=waits)
    assert (v.verdict, v.value, v.ok) == ("EXACT", "0.2", True)
    assert len(v.attempts) == 1 and waits.gaps == []


def test_a_dropped_reply_is_read_again_and_an_exact_answer_continues() -> None:
    waits = _Waits()
    v = st.bounded_read(_answers(None, None, "0.2"), lambda a: a == "0.2", what="cap", wait=waits)
    assert v.verdict == "EXACT" and v.ok
    assert len(v.attempts) == 3 and waits.gaps == [5.0, 5.0]
    assert "no answer" in v.attempts[0] and "exact" in v.attempts[2]


def test_four_silences_are_no_answer_and_every_attempt_is_recorded() -> None:
    waits = _Waits()
    v = st.bounded_read(_answers(None, None, None, None, "0.2"), lambda a: True, what="cap",
                        wait=waits)
    assert v.verdict == "NO ANSWER" and not v.ok and v.value is None
    assert len(v.attempts) == 4 and waits.gaps == [5.0, 5.0, 5.0]


def test_a_wrong_answer_is_never_read_again() -> None:
    waits = _Waits()
    v = st.bounded_read(_answers("0.5", "0.2"), lambda a: a == "0.2", what="cap", wait=waits)
    assert (v.verdict, v.value, v.ok) == ("MISMATCH", "0.5", False)
    assert len(v.attempts) == 1 and waits.gaps == []


def test_a_stop_during_the_gap_ends_the_read_as_halted() -> None:
    v = st.bounded_read(_answers(None, "0.2"), lambda a: True, what="cap", wait=_Waits(1))
    assert v.verdict == "HALTED" and not v.ok


def test_a_call_whose_reply_was_dropped_is_made_again() -> None:
    waits = _Waits()
    v = st.bounded_call(_answers(None, True), what="activate", wait=waits)
    assert v.verdict == "DONE" and v.ok and waits.gaps == [5.0]


def test_a_refused_call_is_not_made_again() -> None:
    waits = _Waits()
    v = st.bounded_call(_answers(False, True), what="activate", wait=waits)
    assert v.verdict == "REFUSED" and not v.ok and waits.gaps == []


def test_a_call_never_answered_is_no_answer_after_the_ruled_tries() -> None:
    waits = _Waits()
    v = st.bounded_call(_answers(None, None, None, None, True), what="activate", wait=waits)
    assert v.verdict == "NO ANSWER" and len(v.attempts) == 4 and waits.gaps == [5.0] * 3


# ── the mask, turned into this boot's frame ─────────────────────────────────

def _room_mask(w: int = 20, h: int = 10, res: float = 0.1) -> st.Mask:
    """A w x h mask, origin so that world (0, 0) is inside, with a one-cell keep-out border
    and everything else allowed."""
    keep = [x in (0, w - 1) or y in (0, h - 1) for y in range(h) for x in range(w)]
    return st.Mask(width=w, height=h, res=res, ox=-0.5, oy=-0.5, keep=keep)


def _keep_at(mask: st.Mask, x: float, y: float) -> bool:
    ix, iy = math.floor((x - mask.ox) / mask.res), math.floor((y - mask.oy) / mask.res)
    if not (0 <= ix < mask.width and 0 <= iy < mask.height):
        return True
    return mask.keep[iy * mask.width + ix]


def test_a_zero_turn_leaves_the_mask_as_it_was() -> None:
    m = _room_mask()
    r = st.rotate_mask(m, 0.0)
    assert (r.width, r.height, r.keep) == (m.width, m.height, m.keep)
    assert (r.ox, r.oy) == pytest.approx((m.ox, m.oy))


def test_a_room_point_lands_where_the_turned_frame_puts_it() -> None:
    m = _room_mask()
    # An allowed point at (1.0, 0.2) in the mask's frame is at R(90 deg) of it in this boot's.
    assert not _keep_at(m, 1.0, 0.2)
    r = st.rotate_mask(m, 90.0)
    assert not _keep_at(r, -0.2, 1.0)
    assert _keep_at(r, 1.0, 0.2)            # where it was is now outside the room


def test_outside_the_old_mask_is_keep_out_and_the_mark_stays_allowed() -> None:
    m = _room_mask()
    r = st.rotate_mask(m, 37.0)
    assert not _keep_at(r, 0.0, 0.0)        # the mark: the turn is about it
    assert _keep_at(r, r.ox + 0.5 * r.res, r.oy + 0.5 * r.res)  # a corner of the new extent


def test_a_turn_there_and_back_keeps_the_allowed_area() -> None:
    m = _room_mask(40, 30, 0.05)
    back = st.rotate_mask(st.rotate_mask(m, 90.0), -90.0)
    assert st.allowed_area_m2(back) == pytest.approx(st.allowed_area_m2(m), rel=0.02)


def test_a_mask_with_other_values_is_refused() -> None:
    with pytest.raises(ValueError, match="0 and 205"):
        st.mask_from_pgm(b"P5\n2 1\n255\n\x00\xfe", res=0.05, ox=0.0, oy=0.0)


def test_a_mask_round_trips_through_its_pgm() -> None:
    m = _room_mask()
    back = st.mask_from_pgm(st.mask_to_pgm(m), res=m.res, ox=m.ox, oy=m.oy)
    assert back.keep == m.keep


def test_an_operator_keep_out_is_drawn_into_the_mask() -> None:
    m = _room_mask()
    ledge = st.Keepout(name="ledge-1", reason="a ledge below the lidar plane",
                       polygon=((0.5, 0.0), (0.9, 0.0), (0.9, 0.3), (0.5, 0.3)))
    out = st.apply_keepouts(m, [ledge])
    assert _keep_at(out, 0.7, 0.15) and not _keep_at(m, 0.7, 0.15)
    assert not _keep_at(out, 0.2, 0.15)     # outside the polygon: unchanged
    assert st.allowed_area_m2(out) < st.allowed_area_m2(m)


# ── the bound, proven in the costmap ────────────────────────────────────────

def _grids(mask: st.Mask, *, leak: bool = False, erase_unknown: bool = False):
    """/map and the global costmap on the mask's own grid: /map knows a strip near the robot,
    unknown elsewhere; the costmap has every keep-out cell lethal (unless `leak`) and keeps
    /map's unknown (unless `erase_unknown`, the filter-writes-free failure)."""
    n = mask.width * mask.height
    slam = [-1] * n
    for i in range(n):
        if (i % mask.width) < 4:
            slam[i] = 0
    cost = []
    for i in range(n):
        if mask.keep[i]:
            cost.append(0 if leak and i == 1 else 100)  # (1, 0): the bottom border
        elif slam[i] == -1:
            cost.append(0 if erase_unknown else -1)
        else:
            cost.append(0)
    geo = {"width": mask.width, "height": mask.height, "res": mask.res, "ox": mask.ox, "oy": mask.oy}
    return st.Grid(data=cost, **geo), st.Grid(data=slam, **geo)


def test_the_bound_is_in_force_when_keep_out_is_lethal_and_unknown_is_kept() -> None:
    m = _room_mask(60, 60, 0.1)
    cost, slam = _grids(m)
    ok, lines = st.grade_bound(cost, slam, m, robot=(0.0, 0.0), reach_m=1.0)
    assert ok, lines


def test_a_keep_out_cell_that_is_not_lethal_fails_the_bound() -> None:
    m = _room_mask(60, 60, 0.1)
    cost, slam = _grids(m, leak=True)
    ok, lines = st.grade_bound(cost, slam, m, robot=(0.0, 0.0), reach_m=1.0)
    assert not ok and any("not lethal: 1" in ln for ln in lines)


def test_a_filter_that_frees_unknown_fails_the_bound() -> None:
    # A FREE mask value erases unknown cells, and with them every frontier.
    m = _room_mask(60, 60, 0.1)
    cost, slam = _grids(m, erase_unknown=True)
    ok, _ = st.grade_bound(cost, slam, m, robot=(0.0, 0.0), reach_m=1.0)
    assert not ok


def test_unknown_within_the_lidars_reach_is_not_judged() -> None:
    # A still robot's /map holds one scan while the costmap clears from live scans: near the
    # robot the two disagree for a reason that is not the filter's.
    m = _room_mask(60, 60, 0.1)
    cost, slam = _grids(m, erase_unknown=True)
    ok, lines = st.grade_bound(cost, slam, m, robot=(0.0, 0.0), reach_m=100.0)
    assert ok, lines


def test_a_resolution_mismatch_fails_the_bound() -> None:
    m = _room_mask(60, 60, 0.1)
    cost, slam = _grids(m)
    cost.res = 0.05
    ok, lines = st.grade_bound(cost, slam, m)
    assert not ok and "resolution" in lines[0]


def test_covered_counts_known_cells_inside_the_bound_only() -> None:
    m = _room_mask(10, 10, 0.1)
    known = [0] * 100                        # the whole grid is known, border included
    slam = st.Grid(width=10, height=10, res=0.1, ox=m.ox, oy=m.oy, data=known)
    assert st.covered_m2(slam, m) == pytest.approx(st.allowed_area_m2(m))
    assert st.allowed_area_m2(m) == pytest.approx(64 * 0.01)


# ── the supervisor: battery, and stop -> wait paused -> return ───────────────

def _feed(s: st.Supervisor, t0: float, t1: float, mv: float, dt: float = 1.0) -> None:
    t = t0
    while t <= t1:
        s.battery(t, mv)
        t += dt


def test_one_sag_is_not_a_threshold() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, sustain_s=30)
    _feed(s, 0, 40, 7900)
    s.battery(41, 7700)  # one sample under motor load
    assert s.step(41) == [] and s.state == st.EXPLORING


def test_the_window_must_be_full_before_it_decides() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, sustain_s=30)
    _feed(s, 0, 10, 7800)
    assert s.step(10) == [] and s.state == st.EXPLORING


def test_the_return_threshold_stops_then_waits_for_the_explorer_then_sends_the_mark() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, sustain_s=30)
    _feed(s, 0, 31, 7830)
    acts = s.step(31, "exploration_in_progress")
    assert ("stop_explore",) in acts and s.state == st.STOPPING
    assert s.step(32, "exploration_in_progress") == []   # it has not said it stopped
    assert s.step(33, "exploration_paused") == []        # said it, but cancels AFTER saying it
    assert s.step(34.5, "exploration_paused") == []
    acts = s.step(35.0, "exploration_paused")            # the settle after the status
    assert ("send_mark",) in acts and s.state == st.RETURNING


def test_the_stop_wait_floor_sends_the_mark_without_a_status() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, sustain_s=30)
    _feed(s, 0, 31, 7830)
    s.step(31, "")
    assert s.step(35, "") == []
    assert ("send_mark",) in s.step(36.1, "")


def test_exploration_complete_returns_without_stopping_the_explorer_again() -> None:
    # A second stop made the explorer cancel every goal again -- the return goal among them.
    s = st.Supervisor(return_mv=7835, hard_mv=7750)
    _feed(s, 0, 40, 8100)
    acts = s.step(40, "exploration_complete")
    assert ("stop_explore",) not in acts and "exploration_complete" in s.reason
    assert s.step(41, "exploration_complete") == []
    assert ("send_mark",) in s.step(42, "exploration_complete")


def test_the_time_box_returns_like_the_battery() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, time_box_s=60)
    _feed(s, 0, 70, 8100)
    assert s.step(0, "") == []
    acts = s.step(61, "")
    assert ("stop_explore",) in acts and "time box" in s.reason


def test_an_exhausted_area_returns_like_the_battery() -> None:
    # The explorer cannot be told to avoid a region, so an exhausted area ends the segment:
    # the next segment's mask keeps the area out.
    s = st.Supervisor(return_mv=7835, hard_mv=7750)
    _feed(s, 0, 40, 8100)
    s.end_exploring("exhausted: area x 1.0..2.0 m, y 0.0..1.0 m")
    acts = s.step(40, "exploration_in_progress")
    assert ("stop_explore",) in acts and s.state == st.STOPPING
    assert "exhausted: area x 1.0..2.0 m" in s.reason
    assert ("send_mark",) in s.step(46, "")


def test_ending_exploring_after_it_has_ended_changes_nothing() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, time_box_s=10)
    _feed(s, 0, 20, 8100)
    s.step(0, "")
    s.step(11, "")
    s.end_exploring("exhausted: somewhere")
    assert "time box" in s.reason and s.state == st.STOPPING


def test_an_exhausted_area_is_a_square_polygon_in_the_segments_frame() -> None:
    c = st.AreaCounter(2)
    assert c.add(1.2, 0.3) is None
    name = c.add(1.9, 0.1)
    assert name == "area x 1.0..2.0 m, y 0.0..1.0 m"
    assert c.exhausted_polygons() == [(name, ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)))]


def test_the_return_result_decides_at_mark_or_failed() -> None:
    for result, want in (("SUCCEEDED", st.AT_MARK), ("ABORTED", st.RETURN_FAILED)):
        s = st.Supervisor(return_mv=7835, hard_mv=7750)
        _feed(s, 0, 40, 8100)
        s.step(40, "exploration_complete")
        s.step(42, "exploration_complete")
        assert s.state == st.RETURNING
        assert s.step(42, "exploration_complete", None) == []
        s.step(43, "exploration_complete", result)
        assert s.state == want


def test_the_hard_stop_wins_from_any_state_including_returning() -> None:
    s = st.Supervisor(return_mv=7835, hard_mv=7750, sustain_s=30)
    _feed(s, 0, 31, 7830)
    s.step(31, "")
    s.step(37, "")
    assert s.state == st.RETURNING
    _feed(s, 38, 80, 7740)
    acts = s.step(80, "")
    assert s.state == st.HARD_STOP
    assert ("stop_explore",) in acts and ("cancel_nav",) in acts
    assert s.step(81, "") == []  # once


def test_the_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        st.Supervisor(return_mv=7750, hard_mv=7750)


# ── exhausted areas ─────────────────────────────────────────────────────────

def test_n_recoveries_in_one_area_exhaust_it_once_and_name_it() -> None:
    c = st.AreaCounter(limit=3)
    assert c.add(1.2, 0.4) is None and c.add(1.7, 0.1) is None
    name = c.add(1.1, 0.9)
    assert name == "area x 1.0..2.0 m, y 0.0..1.0 m"
    assert c.add(1.5, 0.5) is None           # named once
    assert c.exhausted == [name]


def test_recoveries_in_different_areas_do_not_add_up() -> None:
    c = st.AreaCounter(limit=2)
    assert c.add(0.5, 0.5) is None and c.add(-0.5, 0.5) is None
    assert c.exhausted == []


# ── ALIGNED: the live map is in the saved map's frame ───────────────────────

def _walls(cells, n: int = 100, res: float = 0.05, ox: float = 0.0, oy: float = 0.0,
           free=()) -> st.Grid:
    """A grid with 100 on `cells` (ix, iy), 0 on `free`, -1 elsewhere."""
    data = [-1] * (n * n)
    for ix, iy in free:
        data[iy * n + ix] = 0
    for ix, iy in cells:
        data[iy * n + ix] = 100
    return st.Grid(width=n, height=n, res=res, ox=ox, oy=oy, data=data)


def _square(lo: int = 20, hi: int = 80):
    return ({(x, lo) for x in range(lo, hi)} | {(x, hi) for x in range(lo, hi)}
            | {(lo, y) for y in range(lo, hi)} | {(hi, y) for y in range(lo, hi)})


def test_alignment_passes_in_frame_and_fails_when_shifted() -> None:
    ref = _walls(_square())
    ok, line = st.align(_walls(_square()), ref)
    assert ok, line
    shifted = _walls(_square(), ox=0.30)             # booted 0.30 m off the mark
    ok, line = st.align(shifted, ref)
    assert ok is False and "want >= 80%" in line


def test_alignment_with_too_little_wall_is_not_proven_never_misaligned() -> None:
    ref = _walls({(x, 10) for x in range(10, 20)}, n=50)
    ok, line = st.align(ref, ref)
    assert ok is None and "only 10" in line


def test_alignment_follows_the_boot_rotation() -> None:
    res, ox, oy, n = 0.05, -3.0, -3.0, 120
    ell = {(x, 30) for x in range(20, 100)} | {(20, y) for y in range(30, 90)}
    ref = _walls(ell, n=n, res=res, ox=ox, oy=oy)
    d = math.radians(128.6)
    turned = set()
    for ix, iy in ell:
        wx, wy = ox + (ix + 0.5) * res, oy + (iy + 0.5) * res
        nx, ny = math.cos(d) * wx - math.sin(d) * wy, math.sin(d) * wx + math.cos(d) * wy
        cx, cy = math.floor((nx - ox) / res), math.floor((ny - oy) / res)
        if 0 <= cx < n and 0 <= cy < n:
            turned.add((cx, cy))
    live = _walls(turned, n=n, res=res, ox=ox, oy=oy)
    assert st.align(live, ref)[0] is False                  # unturned: not aligned
    ok, line = st.align(live, ref, delta_deg=128.6)
    assert ok is True, line


def test_duplicate_walls_on_the_saved_maps_free_space_fail_a_resume() -> None:
    inside = {(x, y) for x in range(21, 80) for y in range(21, 80)}
    ref = _walls(_square(), free=inside)
    ok, line = st.align(_walls(_square()), ref, max_new_frac=0.05)
    assert ok and "free space: 0 " in line
    doubled = _walls(_square() | {(x, 50) for x in range(25, 75)})
    ok, line = st.align(doubled, ref, max_new_frac=0.05)
    assert not ok and "DUPLICATE WALLS" in line


def test_new_territory_is_not_counted_as_disagreement() -> None:
    west = ({(x, 20) for x in range(20, 50)} | {(x, 80) for x in range(20, 50)}
            | {(20, y) for y in range(20, 80)})
    ref = _walls(west, free={(x, y) for x in range(21, 50) for y in range(21, 80)})
    east = ({(x, 20) for x in range(50, 90)} | {(x, 80) for x in range(50, 90)}
            | {(89, y) for y in range(20, 80)})
    ok, line = st.align(_walls(west | east), ref, max_new_frac=0.05)
    assert ok is True and "new territory" in line, line


def test_a_saved_map_reads_back_from_its_pgm() -> None:
    g = _walls({(1, 1)}, n=3, free={(0, 0)})
    back = st.grid_from_pgm(st.grid_to_pgm(g), res=g.res, ox=g.ox, oy=g.oy)
    assert back.data == g.data
