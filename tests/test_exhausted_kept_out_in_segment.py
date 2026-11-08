# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""An exhausted area is kept out of the running segment, and exploring goes on.

N recovery episodes in one area used to end the whole segment: the robot returned and every
other frontier waited for the next segment, and so for the next charge. Now the robot is
stopped and drives back to the mark with the area not yet drawn in (the bound, the keep-out set
and the keep-outs measured by hand stay in force); on the mark the whole area is drawn into the
running segment's keep-out mask, the mask is reloaded, the bound is proven again in the costmap
with the robot still, and only then does the explorer start again, toward what is left.

The segment ends as before only when no frontier is left, the battery says return, the time box
runs out -- or the bound cannot be proven again, and then nothing more moves. No area within
0.35 m of the start mark is exhausted.
"""

from __future__ import annotations

import json

from test_explore_job import FakeClock, _job

from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SIM_THETA0_DEG, SimulatedExploreStack

AREA = "area x 1.0..2.0 m, y 0.0..1.0 m"
OTHER_AREA = "area x -2.0..-1.0 m, y 2.0..3.0 m"
FIRST = [(5, 1.2, 0.3), (6, 1.4, 0.6), (7, 1.9, 0.1)]
SECOND = [(20, -1.5, 2.2), (21, -1.4, 2.5), (22, -1.2, 2.8)]


def _stack(clock, recoveries=FIRST, **kw):
    # This boot's frame is the bound's, so the areas read as named.
    return SimulatedExploreStack(clock=clock, boot_yaw_deg=SIM_THETA0_DEG, explore_seconds=60,
                                 recoveries=recoveries, **kw)


def _run(tmp_path, stack, clock, doc=None):
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"exhausted_after_recoveries": 3, **(doc or {})})
    job.run()
    return stack, emits


def _starts(stack) -> list[int]:
    return [i for i, step in enumerate(stack.log) if step == "start_explorer"]


def test_one_exhausted_area_and_exploring_goes_on_to_the_frontiers_left(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock), clock)
    state, message, _, _ = emits.last
    assert state == "SUCCEEDED", emits.updates
    # The explorer ran to its own end: no frontier was left behind by the area.
    assert "frontier: none-left" in message and f"exhausted: {AREA}" in message
    assert len(_starts(stack)) == 2
    assert any("kept out of this segment" in m and AREA in m for m in emits.messages())
    assert not any(m.startswith("explore-floor: returning: exhausted") for m in emits.messages())
    seg = json.loads((tmp_path / "lab" / "explore-first" / "segment.json").read_text())
    assert [a["name"] for a in seg["exhausted_areas"]] == [AREA]    # and kept out of the next


def test_the_reload_proves_the_bound_again_before_the_next_goal(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock), clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    first, again = _starts(stack)
    after = stack.log[first + 1:again]
    want = ["stop_explorer", "hold", "load_keepout_mask", "reload_keepout_mask",
            "reset_costmap", "read_costmap"]
    positions = [after.index(step) for step in want]
    assert positions == sorted(positions), after
    proven = next(m for m in emits.messages() if "kept out of this segment" in m)
    assert "not lethal: 0" in proven


def test_the_area_is_closed_whole_once_the_robot_has_left_it(tmp_path) -> None:
    # The robot stopped at (1.9, 0.1), its last recovery. It drives back to the mark, and only
    # then is the whole area kept out -- where it stood included -- before anything explores
    # again.
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock), clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    for x, y in ((1.5, 0.75), (1.9, 0.1), (1.2, 0.05), (1.5, 0.5)):
        assert stack.loaded_mask_keep_at(x, y) is True, (x, y)
    assert stack.loaded_mask_keep_at(0.5, 0.5) is False


def test_the_robot_leaves_for_the_mark_before_the_area_is_drawn_in(tmp_path) -> None:
    # Superseded order (a way out drawn and proven first, then the leave): a way out drawn
    # through the area left too little room for Nav2's recoveries (maintainer's ruling, 2026-10-02).
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock), clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    first, again = _starts(stack)
    after = stack.log[first + 1:again]
    leave = after.index("go_to_mark")
    assert "load_keepout_mask" not in after[:leave] and "reload_keepout_mask" not in after[:leave]
    want = ["hold", "go_to_mark", "load_keepout_mask", "reload_keepout_mask", "reset_costmap",
            "read_costmap"]
    i = 0
    for step in after:
        if i < len(want) and step == want[i]:
            i += 1
    assert i == len(want), after
    text = " | ".join(emits.messages())
    assert text.find("leaving " + AREA + " for the mark") < text.find("kept out of this segment (")


def test_a_robot_that_cannot_leave_the_area_stops_where_it_stands(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock, mark_result="ABORTED"), clock)
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert AREA in message and "could not leave" in message and "stopped where it stands" in message
    assert len(_starts(stack)) == 1


def test_a_bound_that_cannot_be_proven_again_refuses_motion(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock, reload_fails=True), clock)
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "bound not in force" in message and AREA in message
    assert "stopped where it stands" in message
    # The robot has left the area for the mark first; the proof on the mark failed, so the
    # explorer never started again and nothing more moves.
    assert len(_starts(stack)) == 1
    assert stack.log.index("go_to_mark") < stack.log.index("reload_keepout_mask")
    assert not stack.moving()


def test_a_second_exhausted_area_is_kept_out_the_same_way(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock, recoveries=FIRST + SECOND), clock)
    state, message, _, _ = emits.last
    assert state == "SUCCEEDED", emits.updates
    assert f"exhausted: {AREA}, {OTHER_AREA}" in message and "frontier: none-left" in message
    assert len(_starts(stack)) == 3
    # One reload an area: whole, once the robot is back on the mark.
    assert stack.log.count("reload_keepout_mask") == 2
    assert stack.loaded_mask_keep_at(1.5, 0.75) is True        # the first stays out
    assert stack.loaded_mask_keep_at(-1.8, 2.9) is True


def test_the_battery_still_ends_a_segment_after_an_area_is_kept_out(tmp_path) -> None:
    clock = FakeClock()
    stack = _stack(clock, start_mv=7400.0, drain_mv_per_s=8.0)
    stack, emits = _run(tmp_path, stack, clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    assert "frontier: remaining" in emits.last[1]
    assert any("returning: battery" in m for m in emits.messages())


def test_recoveries_at_the_mark_never_exhaust_it_or_reload_the_mask(tmp_path) -> None:
    clock = FakeClock()
    stack, emits = _run(tmp_path, _stack(clock, recoveries=[
        (5, 0.30, -0.20), (6, 0.32, -0.25), (7, 0.25, -0.20)]), clock)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    assert "exhausted: none" in emits.last[1]
    assert "reload_keepout_mask" not in stack.log and len(_starts(stack)) == 1
    assert any("start mark" in m and "never exhausted" in m for m in emits.messages())


def test_the_way_out_is_left_open_only_inside_the_area() -> None:
    m = steps.Mask(40, 40, 0.1, -1.0, -1.0, [False] * 1600)
    square = ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0))
    out = steps.keep_out_area(m, square, (1.5, 0.5), (0.0, 0.0), 0.35)
    assert steps._mask_keep_at(out, 1.85, 0.85) is True
    assert steps._mask_keep_at(out, 1.5, 0.5) is False         # where the robot stands
    assert steps._mask_keep_at(out, 1.05, 0.35) is False        # on its way to the mark
    assert steps._mask_keep_at(out, 0.5, 0.2) is False          # outside the area: unchanged
    walled = steps.Mask(40, 40, 0.1, -1.0, -1.0, [True] * 1600)
    assert steps._mask_keep_at(steps.keep_out_area(walled, square, (1.5, 0.5), (0.0, 0.0),
                                                   0.35), 1.5, 0.5) is True   # never cleared
