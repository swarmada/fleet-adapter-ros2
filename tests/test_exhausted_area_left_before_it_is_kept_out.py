# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""An exhausted area is left for the mark first, and kept out whole once the robot is there.

A way out drawn through an exhausted area -- a strip along the line to the mark -- left the
robot too little room for Nav2's own recoveries. On a robot day the retreat's first goal failed
TEB's feasibility check, the navigator's BackUp completed, and the Spin after it was refused
("Collision Ahead"): the goal aborted with the robot still in the area. No strip wide enough to
turn in after a back-up left most of the area kept out. So the robot now drives back to the
mark with nothing of the exhausted area drawn in, and the area is drawn in whole, reloaded and
proven on the mark before the explorer starts again.

What stays in force on the way: the room's bound, the keep-out set and the keep-outs measured
by hand. Every plan the planner or the controller publishes during the leave is held against
them, as while exploring, and one that enters them stops the robot where it stands. A plan
through the exhausted area itself is not stopped on the way: that is the ruled cost.
"""

from __future__ import annotations

import math

from test_explore_job import FakeClock, _job

from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SIM_THETA0_DEG, SimulatedExploreStack

AREA = "area x 1.0..2.0 m, y 0.0..1.0 m"
# The third recovery, at (1.9, 0.1), exhausts x 1..2, y 0..1.
RECOVERIES = [(5, 1.2, 0.3), (6, 1.4, 0.6), (7, 1.9, 0.1)]
# A keep-out measured by hand that reaches into the exhausted square (x 1..1.6, y 0.8..1.0).
HEARTH = steps.Keepout("hearth", "a raised slab below the lidar plane",
                       ((0.6, 0.8), (1.6, 0.8), (1.6, 2.0), (0.6, 2.0)))


def _line(a, b, step=0.05):
    n = max(1, math.ceil(math.dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def _run(tmp_path, *, leave_plan=None, hand=(HEARTH,), **kw):
    """A segment that exhausts x 1..2, y 0..1, every mask it loads kept. ``leave_plan``: one
    path the planner publishes while the robot leaves the area for the mark."""
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=SIM_THETA0_DEG, explore_seconds=60,
                                  recoveries=RECOVERIES, hand_keepouts=hand, **kw)
    loaded: list[steps.Mask] = []
    load, plans = stack.load_keepout_mask, stack.plans
    sent = []

    def keep_each(mask, delta_deg):
        loaded.append(mask)
        return load(mask, delta_deg)

    def plans_while_leaving():
        out = plans()
        if (leave_plan is not None and not sent and "go_to_mark" in stack.log
                and "load_keepout_mask" not in stack.log[stack.log.index("skip_area"):]):
            sent.append(1)
            out.append(leave_plan)
        return out

    stack.load_keepout_mask, stack.plans = keep_each, plans_while_leaving
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"exhausted_after_recoveries": 3})
    job.run()
    return stack, emits, loaded, sent


def test_nothing_of_the_area_is_drawn_in_until_the_robot_is_on_the_mark(tmp_path) -> None:
    stack, emits, loaded, _ = _run(tmp_path)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    i, j = stack.log.index("skip_area"), stack.log.index("go_to_mark")
    assert "load_keepout_mask" not in stack.log[i:j]
    assert any(f"leaving {AREA} for the mark" in m for m in emits.messages())
    # The mask in force during the leave is the one proven before motion: the area is open.
    before = loaded[0]
    assert steps._mask_keep_at(before, 1.9, 0.1) is False
    assert steps._mask_keep_at(before, 1.5, 0.5) is False


def test_the_whole_area_is_proven_on_the_mark_before_exploring_again(tmp_path) -> None:
    stack, emits, loaded, _ = _run(tmp_path)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    log = stack.log
    mark = log.index("go_to_mark")
    load = log.index("load_keepout_mask", mark)
    reload = log.index("reload_keepout_mask", load)
    reset = log.index("reset_costmap", reload)
    read = log.index("read_costmap", reset)
    again = [k for k, s in enumerate(log) if s == "start_explorer"][1]
    assert mark < load < reload < reset < read < again
    whole = loaded[-1]
    for x, y in ((1.9, 0.1), (1.5, 0.5), (1.05, 0.05), (1.95, 0.75)):
        assert steps._mask_keep_at(whole, x, y) is True, (x, y)


def test_a_failed_proof_on_the_mark_ends_the_segment(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=SIM_THETA0_DEG, explore_seconds=60,
                                  recoveries=RECOVERIES, hand_keepouts=(HEARTH,))
    reload = stack.reload_keepout_mask
    calls = []

    def refuse_on_the_mark():
        calls.append(1)
        return False if "go_to_mark" in stack.log else reload()

    stack.reload_keepout_mask = refuse_on_the_mark
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"exhausted_after_recoveries": 3})
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED", emits.updates
    assert "bound not in force after keeping out" in message and AREA in message
    assert "stopped where it stands" in message
    assert stack.log.count("start_explorer") == 1
    assert not stack.moving()


def test_a_plan_into_the_hearth_while_leaving_stops_the_robot(tmp_path) -> None:
    stack, emits, _, sent = _run(tmp_path, leave_plan=_line((1.1, 0.5), (1.1, 1.6)))
    state, message, _, _ = emits.last
    assert sent, "the plan was not published during the leave"
    assert state == "FAILED", emits.updates
    assert "keep-out 'hearth'" in message and "stopped where it stands" in message
    assert stack.log.count("start_explorer") == 1


def test_a_plan_out_of_the_room_while_leaving_stops_the_robot(tmp_path) -> None:
    _, emits, _, sent = _run(tmp_path, leave_plan=_line((0.0, 0.0), (0.0, -3.0)))
    state, message, _, _ = emits.last
    assert sent and state == "FAILED", emits.updates
    assert "the room's bound" in message and "stopped where it stands" in message


def test_a_plan_through_the_exhausted_area_while_leaving_is_not_stopped(tmp_path) -> None:
    # The ruled cost: on the way out the exhausted area itself is not kept out.
    _, emits, _, sent = _run(tmp_path, leave_plan=_line((1.9, 0.1), (1.5, 0.5)))
    assert sent and emits.last[0] == "SUCCEEDED", emits.updates


def test_the_hearth_and_the_bound_are_kept_out_in_every_mask_loaded(tmp_path) -> None:
    _, emits, loaded, _ = _run(tmp_path)
    assert emits.last[0] == "SUCCEEDED", emits.updates
    assert len(loaded) >= 2
    for m in loaded:
        for x, y in ((1.1, 1.4), (1.3, 0.9), (0.9, 1.9)):        # the hearth, in the square too
            assert steps._mask_keep_at(m, x, y) is True, (x, y)
        assert steps._mask_keep_at(m, 1.0, -1.45) is True        # the room's bound (its edge)
