# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""An explore-floor segment starts only when every costmap a planner plans on carries the
keep-out filter.

Nav2 gives each costmap its own filters. The global costmap carries the planner's path; the local
costmap carries the controller's trajectory (TEB plans on it). A keep-out filter on the global
costmap alone lets the controller's trajectory enter a keep-out the path avoids, and the watch on
the controller's local plan then has to stop the robot where it stands: the segment ends failed,
not returned. So the segment-start proof reads, from the live nodes, the filters of both
costmaps, and refuses before anything moves -- naming the costmap -- if one lacks a keep-out
filter, has it disabled, or reads another filter-info topic than the other (two masks).
"""

from __future__ import annotations

import threading

import pytest

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

GLOBAL = "/global_costmap/global_costmap"
LOCAL = "/local_costmap/local_costmap"


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.t += seconds
        return event.is_set()


def _run(tmp_path, params=None, drop=None):
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, params=params, drop=drop)
    updates: list[tuple[str, str]] = []
    job = explore.ExploreJob(
        action_id="explore-first",
        doc={"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2},
        stack=stack, limits=stack.battery_limits(),
        artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab",
        clock=clock, emit=lambda state, message, pct=0, end_cause="": updates.append(
            (state.removeprefix("ACTION_STATE_"), message)))
    job.run()
    return stack, updates


def _refused_before_anything_moved(stack, updates) -> str:
    state, message = updates[-1]
    assert state == "FAILED", updates
    assert "no motion" in message
    for step in ("start_slam", "load_keepout_mask", "start_explorer", "go_to_mark"):
        assert step not in stack.log, (step, stack.log)
    assert not stack.moving()
    return message


def test_a_local_costmap_without_the_keepout_filter_refuses_the_segment(tmp_path) -> None:
    # What the robot's overlay gave the local costmap: no filters at all.
    stack, updates = _run(tmp_path, params={(LOCAL, "filters"): "not set"})
    message = _refused_before_anything_moved(stack, updates)
    assert "local costmap" in message and LOCAL in message
    assert "no keep-out filter" in message


def test_a_global_costmap_without_the_keepout_filter_refuses_the_segment(tmp_path) -> None:
    stack, updates = _run(tmp_path, params={(GLOBAL, "filters"): "[]"})
    message = _refused_before_anything_moved(stack, updates)
    assert "global costmap" in message and GLOBAL in message


def test_a_filter_that_is_not_a_keepout_filter_does_not_count(tmp_path) -> None:
    stack, updates = _run(tmp_path, params={
        (LOCAL, "filters"): "[speed_filter]",
        (LOCAL, "speed_filter.plugin"): "nav2_costmap_2d::SpeedFilter"})
    message = _refused_before_anything_moved(stack, updates)
    assert "local costmap" in message and "speed_filter" in message


def test_a_disabled_keepout_filter_refuses_the_segment(tmp_path) -> None:
    stack, updates = _run(tmp_path, params={(LOCAL, "keepout_filter.enabled"): "False"})
    message = _refused_before_anything_moved(stack, updates)
    assert "local costmap" in message and "disabled" in message


def test_two_costmaps_reading_two_filter_info_topics_refuse_the_segment(tmp_path) -> None:
    # One mask is loaded and proven; a costmap filtering from another topic is not held by it.
    stack, updates = _run(tmp_path, params={
        (LOCAL, "keepout_filter.filter_info_topic"): "/other_filter_info"})
    message = _refused_before_anything_moved(stack, updates)
    assert "/other_filter_info" in message and "/costmap_filter_info" in message


def test_both_costmaps_filtered_is_said_before_slam_and_the_segment_runs(tmp_path) -> None:
    stack, updates = _run(tmp_path)
    assert updates[-1][0] == "SUCCEEDED", updates[-1]
    text = " | ".join(m for _, m in updates)
    said = text.find("keep-out filter on every planning costmap")
    assert 0 <= said < text.find("starting SLAM"), text
    assert "global costmap" in text[said:] and "local costmap" in text[said:]
    reads = [c for c in stack.param_reads if c[0] in (GLOBAL, LOCAL)]
    assert (GLOBAL, "filters") in reads and (LOCAL, "filters") in reads


def test_a_dropped_filter_reply_is_read_again(tmp_path) -> None:
    _, updates = _run(tmp_path, drop={"read_param": 3})
    assert updates[-1][0] == "SUCCEEDED", updates[-1]


@pytest.mark.parametrize("node", [GLOBAL, LOCAL])
def test_a_filter_read_never_answered_refuses_after_the_ruled_retries(tmp_path, node) -> None:
    stack, updates = _run(tmp_path, params={(node, "filters"): None})
    message = _refused_before_anything_moved(stack, updates)
    assert "NO ANSWER" in message


# ── the same proof on the ROS 2 stack: read from the live nodes ──────────────

def _ros2_job(tmp_path, local=None):
    from test_explore_job import Emits
    from test_ros2_explore import _bound_dir, _stack, keepout_filter_answers

    _bound_dir(tmp_path)
    st, ros, shell, _, _ = _stack(tmp_path)
    ros.t = 10.0
    ros.answers["/controller_server/get_parameters"] = {"values": [
        {"type": 3, "double_value": 0.2}]}
    ros.answers["/velocity_smoother/get_parameters"] = {"values": [
        {"type": 8, "double_array_value": [0.2, 0.0, 1.0]}]}
    ros.answers[f"{GLOBAL}/get_parameters"] = keepout_filter_answers()
    ros.answers[f"{LOCAL}/get_parameters"] = keepout_filter_answers(local)
    emits = Emits()
    explore.ExploreJob(
        action_id="explore-first",
        doc={"operation": "explore-floor", "bound_ref": "room-1", "max_speed_mps": 0.2},
        stack=st, limits=st.battery_limits(), artifacts=explore.ArtifactDir(str(tmp_path), "m"),
        namespace="lab", clock=st.clock, emit=emits).run()
    return ros, shell, emits


def test_on_the_ros2_stack_the_local_costmap_is_read_from_its_own_node(tmp_path) -> None:
    # The local costmap as the robot ran it on 2026-10-02: no filters declared.
    ros, shell, emits = _ros2_job(tmp_path, local={"filters": {"type": 0}})
    state, message = emits.last[:2]
    assert state == "FAILED" and "local costmap" in message and "no keep-out filter" in message
    asked = [f["names"] for s, _, f in ros.calls if s == f"{LOCAL}/get_parameters"]
    assert ["filters"] in asked
    assert not any(detach for _, detach, _ in shell.runs), "something was launched"


def test_on_the_ros2_stack_both_costmaps_filtered_passes_the_proof(tmp_path) -> None:
    _, _, emits = _ros2_job(tmp_path)
    text = " | ".join(emits.messages())
    assert ("keep-out filter on every planning costmap: global costmap keepout_filter on "
            "/costmap_filter_info; local costmap keepout_filter on /costmap_filter_info") in text
