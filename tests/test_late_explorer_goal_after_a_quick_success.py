# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A goal that supersedes a segment keeps cancelling other senders' goals past its SUCCEEDED.

A Navigate that supersedes an explore segment tells the explorer to stop and sends its own goal,
which cancels every other goal the server lists then. An explorer goal sent just before the stop
and accepted only after the Navigate's goal had succeeded (the robot was already at its target)
was never cancelled: the guard lifted at SUCCEEDED, the action was over, and the robot drove
toward the explorer's frontier with no action at all. Now the guard stays on past that
SUCCEEDED until the segment's processes have been ended and the explorer's settle has passed,
or until a new goal, an explorer release or a hold ends it earlier. After that, as before, a
goal another sender gives an idle robot is left alone.
"""

from __future__ import annotations

import json

import pytest
from test_explore_adapter import StepClock, _assign
from test_ros2_explore import _stack as _ros2_stack

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.ros2_explore import EXPLORER_SETTLE_S

ROBOT = "amr-1"
SEGMENT = json.dumps({"operation": "explore-floor", "bound_ref": "sim-room",
                      "max_speed_mps": 0.2}).encode()
NAVIGATE = b'{"x": 1, "y": 2}'
EXPLORERS = b"\x07" * 16              # a goal the segment's explorer sent while it drove
LATE = b"\x0b" * 16                   # one it sent just before its stop, accepted late
OPERATOR = b"\x0c" * 16               # a goal an operator sent
NEXT_EXPLORERS = b"\x0d" * 16         # a goal of a later segment's own explorer


def _rig(tmp_path, monkeypatch):
    """The Nav2 binding, with the ROS 2 exploration stack, through the adapter, on a clock the
    test moves. The jobs' own threads are not run."""
    monkeypatch.setattr(explore.ExploreJob, "run", lambda self: None)
    st, _ros, _shell, b, _sent = _ros2_stack(tmp_path)
    b._explore = st
    b._status_sub = object()          # subscribed to the server's goal status (start())
    b.cancelled = []                  # the goal ids cancelled by id
    b._cancel_running_goal = b.cancelled.append
    now = [100.0]
    b._now = lambda: now[0]
    a = Adapter(pb, None, b, ROBOT, job_clock=StepClock(),
                artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab")
    return a, b, st, now


def _navigate_that_succeeds_at_once(a, b, token: int, action: str) -> None:
    assert _assign(a, "Navigate", NAVIGATE, token=token, action=action)[0].accepted
    b.on_goal_result(4, epoch=b._goal_epoch)            # SUCCEEDED: the robot was at its target
    a._maybe_emit_task_progress()
    assert a._current_action == "", "the action is over: the robot has none"


def _segment_superseded_by_a_quick_navigate(tmp_path, monkeypatch):
    a, b, st, now = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Custom", SEGMENT, token=1, action="seg-1")[0].accepted
    assert b.release_hold_for_explorer(ROBOT, a._job.may_move)    # its explorer drives
    b.on_nav_status([(EXPLORERS, 2)])
    _navigate_that_succeeds_at_once(a, b, token=2, action="t-2")
    b.cancelled.clear()
    return a, b, st, now


def test_a_late_explorer_goal_after_the_superseding_goal_succeeded_is_cancelled(
        tmp_path, monkeypatch) -> None:
    _a, b, _st, _now = _segment_superseded_by_a_quick_navigate(tmp_path, monkeypatch)
    b.on_nav_status([(LATE, 1)])
    assert b.cancelled == [LATE], "a late explorer goal drove a robot with no action"


def test_it_is_cancelled_after_the_segment_is_ended_until_the_settle_has_passed(
        tmp_path, monkeypatch) -> None:
    _a, b, st, now = _segment_superseded_by_a_quick_navigate(tmp_path, monkeypatch)
    st.end_segment()
    now[0] += EXPLORER_SETTLE_S / 2
    b.on_nav_status([(LATE, 1)])
    assert b.cancelled == [LATE], "a late explorer goal drove a robot with no action"


def test_once_the_settle_has_passed_an_operators_goal_is_left_alone(tmp_path, monkeypatch) -> None:
    _a, b, st, now = _segment_superseded_by_a_quick_navigate(tmp_path, monkeypatch)
    st.end_segment()
    now[0] += EXPLORER_SETTLE_S
    b.on_nav_status([(OPERATOR, 2)])
    assert b.cancelled == [], "an idle robot's goal from another sender was cancelled"


@pytest.mark.parametrize("before", ["no-segment", "a-segment-ended-and-settled"])
def test_a_goal_that_superseded_no_live_explorer_leaves_an_operators_goal_alone_once_it_succeeded(
        tmp_path, monkeypatch, before) -> None:
    a, b, st, now = _rig(tmp_path, monkeypatch)
    if before == "a-segment-ended-and-settled":
        assert _assign(a, "Custom", SEGMENT, token=1, action="seg-1")[0].accepted
        assert b.release_hold_for_explorer(ROBOT, a._job.may_move)
        st.end_segment()
        now[0] += EXPLORER_SETTLE_S
    _navigate_that_succeeds_at_once(a, b, token=2, action="t-2")
    b.cancelled.clear()
    b.on_nav_status([(OPERATOR, 2)])
    assert b.cancelled == []


@pytest.mark.parametrize("ender", ["a-new-goal", "an-explorer-release", "a-hold"])
def test_inside_the_window_a_new_goal_an_explorer_release_or_a_hold_works_as_before(
        tmp_path, monkeypatch, ender) -> None:
    a, b, _st, _now = _segment_superseded_by_a_quick_navigate(tmp_path, monkeypatch)
    if ender == "a-new-goal":
        assert _assign(a, "Navigate", b'{"x": 3, "y": 4}', token=3, action="t-3")[0].accepted
        ours = b._claimed[b._goal_epoch]
        b.cancelled.clear()
        b.on_nav_status([(ours, 2), (LATE, 1)])
        assert b.cancelled == [LATE]                    # its own goal runs; another's does not
    elif ender == "an-explorer-release":
        assert _assign(a, "Custom", SEGMENT, token=3, action="seg-3")[0].accepted
        assert b.release_hold_for_explorer(ROBOT, a._job.may_move)
        b.cancelled.clear()
        b.on_nav_status([(NEXT_EXPLORERS, 2)])
        assert b.cancelled == []                        # the new segment's explorer drives
    else:
        b.command_hold(ROBOT)
        b.cancelled.clear()
        b.on_nav_status([(LATE, 2)])
        assert b.cancelled == [LATE]                    # held: every goal is cancelled
