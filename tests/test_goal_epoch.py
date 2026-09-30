# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A superseded goal's result must not be applied to the goal that replaced it.

The defect these cover, observed against a live stack: `on_goal_result()` wrote
`_goal_status` with no goal identity, so the result callback of an already-cancelled goal
overwrote the status of the goal dispatched after it. The adapter then reported that
terminal state against whatever action was *currently* assigned — `SUCCEEDED` at
`progress=100` for an action assigned 192 ms earlier whose target was 6 m away.

On the wire that is indistinguishable from real completion: the control plane marks the
action done and is free to dispatch the next one, while the robot is still driving toward
the abandoned goal with its lease released. It surfaced only as an intermittent conformance
failure (~11% of runs), which is a very quiet symptom for a correctness defect.

These run at $0 — no ROS, no Nav2 node. `command_move()` bumps the epoch whether or not a
node is live, which is what makes the ordering testable without a runtime.
"""

from __future__ import annotations

from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
_GS_SUCCEEDED, _GS_ABORTED, _GS_CANCELED = 4, 6, 5


def test_stale_goal_result_does_not_overwrite_the_current_goal() -> None:
    b = Nav2Binding()
    b.command_move(ROBOT, 1.0, 0.0)
    first = b._goal_epoch
    b.command_move(ROBOT, 2.0, 0.0)          # supersedes it
    assert b.task_state(ROBOT) == task_map.TASK_RUNNING

    # The FIRST goal's result arrives late. It must be dropped.
    b.on_goal_result(_GS_SUCCEEDED, epoch=first)
    assert b.task_state(ROBOT) == task_map.TASK_RUNNING, (
        "a superseded goal's SUCCEEDED was applied to the goal that replaced it")

    # The CURRENT goal's result is still adopted.
    b.on_goal_result(_GS_SUCCEEDED, epoch=b._goal_epoch)
    assert b.task_state(ROBOT) == task_map.TASK_SUCCEEDED


def test_stale_abort_and_cancel_are_dropped_too() -> None:
    # Not just SUCCEEDED: a stale ABORTED would report a failure that never happened, and a
    # stale CANCELED would mark a live goal cancelled.
    for stale_status in (_GS_ABORTED, _GS_CANCELED):
        b = Nav2Binding()
        b.command_move(ROBOT, 1.0, 0.0)
        stale = b._goal_epoch
        b.command_move(ROBOT, 2.0, 0.0)
        b.on_goal_result(stale_status, epoch=stale)
        assert b.task_state(ROBOT) == task_map.TASK_RUNNING


def test_resume_supersedes_the_cancelled_goals_result() -> None:
    # command_pause cancels the goal, which will deliver CANCELED. command_resume
    # re-dispatches, and that stale CANCELED must not mark the resumed goal cancelled.
    b = Nav2Binding()
    b.command_move(ROBOT, 1.0, 0.0)
    paused = b._goal_epoch
    b.command_pause(ROBOT)
    b.command_resume(ROBOT)
    assert b._goal_epoch != paused

    b.on_goal_result(_GS_CANCELED, epoch=paused)
    assert b.task_state(ROBOT) == task_map.TASK_RUNNING


def test_epoch_none_is_still_accepted_as_a_test_seam() -> None:
    # on_goal_result's positional signature is unchanged; callers that do not identify a
    # goal (the existing unit tests) keep working.
    b = Nav2Binding()
    b.command_move(ROBOT, 1.0, 0.0)
    b.on_goal_result(_GS_SUCCEEDED)
    assert b.task_state(ROBOT) == task_map.TASK_SUCCEEDED
