# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Pause / resume mapping: Swarmada pause/resume → Nav2 goal cancel-and-hold +
re-dispatch. Pause KEEPS the target (unlike command_stop, which drops it), and the
base's rest is CONFIRMED from odom (is_stopped), never inferred. Also covers the
NavigateToPose GoalStatus → TaskState derivation used for completion streaming."""

from __future__ import annotations

from fleet_adapter_ros2 import task
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

# action_msgs/GoalStatus codes.
_GS_SUCCEEDED, _GS_CANCELED, _GS_ABORTED = 4, 5, 6


def test_pause_keeps_target_while_stop_drops_it() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_move("amr-1", 5.0, 2.0)
    assert b.pending_goal("amr-1") is not None
    b.command_pause("amr-1")
    assert b.pending_goal("amr-1") is not None  # KEPT so command_resume can re-dispatch
    b.command_stop("amr-1")
    assert b.pending_goal("amr-1") is None       # stop (RTL/estop path) drops the target


def test_task_state_tracks_goal_status() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_move("amr-1", 1.0, 1.0)
    assert b.task_state("amr-1") == task.TASK_RUNNING  # in-flight
    b.on_goal_result(_GS_SUCCEEDED)
    assert b.task_state("amr-1") == task.TASK_SUCCEEDED
    b.on_goal_result(_GS_ABORTED)
    assert b.task_state("amr-1") == task.TASK_FAILED
    b.on_goal_result(_GS_CANCELED)
    assert b.task_state("amr-1") == task.TASK_CANCELLED
    b.command_move("amr-1", 2.0, 2.0)  # a new goal is in-flight again
    assert b.task_state("amr-1") == task.TASK_RUNNING


def test_sim_pause_halts_and_resume_moves() -> None:
    r = SimulatedRobot()
    r.spawn("amr-1")
    r.command_move("amr-1", 5.0, 0.0)
    assert not r.is_stopped("amr-1")
    r.command_pause("amr-1")
    assert r.is_stopped("amr-1")      # cancel + hold → confirmed at rest
    r.command_resume("amr-1")
    assert not r.is_stopped("amr-1")  # goal re-dispatched
