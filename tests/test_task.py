# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FleetTask → Nav2 task translation (sub-item 3, targets C6.4/C7).

Pure: no ROS install, no generated protobufs — the mappings operate on the
AssignAction's ``destination`` string / ``payload_json`` bytes and on GoalStatus ints.
"""

from __future__ import annotations

import json

from fleet_adapter_ros2 import task
from fleet_adapter_ros2.nav2_binding import Nav2Binding


# ── goal_from_assign ──────────────────────────────────────────────────────────

def test_goal_from_payload_coordinates() -> None:
    payload = json.dumps({"x": 5.0, "y": -3.0, "yaw": 1.57, "frame": "map"}).encode()
    g = task.goal_from_assign("dock-7", payload)
    assert (g.x, g.y, g.yaw, g.frame) == (5.0, -3.0, 1.57, "map")
    assert g.label == "dock-7"  # destination retained as a label


def test_goal_payload_takes_precedence_over_destination_string() -> None:
    # Explicit coordinates in the payload win over a coordinate-looking destination.
    g = task.goal_from_assign("1,1", json.dumps({"x": 9.0, "y": 8.0}).encode())
    assert (g.x, g.y) == (9.0, 8.0)


def test_goal_from_destination_coordinate_string() -> None:
    g = task.goal_from_assign("2.5,4.0,0.5")
    assert (g.x, g.y, g.yaw) == (2.5, 4.0, 0.5)
    assert g.label == ""  # coordinates, not a named location


def test_goal_from_named_location() -> None:
    # A non-coordinate destination is a named location the map server resolves.
    g = task.goal_from_assign("charging-station-A")
    assert g.label == "charging-station-A"
    assert (g.x, g.y) == (0.0, 0.0)


def test_goal_ignores_opaque_or_garbage_payload() -> None:
    # Non-numeric / non-JSON payloads never crash and never leak into coordinates.
    g = task.goal_from_assign("bay-2", b"\xff\xfenot json")
    assert g.label == "bay-2" and (g.x, g.y) == (0.0, 0.0)
    g2 = task.goal_from_assign("bay-2", json.dumps({"x": "over-there"}).encode())
    assert g2.label == "bay-2" and (g2.x, g2.y) == (0.0, 0.0)


def test_goal_empty_assign_is_frame_origin() -> None:
    g = task.goal_from_assign("", b"")
    assert (g.x, g.y, g.label) == (0.0, 0.0, "")


# ── task_state_from_goal_status ───────────────────────────────────────────────

def test_goal_status_maps_to_task_state() -> None:
    # action_msgs/GoalStatus: SUCCEEDED=4, CANCELED=5, ABORTED=6, EXECUTING=2, ACCEPTED=1.
    assert task.task_state_from_goal_status(4) == task.TASK_SUCCEEDED
    assert task.task_state_from_goal_status(5) == task.TASK_CANCELLED  # not FAILED
    assert task.task_state_from_goal_status(6) == task.TASK_FAILED
    assert task.task_state_from_goal_status(2) == task.TASK_RUNNING
    assert task.task_state_from_goal_status(1) == task.TASK_RUNNING


# ── binding: command_move records the NavigateToPose target ────────────────────

def test_binding_command_move_records_goal_without_ros() -> None:
    b = Nav2Binding(robot_id="amr-1")
    assert b.pending_goal("amr-1") is None
    b.command_move("amr-1", 3.0, 4.0, 1.0)  # no rclpy node → cached only, no dispatch
    g = b.pending_goal("amr-1")
    assert (g.x, g.y, g.yaw) == (3.0, 4.0, 1.0)
