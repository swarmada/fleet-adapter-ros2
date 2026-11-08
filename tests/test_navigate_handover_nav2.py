# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The Nav2 binding's handover, without ROS: a fake publisher stands in for ``cmd_vel``.

The gate is armed for a navigate-handover while Nav2 drives and forwards nothing. After Nav2
declines, `hand_over_teleop` turns it to the operator under ``_lock``, the lock every stop takes:
a stop that closed the gate, a lease deadline that passed, or another action's id refuses it.
"""

from __future__ import annotations

import pytest
from test_manual_drive_nav2 import ROBOT, Clock, binding, msg
from test_stop_never_waits import _odom

from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.nav2_binding import _GS_ABORTED

ACTION = "handover-a"


def armed(clock=None, lease_s: float = 60.0):
    b, pub = binding(clock)
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    b.open_teleop(ROBOT, ACTION, drive, lease_s=lease_s)
    with b._lock:                                # the goal as `command_move` records it, unsent
        b._goal = task_map.GoalPose(x=2.0, y=1.0, yaw=0.0)
        b._holding, b._goal_status = False, 0
    return b, pub, drive


def decline(b) -> None:
    """Nav2's ABORTED for the current goal, as `on_goal_result` records it."""
    with b._lock:
        b._goal_status, b._failure = _GS_ABORTED, "no valid path"
    assert b.task_state(ROBOT) == task_map.TASK_FAILED


def test_armed_it_forwards_nothing_while_nav2_drives() -> None:
    b, pub, _ = armed()
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.3))
    assert len(pub.sent) == n and b.teleop_open(ROBOT, ACTION)


def test_after_a_decline_it_hands_over_holds_the_base_and_drops_the_goal() -> None:
    b, pub, drive = armed()
    b._odometry_source = "measured"              # a handover needs rest it can confirm
    decline(b)
    assert b.hold_for_handover(ROBOT, ACTION)
    assert pub.sent[-1] == md.ZERO and b._goal is None and drive.phase == md.NAVIGATING
    b.on_odom(_odom(0.0))
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert pub.sent[-1] == md.ZERO and drive.phase == md.RUNNING
    assert "no valid path" in drive.message
    assert b._goal is None                       # a resume has nothing to re-dispatch
    n = len(pub.sent)
    b.on_teleop(msg(0.5, 0.5, 5.0))
    vx, vy, wz = pub.sent[n]
    assert abs((vx * vx + vy * vy) ** 0.5 - md.CAP_LINEAR_MPS) < 1e-9 and wz == 0.7
    assert not b.hand_over_teleop(ROBOT, ACTION, 6.0, "again")   # once only


@pytest.mark.parametrize("stop", ["command_stop", "command_hold", "command_pause"])
def test_a_stop_before_the_handover_refuses_it(stop) -> None:
    b, pub, drive = armed()
    decline(b)
    getattr(b, stop)(ROBOT)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    assert drive.phase == md.NAVIGATING and not b.teleop_open(ROBOT)
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n


def test_past_the_lease_deadline_the_handover_is_refused() -> None:
    clock = Clock()
    b, pub, drive = armed(clock, lease_s=1.0)
    decline(b)
    clock.t += 1.0
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, "no valid path")
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n and drive.phase == md.NAVIGATING


def test_another_actions_handover_is_refused() -> None:
    b, _, drive = armed()
    decline(b)
    assert not b.hand_over_teleop(ROBOT, "someone-else", 5.0, "no valid path")
    assert drive.phase == md.NAVIGATING
