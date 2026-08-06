# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for the Nav2 binding (sub-item 4): confirmed stop / hold (C5),
shared by lease-expiry self-stop (C4).

The safety-critical property under test: STOPPED is reported ONLY from the robot's
measured odom twist — never inferred from the stop command, a timer, or an absent
message (CONFORMANCE.md C5.3). ``confirm_estop`` is the audited SDK primitive the
adapter uses on both the estop and lease-expiry paths.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from swarmada_sdk.safety import ESTOP_FAILED, ESTOP_STOPPED, confirm_estop

from fleet_adapter_ros2.nav2_binding import Nav2Binding


def _odom(vx=0.0, vy=0.0, wz=0.0):
    """Odometry carrying only the twist that is_stopped reads (pose defaults to 0)."""
    return NS(
        pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                        orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
        twist=NS(twist=NS(linear=NS(x=vx, y=vy, z=0.0), angular=NS(x=0.0, y=0.0, z=wz))),
    )


def test_is_stopped_false_before_any_odom() -> None:
    # No odom observed → no evidence of rest → never claim STOPPED.
    b = Nav2Binding(robot_id="amr-1")
    assert b.is_stopped("amr-1") is False


def test_is_stopped_false_while_moving() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.on_odom(_odom(vx=0.4))                 # translating
    assert b.is_stopped("amr-1") is False
    b.on_odom(_odom(wz=0.3))                 # rotating in place
    assert b.is_stopped("amr-1") is False


def test_is_stopped_true_only_when_twist_near_zero() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.on_odom(_odom(vx=0.005, wz=0.005))     # within sensor noise
    assert b.is_stopped("amr-1") is True


def test_command_stop_cancels_goal_and_holds() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_move("amr-1", 5.0, 5.0)
    assert b.pending_goal("amr-1") is not None
    b.command_stop("amr-1")
    assert b.pending_goal("amr-1") is None    # active goal cancelled
    # A stop is only a REQUEST: without odom confirming rest, not yet STOPPED.
    assert b.is_stopped("amr-1") is False


def test_confirm_estop_returns_stopped_when_odom_confirms_rest() -> None:
    # confirm_estop → command_stop, then polls is_stopped; odom shows rest → STOPPED.
    b = Nav2Binding(robot_id="amr-1")
    b.on_odom(_odom(vx=0.0, wz=0.0))
    assert confirm_estop(b, "amr-1") == ESTOP_STOPPED


def test_confirm_estop_fails_when_robot_still_moving() -> None:
    # The stop is commanded but odom still shows motion → FAILED, never a false
    # STOPPED. This is the C5.3 guarantee: rest is confirmed, not assumed.
    b = Nav2Binding(robot_id="amr-1")
    b.on_odom(_odom(vx=0.5))
    assert confirm_estop(b, "amr-1", tick_dt=0.0, max_ticks=5) == ESTOP_FAILED


def test_new_goal_releases_hold() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_stop("amr-1")
    b.command_move("amr-1", 1.0, 2.0)         # dispatching a fresh task clears the hold
    assert b.pending_goal("amr-1") == b.pending_goal("amr-1")
    assert b.pending_goal("amr-1") is not None
