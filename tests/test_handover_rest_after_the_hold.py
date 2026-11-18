"""The navigate-handover's rest is judged on odometry received after the hold.

`hold_for_handover` holds the base when Nav2 declines; the job then asks `handover_at_rest`, and
`hand_over_teleop` asks again under the stop lock before it turns the gate. Both used to read the
latest odom whatever its age, so a reading from before the hold, from a base that may still have
been driving, could show rest at the first look (without a scan channel), or leave the scan's odom
veto silent (with one). The binding now takes the odom count at the hold, as every stop's
confirmation does (`state_seq`), and judges rest only once a message has been counted after it.
With a scan channel the rest is also refused while the navigation server still runs a goal,
whoever sent it, as `_at_rest_locked` refuses it.
"""

from __future__ import annotations

from scan_room import Lidar, Plant
from test_handover_scan_rest import _armed_nav2_scan, _scans_for
from test_handover_waits_for_rest import WHY
from test_manual_drive_gate import ROBOT
from test_navigate_handover_nav2 import ACTION, armed, decline
from test_scan_rest import T  # noqa: F401 - the fixture: T from two stationary runs
from test_stop_never_waits import _odom

from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import operator_ready as opr

NO_ODOM = "no odometry received since the hold"
GOAL = bytes(range(16))


def _held_without_scan():
    b, pub, drive = armed()
    b._odometry_source = "measured"
    b.on_odom(_odom(0.0))                        # at rest, but before the hold
    decline(b)
    assert b.hold_for_handover(ROBOT, ACTION)
    return b, pub, drive


# ── without a scan channel ───────────────────────────────────────────────────

def test_without_a_scan_odometry_from_before_the_hold_never_hands_over() -> None:
    b, _, drive = _held_without_scan()
    assert b.handover_at_rest(ROBOT) == (False, NO_ODOM)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)
    assert drive.phase != md.RUNNING
    b.on_odom(_odom(0.0))                        # the first reading after the hold: at rest
    assert b.handover_at_rest(ROBOT) == (True, "")
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_without_a_scan_the_first_reading_after_the_hold_showing_motion_refuses_it() -> None:
    b, _, drive = _held_without_scan()
    b.on_odom(_odom(0.3))
    assert not b.handover_at_rest(ROBOT)[0]
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING
    b.on_odom(_odom(0.0))
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_each_hold_needs_its_own_reading() -> None:
    b, _, drive = _held_without_scan()
    b.on_odom(_odom(0.0))
    assert b.handover_at_rest(ROBOT)[0]
    assert b.hold_for_handover(ROBOT, ACTION)    # held again: the count is taken again
    assert b.handover_at_rest(ROBOT) == (False, NO_ODOM)
    b.on_odom(_odom(0.0))
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_never_held_never_at_rest_for_a_handover() -> None:
    b, _, drive = armed()
    b._odometry_source = "measured"
    decline(b)
    b.on_odom(_odom(0.0))
    ok, why = b.handover_at_rest(ROBOT)
    assert not ok and "no hold" in why
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING


# ── with a scan channel ──────────────────────────────────────────────────────

def test_with_the_scan_odometry_from_before_the_hold_never_hands_over(T) -> None:  # noqa: F811
    b, _, drive, clock = _armed_nav2_scan(T)
    b.on_odom(_odom(0.0))                        # before the hold
    assert b.hold_for_handover(ROBOT, ACTION)
    _scans_for(b, clock, Plant(), Lidar(seed=87), 2.5)        # the scan shows rest
    assert b.handover_at_rest(ROBOT) == (False, NO_ODOM)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING
    b.on_odom(_odom(0.0))
    assert b.handover_at_rest(ROBOT) == (True, "")
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_with_the_scan_moving_odometry_from_before_the_hold_does_not_decide_it(T) -> None:  # noqa: F811
    """The veto is judged on a reading after the hold too: motion before it is not the answer,
    and nor is its absence."""
    b, _, _drive, clock = _armed_nav2_scan(T)
    b.on_odom(_odom(0.3))                        # driving, before the hold
    assert b.hold_for_handover(ROBOT, ACTION)
    _scans_for(b, clock, Plant(), Lidar(seed=88), 2.5)
    assert b.handover_at_rest(ROBOT) == (False, NO_ODOM)
    b.on_odom(_odom(0.0))
    assert b.handover_at_rest(ROBOT) == (True, "")


def test_with_the_scan_a_goal_the_server_still_runs_refuses_it(T) -> None:  # noqa: F811
    b, _, drive, clock = _armed_nav2_scan(T)
    b._cancel_running_goal = lambda goal_id: None
    b.on_nav_status([])                          # the server's status: nothing runs
    assert b.hold_for_handover(ROBOT, ACTION)
    b.on_odom(_odom(0.0))
    _scans_for(b, clock, Plant(), Lidar(seed=89), 2.5)
    b.on_nav_status([(GOAL, 2)])                 # another sender's goal, running
    ok, why = b.handover_at_rest(ROBOT)
    assert not ok and "still runs a goal" in why
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING
    b.on_nav_status([(GOAL, 5)])                 # CANCELED: nothing runs
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_with_the_operators_ready_odometry_from_before_the_hold_never_turns_the_gate(T) -> None:  # noqa: F811
    b, _, drive, clock = _armed_nav2_scan(T)
    b._ready = opr.ReadyLatch(log=lambda line: None)
    b.on_odom(_odom(0.0))
    assert b.hold_for_handover(ROBOT, ACTION)
    drive.decline(clock.t, WHY)                  # as the job records it after the hold
    _scans_for(b, clock, Plant(), Lidar(seed=90), 2.5)        # the scan's full look, no veto
    assert b.on_ready(ACTION) == ""
    assert b.take_ready(ROBOT, ACTION) is not None
    assert b.handover_at_rest(ROBOT) == (False, NO_ODOM)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING
    b.on_odom(_odom(0.0))
    assert b.on_ready(ACTION) == ""              # the refused ready was spent: press again
    assert b.take_ready(ROBOT, ACTION) is not None
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING
