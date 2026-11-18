"""While this binding's own goal runs, a goal another sender gave the navigation server is
cancelled by its id, an operator's included.

An assignment that supersedes an explore segment halts the segment, stops the explorer, and sends
its own goal, which lifts the stop's hold. An explorer goal sent just before, and accepted after the
assignment's goal, could then run in its place: nothing cancelled it, since goals are cancelled by
id only while the robot is held, and the robot drove toward the explorer's frontier. Now, from the
binding's own goal until it succeeds, the explorer is released, or the robot is held, every goal the
server lists as running that the binding did not send is cancelled: when the goal is sent, and as
each status shows one. A goal of the binding's that fails or is cancelled keeps the guard: the
server ends it when a later goal preempts it.
"""

from __future__ import annotations

from test_segment_motion_after_a_stop import EXPLORERS, ROBOT, _binding

LATE = b"\x0b" * 16                   # an explorer goal accepted after the assignment's
OPERATOR = b"\x0c" * 16               # a goal an operator sent


def _ours(b) -> bytes:
    return b._claimed[b._goal_epoch]


def test_a_late_explorer_goal_accepted_after_the_assignments_goal_is_cancelled():
    b = _binding()
    assert b.release_hold_for_explorer(ROBOT, lambda: True)  # a segment's explorer drives
    b.on_nav_status([(EXPLORERS, 2)])
    assert b.cancelled == []
    b.command_hold(ROBOT)                                    # the segment is superseded
    b.command_move(ROBOT, 3.0, 4.0)                          # the assignment's goal
    b.cancelled.clear()
    b.on_nav_status([(_ours(b), 2), (LATE, 1)])              # the late one, accepted after
    assert b.cancelled == [LATE]


def test_a_goal_running_when_the_binding_sends_its_own_is_cancelled_at_once():
    b = _binding()
    assert b.release_hold_for_explorer(ROBOT, lambda: True)
    b.on_nav_status([(EXPLORERS, 2)])
    b.command_move(ROBOT, 3.0, 4.0)
    assert b.cancelled == [EXPLORERS]


def test_the_bindings_own_goal_is_never_cancelled():
    b = _binding()
    b.command_move(ROBOT, 3.0, 4.0)
    b.on_nav_status([(_ours(b), 2)])
    b.command_move(ROBOT, 5.0, 6.0)                          # a newer goal of its own
    b.on_nav_status([(_ours(b), 1)])
    assert b.cancelled == []


def test_a_goals_id_is_the_bindings_before_it_is_sent():
    b = _binding()
    seen = []
    b._send_nav2_goal = lambda goal, epoch: seen.append(b._claimed[epoch] in b._own_ids)
    b.command_move(ROBOT, 3.0, 4.0)
    assert seen == [True]


def test_an_operators_goal_is_cancelled_while_the_assignment_runs_and_not_once_it_succeeded():
    b = _binding()
    b.command_move(ROBOT, 3.0, 4.0)
    b.on_nav_status([(_ours(b), 2), (OPERATOR, 2)])
    assert b.cancelled == [OPERATOR]
    b.on_goal_result(4, epoch=b._goal_epoch)                 # SUCCEEDED: the robot is idle
    b.on_nav_status([(OPERATOR, 2)])
    assert b.cancelled == [OPERATOR]


def test_a_goal_of_ours_ended_by_a_later_goal_keeps_the_guard():
    """Nav2 ends our goal when a later goal preempts it: that later goal is still cancelled."""
    b = _binding()
    b.command_move(ROBOT, 3.0, 4.0)
    b.on_goal_result(6, epoch=b._goal_epoch)                 # ABORTED: preempted
    b.on_nav_status([(LATE, 2)])
    assert b.cancelled == [LATE]


def test_while_a_segment_may_drive_its_explorer_goals_are_not_cancelled():
    b = _binding()
    b.command_move(ROBOT, 3.0, 4.0)                          # a return goal, then the explorer
    b.on_goal_result(4, epoch=b._goal_epoch)
    b.command_move(ROBOT, 1.0, 1.0)
    assert b.release_hold_for_explorer(ROBOT, lambda: True)
    b.on_nav_status([(EXPLORERS, 2)])
    assert b.cancelled == []


def test_a_resumed_goal_guards_again():
    """A resume re-sends the retained goal, and guards again even after the guard was lifted."""
    b = _binding()
    b.command_move(ROBOT, 3.0, 4.0)
    b.on_goal_result(4, epoch=b._goal_epoch)                 # SUCCEEDED: the guard is lifted
    b.on_nav_status([(OPERATOR, 2)])
    assert b.cancelled == []                                 # idle: another's goal is left alone
    b.command_pause(ROBOT)
    b.on_nav_status([(_ours(b), 5)])
    b.cancelled.clear()
    b.command_resume(ROBOT)
    b.on_nav_status([(_ours(b), 2), (OPERATOR, 2)])
    assert b.cancelled == [OPERATOR]
