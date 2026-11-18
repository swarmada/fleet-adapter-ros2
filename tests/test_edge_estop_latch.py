"""An edge estop sets the robot's local latch: no assignment or resume is accepted after it.

RFC-0001 (safety, *Edge / Headless Emergency Stop*): every adapter with an EdgeStream sets the
local latch for each robot an edge stop stops, whatever contract version, and while it is set
every AssignAction is refused (*The Stop Latch*, *While latched*). This adapter keeps it in its
process, as the adapter template does, so only a restart lifts it. Before, the robot was stopped
and the next assignment was accepted and driven.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace as NS

from test_estop_stop_retry import _adapter, _edge_estop, _navigate, _Robot, _until

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter

ROBOT = "amr-1"
LATCHED = "edge estop edge-1 set this robot's local latch"


class _Records(_Robot):
    """A binding that records, at each stop, why motion is refused then, and whether the lease
    lock is held."""

    def __init__(self, a_box: list) -> None:
        super().__init__(fail=0)
        self.a_box, self.at_stop = a_box, []

    def command_stop(self, robot_id: str) -> None:
        a = self.a_box[0]
        self.at_stop.append((a._no_estop_path(), a._lease_lock.locked()))
        super().command_stop(robot_id)


def _refused(a: Adapter) -> tuple:
    res = pb.CommandResult(command_id=7, robot_id=ROBOT)
    a._on_assign(_navigate("t-7", 7), res)
    resumed = pb.CommandResult(command_id=8, robot_id=ROBOT)
    a._on_resume(type(pb.Command().resume)(), resumed)
    return res.assign_action, resumed.resume


def test_an_edge_estop_refuses_every_assignment_and_resume_after_it(capsys):
    robot = _Robot(fail=0)
    a = _adapter(robot)
    try:
        edge = _edge_estop(a, "edge-1")
        assert edge.acks(1)["edge-1"].state == pb.ESTOP_STATE_STOPPED
        assign, resume = _refused(a)
        assert not assign.accepted, assign
        assert assign.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
        assert LATCHED in assign.message and "cleared the latch there" in assign.message, (
            assign.message)
        assert not resume.resumed and LATCHED in resume.message, resume
        assert robot.moves == 0 and robot.resumes == 0
        _edge_estop(a, "edge-2")                     # a later edge estop: the first one is named
        assert LATCHED in a._no_estop_path()
        err = capsys.readouterr().err
        assert err.count("EDGE_LATCH") == 1 and "edge estop edge-1 set the local latch" in err
    finally:
        a._shutdown.set()


def test_the_latch_is_set_before_the_stop_and_the_stop_is_commanded_again_under_the_lease_lock():
    box: list = []
    robot = _Records(box)
    a = _adapter(robot)
    box.append(a)
    try:
        _edge_estop(a, "edge-1")
        assert _until(lambda: len(robot.at_stop) == 2), robot.at_stop
        assert all(LATCHED in why for why, _ in robot.at_stop), robot.at_stop
        assert sorted(locked for _, locked in robot.at_stop) == [False, True], robot.at_stop
    finally:
        a._shutdown.set()


class _NeverAtRest(_Robot):
    """A base whose rest is never confirmed: a goal keeps it driving."""

    def is_stopped(self, robot_id: str) -> bool:
        return False


def test_the_stop_after_the_latch_does_not_wait_for_the_estops_confirmation():
    """A goal sent just after the estop's stop keeps the base from confirming rest for the whole
    stop budget (2 s); the stop is commanded again well before that."""
    robot = _NeverAtRest(fail=0)
    a = _adapter(robot)
    try:
        threading.Thread(target=_edge_estop, args=(a, "edge-1"), daemon=True).start()
        assert _until(lambda: robot.stops >= 2, 0.5), "the second stop waited for the answer"
    finally:
        a._shutdown.set()


def test_a_goal_sent_just_after_the_estops_stop_is_stopped():
    """An assignment that passed its check before the latch, and holds the lease lock to send
    its goal, sends it after the estop's stop: the stop commanded again once the lock is free
    stops it."""
    robot = _Robot(fail=0)
    a = _adapter(robot)
    a._lease_lock.acquire()                          # the dispatch, past its check
    try:
        edge = _edge_estop(a, "edge-1")              # answered: the stop after it waits
        assert edge.acks(1)["edge-1"].state == pb.ESTOP_STATE_STOPPED
        assert robot.stops == 1
        robot.command_move(ROBOT, 3.0, 4.0)          # the dispatch's goal, after the stop
        assert robot._moving.get(ROBOT)
        a._lease_lock.release()
        assert _until(lambda: robot.stops == 2 and not robot._moving.get(ROBOT)), (
            "the late goal kept driving")
    finally:
        if a._lease_lock.locked():
            a._lease_lock.release()
        a._shutdown.set()


def test_a_drill_estop_sets_no_latch():
    """The drill is a scenario's estop, not an edge stop."""
    robot = _Robot(fail=0)
    a = _adapter(robot)
    a._sim = NS(estop_due=lambda: True)
    try:
        a._safety_outbox = queue.Queue()
        a._maybe_estop_drill()
        assert a._no_estop_path() == ""
    finally:
        a._shutdown.set()
