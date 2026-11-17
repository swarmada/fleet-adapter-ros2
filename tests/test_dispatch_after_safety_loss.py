# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""No assignment, segment or resume is dispatched after the SafetyStream goes down.

An assignment, a segment or a resume checks for an estop path first, then runs its other
checks, then dispatches under `_lease_lock`. A SafetyStream loss marks the stream down, then
stops the robot and ends the action under that same lock. Each test here loses the stream in
the gap between the first check and the dispatch -- from a hook the dispatch calls in that gap
-- and requires the dispatch to see it: refused, nothing commanded. The last two require the
other order to hold too: a dispatch that came first is stopped and reported by the loss, and a
resume waits for the lock the loss stops under. The re-check is pinned inside the lock: the
state changes while the test holds it, after the first check has passed. (A segment's own later
phases, such as its return to the mark after a resume, run on its thread and are not covered
here.)
"""
from __future__ import annotations

import queue
import threading
import time

import pytest

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
EXPLORE = b'{"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2}'


class _Counting(SimulatedRobot):
    """Counts the motion commands and stops it is sent."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.moves = self.resumes = self.stops = 0

    def command_move(self, robot_id, x, y, yaw=0.0) -> None:
        self.moves += 1
        super().command_move(robot_id, x, y, yaw)

    def command_resume(self, robot_id) -> None:
        self.resumes += 1
        super().command_resume(robot_id)

    def command_stop(self, robot_id) -> None:
        self.stops += 1
        super().command_stop(robot_id)


def _adapter(robot: _Counting, **kw) -> Adapter:
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, **kw)
    a._safety_required = True
    a._safety_up.set()          # the SafetyStream is up: the first check passes
    return a


def _lose_the_stream_in_the_gap(a: Adapter, method: str) -> None:
    """Make `method`, which the dispatch calls after its first check and before `_lease_lock`,
    lose the SafetyStream first."""
    original = getattr(a, method)

    def hook(*args, **kwargs):
        a._safety_down("test: lost between the check and the dispatch", 10.0)
        return original(*args, **kwargs)

    setattr(a, method, hook)


def _navigate(action_id: str = "t-1", token: int = 1):
    return pb.AssignAction(action_id=action_id, action_type="Navigate",
                           payload_json=b'{"x": 3, "y": 4}', fencing_token=token,
                           lease_generation=1, lease_duration_ms=60_000)


def _statuses(a: Adapter) -> list:
    out = []
    while True:
        try:
            m = a._outbox.get_nowait()
        except queue.Empty:
            return out
        if m.WhichOneof("payload") == "action_status":
            out.append(m.action_status)


def test_an_assignment_is_refused_when_the_stream_is_lost_before_it_dispatches():
    robot = _Counting()
    a = _adapter(robot)
    _lose_the_stream_in_the_gap(a, "_geometry_refusal")
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    a._on_assign(_navigate(), res)
    assert not res.assign_action.accepted
    assert res.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert "the SafetyStream is down" in res.assign_action.message, res.assign_action
    assert robot.moves == 0, "a goal was sent with no estop path"
    assert a._current_action == ""
    assert ROBOT not in a._lease._deadline


def test_a_segment_is_refused_and_never_started_when_the_stream_is_lost_before_it_dispatches(
        tmp_path):
    stack = SimulatedExploreStack()
    robot = _Counting(explore=stack)
    a = _adapter(robot, artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"),
                 namespace="lab")
    _lose_the_stream_in_the_gap(a, "_geometry_refusal")
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    a._on_assign(pb.AssignAction(action_id="explore-1", action_type="Custom",
                                 payload_json=EXPLORE, fencing_token=1, lease_generation=1,
                                 lease_duration_ms=60_000), res)
    assert not res.assign_action.accepted
    assert res.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert "the SafetyStream is down" in res.assign_action.message, res.assign_action
    assert a._running_job() is None and a._current_action == ""
    time.sleep(0.05)
    assert not [t for t in threading.enumerate() if t.name == "job-explore-1"], \
        "the segment's thread was started"
    assert [s for s in _statuses(a) if s.action_id == "explore-1"] == []


def test_a_resume_is_refused_when_the_stream_is_lost_before_it_dispatches():
    robot = _Counting()
    a = _adapter(robot)
    _lose_the_stream_in_the_gap(a, "_running_job")
    res = pb.CommandResult(command_id=2, robot_id=ROBOT)
    a._on_resume(type(pb.Command().resume)(), res)
    assert not res.resume.resumed
    assert "the SafetyStream is down" in res.resume.message, res.resume
    assert robot.resumes == 0, "a goal was re-dispatched with no estop path"


def test_an_assignment_dispatched_before_the_loss_is_stopped_and_reported_by_it():
    robot = _Counting()
    a = _adapter(robot)
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    a._on_assign(_navigate(), res)
    assert res.assign_action.accepted and robot.moves == 1
    stops = robot.stops
    a._safety_down("test: lost after the dispatch", 10.0)
    assert robot.stops == stops + 1, "the dispatched goal was not stopped"
    assert a._current_action == ""
    end = time.monotonic() + 5.0
    ended: list = []
    while not ended and time.monotonic() < end:
        ended = [s for s in _statuses(a) if s.action_id == "t-1"
                 and s.state == pb.ACTION_STATE_FAILED]
        time.sleep(0.02)
    assert len(ended) == 1, "the dispatched action's end was not reported"


def test_a_resume_dispatches_under_the_lock_the_loss_stops_under():
    robot = _Counting()
    a = _adapter(robot)
    res = pb.CommandResult(command_id=2, robot_id=ROBOT)
    t = threading.Thread(target=a._on_resume, args=(type(pb.Command().resume)(), res),
                         daemon=True)
    with a._lease_lock:                     # as `_safety_down` holds it while it stops
        t.start()
        time.sleep(0.1)
        assert robot.resumes == 0 and t.is_alive(), "the resume did not wait for the lease lock"
    t.join(timeout=5.0)
    assert robot.resumes == 1 and res.resume.resumed


def _hold_the_lock_while(a: Adapter, dispatch, change) -> None:
    """Run `dispatch` on its own thread while the test holds `_lease_lock`, so it passes its
    first check and then waits for the lock; make `change` under the lock, as a loss does after
    marking the stream down; then let the dispatch go."""
    t = threading.Thread(target=dispatch, daemon=True)
    with a._lease_lock:
        t.start()
        time.sleep(0.1)                     # past the first check, waiting for the lock
        assert t.is_alive(), "the dispatch did not wait for the lease lock"
        change()
    t.join(timeout=5.0)
    assert not t.is_alive()


def _stream_lost(a: Adapter):
    return a._safety_up.clear               # what `_safety_down` does before it takes the lock


def _stop_failing(a: Adapter):
    def change() -> None:
        a._stop_failing = "test: the stop raised"
    return change


@pytest.mark.parametrize(("change", "says"), [
    (_stream_lost, "the SafetyStream is down"),
    (_stop_failing, "the robot's stop is failing"),
])
def test_an_assignment_checks_again_inside_the_lease_lock(change, says):
    robot = _Counting()
    a = _adapter(robot)
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    _hold_the_lock_while(a, lambda: a._on_assign(_navigate(), res), change(a))
    assert not res.assign_action.accepted
    assert res.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert says in res.assign_action.message, res.assign_action
    assert robot.moves == 0 and a._current_action == ""


def test_a_segment_checks_again_inside_the_lease_lock(tmp_path):
    stack = SimulatedExploreStack()
    robot = _Counting(explore=stack)
    a = _adapter(robot, artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"),
                 namespace="lab")
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    task = pb.AssignAction(action_id="explore-2", action_type="Custom", payload_json=EXPLORE,
                           fencing_token=1, lease_generation=1, lease_duration_ms=60_000)
    _hold_the_lock_while(a, lambda: a._on_assign(task, res), _stream_lost(a))
    assert not res.assign_action.accepted
    assert "the SafetyStream is down" in res.assign_action.message, res.assign_action
    assert a._running_job() is None and a._current_action == ""
    assert not [t for t in threading.enumerate() if t.name == "job-explore-2"]
