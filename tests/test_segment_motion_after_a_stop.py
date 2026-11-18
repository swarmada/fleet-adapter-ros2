# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""No explore segment sets the robot moving after a stop or a pause.

A segment's own motion, its return to the mark and the explorer's start, is sent from the job's
thread. The job used to check whether it was stopped or paused first and send later in the same
pass, outside the lock the stop takes, so a stop or a pause landing in between let the motion
out. Now every stop and pause marks the job (`halted`, paused) before it takes the lock it stops
under, and the motion is checked against the mark under that same lock (`_Job.may_move`): the
stop either finds the motion and stops it, or the motion is refused. The tests here land the
stop inside the stack's call, just before it takes its lock, after every check the job thread
makes itself.

Also here: a pause that comes right after a resume stays a pause; the explorer's own navigation
goals are cancelled while the robot is held and keep rest from being confirmed while they run;
a job stopped before its thread starts runs nothing; the pause's stop token is read after the
hold; and shutdown marks the job before its first hold.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest
from conftest import mark_transport_live, odom_after_stop, odom_msg
from test_ros2_explore import _stack as _ros2_stack

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
DOC = {"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2,
       "battery_return_mv": 7835}


# ── a segment on the simulated stack, with hooks in its gaps ────────────────

class _Clock:
    """A fake clock whose waits advance it, never sleep, and run the test's hooks: each hook is
    (waits to skip, callable), run once that many waits have passed."""

    def __init__(self) -> None:
        self.t = 0.0
        self.hooks: list[list] = []

    def now(self) -> float:
        return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.t += seconds
        for h in list(self.hooks):
            h[0] -= 1
            if h[0] < 0:
                self.hooks.remove(h)
                h[1]()
        return event.is_set()

    def after(self, waits: int, fn) -> None:
        self.hooks.append([waits, fn])


class _HookedLock:
    """The stack's lock, with a one-shot hook run just before the next acquire: a stop or a
    pause that lands inside the stack's call, after every check the job thread made."""

    def __init__(self, lock) -> None:
        self._lock, self.hook = lock, None

    def __enter__(self):
        hook, self.hook = self.hook, None
        if hook is not None:
            hook()
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


class _Emits:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str]] = []

    def __call__(self, state: str, message: str, pct: int = 0, end_cause: str = "") -> None:
        self.updates.append((state.removeprefix("ACTION_STATE_"), message))


def _segment(tmp_path):
    """A segment that explores until the battery reaches its return threshold, then returns."""
    clock = _Clock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, start_mv=7900,
                                  drain_mv_per_s=1.0)
    stack._lock = _HookedLock(stack._lock)
    emits = _Emits()
    job = explore.ExploreJob(
        action_id="explore-gap", doc=dict(DOC), stack=stack, limits=stack.battery_limits(),
        artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab",
        clock=clock, emit=emits)
    return job, stack, clock, emits


def _end_the_job(job, stack) -> None:
    """As the adapter's `_halt_job` does: the job marked, then the stack halted."""
    job.halted.set()
    stack.halt()


def _halt_the_stack(job, stack) -> None:
    """As a binding's `command_stop` does: the stack halted, which marks the job first."""
    stack.halt()


STOPS = {"the adapter ends the job": _end_the_job, "the binding's stop": _halt_the_stack}


def _pause_once_the_return_is_sent(job, stack, clock) -> None:
    """Pause the segment as soon as its return goal is sent, and resume it at the next wait."""
    send = stack.go_to_mark
    done = [False]

    def go(*args, **kw):
        ok = send(*args, **kw)
        if ok is not False and not done[0]:
            done[0] = True
            job.pause()                   # the return goal stops; the segment waits
            clock.after(0, job.resume)
        return ok
    stack.go_to_mark = go


def _when_the_pause_is_waited_out(job, then) -> None:
    """Run `then` once, right after the job thread has waited out the pause, before it sends
    anything again."""
    hold = job._hold_while_paused
    done = [False]

    def wrapped():
        waited = hold()
        if waited and not done[0]:
            done[0] = True
            then()
        return waited
    job._hold_while_paused = wrapped


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_return_goal_is_refused_when_a_stop_lands_after_the_pause_is_waited_out(
        tmp_path, stop):
    job, stack, clock, emits = _segment(tmp_path)
    _pause_once_the_return_is_sent(job, stack, clock)
    seen: list = []

    def stop_in_the_send() -> None:
        STOPS[stop](job, stack)
        seen.append(len(emits.updates))
    _when_the_pause_is_waited_out(job, lambda: setattr(stack._lock, "hook", stop_in_the_send))
    job.run()
    assert seen, "the stop never landed"
    assert not stack.moving(), "the return goal was sent after the stop"
    assert "go_to_mark refused" in stack.log
    assert len(emits.updates) == seen[0], "the job reported after the stop"


def test_a_return_goal_is_refused_when_a_pause_lands_after_the_last_pause_is_waited_out(
        tmp_path):
    job, stack, clock, emits = _segment(tmp_path)
    _pause_once_the_return_is_sent(job, stack, clock)
    at_resume: list = []

    def pause_in_the_send() -> None:
        job.pause()

        def resume() -> None:
            at_resume.append(stack.moving())
            job.resume()
        clock.after(3, resume)
    _when_the_pause_is_waited_out(job, lambda: setattr(stack._lock, "hook", pause_in_the_send))
    job.run()
    assert at_resume == [False], "the return goal was sent while the segment was paused"
    assert "go_to_mark refused" in stack.log
    assert emits.updates[-1][0] == "SUCCEEDED", emits.updates[-1]


class _PauseOnClear(threading.Event):
    """The job's paused flag, running a hook once, right after it is next cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = None

    def clear(self) -> None:
        super().clear()
        hook, self.hook = self.hook, None
        if hook is not None:
            hook()


def test_a_pause_right_after_a_resume_stays_a_pause(tmp_path):
    job, stack, clock, emits = _segment(tmp_path)
    job._paused = _PauseOnClear()
    at_resume: list = []

    def pause_again() -> None:
        job.pause()

        def resume() -> None:
            at_resume.append(stack.moving())
            job.resume()
        clock.after(3, resume)

    send = stack.go_to_mark
    done = [False]

    def go(*args, **kw):
        ok = send(*args, **kw)
        if ok is not False and not done[0]:
            done[0] = True
            job.pause()

            def resume() -> None:
                job._paused.hook = pause_again   # the next pause lands as the flag clears
                job.resume()
            clock.after(0, resume)
        return ok
    stack.go_to_mark = go
    job.run()
    assert at_resume == [False], "the second pause was lost: the segment moved while paused"
    assert emits.updates[-1][0] == "SUCCEEDED", emits.updates[-1]


class _PauseWhenSeenOver(threading.Event):
    """The job's paused flag, running a hook once, the next time it is read as clear: a pause
    that lands just as the job thread sees the last one end."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = None

    def is_set(self) -> bool:
        was = super().is_set()
        if not was and self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return was


def test_a_pause_as_the_job_sees_the_last_one_end_stays_a_pause(tmp_path):
    job, stack, clock, emits = _segment(tmp_path)
    job._paused = _PauseWhenSeenOver()
    at_resume: list = []

    def pause_again() -> None:
        job.pause()

        def resume() -> None:
            at_resume.append(stack.moving())
            job.resume()
        clock.after(3, resume)

    send = stack.go_to_mark
    done = [False]

    def go(*args, **kw):
        ok = send(*args, **kw)
        if ok is not False and not done[0]:
            done[0] = True
            job.pause()

            def resume() -> None:
                job.resume()
                job._paused.hook = pause_again
            clock.after(2, resume)       # while the thread waits inside the pause
        return ok
    stack.go_to_mark = go
    job.run()
    assert at_resume == [False], "the second pause was lost: the segment moved while paused"
    assert emits.updates[-1][0] == "SUCCEEDED", emits.updates[-1]


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_the_explorer_is_not_restarted_when_a_stop_lands_after_the_pause_is_waited_out(
        tmp_path, stop):
    job, stack, clock, emits = _segment(tmp_path)

    def pause_when_it_starts() -> None:
        stack.on_explorer_started = None
        job.pause()
        clock.after(0, job.resume)
    stack.on_explorer_started = pause_when_it_starts
    seen: list = []

    def stop_in_the_start() -> None:
        STOPS[stop](job, stack)
        seen.append(len(emits.updates))
    _when_the_pause_is_waited_out(job, lambda: setattr(stack._lock, "hook", stop_in_the_start))
    job.run()
    assert seen, "the stop never landed"
    assert not stack.moving(), "the explorer was started after the stop"
    assert "start_explorer refused" in stack.log
    assert len(emits.updates) == seen[0], "the job reported after the stop"


def test_a_job_halted_before_its_thread_starts_runs_nothing(tmp_path):
    job, stack, _, emits = _segment(tmp_path)
    job.halted.set()
    job.run()
    assert stack.log == [] and emits.updates == [], "it ran, or ended a segment it never began"


@pytest.mark.parametrize("motion", ["the return", "the explorer"])
def test_a_motion_refused_for_a_pause_restarts_its_phase_even_once_the_pause_is_over(
        tmp_path, motion):
    job, _, _, _ = _segment(tmp_path)
    job.arm()
    job._paused.set()
    job._pauses = job._pauses_seen = 1    # the thread already counted this pause as waited out
    if motion == "the return":
        assert job._go_to_mark() is False
    else:
        assert job._start_explorer("explorer restart") is None
    job._paused.clear()                    # resumed before the thread looks again
    assert job._hold_while_paused() is True, "the refused motion would never be sent"


def test_a_stop_before_a_segments_thread_starts_marks_that_segment(tmp_path, monkeypatch):
    stack = SimulatedExploreStack()
    robot = SimulatedRobot(explore=stack)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, artifacts=explore.ArtifactDir(str(tmp_path), "maps"),
                namespace="lab")
    gate = threading.Event()
    run = explore.ExploreJob.run
    monkeypatch.setattr(explore.ExploreJob, "run", lambda job: (gate.wait(5.0), run(job)))
    res = pb.CommandResult(command_id=1, robot_id=ROBOT)
    a._on_assign(pb.AssignAction(
        action_id="explore-armed", action_type="Custom", payload_json=(
            b'{"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2}'),
        fencing_token=1, lease_generation=1, lease_duration_ms=60_000), res)
    assert res.assign_action.accepted, res.assign_action
    job = a._job
    robot.command_stop(ROBOT)            # an estop's stop, before the segment's thread runs
    assert job.halted.is_set(), "the stop marked another job, or none"
    gate.set()
    for t in threading.enumerate():
        if t.name == "job-explore-armed":
            t.join(timeout=5.0)
    assert not stack.moving()
    assert "start_explorer" not in stack.log and "go_to_mark" not in stack.log


# ── the ROS 2 stack: the explorer's start and stop, and the return goal ───────

def test_the_explorer_is_not_resumed_for_a_job_stopped_or_paused_first():
    st, ros, shell, b, _ = _ros2_stack()
    shell.on("pgrep -fc lib/explore_lite/explore", (0, b"1\n", b""))
    b.command_hold(ROBOT)                                    # a stop holds the robot
    assert st.start_explorer(lambda: False) is False
    assert ("explore/resume", {"data": True}) not in ros.published
    assert b._nav_held, "the stop's hold was released for an explorer that never started"


def test_a_stop_between_the_hold_release_and_the_resume_keeps_the_explorer_stopped():
    st, ros, shell, _, _ = _ros2_stack()
    shell.on("pgrep -fc lib/explore_lite/explore", (0, b"1\n", b""))
    answers = iter([True, False])                            # the stop lands after the release
    assert st.start_explorer(lambda: next(answers)) is False
    assert ("explore/resume", {"data": True}) not in ros.published


@pytest.mark.parametrize("what", ["paused", "halted"])
def test_a_stop_during_a_launch_is_sent_again_once_the_explorer_is_up(what):
    st, ros, shell, _, _ = _ros2_stack()
    halted = threading.Event()
    st.arm(halted)
    counts = iter([b"0\n", b"1\n"])
    shell.on("pgrep -fc lib/explore_lite/explore", lambda s: (0, next(counts), b""))
    paused = [False]

    def launched(script):
        # The stop lands as the node starts, before it subscribes: its publish is lost.
        (halted.set if what == "halted" else lambda: paused.__setitem__(0, True))()
        st.stop_explorer()
        return (0, b"", b"")
    shell.on("ros2 run explore_lite explore", launched)
    got = st.start_explorer(lambda: not halted.is_set() and not paused[0])
    assert got is (None if what == "halted" else False)
    resumes = [f for t, f in ros.published if t == "explore/resume"]
    assert resumes[-1] == {"data": False} and len(resumes) >= 2, resumes


def test_the_stacks_halt_marks_the_job_before_it_stops_the_explorer():
    st, ros, _, _, _ = _ros2_stack()
    halted = threading.Event()
    st.arm(halted)
    marked: list = []
    publish = ros.publish

    def recording(topic, msg_type, fields):
        if topic == "explore/resume" and fields == {"data": False}:
            marked.append(halted.is_set())
        publish(topic, msg_type, fields)
    ros.publish = recording
    st.halt()
    assert marked == [True]


def test_the_return_goal_is_sent_only_while_the_job_may_move():
    st, _, _, b, sent = _ros2_stack()
    assert st.go_to_mark(1.0, 2.0, 0.0, lambda: False) is False
    assert sent == [] and b.pending_goal(ROBOT) is None
    assert st.go_to_mark(1.0, 2.0, 0.0, lambda: True) is True
    assert len(sent) == 1 and b.pending_goal(ROBOT) is not None


# ── the binding: every goal on the navigation server, whoever sent it ───────

def _binding() -> Nav2Binding:
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    b._status_sub = object()          # subscribed to the server's goal status (start())
    b.cancelled = []                  # the goal ids cancelled by id
    b._cancel_running_goal = b.cancelled.append
    b.sent = []
    b._send_nav2_goal = lambda goal, epoch: b.sent.append(goal)
    b._server_ready = lambda: True
    return b


EXPLORERS = b"\x07" * 16              # a goal id the explorer sent


def test_every_stop_cancels_every_goal_the_server_runs():
    b = _binding()
    b.on_nav_status([(EXPLORERS, 2)])                        # running; nothing held yet
    assert b.cancelled == []
    b.command_hold(ROBOT)
    assert b.cancelled == [EXPLORERS]
    b.command_pause(ROBOT)
    assert b.cancelled == [EXPLORERS, EXPLORERS]


def test_a_goal_seen_running_while_held_is_cancelled_and_not_once_released():
    b = _binding()
    b.command_pause(ROBOT)
    b.on_nav_status([(EXPLORERS, 2)])                        # accepted after the pause
    assert b.cancelled == [EXPLORERS]
    b.on_nav_status([(EXPLORERS, 5)])                        # CANCELED: nothing runs
    assert b.cancelled == [EXPLORERS]
    assert b.release_hold_for_explorer(ROBOT, lambda: True)
    b.on_nav_status([(EXPLORERS, 2)])                        # the explorer's own, allowed
    assert b.cancelled == [EXPLORERS]


def test_a_goal_sent_just_after_a_stop_is_never_cancelled_by_it():
    b = _binding()
    b.on_nav_status([(EXPLORERS, 2)])
    record = b._cancel_running_goal

    def cancel(goal_id):
        if not b.sent:                   # the next goal is sent as the stop's cancels go out
            b.command_move(ROBOT, 1.0, 2.0)
        record(goal_id)
    b._cancel_running_goal = cancel
    b.command_hold(ROBOT)
    ours = b._claimed[b._goal_epoch]                         # the id the new goal was sent with
    b.on_nav_status([(EXPLORERS, 3), (ours, 2)])            # the new goal runs: not held now
    assert ours not in b.cancelled, "a goal sent after the stop was cancelled"
    assert set(b.cancelled) == {EXPLORERS}    # by the stop, and by the new goal's own guard
    assert len(b.sent) == 1 and b.pending_goal(ROBOT) is not None


def test_no_rest_is_confirmed_while_the_server_runs_any_goal():
    b = _binding()
    b.on_odom(odom_msg())                                    # at rest by odom
    b.on_nav_status([(EXPLORERS, 2)])
    assert not b.is_stopped(ROBOT) and not b.stop_settled(ROBOT, 0)
    b.on_nav_status([(EXPLORERS, 4)])                        # SUCCEEDED
    assert b.is_stopped(ROBOT) and b.stop_settled(ROBOT, 0)


def _pause(a: Adapter, timeout: float = 5.0):
    a._handle_command(pb.Command(command_id=3, robot_id=ROBOT, pause=pb.PauseCapabilities(
        reason="test", require_stop_before_ack=True)))
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = a._outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        if m.WhichOneof("payload") == "command_result":
            return m.command_result.pause
    raise AssertionError("the pause was never answered")


@pytest.mark.parametrize("ends", [False, True])
def test_a_pause_is_not_acknowledged_while_an_explorer_goal_accepted_after_it_runs(
        monkeypatch, ends):
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.005)
    b = odom_after_stop(_binding())
    b.on_odom(odom_msg())
    pause = b.command_pause

    def accepted_after_it(robot_id):
        pause(robot_id)
        b.on_nav_status([(EXPLORERS, 2)])                    # the server says it runs
    b.command_pause = accepted_after_it
    tick, looks = b.tick, [0]

    def tick_and_end(dt):
        looks[0] += 1
        if ends and looks[0] == 3:                           # cancelled by the third look
            b.on_nav_status([(EXPLORERS, 5)])
        return tick(dt)
    b.tick = tick_and_end
    a = Adapter(pb, None, b, ROBOT)
    a._current_action, a._current_fencing_token = "t-1", 7
    got = _pause(a)
    assert b.cancelled == [EXPLORERS], "the explorer's goal was not cancelled by its id"
    if ends:
        assert got.paused is True, got.message
    else:
        assert got.paused is False and "still runs a goal" in got.message, got.message


# ── the pause's token, and shutdown ─────────────────────────────────────────

class _HoldStack:
    """A stack whose hold sees one odom message arrive while it is being commanded, then holds
    the base. Nothing arrives after."""

    def __init__(self, b: Nav2Binding) -> None:
        self.b = b

    def battery_limits(self):
        return explore.BatteryLimits(hard_mv=7750, full_mv=8300, return_mv=7835)

    def arm(self, halted) -> None:
        pass

    def hold(self) -> None:
        self.b.on_odom(odom_msg())                          # in flight as the hold goes out
        self.b.command_pause(ROBOT)


def _explore_job(a: Adapter, stack, tmp_path) -> explore.ExploreJob:
    job = explore.ExploreJob(action_id="explore-1", doc=dict(DOC), stack=stack,
                             limits=stack.battery_limits(),
                             artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"),
                             namespace="lab", clock=_Clock(), emit=lambda *a, **k: None)
    a._job, a._current_action, a._current_fencing_token = job, job.action_id, 7
    return job


def test_a_segments_pause_is_confirmed_only_from_odom_counted_after_its_hold(
        monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.005)
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    b.on_odom(odom_msg(vx=0.3))                             # moving before the pause
    a = Adapter(pb, None, b, ROBOT)
    _explore_job(a, _HoldStack(b), tmp_path)
    got = _pause(a)
    assert got.paused is False, "rest was confirmed from odom received during the hold"


def test_shutdown_marks_the_running_job_before_it_holds_the_base(tmp_path):
    stack = SimulatedExploreStack(clock=_Clock())
    robot = SimulatedRobot(explore=stack)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    job = _explore_job(a, stack, tmp_path)
    seen: list = []
    hold = getattr(robot, "command_hold", None)
    name = "command_hold" if hold is not None else "command_stop"
    original = getattr(robot, name)

    def holding(robot_id):
        original(robot_id)
        # The job thread sends its return right after the hold, under the stack's lock.
        seen.append(stack.go_to_mark(0.0, 0.0, 0.0, job.may_move))
    setattr(robot, name, holding)
    a.shutdown()
    assert seen == [False], "a return goal sent after shutdown's hold was not refused"
    assert not stack.moving()
