# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A robot that stops making progress is reported FAILED, not RUNNING forever.

Driven through the adapter with the real protobuf messages, on a ROS-free Nav2Binding whose
motion outputs are recorded: the NavigateToPose dispatch, the goal handle's cancel, and every
/cmd_vel Twist. On a stall the goal is cancelled, one zero Twist (the halt) is published and
nothing after it, and the action is reported FAILED through the same terminal path an ABORTED
goal takes, which releases the lease. A paused goal, and the time before the first feedback,
are not stalls. `--binding ros2` without thresholds refuses to start.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2 import task
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
WINDOW, MIN_PROGRESS = 2.0, 0.1


def _limits(window: float = WINDOW, min_progress: float = MIN_PROGRESS):
    from fleet_adapter_ros2.watchdog import StallLimits
    return StallLimits(window_s=window, min_progress_m=min_progress)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Twist:
    """Stands in for geometry_msgs/Twist: zero on construction, like the real one."""

    def __init__(self) -> None:
        self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _Publisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _Handle:
    """An accepted NavigateToPose goal handle: records cancels and the result callback."""

    accepted = True

    def __init__(self) -> None:
        self.cancels = 0
        self.on_result = None

    def cancel_goal_async(self) -> None:
        self.cancels += 1

    def get_result_async(self):
        handle = self

        class _F:
            def add_done_callback(self, cb) -> None:
                handle.on_result = cb
        return _F()


class _Done:
    def __init__(self, value) -> None:
        self._v = value

    def result(self):
        return self._v


def _spied(clock=None, limits="default"):
    b = mark_transport_live(Nav2Binding(
        ROBOT, stall_limits=_limits() if limits == "default" else limits, clock=clock))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append((goal, epoch))
    b._server_ready = lambda: True                        # the server is up: goals go at once
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    return Adapter(pb, None, b, ROBOT), b, sent


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _statuses(out: list) -> list:
    return [m.action_status for m in out if m.WhichOneof("payload") == "action_status"]


def _command(a: Adapter, n: int, **arm) -> list:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=n, robot_id=ROBOT, **arm)), None)
    return _drain(a)


def _assign_and_accept(a: Adapter, b: Nav2Binding, sent: list, token: int = 7) -> _Handle:
    """Assign a served Navigate, then have the server accept it and hand back a handle."""
    out = _command(a, token, assign_action=pb.AssignAction(
        action_id="t-1", assignment_id="a-1", action_type="Navigate",
        payload_json=b'{"x": 3, "y": 0}', fencing_token=token, lease_generation=1,
        lease_duration_ms=60_000))
    assert [s.state for s in _statuses(out)] == [pb.ACTION_STATE_RUNNING]
    h = _Handle()
    b._on_goal_response(_Done(h), sent[-1][1])
    return h


def _is_zero(tw) -> bool:
    return all(getattr(v, k) == 0.0 for v in (tw.linear, tw.angular) for k in "xyz")


# ── the stall: FAILED, cancelled, halted, and nothing after ─────────────────

def test_a_stall_cancels_the_goal_halts_once_and_reports_failed_with_the_reason() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)
    b.on_feedback(3.0)
    first = c.t
    c.t += 0.5
    b.on_feedback(2.96)                      # 4 cm, then nothing: the stall shape
    while c.t + 0.1 < first + WINDOW:        # up to 1.9 s after the first feedback: not yet
        c.t += 0.1
        assert b.check_progress() == ""
        a._maybe_emit_task_progress()
    assert b.task_state(ROBOT) == task.TASK_RUNNING and h.cancels == 0
    c.t = first + WINDOW
    reason = b.check_progress()
    assert reason.startswith("stalled: distance to go improved 0.040 m in the last 2.0 s")

    # The goal is cancelled, and the only /cmd_vel is the halt.
    assert h.cancels == 1
    assert len(b._cmd_vel.published) == 1 and _is_zero(b._cmd_vel.published[0])
    assert b.pending_goal(ROBOT) is None

    # Reported FAILED, once, with the reason and the cause, through the terminal path.
    a._maybe_emit_task_progress()
    st = _statuses(_drain(a))
    assert len(st) == 1
    assert st[0].action_id == "t-1" and st[0].state == pb.ACTION_STATE_FAILED
    assert st[0].message == reason
    assert "less than the 0.1 m required within 2 s" in st[0].message
    assert st[0].end_cause == pb.ACTION_END_CAUSE_EXECUTION_FAILURE
    assert st[0].fencing_token == 7
    assert a._current_action == "" and ROBOT not in a._lease._deadline   # lease released

    # Nothing further: no /cmd_vel, no re-dispatch, no second report.
    for _ in range(100):
        c.t += 0.1
        assert b.check_progress() == ""
        a._maybe_emit_task_progress()
    assert len(b._cmd_vel.published) == 1 and len(sent) == 1 and h.cancels == 1
    assert _statuses(_drain(a)) == []


def test_the_cancelled_goals_result_does_not_turn_failed_into_cancelled() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)
    b.on_feedback(3.0)
    c.t += WINDOW
    assert b.check_progress()
    h.on_result(_Done(SimpleNamespace(status=5)))   # the server answers the cancel: CANCELED
    assert b.task_state(ROBOT) == task.TASK_FAILED


def test_a_goal_accepted_after_a_stall_was_declared_is_cancelled_on_arrival() -> None:
    # No feedback, and the server has not answered: the stall comes first. The goal the server
    # then accepts would drive a robot already reported FAILED.
    c = Clock()
    a, b, sent = _spied(c)
    _command(a, 7, assign_action=pb.AssignAction(
        action_id="t-1", assignment_id="a-1", action_type="Navigate",
        payload_json=b'{"x": 3, "y": 0}', fencing_token=7, lease_duration_ms=60_000))
    c.t += WINDOW
    assert "no progress feedback" in b.check_progress()
    h = _Handle()
    b._on_goal_response(_Done(h), sent[-1][1])
    assert h.cancels == 1 and b._goal_handle is None and h.on_result is None


def test_a_stall_is_reported_within_the_window_plus_one_period_on_a_real_clock() -> None:
    period = 0.1                               # nav2_binding._WATCHDOG_PERIOD_S
    window = 0.3
    a, b, sent = _spied(limits=_limits(window, MIN_PROGRESS))
    _assign_and_accept(a, b, sent)
    b.on_feedback(3.0)
    last_progress = time.monotonic()
    failed_at = None
    while time.monotonic() - last_progress < 5 * window:
        time.sleep(period)
        b.check_progress()                     # what the executor timer does
        a._maybe_emit_task_progress()          # what the telemetry loop does
        if any(s.state == pb.ACTION_STATE_FAILED for s in _statuses(_drain(a))):
            failed_at = time.monotonic()
            break
    assert failed_at is not None
    assert window <= failed_at - last_progress <= window + period + 0.1


# ── not a stall ─────────────────────────────────────────────────────────────

def test_steady_progress_never_fires() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    _assign_and_accept(a, b, sent)
    d = 30.0
    for _ in range(500):                       # 50 s at 0.5 m/s, 25 windows
        c.t += 0.1
        d -= 0.05
        b.on_feedback(d)
        assert b.check_progress() == ""
        a._maybe_emit_task_progress()
    assert b.task_state(ROBOT) == task.TASK_RUNNING
    assert not any(s.state == pb.ACTION_STATE_FAILED for s in _statuses(_drain(a)))
    assert a._current_action == "t-1" and ROBOT in a._lease._deadline


def test_a_paused_goal_never_fires_and_resume_watches_it_again() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    _assign_and_accept(a, b, sent)
    b.on_feedback(3.0)
    c.t += 1.0
    out = _command(a, 8, pause=pb.PauseCapabilities(reason="operator"))
    res = [m.command_result for m in out if m.WhichOneof("payload") == "command_result"][0]
    assert res.WhichOneof("result") == "pause"           # C11.1: answered
    for _ in range(600):                                 # held for 60 s, 30 windows
        c.t += 0.1
        assert b.check_progress() == ""
        a._maybe_emit_task_progress()
    assert not any(s.state == pb.ACTION_STATE_FAILED for s in _statuses(_drain(a)))
    assert a._current_action == "t-1"

    _command(a, 9, resume=pb.ResumeCapabilities(reason="operator"))
    assert len(sent) == 2                                # the goal was re-dispatched
    c.t += WINDOW - 0.1
    assert b.check_progress() == ""                      # a fresh wait for feedback
    b.on_feedback(3.0)
    c.t += WINDOW
    assert b.check_progress()                            # and watched again once it moves on


def test_the_time_before_the_first_feedback_is_bounded_by_one_window() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    _assign_and_accept(a, b, sent)
    c.t += WINDOW - 0.01
    assert b.check_progress() == ""
    c.t += 0.01
    assert b.check_progress().startswith("stalled: no progress feedback arrived within 2.0 s")
    a._maybe_emit_task_progress()
    assert [s.state for s in _statuses(_drain(a))] == [pb.ACTION_STATE_FAILED]


def test_a_goal_that_ends_is_no_longer_watched() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)
    b.on_feedback(0.0)
    h.on_result(_Done(SimpleNamespace(status=4)))        # SUCCEEDED
    c.t += 10 * WINDOW
    assert b.check_progress() == "" and b.task_state(ROBOT) == task.TASK_SUCCEEDED


def test_an_unarmed_binding_never_fires() -> None:
    c = Clock()
    a, b, sent = _spied(c, limits=None)
    _assign_and_accept(a, b, sent)
    c.t += 3600
    assert b.check_progress() == "" and b.task_state(ROBOT) == task.TASK_RUNNING


# ── the one terminal path ───────────────────────────────────────────────────

def test_an_aborted_goal_releases_the_lease_on_the_same_path_with_the_servers_reason() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)
    h.on_result(_Done(SimpleNamespace(status=6)))        # ABORTED by the server
    a._maybe_emit_task_progress()
    st = _statuses(_drain(a))
    assert [s.state for s in st] == [pb.ACTION_STATE_FAILED]
    assert st[0].message == "the navigation server aborted the goal"
    assert st[0].end_cause == pb.ACTION_END_CAUSE_EXECUTION_FAILURE
    assert ROBOT not in a._lease._deadline and a._current_action == ""


def test_renewal_after_a_stall_says_the_action_is_not_running() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    _assign_and_accept(a, b, sent)
    c.t += WINDOW
    assert b.check_progress()
    a._maybe_emit_task_progress()
    out = _command(a, 10, renew_lease=pb.RenewActionLease(
        action_id="t-1", lease_generation=1, lease_duration_ms=60_000))
    res = [m.command_result.renew_lease for m in out
           if m.WhichOneof("payload") == "command_result"][0]
    assert res.running is False                         # C4.4


# ── start-up: no thresholds, no ros2 adapter ────────────────────────────────

class _Stop(Exception):
    pass


def _main(monkeypatch, capsys, *argv: str):
    """Run adapter.main() up to the point it would connect; return (exit code, stderr, robot)."""
    built = {}

    def _adapter(pb_, pb_grpc, robot, robot_id, **_kw):
        built["robot"] = robot
        raise _Stop

    monkeypatch.setattr(sys, "argv", ["adapter", *argv])
    monkeypatch.setattr(adapter_mod, "fleet_adapter_ros2Adapter", _adapter)
    monkeypatch.setattr(adapter_mod.grpc, "insecure_channel",
                        lambda *_: pytest.fail("the adapter connected"))
    code = None
    try:
        adapter_mod.main()
    except SystemExit as exc:
        code = exc.code
    except _Stop:
        pass
    return code, capsys.readouterr().err, built.get("robot")


def test_ros2_without_thresholds_refuses_to_start_naming_the_settings(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2")
    assert code == 2 and robot is None
    assert err.startswith("REFUSED: --binding ros2 needs --stall-window and --stall-min-progress")


def test_ros2_with_one_threshold_refuses_to_start(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5")
    assert code == 2 and robot is None and "--stall-min-progress" in err


def test_ros2_with_thresholds_starts_armed_and_says_so(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2",
                             "--stall-window", "5", "--stall-min-progress", "0.05",
                             "--goal-response-bound", "1",
                             "--odometry-source", "measured")
    assert code is None and isinstance(robot, Nav2Binding)
    assert robot.stall_limits == _limits(5.0, 0.05)
    assert "progress watchdog: armed" in err and "--stall-window 5" in err


def test_the_simulated_binding_starts_unarmed_and_says_so(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "simulated")
    assert code is None and robot is not None
    assert "progress watchdog: not armed" in err


# ── a stale first feedback: shorter than the straight line to the goal ───────
# Nav2's first feedback for a new goal can still carry the previous plan's distance. On a MentorPi
# it read 0.20 m with the robot 2.96 m from the goal, then 3.01 m once the goal's own plan was in.
# Taken as the baseline, that 0.20 made every real distance look like no progress, and a goal that
# was getting closer around an obstacle was cancelled as stalled. A path can never be shorter than
# the straight line to its goal, so such a distance is not this goal's.

def _feedback(d: float, x: float, y: float):
    """A NavigateToPose FeedbackMessage as rclpy hands it to the feedback callback."""
    pos = SimpleNamespace(x=x, y=y, z=0.0)
    return SimpleNamespace(feedback=SimpleNamespace(
        distance_remaining=d, current_pose=SimpleNamespace(pose=SimpleNamespace(position=pos))))


# The recorded shape, scaled to this file's 2 s window: stale 0.20, the real 3.01, a stop at the
# obstacle (flat for 1.2 s), then the detour closing the distance.
_DETOUR = [(0.00, 0.20, 0.03, 0.0), (0.02, 3.01, 0.03, 0.0), (0.4, 2.61, 0.4, 0.0),
           (0.8, 2.18, 0.8, 0.0), (1.2, 2.10, 0.9, 0.0), (2.4, 2.12, 0.9, 0.1),
           (3.0, 1.83, 1.2, 0.3), (3.6, 1.41, 1.6, 0.3), (4.2, 1.13, 1.9, 0.2), (4.8, 1.07, 1.95, 0.1)]


def test_a_stale_first_feedback_is_ignored_and_a_goal_that_is_getting_closer_is_not_a_stall() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)                  # goal (3, 0)
    t0 = c.t
    for dt, d, x, y in _DETOUR:
        c.t = t0 + dt
        b._on_nav_feedback(_feedback(d, x, y))
        assert b.check_progress() == ""
    assert h.cancels == 0 and b.task_state(ROBOT) == task.TASK_RUNNING
    assert b.task_progress(ROBOT) > 0                   # the baseline is 3.01, not the stale 0.20


def test_a_real_stall_after_a_stale_first_feedback_still_fires() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    h = _assign_and_accept(a, b, sent)
    t0 = c.t
    b._on_nav_feedback(_feedback(0.20, 0.03, 0.0))      # stale: ignored
    c.t = t0 + 0.02
    b._on_nav_feedback(_feedback(3.01, 0.03, 0.0))      # the real baseline
    while c.t < t0 + 0.02 + WINDOW:
        c.t += 0.1
        b._on_nav_feedback(_feedback(3.0, 0.03, 0.0))   # no progress at all
    assert b.check_progress().startswith("stalled: distance to go improved 0.010 m")
    assert h.cancels == 1


def test_a_distance_at_or_above_the_straight_line_is_used() -> None:
    c = Clock()
    a, b, sent = _spied(c)
    _assign_and_accept(a, b, sent)                      # goal (3, 0)
    b.on_feedback(2.95, pose=(0.0, 0.0))                # 5 cm under 3.00: within the margin, used
    assert b.task_progress(ROBOT) == 0 and b._nav_initial == 2.95
