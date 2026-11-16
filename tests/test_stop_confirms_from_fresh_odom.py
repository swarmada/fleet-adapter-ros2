# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Every stop is confirmed only from an odom message received after it.

`is_stopped()` answers from the last odom message the binding received. Before this, every stop
path confirmed rest from that message whatever its age: a robot at rest a moment before an
estop, a cancel, a lease lapse, the SafetyStream lost or a shutdown, and moving again by the
time the stop was commanded, was reported STOPPED (or its cancel acknowledged STOPPED_SAFELY,
which lets the control plane give the action to another robot) on a reading from before the
stop.

Now `Nav2Binding.state_seq()` counts odom messages, and every stop reads it once the stop has
been commanded (`command_estop` for the estop, `_stop_token` for the other four) and keeps that
token for every look at that stop. Only a message counted after it confirms the stop (the SDK's
`await_estop_confirmation(..., since=)`, swarmada 5792bb5). A message received while the stop is
being commanded does not count; one received between two looks at the same stop does.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from conftest import ThreadedOdom, mark_transport_live, odom_after_stop, odom_msg
from swarmada_sdk import safety
from swarmada_sdk.safety import ESTOP_FAILED, ESTOP_STOPPED, confirm_estop

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"
ACTION = "t-fresh-odom"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _short_budget(monkeypatch):
    """The stop budget's looks 5 ms apart instead of 50 ms, so a stop that is never confirmed
    answers in 0.2 s. The number of looks, and every rule about which report counts, is
    unchanged."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.005)


def _binding(**kw) -> Nav2Binding:
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source=kw.pop("odometry_source",
                                                                       "measured"), **kw))
    b.on_odom(odom_msg())          # at rest, BEFORE any stop: never enough to confirm one
    return b


def _adapter(b) -> Adapter:
    return Adapter(pb, None, b, ROBOT)


def _estop(a: Adapter, on_ack=None) -> list:
    """One estop through the adapter's own path; the EstopAck states, in order."""
    acks: list = []

    def send(ack) -> None:
        acks.append(ack.state)
        if on_ack is not None:
            on_ack(ack.state)
    a._estop("e-1", send)
    return acks


def _wait_for(a: Adapter, kind: str, timeout: float = 5.0):
    """The first message of `kind` the adapter sent, waiting up to `timeout`. Messages of other
    kinds read on the way are kept for a later call, in order."""
    kept = a.__dict__.setdefault("_test_kept", [])
    for m in kept:
        if m.WhichOneof("payload") == kind:
            kept.remove(m)
            return getattr(m, kind)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = a._outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        if m.WhichOneof("payload") == kind:
            return getattr(m, kind)
        kept.append(m)
    raise AssertionError(f"no {kind} within {timeout} s")


def _running(a: Adapter) -> None:
    """An action running on the robot under a lease, without a goal sent to a server: a goal
    still unanswered at the stop refutes rest by itself (`_stop_pending`), which is another
    test's subject."""
    a._current_action = ACTION
    a._current_fencing_token = 7


def _cancel(a: Adapter):
    result = pb.CommandResult(command_id=2, robot_id=ROBOT)
    a._on_cancel(pb.CancelAction(action_id=ACTION, cause=pb.CANCEL_CAUSE_OPERATOR), result)
    return _wait_for(a, "command_result").cancel_action


def _odom_during_command_stop(b: Nav2Binding, vx: float = 0.0) -> None:
    """The binding receives an at-rest odom message WHILE `command_stop` runs: one the robot
    sent before the stop reached it."""
    original = b.command_stop

    def command_stop(robot_id):
        b.on_odom(odom_msg(vx))
        return original(robot_id)
    b.command_stop = command_stop


# ── the binding's count ─────────────────────────────────────────────────────

def test_state_seq_counts_odom_messages_and_nothing_else():
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured",
                                        rest_scan_topic="scan", rest_scan_threshold_m=0.05))
    assert b.state_seq(ROBOT) == 0
    b.on_odom(odom_msg())
    b.on_odom(odom_msg(vx=0.3))
    assert b.state_seq(ROBOT) == 2
    b.on_scan(NS(header=NS(stamp=NS(sec=1, nanosec=0)), ranges=[1.0, 1.0, 1.0],
                 range_min=0.1, range_max=10.0, angle_min=0.0, angle_increment=0.1))
    b.on_battery(NS(percentage=0.5, power_supply_status=0, voltage=12.0))
    b.on_diagnostics(NS(status=[]))
    b.on_map_to_odom(None)
    b.on_feedback(2.0)
    b.tick(0.05)
    b.command_stop(ROBOT)
    assert b.state_seq(ROBOT) == 2


def test_state_seq_is_an_int_and_is_0_for_a_robot_the_binding_does_not_drive():
    b = _binding()
    assert type(b.state_seq(ROBOT)) is int and b.state_seq(ROBOT) == 1
    assert b.state_seq("amr-2") == 0


# ── estop ───────────────────────────────────────────────────────────────────

def test_an_estop_is_not_confirmed_from_odom_received_before_it():
    """At rest a moment before the estop, and no odom since: nothing says it is at rest NOW."""
    a = _adapter(_binding())
    assert _estop(a) == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]


def test_an_estop_is_not_confirmed_while_the_odom_after_it_shows_motion():
    a = _adapter(odom_after_stop(_binding(), vx=0.4))
    assert _estop(a) == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]


def test_an_estop_is_confirmed_from_odom_received_after_it():
    a = _adapter(odom_after_stop(_binding()))
    assert _estop(a) == [pb.ESTOP_STATE_STOPPED]


def test_odom_received_while_the_stop_is_commanded_does_not_confirm_the_estop():
    b = _binding()
    _odom_during_command_stop(b)
    assert _estop(_adapter(b)) == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]


def test_odom_received_while_the_stopping_receipt_goes_out_confirms_the_estop():
    """The token is read once, when the stop is commanded, and both waits wait past it. A
    second wait that read the count anew would not count this message, and with no other one
    would answer FAILED for a robot at rest."""
    b = _binding()

    def on_ack(state) -> None:
        if state == pb.ESTOP_STATE_STOPPING:
            b.on_odom(odom_msg())
    assert _estop(_adapter(b), on_ack) == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_STOPPED]


def test_odom_received_between_the_stop_and_the_first_look_confirms_the_estop_at_once():
    """The first look waits past the stop's token too: a message that lands after the stop and
    before the look confirms it in a single STOPPED."""
    b = _binding()
    gap = b.rest_evidence_gap

    def rest_evidence_gap(robot_id):          # the adapter asks this between the two
        b.on_odom(odom_msg())
        return gap(robot_id)
    b.rest_evidence_gap = rest_evidence_gap
    assert _estop(_adapter(b)) == [pb.ESTOP_STATE_STOPPED]


def test_fresh_odom_never_confirms_a_stop_on_command_derived_odometry():
    """ADR-0071: a twist that is the command echoed back reads zero the moment the stop is
    commanded. Counting it as fresh changes nothing."""
    a = _adapter(odom_after_stop(_binding(odometry_source="command-derived")))
    assert _estop(a) == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]


# ── cancel ──────────────────────────────────────────────────────────────────

def test_a_cancel_is_not_acknowledged_from_odom_received_before_it():
    a = _adapter(_binding())
    _running(a)
    got = _cancel(a)
    assert got.acknowledged is False
    end = _wait_for(a, "action_status")
    assert end.state == pb.ACTION_STATE_FAILED


def test_a_cancel_is_acknowledged_from_odom_received_after_it():
    a = _adapter(odom_after_stop(_binding()))
    _running(a)
    got = _cancel(a)
    assert got.acknowledged is True
    assert got.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert _wait_for(a, "action_status").state == pb.ACTION_STATE_CANCELLED


def test_odom_received_while_the_cancel_commands_the_stop_does_not_acknowledge_it():
    b = _binding()
    _odom_during_command_stop(b)
    a = _adapter(b)
    _running(a)
    assert _cancel(a).acknowledged is False


def test_a_cancels_later_looks_wait_past_the_token_read_at_the_stop():
    """The first look finds nothing new; the only fresh message arrives after it, before the
    looks the stop reporter takes. Those looks must count it: the token they wait past is the
    one read when the cancel commanded the stop, not one read when they begin."""
    b = _binding()
    a = _adapter(b)
    _running(a)
    defer = a._defer_stop_report

    def deliver_then_defer(job) -> None:
        b.on_odom(odom_msg())
        defer(job)
    a._defer_stop_report = deliver_then_defer
    got = _cancel(a)
    assert got.acknowledged is True
    assert _wait_for(a, "action_status").state == pb.ACTION_STATE_CANCELLED


# ── pause (C11.1: with require_stop_before_ack, paused=true only once confirmed at rest) ──

def _pause(a: Adapter, require_stop: bool = True):
    """A pause through the adapter's own command handling; its PauseResult, when it comes."""
    a._handle_command(pb.Command(command_id=3, robot_id=ROBOT, pause=pb.PauseCapabilities(
        reason="test", require_stop_before_ack=require_stop)))
    return _wait_for(a, "command_result").pause


def test_a_pause_is_not_acknowledged_paused_from_odom_received_before_it():
    a = _adapter(_binding())
    _running(a)
    got = _pause(a)
    assert got.paused is False
    assert "rest was not confirmed" in got.message


def test_a_pause_is_acknowledged_paused_at_once_from_odom_received_after_it():
    """The first look finds a fresh message at rest: the pause is answered on the control
    thread, before the handler returns."""
    a = _adapter(odom_after_stop(_binding()))
    _running(a)
    a._handle_command(pb.Command(command_id=3, robot_id=ROBOT, pause=pb.PauseCapabilities(
        reason="test", require_stop_before_ack=True)))
    sent = []
    while not a._outbox.empty():                      # already sent: nothing is waited for
        sent.append(a._outbox.get_nowait())
    got = [m.command_result.pause for m in sent if m.WhichOneof("payload") == "command_result"]
    assert len(got) == 1, "the pause was not answered before the handler returned"
    assert got[0].paused is True and got[0].message == "nav2: goal cancelled + hold"


def test_a_pause_not_confirmed_at_its_first_look_is_answered_paused_once_fresh_odom_shows_rest():
    """The follow-up: nothing new at the first look, so the answer waits, and is paused=true as
    soon as a message received after the pause shows rest -- not left at "awaiting"."""
    b = _binding()
    a = _adapter(b)
    _running(a)
    deferred: list = []
    defer = a._defer_stop_report

    def deliver_then_defer(job) -> None:
        deferred.append(job)
        b.on_odom(odom_msg())
        defer(job)
    a._defer_stop_report = deliver_then_defer
    got = _pause(a)
    assert len(deferred) == 1, "the pause was answered at its first look"
    assert got.paused is True and got.message == "nav2: goal cancelled + hold"


def _results_by_id(a: Adapter, ids: set) -> dict:
    """The CommandResults for these command ids, in the order sent, waiting up to 5 s for each."""
    got: dict = {}
    order: list = []
    while set(got) != ids:
        r = _wait_for(a, "command_result")
        got[r.command_id] = r
        order.append(r.command_id)
    got["order"] = order
    return got


def _command_while_the_pause_waits(a: Adapter, b: Nav2Binding, command) -> dict:
    """A pause whose first look finds nothing new; before its later looks, `command` is read and
    acted on, and then a fresh at-rest message arrives. Returns the CommandResults by id."""
    defer = a._defer_stop_report

    def command_then_rest_then_defer(job) -> None:
        a._handle_command(command)
        b.on_odom(odom_msg())          # rest seen after the command: not the pause's rest
        defer(job)
    a._defer_stop_report = command_then_rest_then_defer
    a._handle_command(pb.Command(command_id=3, robot_id=ROBOT, pause=pb.PauseCapabilities(
        reason="test", require_stop_before_ack=True)))
    return _results_by_id(a, {3, command.command_id})


def test_a_resume_while_a_pause_waits_for_rest_supersedes_it():
    """C11.1: paused=true says the robot is held and at rest. After a resume it is not held, so
    rest seen then is not the pause's, and the pause answers paused=false -- never paused=true
    after resumed=true."""
    b = _binding()
    a = _adapter(b)
    _running(a)
    got = _command_while_the_pause_waits(a, b, pb.Command(
        command_id=4, robot_id=ROBOT, resume=pb.ResumeCapabilities(reason="test")))
    assert got[4].resume.resumed is True
    assert got[3].pause.paused is False
    assert "a later resume or assignment" in got[3].pause.message


def test_an_assignment_while_a_pause_waits_for_rest_supersedes_it():
    b = _binding()
    b._send_nav2_goal = lambda goal, epoch: None     # a server that takes the goal
    b._server_ready = lambda: True
    a = _adapter(b)
    _running(a)
    got = _command_while_the_pause_waits(a, b, pb.Command(
        command_id=5, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="t-next", assignment_id="a-next", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=12, lease_generation=1,
            lease_duration_ms=60_000)))
    assert got[5].assign_action.accepted is True
    assert got[3].pause.paused is False
    assert "a later resume or assignment" in got[3].pause.message


def test_a_job_assigned_while_a_pause_waits_for_rest_supersedes_it():
    """A job (here a manual-drive, which opens the teleop gate) moves the robot from its own
    thread: its assignment counts as setting the robot moving, as a Navigate's does."""
    b = _binding(teleop_topic="/cmd_vel_teleop")
    b._cmd_vel, b._twist_type = NS(publish=lambda _msg: None), lambda: NS(
        linear=NS(x=0.0, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))
    a = _adapter(b)
    _running(a)
    body = json.dumps({"operation": "manual-drive", "x": 1.0, "y": 0.0, "frame": "map",
                       "radius_m": 0.25}).encode()
    got = _command_while_the_pause_waits(a, b, pb.Command(
        command_id=6, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="manual-next", action_type="Custom", assignment_id="a-manual",
            payload_json=body, fencing_token=13, lease_generation=1,
            lease_duration_ms=60_000)))
    assert got[6].assign_action.accepted is True
    assert got[3].pause.paused is False
    assert "a later resume or assignment" in got[3].pause.message


def test_odom_received_while_the_pause_is_commanded_does_not_acknowledge_it():
    b = _binding()
    original = b.command_pause

    def command_pause(robot_id):
        b.on_odom(odom_msg())
        return original(robot_id)
    b.command_pause = command_pause
    a = _adapter(b)
    _running(a)
    assert _pause(a).paused is False


def test_a_pause_without_require_stop_before_ack_claims_the_hold_at_once():
    """Without require_stop_before_ack the answer claims the hold, not rest: paused=true at
    once, whatever odom has or has not said."""
    a = _adapter(_binding())
    _running(a)
    got = _pause(a, require_stop=False)
    assert got.paused is True and got.message == "nav2: goal cancelled + hold"


# ── lease self-stop ─────────────────────────────────────────────────────────

def _lapse(a: Adapter):
    """The lease lapses and the adapter's lease look runs: the stop, then its report."""
    _running(a)
    a._lease.grant(ROBOT, -1.0, 1)             # its deadline is already past
    a._tick_lease()
    return _wait_for(a, "action_status")


def test_a_lease_self_stop_is_not_reported_stopped_from_odom_received_before_it():
    end = _lapse(_adapter(_binding()))
    assert end.state == pb.ACTION_STATE_FAILED


def test_a_lease_self_stop_is_reported_stopped_from_odom_received_after_it():
    end = _lapse(_adapter(odom_after_stop(_binding())))
    assert end.state == pb.ACTION_STATE_STOPPED


def test_odom_received_while_the_lease_lapse_commands_the_stop_does_not_confirm_it():
    b = _binding()
    _odom_during_command_stop(b)
    assert _lapse(_adapter(b)).state == pb.ACTION_STATE_FAILED


def test_a_lease_self_stop_waits_past_the_token_read_at_the_lapse():
    """A message received between the stop at the lapse and the report's looks counts: the
    token is read where the stop is commanded, under the lease lock, not where it is reported."""
    b = _binding()
    a = _adapter(b)
    self_stop = a._self_stop

    def deliver_then_report(*args) -> None:
        b.on_odom(odom_msg())
        self_stop(*args)
    a._self_stop = deliver_then_report
    assert _lapse(a).state == pb.ACTION_STATE_STOPPED


# ── the SafetyStream lost ───────────────────────────────────────────────────

def _lose_safety_stream(a: Adapter):
    _running(a)
    a._safety_up.set()
    a._safety_down("test: the stream ended", open_for=10.0)
    return _wait_for(a, "action_status")


def test_a_lost_safety_stream_does_not_report_rest_from_odom_received_before_the_stop():
    end = _lose_safety_stream(_adapter(_binding()))
    assert end.state == pb.ACTION_STATE_FAILED
    assert "rest was not observed" in end.message


def test_a_lost_safety_stream_reports_rest_from_odom_received_after_the_stop():
    end = _lose_safety_stream(_adapter(odom_after_stop(_binding())))
    assert end.state == pb.ACTION_STATE_FAILED          # the action ends FAILED either way
    assert end.message.endswith("no estop could reach the robot")


def test_a_lost_safety_stream_waits_past_the_token_read_at_the_stop():
    """A message received after the stop, while the running job is ended and before the report
    thread looks, counts: the token is read where the stop is commanded, not in that thread."""
    b = _binding()
    a = _adapter(b)
    halt_job = a._halt_job

    def deliver_then_halt() -> None:
        b.on_odom(odom_msg())
        halt_job()
    a._halt_job = deliver_then_halt
    assert _lose_safety_stream(a).message.endswith("no estop could reach the robot")


# ── shutdown ────────────────────────────────────────────────────────────────

def test_shutdown_does_not_report_rest_from_odom_received_before_the_hold(capsys):
    _adapter(_binding()).shutdown()
    assert "shutdown: rest was not confirmed" in capsys.readouterr().err


def test_shutdown_reports_rest_from_odom_received_after_the_hold(capsys):
    _adapter(odom_after_stop(_binding())).shutdown()
    assert "shutdown: the robot is at rest" in capsys.readouterr().err


# ── the simulated robot ─────────────────────────────────────────────────────

def test_the_simulated_robot_counts_its_ticks_as_state_reports_for_robots_it_spawned():
    r = SimulatedRobot()
    assert r.state_seq(ROBOT) == 0
    r.spawn(ROBOT)
    r.tick(0.1)
    r.tick(0.1)
    assert r.state_seq(ROBOT) == 2
    assert r.state_seq("never-spawned") == 0


def test_the_simulated_robot_still_confirms_an_estop_at_the_first_look(monkeypatch, caplog):
    """Each tick is a report after the stop, so the simulated estop answers as before, and the
    SDK has no binding without a count to warn about."""
    monkeypatch.setattr(safety, "_warned_without_state_seq", set())
    r = SimulatedRobot()
    r.spawn(ROBOT)
    with caplog.at_level(logging.WARNING, logger=safety.__name__):
        assert _estop(_adapter(r)) == [pb.ESTOP_STATE_STOPPED]
    assert "has no state_seq" not in caplog.text


def test_the_simulated_robot_never_confirms_a_stop_of_a_robot_it_never_spawned():
    assert confirm_estop(SimulatedRobot(), "never-spawned", tick_dt=0.0, max_ticks=5,
                         sleep=lambda _s: None) == ESTOP_FAILED
    r = SimulatedRobot()
    r.spawn(ROBOT)
    assert confirm_estop(r, ROBOT, tick_dt=0.0, max_ticks=5,
                         sleep=lambda _s: None) == ESTOP_STOPPED


# ── an SDK without EstopToken ───────────────────────────────────────────────

def test_the_adapter_refuses_an_sdk_without_estop_token(tmp_path):
    """Everything this adapter imports from the SDK, as an SDK from before 5792bb5 has it."""
    sdk = tmp_path / "swarmada_sdk"
    sdk.mkdir()
    (sdk / "__init__.py").write_text("")
    (sdk / "safety.py").write_text(textwrap.dedent("""
        import enum
        ESTOP_STOPPED, ESTOP_FAILED = "STOPPED", "FAILED"
        class FenceDecision(enum.Enum):
            ACCEPT = "accept"
        class FenceGuard:
            def adopt(self, robot, token):
                return False
        class LeaseMonitor:
            pass
        def command_estop(sim, robot):
            sim.command_stop(robot)
        def await_estop_confirmation(sim, robot, tick_dt=0.05, max_ticks=40, sleep=None):
            return ESTOP_FAILED
    """))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]))
    run = subprocess.run([sys.executable, "-c", "import fleet_adapter_ros2.adapter"],
                         capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60,
                         check=False)
    assert run.returncode != 0
    assert "swarmada_sdk.safety has no EstopToken" in run.stderr


# ── how long an unconfirmed stop takes to answer ────────────────────────────

# The harness's liveness bound on a terminal estop answer (C5.6, _ESTOP_TERMINAL_BUDGET_S). The
# four stop paths below answer an unconfirmed stop after the stop budget's 40 looks 0.05 s apart,
# 2.00 s: a target, not a spec value (C5.5 bounds only the first EstopAck). Their own harness
# waits are 4.0 s (cancel, C16.1; lease, C4.2), so this is the stricter of the bounds.
ANSWER_BOUND_S = 3.0


def _answer_cancel(a: Adapter) -> None:
    _running(a)
    assert _cancel(a).acknowledged is False


def _answer_lease(a: Adapter) -> None:
    assert _lapse(a).state == pb.ACTION_STATE_FAILED


def _answer_safety_lost(a: Adapter) -> None:
    assert "rest was not observed" in _lose_safety_stream(a).message


def _answer_shutdown(a: Adapter) -> None:
    a.shutdown()


def _answer_pause(a: Adapter) -> None:
    _running(a)
    assert _pause(a).paused is False


@pytest.mark.parametrize("answer", [_answer_cancel, _answer_lease, _answer_safety_lost,
                                    _answer_shutdown, _answer_pause],
                         ids=["cancel", "lease-self-stop", "safety-stream-lost", "shutdown",
                              "pause"])
def test_an_unconfirmed_stop_answers_within_the_bound(monkeypatch, answer):
    """At the real look spacing, with no odom after the stop: the answer is the unconfirmed one,
    and it arrives before ANSWER_BOUND_S, measured from the call that commands the stop."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.05)
    a = _adapter(_binding())
    start = time.monotonic()
    answer(a)
    took = time.monotonic() - start
    assert 1.9 <= took < ANSWER_BOUND_S, f"answered in {took:.2f} s"


# ── on a live odom topic, at its own rate ───────────────────────────────────

# The plant's speed falls below is_stopped's 0.01 m/s about 2.5 % before the end of its 0.3 s
# braking, so rest can be seen from about 0.29 s after the stop.
BRAKED_S = 0.25


@pytest.fixture
def live(monkeypatch):
    """The adapter on a binding whose odom comes from its own thread at 30 Hz (the Gazebo rig's
    measured rate), over a base that cruises until a stop and brakes to rest in 0.3 s; the
    looks at the real 0.05 s spacing."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.05)
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    topic = ThreadedOdom(b)
    a = _adapter(b)
    yield a, b, topic
    topic.close()


def test_on_a_live_odom_topic_an_estop_is_receipted_then_confirmed_once_the_base_has_braked(live):
    a, _, topic = live
    times: list = []
    acks = _estop(a, on_ack=lambda _state: times.append(time.monotonic()))
    assert acks == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_STOPPED]
    assert times[0] - topic.stopped_at < 0.5                    # C5.5: the receipt at once
    assert BRAKED_S <= times[1] - topic.stopped_at < 1.0        # rest, once the base is at rest


def test_on_a_live_odom_topic_a_cancel_is_deferred_then_acknowledged_within_the_harness_wait(live):
    """The harness waits 4.0 s for a cancel's end (C16.1, _TERMINAL_STATUS_WAIT)."""
    a, _, topic = live
    _running(a)
    deferred: list = []
    defer = a._defer_stop_report
    a._defer_stop_report = lambda job: (deferred.append(job), defer(job))[1]
    got = _cancel(a)
    took = time.monotonic() - topic.stopped_at
    assert len(deferred) == 1, "the cancel was confirmed at its first look"
    assert got.acknowledged is True
    assert got.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert _wait_for(a, "action_status").state == pb.ACTION_STATE_CANCELLED
    assert BRAKED_S <= took < 4.0


def test_on_a_live_odom_topic_a_pause_is_answered_paused_once_the_base_has_braked(live):
    a, _, topic = live
    _running(a)
    got = _pause(a)
    assert got.paused is True
    assert BRAKED_S <= time.monotonic() - topic.stopped_at < 2.0
