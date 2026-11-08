# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A robot whose odometry is the command it was given cannot confirm its own stop.

`Nav2Binding` confirms rest from the odom twist. On some bases that twist is not a
measurement: the drive controller integrates the velocity command and publishes it back, with
no encoder anywhere in the path. On such a base the twist reads **zero the instant a stop is
commanded**, whatever the wheels are doing -- so `is_stopped()` answers the command, not the
robot, and every path that "confirms rest" from it (the estop's STOPPED, the lease self-stop's
STOPPED, and the cancel's STOPPED_SAFELY, which lets a control plane requeue the action to a
second robot) reports a confirmation the platform cannot give. CONFORMANCE.md C5.3 forbids
exactly that, and ADR-0071 states the precondition: a STOPPED needs a motion channel
INDEPENDENT OF THE COMMAND PATH.

Nothing on the wire or in the ROS graph tells the two kinds of odometry apart -- same topic,
same message type, same frame -- and the difference only shows when the drive disobeys, which
is the one moment a stop report matters. So the adapter is told, by `--odometry-source`, with
no default and required on `--binding ros2`, exactly as the watchdog's thresholds and the goal
response bound are.

The rig below is that platform in miniature: the binding's odom goes to zero the moment a stop
is commanded, while `plant_moving` records that the wheels are still turning. Declared
`measured`, the adapter believes it and reports a confirmed stop -- which is the defect, and it
is asserted here so the declaration is shown to be what changes the answer. Declared
`command-derived`, no path reports a confirmed stop, and every report says why.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from test_stall_reporting import _main
from test_stop_never_waits import (  # noqa: F401 - ros_msgs is a fixture
    ROBOT,
    _assign,
    _ControlPlane,
    _first,
    _Handle,
    _odom,
    _Publisher,
    _SafetyStub,
    _Server,
    _Twist,
    ros_msgs,
)

from conftest import mark_transport_live
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

STOP_S = 2.0    # the confirmed-stop budget, 40 looks 0.05 s apart
CRUISE = 0.4    # m/s, what the plant is actually doing when the stop arrives


class _CommandDerivedBinding(Nav2Binding):
    """A binding whose odom twist is the last velocity command, as some drive controllers
    publish it back: the stop command zeroes it at once, and nothing else ever does.

    `plant_moving` is what the WHEELS are doing. No channel on this platform reports it --
    that is the whole point -- so it exists only for the assertions to read.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plant_moving = False

    def command_stop(self, robot_id: str) -> None:
        super().command_stop(robot_id)
        self.on_odom(_odom(0.0))      # the command, echoed back, before a wheel has slowed

    def drive(self) -> None:
        """The robot is cruising: the command is out, and the echo reports it."""
        self.plant_moving = True
        self.on_odom(_odom(CRUISE))


def _rig(source: str | None):
    """The adapter, its control and lease threads running, on a command-derived binding
    declaring `source`. Returns (adapter, binding, server, control-plane, teardown)."""
    server = _Server(up=True)
    b = mark_transport_live(_CommandDerivedBinding(ROBOT, odometry_source=source))
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b._cancel_goal_by_id = lambda _id: None
    b.start = lambda: None
    b.on_odom(_odom(0.0))
    a = Adapter(pb, None, b, ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    assert cp.opened.wait(2.0)

    def teardown() -> None:
        a._shutdown.set()
        cp.q.put(None)
        control.join(timeout=5.0)
    return a, b, server, cp, teardown


@pytest.fixture(params=["command-derived", "measured"])
def rig(request, ros_msgs):  # noqa: F811 - the imported fixture
    """Both declarations over the same plant, so every test states what each one answers."""
    a, b, server, cp, teardown = _rig(request.param)
    yield request.param, a, b, server, cp
    teardown()


def _result(a, kind: str, within: float):
    return _first(a._outbox, lambda m: m.WhichOneof("payload") == "command_result"
                  and m.command_result.WhichOneof("result") == kind, within)


def _statuses(a, action_id: str) -> list:
    out = []
    while not a._outbox.empty():
        m = a._outbox.get_nowait()
        if m.WhichOneof("payload") == "action_status" and m.action_status.action_id == action_id:
            out.append(m.action_status)
    return out


def _terminal_states():
    return (pb.ACTION_STATE_SUCCEEDED, pb.ACTION_STATE_FAILED,
            pb.ACTION_STATE_CANCELLED, pb.ACTION_STATE_STOPPED)


def _answer_and_terminal(a, action_id: str, within: float):
    """The CancelActionResult and the TERMINAL action_status for `action_id`, in whichever
    order they arrive. Both are read from the one outbox, so a helper that took only the
    first would drop the other -- which is precisely the evidence this test is about."""
    res = term = None
    deadline = time.monotonic() + within
    while (left := deadline - time.monotonic()) > 0 and (res is None or term is None):
        try:
            m = a._outbox.get(timeout=left)
        except Exception:
            break
        kind = m.WhichOneof("payload")
        if (kind == "command_result"
                and m.command_result.WhichOneof("result") == "cancel_action"):
            res = m.command_result.cancel_action
        elif (kind == "action_status" and m.action_status.action_id == action_id
                and m.action_status.state in _terminal_states()):
            term = m.action_status
    return res, term


def _driving(a, b, server, cp, lease_ms: int = 60_000):
    """t-1 assigned, accepted, and the robot cruising with the command echoed back as odom."""
    cp.send(1, assign_action=_assign("t-1", 1, lease_ms=lease_ms))
    assert _result(a, "assign_action", 2.0)
    h = _Handle()
    server.futures[0].cb(NS(result=lambda: h))
    assert b._goal_handle is h
    b.drive()
    return h


# ── the binding itself ───────────────────────────────────────────────────────────────────

def test_odom_that_mirrors_the_command_never_confirms_rest_when_declared(ros_msgs):  # noqa: F811
    b = mark_transport_live(_CommandDerivedBinding(ROBOT, odometry_source="command-derived"))
    b.drive()
    b.on_odom(_odom(0.0))                       # the echo of a stop, with the wheels turning
    assert b.plant_moving is True
    assert b.is_stopped(ROBOT) is False
    assert "command-derived" in b.rest_evidence_gap(ROBOT)


def test_the_same_odom_declared_measured_is_believed(ros_msgs):  # noqa: F811
    """The defect, stated as a test: nothing but the declaration separates these two."""
    b = mark_transport_live(_CommandDerivedBinding(ROBOT, odometry_source="measured"))
    b.drive()
    b.on_odom(_odom(0.0))
    assert b.plant_moving is True
    assert b.is_stopped(ROBOT) is True          # believed, because it was declared a measurement
    assert b.rest_evidence_gap(ROBOT) == ""


def test_an_undeclared_binding_has_a_gap_because_nobody_said_which_it_is(ros_msgs):  # noqa: F811
    b = mark_transport_live(_CommandDerivedBinding(ROBOT))
    assert "undeclared" in b.rest_evidence_gap(ROBOT)


def test_the_simulated_bindings_plant_model_is_its_own_witness():
    assert SimulatedRobot().rest_evidence_gap(ROBOT) == ""


@pytest.mark.parametrize("value", ["", "measured-ish", "encoders", 1, True])
def test_the_binding_refuses_an_odometry_source_it_does_not_define(value):
    with pytest.raises(ValueError, match="odometry_source must be"):
        Nav2Binding(ROBOT, odometry_source=value)


# ── every stop path (C5.2/C5.3, C4.2, C3.6) ──────────────────────────────────────────────

def test_an_estop_is_never_stopped_on_command_derived_odometry(rig):
    source, a, b, server, cp = rig
    _driving(a, b, server, cp)
    safety = threading.Thread(
        target=a.run_safety_stream, args=(_SafetyStub(),), daemon=True)
    safety.start()
    deadline = time.time() + STOP_S + 1.0
    states: list = []
    while time.time() < deadline and (not states or states[-1] == pb.ESTOP_STATE_STOPPING):
        try:
            m = a._safety_outbox.get(timeout=0.2)
        except Exception:
            continue
        if m.WhichOneof("payload") == "estop_ack":
            states.append(m.estop_ack.state)
            last = m.estop_ack
    assert states, "no EstopAck at all"
    assert b.plant_moving is True, "the rig's plant never moved; the test proves nothing"
    if source == "command-derived":
        assert states[-1] == pb.ESTOP_STATE_FAILED
        assert pb.ESTOP_STATE_STOPPED not in states
        assert "command-derived" in last.message
    else:
        assert pb.ESTOP_STATE_STOPPED in states, "the defect this fixes is not reproduced"


def test_a_lease_self_stop_is_failed_not_stopped_on_command_derived_odometry(rig):
    source, a, b, server, cp = rig
    _driving(a, b, server, cp, lease_ms=250)
    deadline = time.time() + 6.0
    seen: list = []
    while time.time() < deadline and not seen:
        a._tick_lease()
        seen = [s for s in _statuses(a, "t-1")
                if s.state in (pb.ACTION_STATE_STOPPED, pb.ACTION_STATE_FAILED)]
        time.sleep(0.05)
    assert seen, "no terminal action_status for the lapsed lease (C4.2)"
    assert b.plant_moving is True
    if source == "command-derived":
        assert seen[-1].state == pb.ACTION_STATE_FAILED
        assert "command-derived" in seen[-1].message
    else:
        assert seen[-1].state == pb.ACTION_STATE_STOPPED


def test_a_cancel_is_not_acknowledged_stopped_safely_on_command_derived_odometry(rig):
    source, a, b, server, cp = rig
    _driving(a, b, server, cp)
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    res = _result(a, "cancel_action", STOP_S + 1.0)
    assert res is not None, "no CancelActionResult"
    c = res.command_result.cancel_action
    assert b.plant_moving is True
    if source == "command-derived":
        assert c.acknowledged is False
        assert c.disposition != pb.CANCEL_DISPOSITION_STOPPED_SAFELY
        assert "command-derived" in c.message
        # The action and its lease are KEPT, so the control plane can retry and the lease
        # still self-stops -- a cancel that cannot be confirmed must not look like one that was.
        assert a._current_action == "t-1"
    else:
        assert c.acknowledged is True
        assert c.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY



def test_a_cancelled_task_reports_a_terminal_state_even_when_rest_cannot_be_confirmed(rig):
    """C16.1: the task the control plane CANCELLED must reach a terminal ActionStatusUpdate.

    Confirming rest is what decides whether the cancel may be ACKNOWLEDGED -- it must never
    claim a safe stop the platform cannot witness. It is not what decides whether the task
    ends. A cancelled task that reports no terminal state leaves the control plane nothing to
    resolve the action with but lease expiry, which is the failure C16.1 names, and on this
    platform rest is never confirmable, so the lease is the ONLY resolution there has been.

    The lease self-stop already answers this shape (C4.2): stop at the lowest level, then
    report FAILED with the reason rest could not be confirmed. The cancel takes the same rule.
    """
    source, a, b, server, cp = rig
    _driving(a, b, server, cp)
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    res, term = _answer_and_terminal(a, "t-1", STOP_S + 1.0)
    assert res is not None, "no CancelActionResult"
    assert b.plant_moving is True, "the rig's plant never moved; the test proves nothing"
    assert term is not None, ("the cancelled task reported no terminal state: it is "
                              "resolvable only by lease expiry")
    if source == "command-derived":
        assert res.acknowledged is False, "a stop this platform cannot witness was claimed"
        assert term.state == pb.ACTION_STATE_FAILED
        assert term.state != pb.ACTION_STATE_STOPPED
        assert "command-derived" in term.message
        assert term.end_cause == pb.ACTION_END_CAUSE_CANCELLED_BY_OPERATOR
    else:
        assert res.acknowledged is True
        assert term.state == pb.ACTION_STATE_CANCELLED



def test_a_cancel_the_plane_retries_does_not_end_the_task_a_second_time(rig):
    """An action ends once. The cancel is kept retryable -- on this platform the plane will
    never get an acknowledgement, so it may well retry -- but a second terminal update, in a
    different state, would tell the plane the action ended twice and differently."""
    source, a, b, server, cp = rig
    _driving(a, b, server, cp)
    ends = []
    for command_id in (2, 3):
        cp.send(command_id, cancel_action=pb.CancelAction(
            action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
        res, term = _answer_and_terminal(a, "t-1", STOP_S + 1.0)
        assert res is not None, f"cancel {command_id} was not answered at all"
        if term is not None:
            ends.append(term)
    assert len(ends) == 1, f"the task ended {len(ends)} times: {[e.state for e in ends]}"


def test_the_stop_is_still_commanded_at_once_whatever_can_be_confirmed(rig):
    """Safety first: what the declaration changes is the REPORT, never the stop."""
    source, a, b, server, cp = rig
    h = _driving(a, b, server, cp)
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    assert _result(a, "cancel_action", STOP_S + 1.0) is not None
    assert h.cancels == 1, "the navigation goal was not cancelled"
    assert b._cmd_vel.published, "no zero Twist was published"
    last = b._cmd_vel.published[-1]
    assert last.linear.x == 0.0 and last.angular.z == 0.0


def test_a_pause_says_it_cannot_be_acknowledged_stopped(rig):
    source, a, b, server, cp = rig
    _driving(a, b, server, cp)
    cp.send(2, pause=pb.PauseCapabilities(require_stop_before_ack=True))
    res = _result(a, "pause", 2.0)
    assert res is not None
    # The robot is cruising either way, so neither declaration acks it paused. What differs
    # is what the operator is told: "awaiting" promises an answer that is coming, and on this
    # platform it never is.
    assert res.command_result.pause.paused is False
    if source == "command-derived":
        assert "cannot be acknowledged stopped" in res.command_result.pause.message
        assert "command-derived" in res.command_result.pause.message
    else:
        assert res.command_result.pause.message == "nav2: pausing (awaiting confirmed rest)"


# ── the setting: required on ros2, no default ────────────────────────────────────────────

def test_ros2_without_an_odometry_source_refuses_to_start_naming_it(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", "1")
    assert code == 2 and robot is None
    assert err.startswith("REFUSED: --binding ros2 needs --odometry-source")
    assert "command-derived" in err and "C5.3" in err


@pytest.mark.parametrize("value", ["yes", "encoder", "MEASURED"])
def test_an_odometry_source_outside_the_two_is_refused(monkeypatch, capsys, value):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", "1",
                             "--odometry-source", value)
    assert code == 2 and robot is None


def test_ros2_declaring_command_derived_says_no_stop_will_be_confirmed(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", "1",
                             "--odometry-source", "command-derived")
    assert code is None and isinstance(robot, Nav2Binding)
    assert "NO STOP IS EVER REPORTED CONFIRMED" in err
    assert "--odometry-source command-derived" in err
    assert robot.is_stopped(ROBOT) is False


def test_ros2_declaring_measured_says_what_it_will_wait_for(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "ros2", "--stall-window", "5",
                             "--stall-min-progress", "0.1", "--goal-response-bound", "1",
                             "--odometry-source", "measured")
    assert code is None and isinstance(robot, Nav2Binding)
    assert "rest evidence: odometry declared measured" in err
    assert robot.rest_evidence_gap(ROBOT) == ""


def test_the_simulated_binding_needs_no_odometry_source(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, "--binding", "simulated")
    assert code is None and robot is not None and robot.rest_evidence_gap(ROBOT) == ""
