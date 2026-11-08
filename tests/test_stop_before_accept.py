# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A stop issued before Nav2 has accepted the goal still stops it.

`send_goal_async` returns before the goal handle exists. A stop in that gap used to find no
handle, cancel nothing, publish one zero Twist and report itself done; the server then
accepted the goal and the robot drove to it. Every stop path now records the stop against the
unanswered goal, the goal is cancelled the moment its handle arrives and is never adopted, and
no path reports a stop until that has happened:

* cancel: acknowledged STOPPED_SAFELY only once settled and observed at rest; otherwise NOT
  acknowledged (tests/test_cancel_confirms_rest.py);
* estop: the first ack is the STOPPING receipt, and STOPPED only from the robot, once settled;
* lease self-stop: STOPPED once confirmed at rest by the estop's rule; otherwise FAILED, never
  UNKNOWN (C4.2, tests/test_self_stop_confirms_rest.py);
* reconcile: the plane's phase once settled; otherwise UNKNOWN.

Each path is driven through its real entry point, with a ROS-free Nav2Binding whose goal
dispatch, goal cancels and /cmd_vel are recorded. The server's answer is delivered from another
thread while the stop path waits, as the executor delivers it.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from test_stop_never_waits import _SafetyStub
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"


@pytest.fixture(autouse=True)
def _short_budgets(monkeypatch):
    # The same rules on a shorter clock: how long a stop waits for the server, and the
    # confirmed-stop budget. raising=False: the red run predates both names.
    monkeypatch.setattr(adapter_mod, "_STOP_SETTLE_S", 0.3, raising=False)
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.01, raising=False)


class _Twist:
    def __init__(self) -> None:
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class _Publisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _Handle:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
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
    def __init__(self, v) -> None:
        self._v = v

    def result(self):
        return self._v


def _odom(vx: float = 0.0):
    return NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))))


def _rig():
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(epoch)
    b._server_ready = lambda: True                        # the server is up: goals go at once
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b.on_odom(_odom(0.0))                       # the robot has not moved yet: odom says rest
    return Adapter(pb, None, b, ROBOT), b, sent


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _statuses(out):
    return [m.action_status for m in out if m.WhichOneof("payload") == "action_status"]


def _results(out):
    return [m.command_result for m in out if m.WhichOneof("payload") == "command_result"]


def _drain_until(a: Adapter, done, within: float = 2.0) -> list:
    """Everything on the outbox until `done(out)` holds: a report that waits for its stop to
    settle is sent from the stop reporter thread, not before the call returns."""
    out = []
    deadline = time.monotonic() + within
    while not done(out) and time.monotonic() < deadline:
        time.sleep(0.005)
        out += _drain(a)
    return out + _drain(a)


def _assign(a: Adapter, lease_ms: int = 60_000) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="t-1", assignment_id="a-1", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=1, lease_generation=1,
            lease_duration_ms=lease_ms))), None)
    _drain(a)


def _answer_later(b: Nav2Binding, epoch: int, handle: _Handle, after: float = 0.05):
    """The server answers the goal `after` s from now, on another thread (the executor)."""
    t = threading.Timer(after, lambda: b._on_goal_response(_Done(handle), epoch))
    t.start()
    return t


def _no_motion_after_the_stop(b: Nav2Binding, sent: list, h: _Handle) -> None:
    assert h.cancels == 1                                  # cancelled on arrival
    assert b._goal_handle is None and h.on_result is None  # never adopted
    assert len(sent) == 1                                  # nothing re-dispatched
    assert all(t.linear.x == 0.0 and t.angular.z == 0.0 for t in b._cmd_vel.published)


# ── cancel ──────────────────────────────────────────────────────────────────

def _cancel(a: Adapter, cause=pb.CANCEL_CAUSE_OPERATOR) -> list:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=2, robot_id=ROBOT, cancel_action=pb.CancelAction(
            action_id="t-1", fencing_token=1, cause=cause))), None)
    return _drain_until(a, _results)


def test_cancel_before_accept_cancels_the_goal_on_arrival_then_reports_stopped_safely():
    a, b, sent = _rig()
    _assign(a)
    h = _Handle()
    _answer_later(b, sent[0], h)
    out = _cancel(a)
    _no_motion_after_the_stop(b, sent, h)
    res = _results(out)[0].cancel_action
    assert res.acknowledged is True
    assert res.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert [s.state for s in _statuses(out)] == [pb.ACTION_STATE_CANCELLED]
    assert ROBOT not in a._lease._deadline


def test_cancel_while_the_goal_is_still_unanswered_is_not_acknowledged():
    a, b, sent = _rig()
    _assign(a)
    out = _cancel(a)                                       # the server never answers in time
    res = _results(out)[0].cancel_action
    assert res.acknowledged is False                       # every disposition says "stopped"
    assert "not known to be stopped" in res.message
    # The ACTION still ends here, and says so (C16.1) -- FAILED with the same reason, never
    # CANCELLED, which would read as a cancel that went cleanly to a stop.
    ended = _statuses(out)
    assert [s.state for s in ended] == [pb.ACTION_STATE_FAILED]
    assert "not known to be stopped" in ended[0].message
    assert a._current_action == "t-1" and ROBOT in a._lease._deadline   # lease still guards
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])                 # the answer arrives late
    _no_motion_after_the_stop(b, sent, h)


def test_cancel_of_an_accepted_goal_is_unchanged():
    a, b, sent = _rig()
    _assign(a)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])                 # accepted and adopted first
    assert b._goal_handle is h
    out = _cancel(a)
    assert h.cancels == 1 and len(sent) == 1
    res = _results(out)[0].cancel_action
    assert res.acknowledged is True and res.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    assert [s.state for s in _statuses(out)] == [pb.ACTION_STATE_CANCELLED]


# ── estop ───────────────────────────────────────────────────────────────────

def _estop(a: Adapter) -> list:
    stub = _SafetyStub(a)
    a.run_safety_stream(stub)
    return [m.estop_ack for m in stub.acks]


def test_estop_before_accept_receipts_stopping_and_confirms_only_after_the_goal_is_cancelled():
    a, b, sent = _rig()
    _assign(a)
    h = _Handle()
    _answer_later(b, sent[0], h, after=0.1)
    acks = _estop(a)
    _no_motion_after_the_stop(b, sent, h)
    # odom said rest all along, and still the first ack is only the receipt: a goal the server
    # might yet accept refutes rest until it has been cancelled.
    assert [x.state for x in acks] == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_STOPPED]
    assert all(x.estop_id == "e-1" and x.stop_initiated_at > 0 for x in acks)
    # The estop did not wait for the handle: the zero Twist went out before the goal arrived.
    assert len(b._cmd_vel.published) == 2


def test_estop_whose_goal_is_never_answered_ends_failed_never_stopped():
    a, b, sent = _rig()
    _assign(a)
    acks = _estop(a)
    assert [x.state for x in acks] == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]
    assert len(b._cmd_vel.published) == 1                  # the halt, at once


def test_estop_of_an_accepted_goal_at_rest_is_unchanged_a_single_stopped():
    a, b, sent = _rig()
    _assign(a)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    acks = _estop(a)
    assert h.cancels == 1
    assert [x.state for x in acks] == [pb.ESTOP_STATE_STOPPED]


def test_estop_of_a_robot_still_moving_receipts_stopping_then_fails():
    # Not stop-before-accept: the other case in which rest is not confirmed at once. The
    # terminal answer is unchanged (FAILED, never an inferred STOPPED); it is now preceded by
    # the receipt, as C5.2 requires when rest cannot be confirmed within 500 ms.
    a, b, sent = _rig()
    _assign(a)
    b._on_goal_response(_Done(_Handle()), sent[0])
    b.on_odom(_odom(0.4))
    acks = _estop(a)
    assert [x.state for x in acks] == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED]


def test_an_edge_estop_before_accept_takes_the_same_path(monkeypatch):
    class _EdgeStub:
        def __init__(self, _channel) -> None:
            pass

        def EdgeStream(self, requests):  # noqa: N802
            self.requests = requests
            return iter([pb.EdgeControlMessage(estop=pb.Estop(estop_id="edge-1"))])

    stubs = []
    grpc_mod = NS(EdgeServiceStub=lambda ch: stubs.append(_EdgeStub(ch)) or stubs[-1])
    monkeypatch.setattr(adapter_mod.grpc, "insecure_channel", lambda addr: object())
    a, b, sent = _rig()
    a._pb_grpc = grpc_mod
    _assign(a)
    h = _Handle()
    _answer_later(b, sent[0], h, after=0.1)
    a._edge_addr.put("edge:1")
    a._edge_loop()
    _no_motion_after_the_stop(b, sent, h)
    msgs = []
    req = stubs[0].requests
    for _ in range(3):                                     # the position frame, then the acks
        msgs.append(next(req))
    acks = [m.estop_ack for m in msgs if m.WhichOneof("msg") == "estop_ack"]
    assert [x.state for x in acks] == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_STOPPED]


# ── lease self-stop ─────────────────────────────────────────────────────────

def test_lease_self_stop_before_accept_reports_stopped_once_the_goal_is_cancelled():
    a, b, sent = _rig()
    _assign(a, lease_ms=1)
    h = _Handle()
    time.sleep(0.01)
    _answer_later(b, sent[0], h)
    a._tick_lease()                                        # the lease lapses: self-stop
    _no_motion_after_the_stop(b, sent, h)
    st = _statuses(_drain(a))
    assert [s.state for s in st] == [pb.ACTION_STATE_STOPPED]
    assert st[0].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED


def test_lease_self_stop_while_the_goal_is_still_unanswered_reports_failed():
    # Odom says rest, but the unanswered goal refutes it, as it does for an estop: FAILED, which
    # C4.2 counts as terminal, never UNKNOWN; and the report says why.
    a, b, sent = _rig()
    _assign(a, lease_ms=1)
    time.sleep(0.01)
    a._tick_lease()
    st = _statuses(_drain(a))
    assert [s.state for s in st] == [pb.ACTION_STATE_FAILED]
    assert st[0].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    assert "not known to be stopped" in st[0].message
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    _no_motion_after_the_stop(b, sent, h)


# ── reconcile on registration ───────────────────────────────────────────────

def _register_ack(a: Adapter, phase) -> list:
    a._handle_control_plane(pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
        accepted=True, authoritative_action_state=pb.RobotActionState(
            phase=phase, action_id="t-1", fencing_token=1))), None)
    return _drain_until(a, _statuses)


def test_reconcile_before_accept_cancels_the_goal_on_arrival_then_reports_the_plane_phase():
    a, b, sent = _rig()
    _assign(a)
    h = _Handle()
    _answer_later(b, sent[0], h)
    out = _register_ack(a, pb.ROBOT_ACTION_PHASE_CANCELLED)
    _no_motion_after_the_stop(b, sent, h)
    assert [s.state for s in _statuses(out)] == [pb.ACTION_STATE_CANCELLED]


def test_reconcile_while_the_goal_is_still_unanswered_reports_unknown():
    a, b, sent = _rig()
    _assign(a)
    out = _register_ack(a, pb.ROBOT_ACTION_PHASE_CANCELLED)
    st = _statuses(out)
    assert [s.state for s in st] == [pb.ACTION_STATE_UNKNOWN]
    assert "not known to be stopped" in st[0].message
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    _no_motion_after_the_stop(b, sent, h)


# ── the one mechanism, below the paths ──────────────────────────────────────

def test_pause_before_accept_cancels_on_arrival_and_resume_sends_a_fresh_goal():
    a, b, sent = _rig()
    _assign(a)
    b.command_pause(ROBOT)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])
    assert h.cancels == 1 and b._goal_handle is None
    b.command_resume(ROBOT)
    assert len(sent) == 2 and sent[1] != sent[0]
    h2 = _Handle()
    b._on_goal_response(_Done(h2), sent[1])
    assert b._goal_handle is h2 and h2.cancels == 0      # the resumed goal is adopted


def test_a_refused_goal_settles_the_stop_with_nothing_to_cancel():
    a, b, sent = _rig()
    _assign(a)
    b.command_stop(ROBOT)
    assert b.is_stopped(ROBOT) is False                    # refuted while unanswered
    b._on_goal_response(_Done(_Handle(accepted=False)), sent[0])
    assert b.stop_settled(ROBOT, 0) is True and b.is_stopped(ROBOT) is True


def test_a_stop_records_every_unanswered_goal_including_a_superseded_one():
    a, b, sent = _rig()
    b.command_move(ROBOT, 1, 0)
    b.command_move(ROBOT, 2, 0)                            # supersedes before either answer
    b.command_stop(ROBOT)
    h1, h2 = _Handle(), _Handle()
    b._on_goal_response(_Done(h2), sent[1])
    assert b.stop_settled(ROBOT, 0) is False               # the first is still out there
    b._on_goal_response(_Done(h1), sent[0])
    assert h1.cancels == 1 and h2.cancels == 1 and b.stop_settled(ROBOT, 0) is True


def test_without_a_stop_a_superseded_goal_is_still_dropped_not_cancelled():
    a, b, sent = _rig()
    b.command_move(ROBOT, 1, 0)
    b.command_move(ROBOT, 2, 0)
    h1 = _Handle()
    b._on_goal_response(_Done(h1), sent[0])
    assert h1.cancels == 0 and b._goal_handle is None      # the server preempts it


def test_a_send_that_raises_leaves_nothing_for_a_stop_to_wait_on():
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))

    def _boom(goal, epoch):
        raise RuntimeError("no action server")
    b._send_nav2_goal = _boom
    b._server_ready = lambda: True
    with pytest.raises(RuntimeError):
        b.command_move(ROBOT, 1, 0)
    b.command_stop(ROBOT)
    assert b.stop_settled(ROBOT, 0) is True
