# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Every stop holds the base, whatever the exploration stack does.

`Nav2Binding.command_stop` is the one stop shared by the confirmed estop, the lease self-stop, a
cancel and the RegisterAck reconciliation. It used to call the exploration stack's `halt` first,
and a `halt` that raised -- the rclpy facade builds its subscriptions on first use, and on the
stand-in it raised `ModuleNotFoundError` for the explorer's messages when the overlay was not
sourced -- left before the base was told to hold and before the Nav2 goal was cancelled. It also
left the adapter: the reconciliation at admission ended the control-stream loop.

Now each layer keeps its own promise:

* the binding holds the base and cancels the goal whatever the stack's `halt` does, and reports
  a `halt` that raised on stderr with its class and message;
* `Ros2ExploreStack.halt`, `hold` and `stop_explorer` never raise: they report and return, and
  `halt` still ends the job and `hold` still pauses the base;
* the zero Twist is published even when the goal's cancel raises, and the answer watch holds
  the base before it cancels abandoned goals by their ids.
"""

from __future__ import annotations

import threading
import time

import pytest
from conftest import mark_transport_live
from test_ros2_explore import _stack
from test_stop_never_waits import _halted, _SafetyStub
from test_unanswered_goal import (  # noqa: F401 - rig and ros_msgs are fixtures
    BOUND_S,
    ROBOT,
    _answer,
    _assign,
    _first,
    _Handle,
    _result,
    rig,
    ros_msgs,
)

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

BROKEN = "No module named 'explore_lite_msgs'"


class _BrokenStack:
    """An exploration stack whose every stop raises, as the facade did on the stand-in."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _raise(self, name: str):
        self.calls.append(name)
        raise ModuleNotFoundError(BROKEN)

    def halt(self) -> None:
        self._raise("halt")

    def hold(self) -> None:
        self._raise("hold")

    def stop_explorer(self) -> None:
        self._raise("stop_explorer")


def _reported(err: str) -> bool:
    return "ModuleNotFoundError" in err and BROKEN in err


# ── the binding ──────────────────────────────────────────────────────────────

def test_command_stop_holds_the_base_even_if_the_explore_halt_raises(capsys) -> None:
    b = mark_transport_live(Nav2Binding("amr-1"))
    b._explore = _BrokenStack()
    cancelled = []
    b._cancel_nav2_goal = cancelled.append
    try:
        b.command_stop("amr-1")
    except Exception as exc:  # noqa: BLE001
        raised = exc
    else:
        raised = None
    assert b._holding, f"the base was never told to hold (the explore halt raised {raised!r} first)"
    assert raised is None, f"a stop path raised: {raised!r}"
    assert cancelled == [None], "the Nav2 goal was not cancelled"
    assert b._explore.calls == ["halt"]
    assert _reported(capsys.readouterr().err), "the failed halt was swallowed silently"


def test_the_zero_twist_goes_out_even_if_the_goal_cancel_raises(rig, capsys) -> None:  # noqa: F811
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    h = _Handle()
    h.cancel_goal_async = lambda: (_ for _ in ()).throw(RuntimeError("context invalid"))
    _answer(server, 0, h)
    assert b._goal_handle is h
    b.command_stop(ROBOT)
    assert b._holding and _halted(b), "no zero Twist: the base was not held"
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "context invalid" in err


def test_the_answer_watch_holds_the_base_before_it_cancels_by_id(rig, capsys) -> None:  # noqa: F811
    a, b, _server, cp = rig

    def cancel_by_id(goal_id):
        raise RuntimeError("cancel by id failed")
    b._cancel_goal_by_id = cancel_by_id
    cp.send(1, assign_action=_assign("t-1", 1))
    assert _result(a, "assign_action", 2.0)
    deadline = time.monotonic() + BOUND_S + 2.0
    while not _halted(b):                                  # the answer never comes
        assert time.monotonic() < deadline, "the abandoned goal's stop published no zero Twist"
        time.sleep(0.01)
    assert b._holding
    time.sleep(0.05)
    assert "cancel by id failed" in capsys.readouterr().err
    assert b._answer_watch.is_alive(), "the answer watch died on a failed cancel"


def test_the_simulated_robot_stops_even_if_its_explore_halt_raises(capsys) -> None:
    robot = SimulatedRobot(explore=_BrokenStack())
    robot.spawn(ROBOT)
    robot.command_move(ROBOT, 1.0, 2.0)
    robot.command_stop(ROBOT)
    assert not robot._moving[ROBOT]
    assert _reported(capsys.readouterr().err)


# ── every stop path through the adapter, on a binding whose stack raises ─────

def _executing(a, b, server, cp, lease_ms: int = 60_000):
    """t-1 assigned, sent and accepted; the binding's exploration stack raising from now on."""
    cp.send(1, assign_action=_assign("t-1", 1, lease_ms=lease_ms))
    assert _result(a, "assign_action", 2.0)
    h = _Handle()
    _answer(server, 0, h)
    assert b._goal_handle is h
    b._explore = _BrokenStack()
    return h


def _stop_by(path: str, a, cp) -> None:
    if path == "estop":
        threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),), daemon=True).start()
        ack = _first(a._safety_outbox, lambda m: m.WhichOneof("payload") == "estop_ack"
                     and m.estop_ack.state != pb.ESTOP_STATE_STOPPING, 3.0)
        assert ack is not None, "the safety stream sent no terminal EstopAck"
    elif path == "lease":
        time.sleep(0.25)
        a._tick_lease()
    elif path == "cancel":
        cp.send(2, cancel_action=pb.CancelAction(action_id="t-1", fencing_token=1,
                                                 cause=pb.CANCEL_CAUSE_OPERATOR))
        assert _result(a, "cancel_action", 3.0) is not None, "the cancel was never answered"
    elif path == "reconcile":
        ack = pb.RegisterAck(accepted=True)
        ack.authoritative_action_state.CopyFrom(pb.RobotActionState(
            phase=pb.ROBOT_ACTION_PHASE_CANCELLED, action_id="t-1", fencing_token=1))
        cp.q.put(pb.ControlPlaneMessage(register_ack=ack))


@pytest.mark.parametrize("path", ["estop", "lease", "cancel", "reconcile"])
def test_every_stop_path_holds_the_base_and_the_loop_survives(rig, capsys, path) -> None:  # noqa: F811
    a, b, server, cp = rig
    h = _executing(a, b, server, cp, lease_ms=100 if path == "lease" else 60_000)
    _stop_by(path, a, cp)
    # The control-stream loop still reads and answers: a validate sent after the stop.
    cp.send(9, validate_action=pb.ValidateAction(action_type="Navigate",
                                                 payload_json=b'{"x": 1, "y": 1}'))
    assert _result(a, "validate_action", 3.0) is not None, "the control-stream loop died"
    assert b._holding, "the base was never told to hold"
    assert _halted(b), "no zero Twist was published"
    assert h.cancels >= 1, "the Nav2 goal was not cancelled"
    assert b._explore.calls and b._explore.calls[0] == "halt"
    assert _reported(capsys.readouterr().err)


# ── the ROS 2 exploration stack never raises from a stop ─────────────────────

def _broken_publish(st) -> None:
    def publish(topic, msg_type, fields):
        raise ModuleNotFoundError(BROKEN)
    st.ros.publish = publish


def test_stop_explorer_reports_and_returns(capsys) -> None:
    st, *_ = _stack()
    _broken_publish(st)
    st.stop_explorer()
    assert _reported(capsys.readouterr().err)


def test_halt_ends_the_job_even_if_the_explorer_cannot_be_told(capsys) -> None:
    st, *_ = _stack()
    halted = threading.Event()
    st.arm(halted)
    _broken_publish(st)
    st.halt()
    assert halted.is_set(), "the job was not ended"
    assert _reported(capsys.readouterr().err)


def test_hold_pauses_the_base_even_if_the_explorer_cannot_be_told(capsys) -> None:
    st, _, _, b, _ = _stack()
    _broken_publish(st)
    st.hold()
    assert b._holding, "the base was not paused"
    assert _reported(capsys.readouterr().err)


def test_hold_reports_a_pause_that_raises_and_returns(capsys) -> None:
    st, _, _, b, _ = _stack()
    b.command_pause = lambda robot_id: (_ for _ in ()).throw(RuntimeError("pause failed"))
    st.hold()
    assert "RuntimeError" in capsys.readouterr().err


def test_the_binding_and_the_ros2_stack_together_hold_and_report(capsys) -> None:
    st, _, _, b, _ = _stack()
    b._explore = st
    halted = threading.Event()
    st.arm(halted)
    _broken_publish(st)
    b.command_stop(ROBOT)
    assert b._holding and halted.is_set()
    assert _reported(capsys.readouterr().err)


def test_a_stack_stop_that_succeeds_says_nothing(capsys) -> None:
    st, ros, _, b, _ = _stack()
    b._explore = st
    b.command_stop(ROBOT)
    assert ros.published == [("explore/resume", {"data": False})]
    assert capsys.readouterr().err == ""
