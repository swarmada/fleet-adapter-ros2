# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A progress-watchdog stall during a navigate-handover is Nav2's decline, as an abort is.

On the robot, TEB answered "trajectory is not feasible" and Nav2 spun in recovery; at the window
the watchdog cancelled the goal. The cancel went through the stop path, which closes the teleop
gate, so the job read a closed gate before it read the FAILED goal and ended the action with "a
stop closed the gate". The operator's turn never opened, where an ABORTED goal does open it.

A stall is not a stop: no estop, lease or cancel asked for it, and it is reported through the
same FAILED path an abort takes. So, while a navigate-handover's gate is armed for its Nav2 goal,
the watchdog cancels the goal and holds the base -- one zero Twist -- and leaves the gate armed,
forwarding nothing. The job then reads the decline, holds, waits for rest, and turns the gate to
the operator, saying "stalled". Every stop path still closes the gate; a stall with no handover
armed still ends the goal as it did.

Run through the adapter on a ROS-free Nav2Binding, as `test_stall_reporting.py` runs it: the
dispatch, the goal handle's cancel and every /cmd_vel Twist are recorded.
"""

from __future__ import annotations

import json
import time

from conftest import mark_transport_live
from test_manual_drive_gate import StepClock, collect, command, drain, statuses, terminals
from test_manual_drive_nav2 import TOPIC, Publisher, Twist, msg
from test_stall_reporting import WINDOW, Clock, _Done, _Handle, _limits
from test_stop_never_waits import _odom

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
ACTION = "handover-a"
TARGET = (2.0, 1.0)


def rig():
    """The adapter on a Nav2Binding with the watchdog armed, a teleop topic, measured odometry,
    and a pose in `map` at the origin (moved by the test)."""
    c = Clock()
    b = mark_transport_live(Nav2Binding(ROBOT, stall_limits=_limits(), clock=c,
                                        teleop_topic=TOPIC))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append((goal, epoch))
    b._server_ready = lambda: True
    b._cmd_vel, b._twist_type = Publisher(), Twist
    b._odometry_source = "measured"
    b.spawn(ROBOT)
    where = {"xy": (0.0, 0.0)}
    b.pose = lambda rid: (*where["xy"], 0)
    b.pose_frame = lambda rid: "map"
    a = Adapter(pb, None, b, ROBOT, job_clock=StepClock())
    return a, b, c, sent, where


def assign(a: Adapter) -> list:
    body = json.dumps({"operation": ho.OPERATION, "x": TARGET[0], "y": TARGET[1],
                       "frame": "map", "radius_m": 0.3, "timeout_s": 300}).encode()
    command(a, 1, assign_action=pb.AssignAction(
        action_id=ACTION, action_type="Custom", assignment_id="a-1", fencing_token=1,
        lease_generation=1, lease_duration_ms=60_000, payload_json=body))
    out = drain(a)
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"
           and m.command_result.HasField("assign_action")]
    assert len(res) == 1 and res[0].accepted
    return list(out)


def stall(b: Nav2Binding, c: Clock, sent: list) -> tuple[_Handle, str]:
    """The server accepts the goal, one feedback arrives, then no progress for the window."""
    h = _Handle()
    b._on_goal_response(_Done(h), sent[-1][1])
    b.on_feedback(3.0)
    c.t += WINDOW
    reason = b.check_progress()
    assert reason.startswith("stalled:"), reason
    return h, reason


def texts(msgs) -> list[str]:
    return [s.message for s in statuses(msgs, ACTION)]


def turned(msgs) -> bool:
    return any("operator's turn" in t for t in texts(msgs))


def test_a_stall_holds_the_base_and_leaves_the_gate_armed_forwarding_nothing() -> None:
    a, b, c, sent, _ = rig()
    msgs = assign(a)
    n = len(b._cmd_vel.sent)
    h, _ = stall(b, c, sent)
    assert h.cancels == 1                                 # the goal is cancelled
    assert b._cmd_vel.sent[n:][:1] == [md.ZERO]           # the base is held
    # No odometry yet, so no rest: the gate is still this action's, armed, and nothing the
    # operator sends reaches the base.
    assert b.teleop_open(ROBOT, ACTION)
    m = len(b._cmd_vel.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert all(s == md.ZERO for s in b._cmd_vel.sent[m:])
    assert not turned(msgs + drain(a))


def test_a_stall_opens_the_operators_turn_and_says_stalled() -> None:
    a, b, c, sent, where = rig()
    msgs = assign(a)
    stall(b, c, sent)
    b.on_odom(_odom(0.0))                                 # measured odometry: at rest
    collect(a, msgs, lambda ms: turned(ms) or terminals(ms, ACTION))
    said = texts(msgs)
    assert not terminals(msgs, ACTION), said
    assert turned(msgs), said
    assert any("stalled" in t for t in said), said
    assert not any("a stop closed the gate" in t for t in said), said
    # The keys now reach the base, clamped, through the gate.
    n = len(b._cmd_vel.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert b._cmd_vel.sent[n:] == [(0.1, 0.0, 0.0)]
    # The operator drives into the circle and stops: Succeeded, the gate closed.
    where["xy"] = TARGET
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED], texts(msgs)
    assert "operator released" in end[0].message
    assert not b.teleop_open(ROBOT)


def test_a_stop_after_the_stall_still_wins_and_closes_the_gate() -> None:
    a, b, c, sent, _ = rig()
    msgs = assign(a)
    stall(b, c, sent)
    b.command_stop(ROBOT)                                 # the stop path, as an estop takes it
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert not turned(msgs)
    assert not b.teleop_open(ROBOT)
    n = len(b._cmd_vel.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(b._cmd_vel.sent) == n


def test_a_stall_with_no_handover_armed_still_closes_the_gate_as_a_stop() -> None:
    """A plain Navigate: nothing armed, so the stall ends the goal through the stop path."""
    a, b, c, sent, _ = rig()
    command(a, 1, assign_action=pb.AssignAction(
        action_id="nav-a", action_type="Navigate", assignment_id="a-1", fencing_token=1,
        lease_generation=1, lease_duration_ms=60_000, payload_json=b'{"x": 3, "y": 0}'))
    drain(a)
    h, _ = stall(b, c, sent)
    assert h.cancels == 1 and not b.teleop_open(ROBOT)
    assert b._cmd_vel.sent[-1] == md.ZERO
    time.sleep(0.01)
