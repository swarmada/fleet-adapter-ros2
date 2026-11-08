# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The handover stand-in a run uses with no test driving it (`--binding simulated --teleop-topic`):
its nav2 declines goals beyond a distance, only the teleop the gate forwards moves it, and the
operator's teleop arrives as lines appended to a file."""

from __future__ import annotations

import threading
import time

from test_manual_drive_gate import ROBOT, StepClock, collect, drain, statuses, terminals
from test_navigate_handover import ACTION, assign, payload, turned

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter


def standin(**kw):
    robot = ho.SimulatedHandoverRobot(decline_beyond_m=3.0, pulse_s=0.5, **kw)
    robot.spawn(ROBOT)
    return Adapter(pb, None, robot, ROBOT, job_clock=StepClock()), robot


def test_a_goal_within_reach_is_driven_and_one_beyond_is_declined_to_the_operator() -> None:
    a, _ = standin()
    res, msgs = assign(a, payload(x=2.0, y=1.0))
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    assert [s.state for s in terminals(msgs, ACTION)] == [pb.ACTION_STATE_SUCCEEDED]
    res, msgs = assign(a, payload(x=6.0, y=1.0), token=2)
    assert res.accepted
    collect(a, msgs, turned)
    (turn,) = [s for s in statuses(msgs, ACTION) if "operator's turn" in s.message]
    assert "more than 3 m" in turn.message


def test_only_forwarded_teleop_moves_the_standin() -> None:
    a, robot = standin()
    robot.stall_next()
    assert assign(a, payload(x=2.0, y=1.0))[0].accepted
    before = robot.pose(ROBOT)[:2]
    robot.teleop(0.2, 0.0, 0.0, ROBOT)          # armed, Nav2 driving: not forwarded
    assert robot.pose(ROBOT)[:2] == before
    a._estop("e-1", lambda _ack: None)
    robot.teleop(0.2, 0.0, 0.0, ROBOT)
    assert robot.pose(ROBOT)[:2] == before


def test_the_operator_file_drives_a_declined_handover_into_its_circle(tmp_path) -> None:
    a, robot = standin()
    f = tmp_path / "operator.teleop"
    f.write_text("0.2 0 0\n")                   # already there before the run: skipped
    stop = threading.Event()
    reader = threading.Thread(target=ho.follow_operator_file,
                              args=(robot, str(f), ROBOT, stop, 0.01), daemon=True)
    reader.start()
    try:
        time.sleep(0.05)
        res, msgs = assign(a, payload(x=4.0, y=0.0, radius_m=0.3, settle_s=0.5))
        assert res.accepted
        collect(a, msgs, turned)
        assert robot.pose(ROBOT)[:2] == (0.0, 0.0)
        with f.open("a") as w:                  # 0.2 m/s x 0.5 s = 0.1 m a line
            w.write("not a twist\n")
            for _ in range(40):
                w.write("0.2 0 0\n")
        collect(a, msgs, lambda ms: terminals(ms, ACTION), timeout=15)
        end = terminals(msgs, ACTION)
        assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED] and "operator" in end[0].message
        x, y = robot.pose(ROBOT)[:2]
        assert abs(x - 4.0) <= 0.3 and abs(y) < 1e-9
    finally:
        stop.set()
        reader.join(1)
    drain(a)
