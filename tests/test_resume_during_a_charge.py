# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A resume during a Charge sends the robot nowhere.

A resume re-dispatched the binding's retained target whatever action ran, unless it was a
segment or a manual-drive. A Charge sends no goal, so the target a resume found there was an
earlier action's: the Navigate before it, or a segment's return to its mark. The robot drove off
the charger toward it. A Charge never moves the robot: its resume now re-dispatches nothing.
"""

from __future__ import annotations

import pytest
from test_explore_adapter import StepClock, _assign, _command, _drain
from test_ros2_explore import _stack as _ros2_stack

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
NAVIGATE = b'{"x": 1, "y": 2}'


def _rig(tmp_path, monkeypatch):
    """The Nav2 binding, with the ROS 2 exploration stack, through the adapter. The Charge's own
    thread is not run: what is tested is what the adapter commands."""
    monkeypatch.setattr(explore.ChargeJob, "run", lambda self: None)
    st, _ros, _shell, b, sent = _ros2_stack(tmp_path)
    b._explore = st
    a = Adapter(pb, None, b, ROBOT, job_clock=StepClock(),
                artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab")
    return a, b, sent


def _resume(a: Adapter, command_id: int):
    _command(a, command_id, resume=pb.ResumeCapabilities())
    got = [m.command_result.resume for m in _drain(a)
           if m.WhichOneof("payload") == "command_result" and m.command_result.HasField("resume")]
    assert len(got) == 1
    return got[0]


@pytest.mark.parametrize("paused", [True, False], ids=["after-a-pause", "without-a-pause"])
def test_a_resume_during_a_charge_sends_no_goal(tmp_path, monkeypatch, paused) -> None:
    a, b, sent = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Navigate", NAVIGATE, token=1, action="t-1")[0].accepted
    b.on_goal_result(4, epoch=b._goal_epoch)                 # SUCCEEDED: at the target
    assert len(sent) == 1
    assert _assign(a, "Charge", b"", token=2, action="charge")[0].accepted
    if paused:
        _command(a, 3, pause=pb.PauseCapabilities())
        _drain(a)
    got = _resume(a, 4)
    assert got.resumed and "never moves" in got.message, got.message
    assert len(sent) == 1, "a resume during a Charge sent an earlier action's goal again"


@pytest.mark.parametrize("paused", [True, False], ids=["after-a-pause", "without-a-pause"])
def test_on_the_simulated_robot_a_resume_during_a_charge_leaves_it_at_rest(
        tmp_path, monkeypatch, paused) -> None:
    monkeypatch.setattr(explore.ChargeJob, "run", lambda self: None)
    clock = StepClock()
    robot = SimulatedRobot(explore=SimulatedExploreStack(clock=clock))
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=clock,
                artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab")
    assert _assign(a, "Navigate", b'{"x": 0, "y": 0}', token=1, action="t-1")[0].accepted
    assert robot.is_stopped(ROBOT)                           # arrived: it was already there
    assert _assign(a, "Charge", b"", token=2, action="charge")[0].accepted
    if paused:
        _command(a, 3, pause=pb.PauseCapabilities())
        _drain(a)
    assert _resume(a, 4).resumed
    assert robot.is_stopped(ROBOT), "a resume during a Charge set the robot moving"


def test_a_resume_of_a_navigate_still_re_dispatches_its_goal(tmp_path, monkeypatch) -> None:
    a, _b, sent = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Navigate", NAVIGATE, token=1, action="t-1")[0].accepted
    _command(a, 2, pause=pb.PauseCapabilities())
    _drain(a)
    assert _resume(a, 3).resumed
    assert len(sent) == 2 and (sent[-1].x, sent[-1].y) == (1.0, 2.0)
