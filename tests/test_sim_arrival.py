# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The simulated robot reaches the goal it is sent to, and says so.

A Navigate on the simulated binding never ended: `command_move` set the robot moving and nothing
ever stopped it, and the binding had no `task_state`, so the adapter's completion stream (C6.4)
had nothing to read. On the simulated binding no action completed, and a check that grades an
action by its completion (C3.9: the preemptor runs to SUCCEEDED) could not pass whatever the
adapter did. The robot now travels to the goal at a fixed speed, comes to rest there, and reports
SUCCEEDED, as a Nav2 goal does. A stop or a pause before then is not an arrival.
"""

from __future__ import annotations

import queue

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import task
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SIM_SPEED_MPS, SimulatedRobot

R = "amr-1"


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _robot():
    clock = _Clock()
    r = SimulatedRobot(clock=clock)
    r.spawn(R)
    return r, clock


def test_a_goal_is_reached_at_the_sim_speed_and_reported_succeeded() -> None:
    r, clock = _robot()
    r.command_move(R, 3.0, 4.0)                      # 5 m away
    travel = 5.0 / SIM_SPEED_MPS
    clock.t += travel - 0.1
    assert not r.is_stopped(R) and r.task_state(R) == task.TASK_RUNNING
    clock.t += 0.2
    assert r.is_stopped(R) and r.task_state(R) == task.TASK_SUCCEEDED


def test_the_next_goal_is_measured_from_where_the_last_one_ended() -> None:
    r, clock = _robot()
    r.command_move(R, 3.0, 4.0)
    clock.t += 5.0 / SIM_SPEED_MPS + 0.1
    r.command_move(R, 3.0, 5.0)                      # 1 m on
    assert r.task_state(R) == task.TASK_RUNNING
    clock.t += 1.0 / SIM_SPEED_MPS + 0.1
    assert r.task_state(R) == task.TASK_SUCCEEDED


def test_a_stopped_goal_is_never_reported_reached() -> None:
    r, clock = _robot()
    r.command_move(R, 3.0, 4.0)
    r.command_stop(R)
    clock.t += 60.0
    assert r.is_stopped(R) and r.task_state(R) != task.TASK_SUCCEEDED


def test_a_paused_goal_keeps_its_remaining_travel_for_the_resume() -> None:
    r, clock = _robot()
    r.command_move(R, 3.0, 4.0)
    clock.t += 4.0
    r.command_pause(R)
    clock.t += 60.0                                  # time paused is not travel
    assert r.is_stopped(R) and r.task_state(R) != task.TASK_SUCCEEDED
    r.command_resume(R)
    clock.t += 5.0 / SIM_SPEED_MPS - 4.0 - 0.1
    assert r.task_state(R) == task.TASK_RUNNING
    clock.t += 0.2
    assert r.task_state(R) == task.TASK_SUCCEEDED


def test_the_adapter_streams_succeeded_for_a_navigate_on_the_simulated_binding() -> None:
    r, clock = _robot()
    a = Adapter(pb, None, r, R)
    res = pb.CommandResult()
    a._on_assign(pb.AssignAction(
        action_id="t-1", action_type="Navigate", payload_json=b'{"x": 3, "y": 4}',
        fencing_token=1, lease_generation=1, lease_duration_ms=60_000), res)
    assert res.assign_action.accepted, res.assign_action.message
    clock.t += 5.0 / SIM_SPEED_MPS + 0.1
    a._maybe_emit_task_progress()
    states = []
    while True:
        try:
            m = a._outbox.get_nowait()
        except queue.Empty:
            break
        if m.WhichOneof("payload") == "action_status":
            states.append(m.action_status.state)
    assert states[-1] == pb.ACTION_STATE_SUCCEEDED
    assert a._current_action == ""
