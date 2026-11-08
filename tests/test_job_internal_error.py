# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A job that raises something it did not expect ends FAILED, and stops what it started.

A job runs on its own thread. An exception other than the two the job raises on purpose (a
stop, a named failure) used to end that thread and nothing else: the action stayed RUNNING at
the control plane with no further update, holding the robot's lease, and SLAM, the keep-out
servers and the explorer kept running -- with nothing watching the battery if the robot had
already moved. The job's boundary now catches it, stops the explorer and the binding's goal,
ends the segment's processes, and reports FAILED with the exception's class and message.
"""

from __future__ import annotations

from test_explore_job import Emits, FakeClock, _job

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack


def _raise(what: str):
    def boom(*_a, **_k):
        raise RuntimeError(what)
    return boom


def test_an_error_before_motion_ends_failed_naming_it_and_stops_the_stack(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock())
    stack.read_param = _raise("the facade has no node")
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, cause = emits.last
    assert state == "FAILED" and cause == "ACTION_END_CAUSE_EXECUTION_FAILURE"
    assert "RuntimeError: the facade has no node" in message and "no motion" in message
    assert "start_explorer" not in stack.log
    assert stack.log[-1] == "end_segment"


def test_an_error_while_exploring_stops_the_explorer_and_the_goal_then_ends(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000)
    stack.on_explorer_started = lambda: setattr(stack, "recoveries", _raise("feedback lost"))
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    state, message, _, cause = emits.last
    assert state == "FAILED" and cause == "ACTION_END_CAUSE_EXECUTION_FAILURE"
    assert "RuntimeError: feedback lost" in message and "no motion" not in message
    started = stack.log.index("start_explorer")
    after = stack.log[started:]
    assert "stop_explorer" in after and "hold" in after
    assert stack.log[-1] == "end_segment"
    assert not stack.moving()


def test_a_stop_that_raises_too_does_not_keep_the_other_stop_or_the_report(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000)

    def explode_then_break_the_stop():
        stack.recoveries = _raise("feedback lost")
        stack.stop_explorer = _raise("explorer unreachable")
    stack.on_explorer_started = explode_then_break_the_stop
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    assert emits.last[0] == "FAILED" and "feedback lost" in emits.last[1]
    assert "hold" in stack.log and stack.log[-1] == "end_segment"


def test_a_charge_that_raises_ends_failed_and_never_moves() -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, start_mv=7900)
    stack.imu_streaming = _raise("imu subscription gone")
    emits = Emits()
    job = explore.ChargeJob(action_id="charge", doc={}, stack=stack,
                            limits=stack.battery_limits(), clock=clock, emit=emits)
    job.run()
    assert emits.last[0] == "FAILED" and "RuntimeError: imu subscription gone" in emits.last[1]
    for motion in ("start_explorer", "go_to_mark", "start_slam"):
        assert motion not in stack.log
