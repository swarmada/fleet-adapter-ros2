# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""No explore-floor segment with the charging cable in.

On this robot the IMU is silent while the charging cable is in, and Charge already reads that
silence as "cable in": it ends only once the battery is full and the IMU streams again. A
segment proved the speed cap, the battery, ALIGNED, the keep-out set and the bound before
motion, but never looked at the cable. On the stand-in a segment refused for a low battery was
dispatched again while the operator had the cable in to charge it; by then the reading was
above the return threshold, and the segment started SLAM and the explorer with the cable in.

Now a segment refuses before motion, before SLAM, while the cable is in, read by the one
predicate Charge ends on: FAILED "the charging cable is in; no motion".
"""

from __future__ import annotations

import pytest
from test_explore_adapter import EXPLORE, _assign, _rig, _terminal
from test_explore_job import Emits, FakeClock, _job

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

CABLE_IN = "explore-floor: the charging cable is in; no motion"


def _plugged(**kw) -> SimulatedExploreStack:
    stack = SimulatedExploreStack(clock=FakeClock(), unplug_after_full_s=None, **kw)
    stack.plug()
    return stack


def _before_slam(stack) -> None:
    for step in ("start_slam", "load_keepout_mask", "start_explorer", "go_to_mark"):
        assert step not in stack.log
    assert not stack.moving()


def test_a_segment_with_the_cable_in_is_refused_before_motion(tmp_path) -> None:
    job, stack, emits, _ = _job(tmp_path, stack=_plugged())
    job.run()
    assert emits.last == ("FAILED", CABLE_IN, 0, "ACTION_END_CAUSE_EXECUTION_FAILURE")
    _before_slam(stack)


def test_the_cable_is_looked_at_before_the_battery(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path, stack=_plugged(start_mv=7100, charge_mv_per_s=0.0))
    job.run()
    assert emits.last[:2] == ("FAILED", CABLE_IN)
    assert not any("battery" in m for m in emits.messages())


def test_a_segment_with_the_cable_out_runs(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    assert emits.last[0] == "SUCCEEDED"
    assert not any("cable" in m for m in emits.messages())


def test_an_imu_that_never_answers_is_not_a_cable_out(tmp_path) -> None:
    class Unread(SimulatedExploreStack):
        answer = None                                     # the IMU never reports

        def imu_streaming(self):
            self.log.append("imu_streaming")
            return self.answer
    stack = Unread(clock=FakeClock())
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "the charging cable is not proven out" in message
    assert message.endswith("; no motion")
    assert stack.log.count("imu_streaming") > 1          # silence is read again, bounded
    _before_slam(stack)


class _Looks(FakeClock):
    """A clock that ends the job after `looks` waits."""

    def __init__(self, looks: int) -> None:
        super().__init__()
        self.left = looks

    def wait(self, event, seconds: float) -> bool:
        self.t += seconds
        self.left -= 1
        if self.left <= 0:
            event.set()
        return event.is_set()


@pytest.mark.parametrize("out", [True, False])
def test_charge_and_the_segment_read_the_cable_by_one_predicate(tmp_path, monkeypatch,
                                                                out) -> None:
    """Whatever `explore.cable_out` answers, Charge and the segment both act on it: there is
    no second reading of the IMU to drift from the first. Each is shown the opposite of what
    the predicate says: the segment a cable in, Charge a cable out."""
    monkeypatch.setattr(explore, "cable_out", lambda imu: out)
    job, _, emits, _ = _job(tmp_path, stack=_plugged())
    job.run()
    assert (emits.last[:2] == ("FAILED", CABLE_IN)) is (not out)

    clock = _Looks(20)
    stack = SimulatedExploreStack(clock=clock, start_mv=8500, unplug_after_full_s=None)
    emits = Emits()
    explore.ChargeJob(action_id="charge", doc={}, stack=stack, limits=stack.battery_limits(),
                      clock=clock, emit=emits).run()
    assert (emits.last[0] == "SUCCEEDED") is out


def test_the_stand_in_timeline_ends_with_the_retry_refused_on_the_cable(tmp_path) -> None:
    """The stand-in's timeline of 2026-09-28, through the adapter: a segment refused for a low battery, then
    the cable in while the control plane would have dispatched it again. The second attempt
    is refused on the cable, and neither attempt started SLAM."""
    a, _, stack = _rig(tmp_path, start_mv=7100, drain_mv_per_s=0.0, charge_mv_per_s=0.0,
                       unplug_after_full_s=None)
    res, out = _assign(a, "Custom", EXPLORE, action="explore-low-battery")
    assert res.accepted
    first = _terminal(a, "explore-low-battery", list(out))
    assert first[0].state == pb.ACTION_STATE_FAILED
    assert "at or below the return threshold" in first[0].message

    stack.plug()                                          # the operator charges the robot
    stack.set_battery(7710)                               # and the reading climbs past return
    res, out = _assign(a, "Custom", EXPLORE, token=2, action="explore-low-battery-retry")
    assert res.accepted
    second = _terminal(a, "explore-low-battery-retry", list(out))
    assert [(s.state, s.message) for s in second] == [(pb.ACTION_STATE_FAILED, CABLE_IN)]
    assert "start_slam" not in stack.log and not stack.moving()


def test_on_the_ros2_stack_a_silent_imu_refuses_the_segment_before_slam(tmp_path) -> None:
    """The robot's own reading: the ROS 2 stack's IMU window, silent while the cable is in."""
    from test_ros2_explore import (
        _bound_dir,
        _imu,
        _stack,
        give_both_costmaps_the_keepout_filter,
    )

    _bound_dir(tmp_path)
    st, ros, shell, _, _ = _stack(tmp_path)
    ros.t = 10.0
    ros.answers["/controller_server/get_parameters"] = {"values": [
        {"type": 3, "double_value": 0.2}]}
    ros.answers["/velocity_smoother/get_parameters"] = {"values": [
        {"type": 8, "double_array_value": [0.2, 0.0, 1.0]}]}
    give_both_costmaps_the_keepout_filter(ros)
    emits = Emits()
    job = explore.ExploreJob(action_id="explore-first", doc={
        "operation": "explore-floor", "bound_ref": "room-1", "max_speed_mps": 0.2},
        stack=st, limits=st.battery_limits(), artifacts=explore.ArtifactDir(str(tmp_path), "m"),
        namespace="lab", clock=st.clock, emit=emits)
    job.run()
    assert emits.last[:2] == ("FAILED", CABLE_IN)
    assert not any(detach for _, detach, _ in shell.runs), "something was launched"

    ros.windows["imu"] = [(9.0 + i * 0.1, _imu(0.001 * (i % 3))) for i in range(10)]
    emits.updates.clear()
    job = explore.ExploreJob(action_id="explore-first", doc={
        "operation": "explore-floor", "bound_ref": "room-1", "max_speed_mps": 0.2},
        stack=st, limits=st.battery_limits(), artifacts=explore.ArtifactDir(str(tmp_path), "m"),
        namespace="lab", clock=st.clock, emit=emits)
    job.run()
    assert "cable" not in emits.last[1]                  # past the cable, to the battery
