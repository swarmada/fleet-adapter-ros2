# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A segment moves only on a battery it can read, and a battery nobody reports is loud.

Two halves of one gap on the robot this adapter drives first:

* The robot publishes no ``sensor_msgs/BatteryState``. Its controller publishes the pack's
  voltage in mV on a ``std_msgs/UInt16`` topic. The binding reads that topic when told to, and
  turns it into a percent between an empty and a full reading the operator states -- no
  default pretends to know them. A battery that never reported is None, left out of
  telemetry and reported as a Failed ``battery`` entry naming the topic; it is never 0.
* An explore-floor segment used to start the explorer without ever having read the battery,
  and skipped every missing reading after, so neither the return threshold nor the hard stop
  could fire. Now: no fresh reading before motion, no segment (the refusal names the topic);
  a reading gone stale while exploring returns the robot to the mark; below the return
  threshold it returns, below the hard stop it stops.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from conftest import mark_transport_live
from test_explore_job import FakeClock, _job
from test_ros2_explore import ROS2, SITE, _site_file, _stack
from test_stall_reporting import _main
from test_telemetry_liveness import _odom_at_rest, _one_cycle

from fleet_adapter_ros2 import telemetry
from fleet_adapter_ros2.nav2_binding import BATTERY_SILENT_S, Nav2Binding
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

TOPIC = "/ros_robot_controller/battery"


class Tick:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _mv_binding(clock=None) -> Nav2Binding:
    return Nav2Binding("amr-1", clock=clock, battery_mv_topic=TOPIC, battery_empty_mv=7000,
                       battery_full_mv=8400)


# ── the binding: the robot's own millivolt topic ────────────────────────────

def test_a_millivolt_reading_is_a_percent_between_empty_and_full_and_its_voltage() -> None:
    b = _mv_binding()
    b.on_battery_mv(NS(data=7700))
    assert b.battery_percent("amr-1") == 50
    assert b.latest_battery("amr-1").voltage == pytest.approx(7.7)
    assert b.latest_battery("amr-1").charging is None        # the topic does not say
    b.on_battery_mv(NS(data=8600))
    assert b.battery_percent("amr-1") == 100
    b.on_battery_mv(NS(data=6000))
    assert b.battery_percent("amr-1") == 0                   # a real reading, clamped


def test_the_source_is_named_and_battery_state_stays_the_default() -> None:
    assert _mv_binding().battery_source() == f"{TOPIC} (std_msgs/UInt16, mV)"
    assert Nav2Binding("amr-1").battery_source() == "battery_state (sensor_msgs/BatteryState)"


@pytest.mark.parametrize("empty, full", [(None, 8400), (7000, None), (8400, 7000),
                                         (7000, float("nan"))])
def test_a_millivolt_topic_without_both_ends_of_its_scale_is_refused(empty, full) -> None:
    with pytest.raises(ValueError, match="battery_empty_mv"):
        Nav2Binding("amr-1", battery_mv_topic=TOPIC, battery_empty_mv=empty,
                    battery_full_mv=full)


def test_a_battery_that_never_reported_is_none_never_zero() -> None:
    assert _mv_binding().battery_percent("amr-1") is None
    assert Nav2Binding("amr-1").battery_percent("amr-1") is None


def test_silence_is_a_failed_battery_entry_naming_the_topic_and_is_said() -> None:
    clock = Tick()
    b = _mv_binding(clock)
    assert b.latest_hardware("amr-1") == []                  # just started: not yet a fault
    clock.t += BATTERY_SILENT_S + 1
    hw = b.latest_hardware("amr-1")
    assert [h.component_name for h in hw] == ["battery"]
    assert hw[0].status == telemetry.STATUS_FAILED
    assert TOPIC in hw[0].reason and "since start" in hw[0].reason
    b.on_battery_mv(NS(data=8000))
    assert b.latest_hardware("amr-1") == []                  # reporting again
    assert b.battery_age_s() == 0.0
    clock.t += BATTERY_SILENT_S + 5
    assert "for 15 s" in b.latest_hardware("amr-1")[0].reason


def test_the_silence_is_said_on_stderr_once_per_change(capsys) -> None:
    clock = Tick()
    b = _mv_binding(clock)
    clock.t += BATTERY_SILENT_S + 1
    b.latest_hardware("amr-1")
    b.latest_hardware("amr-1")
    err = capsys.readouterr().err
    assert err.count("battery MISSING") == 1 and TOPIC in err


def test_telemetry_leaves_an_unread_battery_out_rather_than_sending_zero() -> None:
    b = mark_transport_live(_mv_binding())
    b.on_odom(_odom_at_rest())
    t = _one_cycle(b)[0].telemetry
    assert (t.position.x, t.position.y) == (3.0, 4.0)
    assert t.battery is None
    b.on_battery_mv(NS(data=7700))
    assert _one_cycle(b)[0].telemetry.battery.percent == 50


# ── the stack hands the binding's reading, its age and its source to the job ─

def test_the_stack_reports_the_readings_age_and_source() -> None:
    st, _, _, b, _ = _stack()
    assert st.battery_mv() is None and st.battery_age_s() is None
    b.on_battery(NS(percentage=float("nan"), power_supply_status=0, voltage=8.1))
    assert st.battery_mv() == pytest.approx(8100.0) and st.battery_age_s() is not None
    assert st.battery_source() == "battery_state (sensor_msgs/BatteryState)"


# ── the job: before motion, and while exploring ─────────────────────────────

def _no_motion(stack) -> None:
    for motion in ("start_explorer", "go_to_mark"):
        assert motion not in stack.log
    assert not stack.moving()


def test_no_battery_reading_refuses_the_segment_before_motion_naming_the_source(
        tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), battery_reading=False)
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "no battery reading on the simulated battery" in message
    assert "no motion" in message
    _no_motion(stack)
    assert "start_slam" not in stack.log


def test_a_stale_reading_before_motion_is_no_reading(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, battery_stale_after_s=-100.0)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    assert emits.last[0] == "FAILED" and "no battery reading" in emits.last[1]
    _no_motion(stack)


def test_a_battery_at_the_return_threshold_refuses_the_segment_before_motion(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, start_mv=7200, drain_mv_per_s=0.0)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "at or below the return threshold 7200 mV" in message
    _no_motion(stack)


def test_the_battery_read_before_motion_is_said(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    said = [m for m in emits.messages() if "battery" in m and "on the simulated battery" in m]
    assert said and "return at 7200 mV" in said[0] and "hard stop at 7000 mV" in said[0]
    assert emits.messages().index(said[0]) < next(
        i for i, m in enumerate(emits.messages()) if "starting SLAM" in m)


def test_a_reading_gone_stale_while_exploring_returns_the_robot_to_the_mark(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, drain_mv_per_s=0.0)
    stack.on_explorer_started = lambda: setattr(stack, "battery_stale_after_s", clock.t + 1.0)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    state, message, _, _ = emits.last
    assert state == "SUCCEEDED" and "frontier: remaining" in message
    assert any("returning: battery reading stale" in m and "simulated battery" in m
               for m in emits.messages())
    assert stack.log.index("stop_explorer") < stack.log.index("go_to_mark")


def test_below_the_return_threshold_while_exploring_returns(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, start_mv=7300,
                                  drain_mv_per_s=1.0)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    assert emits.last[0] == "SUCCEEDED"
    assert any("returning: battery sustained at or below 7200 mV" in m
               for m in emits.messages())


# ── start-up ────────────────────────────────────────────────────────────────

def test_a_millivolt_topic_needs_both_ends_of_its_scale_to_start(monkeypatch, capsys) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--battery-mv-topic", TOPIC,
                             "--battery-full-mv", "8400")
    assert code == 2 and robot is None
    assert "--battery-empty-mv" in err and "--battery-full-mv" in err


def test_an_empty_reading_without_a_millivolt_topic_is_refused(monkeypatch, capsys) -> None:
    code, err, _ = _main(monkeypatch, capsys, *ROS2, "--battery-empty-mv", "7000")
    assert code == 2 and "--battery-mv-topic" in err


def test_the_ros2_binding_reads_the_named_millivolt_topic(monkeypatch, capsys, tmp_path) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                             "--battery-mv-topic", TOPIC, "--battery-empty-mv", "7000",
                             "--battery-hard-mv", "7750", "--battery-return-mv", "7835",
                             "--battery-full-mv", "8400")
    assert code is None and isinstance(robot, Nav2Binding)
    assert robot.battery_source() == f"{TOPIC} (std_msgs/UInt16, mV)"
    assert f"battery: read from {TOPIC}" in err and "between 7000 and 8400 mV" in err
    for v in SITE.values():
        assert v not in err


def test_a_hard_stop_below_the_empty_reading_is_refused(monkeypatch, capsys, tmp_path) -> None:
    code, err, _ = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                         "--battery-mv-topic", TOPIC, "--battery-empty-mv", "7800",
                         "--battery-hard-mv", "7750", "--battery-full-mv", "8400")
    assert code == 2 and "--battery-hard-mv" in err
