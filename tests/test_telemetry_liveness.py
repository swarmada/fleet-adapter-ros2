# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter must not publish telemetry it cannot observe (C6.2).

`Nav2Binding` answers every accessor whether or not its transport is up: `pose()` with the
origin, `battery_percent()` with 0. `_telemetry_loop` published those at full cadence with
a fresh `timestamp_ms`, so a blind adapter was indistinguishable on the wire from a robot
parked at (0,0) on a flat battery.

That is a **C6.2 violation** in its own words — *"never send a defaulted 0/false where the
true reading is unknown"* — and not merely a staleness question.

`tick()` raising (see test_transport_liveness.py) was not sufficient on its own, and the
reason is the interesting part: it raises ONCE, so the first thread to reach it consumes
the signal for every other thread. These tests pin the predicate that cannot be consumed.

No ROS install: the binding's un-started state is the real one, and `mark_transport_live`
reaches the started one.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from conftest import mark_transport_live

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import RobotBinding, SimulatedRobot


class _Pb:
    """Just enough of the generated module for the telemetry path."""

    class RobotPhase:
        ROBOT_PHASE_UNSPECIFIED = 0
        ROBOT_PHASE_IDLE = 1
        ROBOT_PHASE_IN_PROGRESS = 3
        ROBOT_PHASE_OFFLINE = 6

    class RobotPosition:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class BatteryStatus:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TelemetryPayload:
        def __init__(self, **kw):
            # Mirrors proto3 message-field presence: absent stays absent.
            self.position = kw.get("position")
            self.battery = kw.get("battery")
            self.phase = kw.get("phase")
            self.robot_id = kw.get("robot_id")
            self.timestamp_ms = kw.get("timestamp_ms")
            self.current_action = kw.get("current_action")

    class AdapterMessage:
        def __init__(self, telemetry=None, **_):
            self.telemetry = telemetry


def _odom_at_rest():
    return NS(pose=NS(pose=NS(position=NS(x=3.0, y=4.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=0.0, y=0.0, z=0.0),
                                angular=NS(x=0.0, y=0.0, z=0.0))))


def _one_cycle(robot) -> list:
    """Run exactly one pass of `_telemetry_loop` and return what it queued."""
    a = Adapter(_Pb(), None, robot, "amr-1")
    a._telemetry_interval = 0            # no real sleeping
    cycles = iter([False, True])         # one pass, then _shutdown looks set

    class _Once:
        def is_set(self):
            return next(cycles)

    a._shutdown = _Once()
    a._telemetry_loop(None)
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def test_an_unobservable_robot_is_reported_OFFLINE_with_no_position_or_battery() -> None:
    """The C6.2 requirement, directly: unknown readings are ABSENT, not 0."""
    msgs = _one_cycle(Nav2Binding("amr-1"))          # never started
    assert len(msgs) == 1
    t = msgs[0].telemetry
    assert t.phase == _Pb.RobotPhase.ROBOT_PHASE_OFFLINE
    assert t.position is None, "a defaulted (0,0,0) position is exactly what C6.2 forbids"
    assert t.battery is None, "a defaulted 0% battery is exactly what C6.2 forbids"
    assert t.robot_id == "amr-1" and t.timestamp_ms   # still a well-formed payload


def test_the_old_behaviour_would_have_published_the_defaults() -> None:
    """Pins WHAT was wrong, so the regression is recognisable rather than abstract. These
    are the values the accessors return on an un-started binding, and they were published
    every 200 ms with a fresh timestamp. The battery no longer defaults: it answers None,
    and the loop leaves the battery out."""
    b = Nav2Binding("amr-1")
    assert b.pose("amr-1") == (0.0, 0.0, 0)
    assert b.battery_percent("amr-1") is None
    assert b.transport_alive() is False    # ...and the adapter now asks this first


def test_an_observable_robot_reports_position_and_battery_normally() -> None:
    """The guard must not degrade a healthy adapter — otherwise it would turn every run
    OFFLINE, which is a worse failure than the one being fixed."""
    b = mark_transport_live(Nav2Binding("amr-1", odometry_source="measured"))
    b.on_odom(_odom_at_rest())
    b.on_battery(NS(percentage=0.77, voltage=24.0, current=-1.0, power_supply_status=2))
    t = _one_cycle(b)[0].telemetry
    assert t.phase == _Pb.RobotPhase.ROBOT_PHASE_IDLE
    assert (t.position.x, t.position.y) == (3.0, 4.0)
    assert t.battery.percent == 77


def test_a_consumed_tick_raise_does_not_let_defaults_through() -> None:
    """THE HOLE THIS PREDICATE EXISTS FOR.

    `tick()` raises once. `main()` starts the safety stream BEFORE `bring_up_robot()`, so
    an estop arriving in that window reaches `_confirm_estop`, which deliberately swallows
    the transport RuntimeError so the control plane still gets an `EstopAck(FAILED)`. That
    SPENDS the binding's single raise. The telemetry loop starts afterwards, on
    `hello_ack`, and would then tick a silent, apparently healthy binding forever —
    measured at 0 raises in 200 ticks, publishing (0,0,0) and 0%.

    An exception is a one-shot signal that one thread can consume for another. A predicate
    cannot be consumed.
    """
    # Declared measured, so the estop reaches the transport rather than stopping at the
    # evidence gap: this test is about a transport that was never started, not about what
    # the odometry is.
    b = Nav2Binding("amr-1", odometry_source="measured")
    pre = Adapter(_Pb(), None, b, "amr-1")
    assert pre._confirm_estop() == "FAILED"        # the estop window
    assert b._never_started_raised is True         # the signal is now spent
    b.tick(0.2)                                   # proof: no longer raises

    t = _one_cycle(b)[0].telemetry                 # ...and telemetry still refuses to lie
    assert t.phase == _Pb.RobotPhase.ROBOT_PHASE_OFFLINE
    assert t.position is None and t.battery is None


def test_a_transport_that_dies_mid_run_stops_reporting_readings() -> None:
    """The transport died mid-run. The cached odom is a real past measurement, not a
    default — but
    the CURRENT reading is unknown, which is what C6.2 is about. The robot may have kept
    moving after the executor died."""
    b = mark_transport_live(Nav2Binding("amr-1", odometry_source="measured"))
    b.on_odom(_odom_at_rest())
    assert _one_cycle(b)[0].telemetry.position is not None   # healthy: reported

    b._record_spin_exit(ValueError("boom"))
    t = _one_cycle(b)[0].telemetry
    assert t.phase == _Pb.RobotPhase.ROBOT_PHASE_OFFLINE
    assert t.position is None, "frozen odom must not be republished as a current reading"


def test_no_terminal_task_status_is_streamed_while_blind() -> None:
    """A frozen `GoalStatus` must not become a terminal report. SUCCEEDED off a stale
    status tells the control plane an action completed when nothing is known about it, and
    it is then free to dispatch the next one."""
    b = Nav2Binding("amr-1")                 # never started → blind
    b.on_goal_result(4)                      # SUCCEEDED, frozen in the cache
    a = Adapter(_Pb(), None, b, "amr-1")
    a._current_action, a._last_task_state = "task-1", "RUNNING"
    a._telemetry_interval = 0
    cycles = iter([False, True])
    a._shutdown = NS(is_set=lambda: next(cycles))
    a._telemetry_loop(None)
    msgs = [a._outbox.get_nowait() for _ in range(a._outbox.qsize())]
    assert all(m.telemetry is not None for m in msgs), \
        "no ActionStatusUpdate may be emitted from a frozen goal status"
    assert a._current_action == "task-1"     # the action was NOT cleared as complete


def test_the_degrade_and_recover_transitions_are_logged_once_each(capsys) -> None:
    """5 lines a second would bury the diagnosis it is delivering."""
    b = Nav2Binding("amr-1")
    a = Adapter(_Pb(), None, b, "amr-1")
    a._report_transport_not_live("detail here")
    a._report_transport_not_live("detail here")
    a._report_transport_not_live("detail here")
    err = capsys.readouterr().err
    assert err.count("telemetry DEGRADED") == 1
    assert "C6.2" in err and "detail here" in err


def test_the_simulated_path_is_untouched() -> None:
    """`--binding simulated` has no transport to lose, so it must never degrade. Its
    `transport_alive()` returning True is the honest answer, not a stub."""
    robot = SimulatedRobot()
    robot.spawn("amr-1")
    assert robot.transport_alive() is True
    t = _one_cycle(robot)[0].telemetry
    assert t.phase == _Pb.RobotPhase.ROBOT_PHASE_IDLE
    assert t.position is not None and t.battery.percent == 100


def test_the_protocol_declares_the_liveness_question() -> None:
    """Asked of every binding through the contract, not discovered with `hasattr` — a
    binding spelling it `is_connected()` would otherwise be treated as always-observable,
    which is the inert-binding failure shape exactly."""
    assert "transport_alive" in RobotBinding.__protocol_attrs__


def test_a_binding_without_the_liveness_method_fails_loudly() -> None:
    """It is part of the contract, so its absence must be an error and never a silent
    assumption that the robot is observable."""
    class _NoLiveness:
        def start(self) -> None: ...
        def tick(self, dt: float) -> None: ...

    with pytest.raises(AttributeError, match="transport_alive"):
        _one_cycle(_NoLiveness())
