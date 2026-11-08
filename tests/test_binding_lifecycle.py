# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter drives the RobotBinding lifecycle.

`Nav2Binding.start()` is what creates the rclpy node, the three subscriptions, the
NavigateToPose action client and the /cmd_vel publisher. The adapter constructed the
binding and never called it, so `--binding ros2` shipped inert: every live path is
guarded on `self._node is not None`, and not one call raised -- `pose()` answered the
origin, `battery_percent()` answered 0 and `is_stopped()` answered False forever, which
is what failed conformance C5.2 / C8.3 / C14.1 for a reason nothing reported.

The root cause was that `RobotBinding` declared no lifecycle, so no call site knew there
was one to call. These tests pin the repair at the seam rather than at the one line:
the Protocol declares `start`/`stop`, and the adapter drives them on the path every run
takes.

NO ROS 2 INSTALL IS NEEDED for any of this, which is the point -- the defect was in what
the adapter does before any transport exists, so the guard against its return must be
provable wherever the tests run, not only where rclpy does.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace as NS

from swarmada_sdk.safety import ESTOP_FAILED, ESTOP_STOPPED

from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import RobotBinding, SimulatedRobot


class _RecordingBinding:
    """Records the ORDER of lifecycle calls. Order is the assertion, not a detail:
    `spawn` is the first thing asked of the robot, and on a real binding nothing can be
    asked before the transport is up."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def spawn(self, robot_id: str) -> None:
        self.calls.append(f"spawn:{robot_id}")


def test_bring_up_starts_the_binding_before_spawning() -> None:
    binding = _RecordingBinding()
    Adapter(None, None, binding, "amr-1").bring_up_robot()
    assert binding.calls == ["start", "spawn:amr-1"]


def test_run_control_stream_brings_the_robot_up() -> None:
    """The gap `bring_up_robot()` alone leaves: a correct bring-up the shipped path
    never calls is exactly the defect again, one level up. `run_control_stream` is what
    `main()` invokes, so it is what must be pinned.

    Driven far enough to observe the bring-up and no further: the stub raises out of the
    first gRPC call, which happens after bring-up and proves the ordering too -- the
    robot is up before the adapter says a word on the wire.
    """
    binding = _RecordingBinding()

    class _Boom(RuntimeError):
        pass

    class _Stub:
        def ControlStream(self, _requests):  # noqa: N802 - mirrors the generated stub
            raise _Boom

    with pytest.raises(_Boom):
        Adapter(None, None, binding, "amr-1").run_control_stream(_Stub())

    assert binding.calls == ["start", "spawn:amr-1"]


def test_adapter_does_not_probe_for_a_lifecycle() -> None:
    """The rejected design, pinned as rejected.

    The alternative fix was for the adapter to DISCOVER the lifecycle
    (`if hasattr(robot, "start"): robot.start()`). Measured across all three reference
    adapters, it would in fact have worked on every binding as they exist today -- all
    three already had a `start()`, and the two that shipped inert (this one, and
    `fleet-adapter-vda5050`) did so because nothing CALLED it, not because it
    was named something else.

    So it is rejected for what it ENCODES, not for a naming divergence anyone has
    observed: `if hasattr(...)` says "a binding with no lifecycle is fine", which is the
    assumption that was wrong, and it cannot tell an absent lifecycle from one spelled
    differently by a binding not yet written. The adapter must call `start()`
    unconditionally and fail loudly on a binding that has none.
    """

    class _NoLifecycle:
        def spawn(self, robot_id: str) -> None: ...

    with pytest.raises(AttributeError, match="start"):
        Adapter(None, None, _NoLifecycle(), "amr-1").bring_up_robot()


def test_a_failing_bring_up_is_fatal_not_swallowed() -> None:
    """`Nav2Binding.start()` raises when rclpy is absent. That must stop the run, not
    leave the adapter streaming plausible telemetry off a binding connected to nothing."""

    class _CannotConnect:
        def start(self) -> None:
            raise RuntimeError("no ROS 2 runtime")

        def spawn(self, robot_id: str) -> None:  # pragma: no cover - must not be reached
            raise AssertionError("spawn ran after start() failed")

    with pytest.raises(RuntimeError, match="no ROS 2 runtime"):
        Adapter(None, None, _CannotConnect(), "amr-1").bring_up_robot()


def test_simulated_robot_satisfies_the_lifecycle_as_a_no_op() -> None:
    """The cost of putting the lifecycle in the contract: a binding with nothing to
    bring up must say so. Cheap, and deliberately visible -- an absent lifecycle was not.
    """
    robot = SimulatedRobot()
    assert robot.start() is None
    assert robot.stop() is None
    assert robot.stop() is None  # idempotent


def test_the_protocol_declares_the_lifecycle() -> None:
    """The seam itself. Without this the adapter's call site is an assumption about
    bindings rather than a requirement on them, which is the state that shipped inert.
    """
    assert "start" in RobotBinding.__protocol_attrs__
    assert "stop" in RobotBinding.__protocol_attrs__


# ── the detection half: an inert binding must not be able to hide ─────────────
# The fix above stops the ADAPTER shipping an un-started binding. These cover the other
# direction: if some other caller hands the adapter one anyway, it must say so rather
# than stream defaults. `Nav2Binding.tick()` is where that is detected, because it is the
# binding's only guaranteed main-thread callback.

def test_an_unstarted_nav2_binding_is_detected_on_tick() -> None:
    """Before this, `tick()` raised only on a transport that had started and DIED. A
    transport that never existed recorded no exit, so `_spin_fault` stayed None and the
    detector saw perfect health. 200 ticks, zero raises, was the measured before-state."""
    from fleet_adapter_ros2.nav2_binding import Nav2Binding

    b = Nav2Binding()
    with pytest.raises(RuntimeError, match="NEVER STARTED"):
        b.tick(0.2)


def test_the_adapters_telemetry_tick_surfaces_an_inert_binding() -> None:
    """Pinned at the adapter's own call site, not just on the binding. `_telemetry_loop`
    ticks the robot every cycle with no try/except, so the raise ends that thread and the
    control plane sees a telemetry GAP instead of a full-cadence stream of defaults —
    which is the trade made deliberately: publishing nothing is recoverable,
    publishing invented readings as fresh is not.
    """
    from fleet_adapter_ros2.nav2_binding import Nav2Binding

    a = Adapter(None, None, Nav2Binding(), "amr-1")
    with pytest.raises(RuntimeError, match="NEVER STARTED"):
        with a._robot_lock:
            a._robot.tick(0.2)


def test_an_inert_transport_still_answers_an_estop_with_FAILED() -> None:
    """The one place the raise must NOT escape.

    `run_safety_stream` and `_edge_loop` call `confirm_estop` inside their response loops,
    and the safety stream is started BEFORE `bring_up_robot()` — so an estop can reach an
    un-started binding. If the raise escaped there, the thread would die and the control
    plane would get NO EstopAck. Silence on an estop is worse than FAILED: FAILED says
    "rest was not confirmed, escalate".

    FAILED is also the truthful answer. A transport delivering no odom cannot confirm
    rest, and C5.3 forbids inferring it from silence.
    """
    from fleet_adapter_ros2.nav2_binding import Nav2Binding

    a = Adapter(None, None, Nav2Binding(), "amr-1")
    assert a._confirm_estop() == ESTOP_FAILED


def test_a_live_transport_still_reaches_the_real_confirm_estop() -> None:
    """The wrapper must not swallow the normal path: a binding that CAN confirm rest must
    still report STOPPED, or the guard above would have turned every estop into FAILED."""
    from conftest import mark_transport_live
    from fleet_adapter_ros2.nav2_binding import Nav2Binding

    b = mark_transport_live(Nav2Binding("amr-1", odometry_source="measured"))
    b.on_odom(NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                                 orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
                 twist=NS(twist=NS(linear=NS(x=0.0, y=0.0, z=0.0),
                                   angular=NS(x=0.0, y=0.0, z=0.0)))))
    a = Adapter(None, None, b, "amr-1")
    assert a._confirm_estop() == ESTOP_STOPPED


def test_the_wrapper_does_not_swallow_a_bug_in_the_safety_primitive() -> None:
    """Only the transport's RuntimeError is absorbed. A binding that fails some other way
    must still crash — a blanket `except Exception` on the estop path would hide exactly
    the class of defect this item is about."""
    class _Broken:
        def command_stop(self, robot_id: str) -> None:
            raise TypeError("command_stop is not callable like that")

    a = Adapter(None, None, _Broken(), "amr-1")
    with pytest.raises(TypeError):
        a._confirm_estop()


def test_simulated_robot_ticks_forever_without_complaint() -> None:
    """The `$0` path must be untouched by all of the above. `SimulatedRobot` has no
    transport, its `start()` is a no-op, and its `tick()` advances the scenario clock —
    there is no un-started state for it to be in, and `--binding simulated` must not
    acquire one."""
    robot = SimulatedRobot()
    robot.start()
    robot.spawn("amr-1")
    for _ in range(200):
        robot.tick(0.2)
    assert robot.elapsed() == pytest.approx(40.0)
