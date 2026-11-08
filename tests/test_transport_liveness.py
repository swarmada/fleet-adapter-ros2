# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The two halves of "the adapter went blind and said nothing".

**Half one, the crash.** ``diagnostic_msgs/DiagnosticStatus.level`` is an IDL
``byte``; rclpy delivers it as a one-element ``bytes``, and ``int(b"\\x00")``
raises ``ValueError`` rather than reading the byte's value. The raise happened
inside the ``/diagnostics`` subscription callback, i.e. inside the rclpy executor.
``tests/test_telemetry.py`` covers the mapping function; these cover the callback
``Nav2Binding`` actually hands to ``create_subscription``, which is the frame the
exception escaped from.

**Half two, the silence.** An exception out of ``rclpy.spin()`` kills the thread
``start()`` spins on. Nothing restarted it and nothing detected it, so the process
stayed up serving pre-crash odom, battery, hardware and goal status forever. These
tests pin the detection: the fault is recorded, ``transport_alive()`` reports it,
``tick()`` re-raises it on the caller's thread, and ``is_stopped()`` refuses to
confirm rest from a transport that has stopped speaking.

No ROS install: ``_record_spin_exit`` is the executor thread's exit path, exposed
as a seam exactly like ``on_goal_result``.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from conftest import mark_transport_live

from fleet_adapter_ros2 import telemetry
from fleet_adapter_ros2.nav2_binding import Nav2Binding


def _diag(*levels):
    return NS(status=[NS(name=f"c{i}", level=lv, message="m", values=[])
                      for i, lv in enumerate(levels)])


def _odom_at_rest():
    return NS(pose=NS(pose=NS(position=NS(x=1.0, y=2.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=0.0, y=0.0, z=0.0),
                                angular=NS(x=0.0, y=0.0, z=0.0))))


# ── half one: the byte level must not raise out of the callback ──────────────

def test_on_diagnostics_accepts_the_ros_byte_level() -> None:
    """The exact message shape that crashed the executor: level as `bytes`.

    Pre-fix this raised `ValueError: invalid literal for int() with base 10: b'\\x00'`
    from inside the subscription callback. `fake_robot_node.py` publishes exactly this
    on an unconditional 1 s timer in both R1 and R2, so it fired on every run.
    """
    b = Nav2Binding("amr-1")
    b.on_diagnostics(_diag(b"\x00", b"\x01", b"\x02", b"\x03"))
    assert [h.status for h in b.latest_hardware("amr-1")] == [
        telemetry.STATUS_HEALTHY,
        telemetry.STATUS_DEGRADED,
        telemetry.STATUS_FAILED,
        telemetry.STATUS_FAILED,
    ]


def test_on_diagnostics_still_accepts_an_int_level() -> None:
    """The int-typed form every other test mocks — which is why the suite was green
    while the live runtime crashed on the first message."""
    b = Nav2Binding("amr-1")
    b.on_diagnostics(_diag(0, 1, 2, 3))
    assert [h.status for h in b.latest_hardware("amr-1")] == [
        telemetry.STATUS_HEALTHY,
        telemetry.STATUS_DEGRADED,
        telemetry.STATUS_FAILED,
        telemetry.STATUS_FAILED,
    ]


def test_bytes_and_int_levels_map_identically() -> None:
    ints = Nav2Binding("amr-1")
    ints.on_diagnostics(_diag(0, 1, 2, 3))
    raw = Nav2Binding("amr-1")
    raw.on_diagnostics(_diag(b"\x00", b"\x01", b"\x02", b"\x03"))
    assert ints.latest_hardware("amr-1") == raw.latest_hardware("amr-1")


# ── half two: a dead executor must be detectable ─────────────────────────────

def test_a_healthy_binding_reports_no_fault() -> None:
    b = mark_transport_live(Nav2Binding("amr-1"))
    assert b.transport_fault() is None
    b.tick(0.1)  # a live transport ticks silently


def test_never_started_is_not_alive_but_is_not_a_fault_either() -> None:
    """`transport_alive()` is False before `start()`; `transport_fault()` is what
    distinguishes "no transport" from "a broken one"."""
    b = Nav2Binding("amr-1")
    assert b.transport_alive() is False
    assert b.transport_fault() is None


# ── the blind spot: a transport that never existed ───────────────────────────
# `transport_fault()` stays None above, correctly — nothing died. But the fault was the
# ONLY thing `tick()` raised on, so the never-started state reached the detector looking
# exactly like perfect health. These pin the second trigger.

def test_tick_raises_on_a_transport_that_was_never_started() -> None:
    """The assertion this file previously made in reverse.

    It used to call `tick()` on an un-started `Nav2Binding` and require silence, commented
    "must not raise on the $0 path, where start() is never called". **The premise was
    wrong: there is no `$0` path through this class.** `--binding simulated` constructs
    `SimulatedRobot`; `Nav2Binding` is built only by `--binding ros2`, which has no
    legitimate un-started state. So the one case the old assertion protected was the
    defect itself, and requiring silence for it is what let an inert binding reach the
    detector undetected.
    """
    b = Nav2Binding("amr-1")
    with pytest.raises(RuntimeError, match="NEVER STARTED") as caught:
        b.tick(0.1)
    # Diagnostic, not just an exception: it must name the consequence and the remedy.
    msg = str(caught.value)
    assert "is_stopped() is False forever" in msg
    assert "bring_up_robot" in msg
    assert caught.value.__cause__ is None   # nothing died, so there is nothing to chain


def test_the_never_started_report_is_made_once_and_the_state_stays_readable() -> None:
    b = Nav2Binding("amr-1")
    with pytest.raises(RuntimeError, match="NEVER STARTED"):
        b.tick(0.1)
    b.tick(0.1)                               # reported once, not every 0.2 s
    assert b.transport_alive() is False       # ...but still answerable
    assert b.transport_fault() is None        # and still NOT a fault: nothing died


def test_never_started_and_executor_death_are_reported_independently() -> None:
    """Two diagnoses with different remedies, so one being spent must not silence the
    other. A binding reported as never-started can still be started late and then die."""
    b = Nav2Binding("amr-1")
    with pytest.raises(RuntimeError, match="NEVER STARTED"):
        b.tick(0.1)
    mark_transport_live(b)                    # a late start()
    b.tick(0.1)                               # now silent
    b._record_spin_exit(ValueError("boom"))   # ...and then the executor dies
    with pytest.raises(RuntimeError, match="executor thread is dead"):
        b.tick(0.1)


def test_a_deliberate_stop_silences_the_never_started_report_too() -> None:
    """`stop()` sets `_stopping` and clears `_node`, so without this guard every clean
    shutdown would raise NEVER STARTED out of whichever tick landed next — and the
    teardown path does still tick on its way out."""
    b = mark_transport_live(Nav2Binding("amr-1"))
    b._stopping, b._node = True, None         # the post-stop() state
    b.tick(0.1)


def test_stop_on_a_binding_that_was_never_started_silences_it() -> None:
    """`stop()` is documented safe on a binding `start()` never ran for. A caller that has
    asked for teardown has said it is done; ticking after that is its own error, not the
    inert-transport one this detector is for."""
    b = Nav2Binding("amr-1")
    b.stop()                                  # idempotent, no node, no thread
    b.tick(0.1)


def test_executor_death_is_recorded_and_reported() -> None:
    b = Nav2Binding("amr-1")
    exc = ValueError("invalid literal for int() with base 10: b'\\x00'")
    b._record_spin_exit(exc)
    assert b.transport_fault() is exc
    assert b.transport_alive() is False


def test_spin_returning_normally_is_also_a_fault() -> None:
    """`rclpy.spin()` returning without a shutdown request means the executor is gone
    just as surely as an exception does — no callback will run again."""
    b = Nav2Binding("amr-1")
    b._record_spin_exit(None)
    assert isinstance(b.transport_fault(), RuntimeError)
    assert b.transport_alive() is False


def test_tick_reraises_the_fault_on_the_callers_thread_exactly_once() -> None:
    """`tick()` is the binding's only guaranteed main-thread callback, so it is where
    a fault raised on the (now dead) executor thread becomes visible to a live one."""
    b = Nav2Binding("amr-1")
    cause = ValueError("boom")
    b._record_spin_exit(cause)

    with pytest.raises(RuntimeError) as caught:
        b.tick(0.1)
    assert caught.value.__cause__ is cause

    b.tick(0.1)  # reported once, not on every tick
    assert b.transport_alive() is False       # ...but the state stays readable
    assert b.transport_fault() is cause


def test_a_dead_transport_is_never_confirmed_at_rest() -> None:
    """C5.3: rest is confirmed from measured motion, never inferred from silence — and
    a dead executor IS silence. A robot that happened to be at rest when the thread
    died is not evidence that it is at rest now."""
    b = Nav2Binding("amr-1")
    b.on_odom(_odom_at_rest())
    assert b.is_stopped("amr-1") is True      # live odom, twist ~0 → confirmed

    b._record_spin_exit(ValueError("boom"))
    assert b.is_stopped("amr-1") is False     # same cached odom, no longer evidence


def test_a_deliberate_stop_is_not_a_fault() -> None:
    """`stop()` makes `rclpy.spin()` return on purpose; that must not be reported as
    the transport dying, or every clean shutdown would raise out of the next tick."""
    b = Nav2Binding("amr-1")
    b._stopping = True
    b._record_spin_exit(None)
    assert b.transport_fault() is None
    b.tick(0.1)


def test_the_first_fault_wins() -> None:
    """A later exit must not overwrite the exception that actually took the executor
    down — that one is the diagnosis."""
    b = Nav2Binding("amr-1")
    first = ValueError("the real cause")
    b._record_spin_exit(first)
    b._record_spin_exit(RuntimeError("noise"))
    assert b.transport_fault() is first
