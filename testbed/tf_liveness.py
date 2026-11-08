#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""TF liveness: assert the INSTRUMENT was healthy while a measurement was taken.

WHY THIS EXISTS

The R2 testbed's TF output was observed decaying over a run -- 10.03 Hz at t+0, 9.3 Hz at
t+3m, 2.7 Hz at t+6m. Nav2's collision_monitor stops the robot when TF goes stale, and
navigation goals abort with an empty error_msg. None of that is the adapter's doing, and
none of it announces itself in a measurement script's output.

Six cancel -> confirmed-rest samples were taken through that instrument. They completed,
so they were probably fine -- but nothing was watching, so "probably" is the strongest
statement available about them. **A measurement taken through a degrading instrument that
does not complain is the same class of problem as a conformance check that cannot fail:**
it produces a number with no way to tell a good one from a bad one.

So: sample the TF rate at the START and END of every measurement run, print BOTH rates
alongside every number reported, and FAIL LOUDLY if the instrument degraded while the
measurement was being taken.

This never makes a passing run pass more easily. It can only turn a silent pass into a
loud failure.

WHAT THIS FILE IS

An INSTRUMENT-SUPPORT module, and nothing in it scores the adapter: it measures the
testbed's TF rate so that a number taken through a decaying apparatus can be recognised as
untrustworthy. It owns no bring-up of its own -- `attach()` takes a node or an
already-live binding from whoever is running the measurement.

The USE block below is the exception worth naming. It is documentation, not a call site,
but a reader copies it, and it used to teach `binding.start()` -- the very line
that let an R1 harness report a full sheet of passes against a transport the shipped
adapter never opened. So the example shows the ADAPTER's bring-up. A measurement script
(`r2_cancel_latency.py`) may legitimately start the binding itself, because starting it is
part of that apparatus; a script that PASSES or FAILS the adapter must not, and the two
kinds of caller read the same example.

USE

    from tf_liveness import TfLiveness

    binding = Nav2Binding()
    fleet_adapter_ros2Adapter(None, None, binding, ROBOT).bring_up_robot()
    tf = TfLiveness.attach(binding)          # subscribes on the binding's own node
    tf.sample_start()
    ... take measurements ...
    tf.sample_end()
    print(tf.summary())                      # ALWAYS printed, not only on failure
    tf.assert_healthy()                      # raises TfDegraded if it decayed
"""

from __future__ import annotations

import threading
import time

# r2_sensors.py publishes odom->base_link at TF_HZ = 50. Consumers (Nav2's controller,
# collision_monitor) run at ~10 Hz, and tf2 refuses to extrapolate, so the floor is set
# well above consumer rate rather than just above zero: at 10 Hz TF the near-misses that
# broke this testbed are guaranteed, not unlucky.
DEFAULT_FLOOR_HZ = 20.0
# End-vs-start ratio below which the run is not trustworthy even if the floor is met.
DEFAULT_MIN_RATIO = 0.6
DEFAULT_WINDOW_S = 2.0


class TfDegraded(AssertionError):
    """The TF instrument decayed while a measurement was being taken."""


class TfLiveness:
    def __init__(self, node, topic: str = "/tf") -> None:
        from tf2_msgs.msg import TFMessage

        self._lock = threading.Lock()
        self._count = 0
        self._topic = topic
        self._start_hz: float | None = None
        self._end_hz: float | None = None
        # Depth 100: TF at 50 Hz into a shallow queue drops messages and would
        # under-report the very rate this is meant to measure.
        self._sub = node.create_subscription(TFMessage, topic, self._on_msg, 100)

    @classmethod
    def attach(cls, binding_or_node, topic: str = "/tf") -> "TfLiveness":
        """Accept either an rclpy Node or a Nav2Binding whose transport is already up.

        This is a precondition, not a bring-up: it deliberately does NOT start anything.
        Whoever owns the run decides how the transport comes up -- a harness through
        `adapter.bring_up_robot()`, a measurement script through its own `start()` -- and
        an instrument that quietly brought its subject up would be making that choice for
        them, which is the inert-adapter mistake in miniature."""
        node = getattr(binding_or_node, "_node", binding_or_node)
        if node is None:
            raise RuntimeError(
                "TfLiveness.attach: the binding has no live rclpy node, so there is no "
                "executor to receive TF at all. Bring the transport up FIRST -- a harness "
                "via adapter.bring_up_robot(), a measurement script via start().")
        return cls(node, topic)

    def _on_msg(self, _msg) -> None:
        with self._lock:
            self._count += 1

    def _measure(self, window_s: float) -> float:
        with self._lock:
            c0 = self._count
        t0 = time.monotonic()
        time.sleep(window_s)
        with self._lock:
            c1 = self._count
        dt = time.monotonic() - t0
        return (c1 - c0) / dt if dt > 0 else 0.0

    def sample_start(self, window_s: float = DEFAULT_WINDOW_S) -> float:
        self._start_hz = self._measure(window_s)
        return self._start_hz

    def sample_end(self, window_s: float = DEFAULT_WINDOW_S) -> float:
        self._end_hz = self._measure(window_s)
        return self._end_hz

    def summary(self) -> str:
        s = "n/a" if self._start_hz is None else f"{self._start_hz:.1f}"
        e = "n/a" if self._end_hz is None else f"{self._end_hz:.1f}"
        ratio = ""
        if self._start_hz and self._end_hz is not None and self._start_hz > 0:
            ratio = f", end/start {self._end_hz / self._start_hz:.2f}"
        return f"{self._topic} rate: start {s} Hz, end {e} Hz{ratio}"

    def assert_healthy(self, floor_hz: float = DEFAULT_FLOOR_HZ,
                       min_ratio: float = DEFAULT_MIN_RATIO) -> None:
        """Raise TfDegraded if the instrument was not trustworthy for this run.

        Two independent conditions, because they catch different failures:
          * FLOOR   -- an absolute rate below which tf2 extrapolation failures are
                       expected regardless of how the run started.
          * RATIO   -- decay DURING the run. A run that starts and ends at 22 Hz is
                       suspicious in a different way from one that starts at 50 and
                       ends at 22; the second says the instrument changed underneath
                       the measurement.
        """
        if self._start_hz is None or self._end_hz is None:
            raise TfDegraded(
                "TF liveness was never sampled at both ends; the run cannot be "
                "trusted and must not be quoted. " + self.summary())
        problems = []
        if self._end_hz < floor_hz:
            problems.append(f"end rate {self._end_hz:.1f} Hz is below the {floor_hz:.0f} Hz "
                            f"floor (consumers run at ~10 Hz and tf2 will not extrapolate)")
        if self._start_hz > 0 and (self._end_hz / self._start_hz) < min_ratio:
            problems.append(f"rate decayed to {self._end_hz / self._start_hz:.0%} of its "
                            f"starting value during the run")
        if problems:
            raise TfDegraded(
                "TF INSTRUMENT DEGRADED -- every number from this run is suspect:\n  "
                + "\n  ".join(problems) + "\n  " + self.summary())
