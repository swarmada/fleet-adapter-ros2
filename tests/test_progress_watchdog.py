# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The progress watchdog's rules, on a fake clock: what is a stall, what is not, and when.

A goal must get closer by `min_progress_m` within every `window_s`. Before the first feedback
the wait is bounded by one window from dispatch. Neither threshold has a default.
"""

from __future__ import annotations

import math

import pytest

from fleet_adapter_ros2.watchdog import ProgressWatchdog, StallLimits, resolve_stall_limits


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _dog(window: float = 2.0, min_progress: float = 0.1) -> tuple[ProgressWatchdog, Clock]:
    c = Clock()
    w = ProgressWatchdog(StallLimits(window_s=window, min_progress_m=min_progress), clock=c)
    w.arm()
    return w, c


def _run(w: ProgressWatchdog, c: Clock, seconds: float, speed: float, d: float,
         dt: float = 0.05) -> tuple[float, str]:
    """Drive `seconds` of a robot closing on the goal at `speed`, feedback and checks every
    dt. Returns the distance left and the first stall reason (or "")."""
    for _ in range(int(round(seconds / dt))):
        c.t += dt
        d = max(0.0, d - speed * dt)
        w.observe(d)
        reason = w.check()
        if reason:
            return d, reason
    return d, ""


# ── the thresholds: required, positive, finite ──────────────────────────────

@pytest.mark.parametrize("window, min_progress", [
    (0, 0.1), (-1, 0.1), (math.nan, 0.1), (math.inf, 0.1), (True, 0.1), (None, 0.1),
    (2.0, 0), (2.0, -0.1), (2.0, math.nan), (2.0, "0.1"),
])
def test_a_threshold_that_is_not_a_positive_finite_number_is_refused(window, min_progress):
    with pytest.raises(ValueError):
        StallLimits(window_s=window, min_progress_m=min_progress)


def test_there_is_no_default_threshold():
    with pytest.raises(TypeError):
        StallLimits()  # type: ignore[call-arg]


# ── before the first feedback ───────────────────────────────────────────────

def test_no_feedback_is_not_a_stall_until_one_window_after_dispatch():
    w, c = _dog(window=2.0)
    c.t += 1.99
    assert w.check() == ""
    c.t += 0.01
    reason = w.check()
    assert reason.startswith("stalled: no progress feedback arrived within 2.0 s")
    assert "limit 2 s" in reason
    assert not w.armed and w.check() == ""   # reported once


def test_the_first_feedback_starts_the_first_measured_window():
    # Feedback arriving late in the pre-feedback window is not judged on the time left over.
    w, c = _dog(window=2.0, min_progress=0.1)
    c.t += 1.9
    w.observe(3.0)
    c.t += 1.9                                 # 3.8 s after dispatch, 1.9 s after feedback
    assert w.check() == ""
    c.t += 0.1
    assert w.check().startswith("stalled: distance to go improved 0.000 m in the last 2.0 s")


# ── a stall, and what its reason says ───────────────────────────────────────

def test_a_goal_that_stops_getting_closer_fails_one_window_after_its_last_progress():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(3.0)
    d, reason = _run(w, c, 1.0, 0.5, 3.0)      # 0.5 m made good in a second: progress
    assert reason == "" and d == pytest.approx(2.5)
    stalled_at = c.t
    d, reason = _run(w, c, 10.0, 0.0, d)       # then nothing
    assert reason
    # The window restarts on each 0.1 m of progress, so the last restart is within one feedback
    # period of `stalled_at`, and the stall is declared one window after that.
    assert 2.0 - 0.05 <= c.t - stalled_at <= 2.0 + 0.05
    assert "less than the 0.1 m required within 2 s" in reason
    assert "2.50 m still to go" in reason and "0.50 m made good" in reason


def test_progress_just_under_the_minimum_is_a_stall():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(3.0)
    d, reason = _run(w, c, 1.95, 0.099 / 1.95, 3.0)   # 0.099 m in the window, then nothing
    assert reason == ""
    _, reason = _run(w, c, 0.2, 0.0, d)
    assert reason.startswith("stalled: distance to go improved 0.099 m")


def test_the_items_stall_after_four_centimetres_is_declared():
    # --stall-after 0.2 --decel 1.0: 0.04 m, then nothing.
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(3.0)
    d, _ = _run(w, c, 0.4, 0.1, 3.0)
    _, reason = _run(w, c, 5.0, 0.0, d)
    assert "improved 0.040 m" in reason


def test_noise_that_moves_the_distance_back_and_forth_is_not_progress():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(3.0)
    for i in range(60):                          # ±4 cm of jitter about a fixed point
        c.t += 0.05
        w.observe(3.0 + (0.04 if i % 2 else -0.04))
        if w.check():
            break
    assert not w.armed


# ── not a stall ─────────────────────────────────────────────────────────────

def test_steady_progress_never_fires():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(10.0)
    d, reason = _run(w, c, 19.0, 0.5, 10.0)
    assert reason == "" and w.armed and d == pytest.approx(0.5)


def test_slow_progress_that_clears_the_minimum_never_fires():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(10.0)
    _, reason = _run(w, c, 30.0, 0.06, 10.0)     # 0.12 m per window
    assert reason == ""


def test_a_disarmed_watchdog_never_fires_however_long():
    w, c = _dog(window=2.0)
    w.disarm()
    c.t += 3600
    assert w.check() == ""


def test_rearming_starts_a_fresh_window_and_a_fresh_wait_for_feedback():
    w, c = _dog(window=2.0, min_progress=0.1)
    w.observe(3.0)
    c.t += 1.5
    w.disarm()                                   # pause
    c.t += 60
    w.arm()                                      # resume
    c.t += 1.99
    assert w.check() == ""                       # the 60 s held and the old 1.5 s count for nothing
    c.t += 0.01
    assert "no progress feedback" in w.check()


# ── start-up: the adapter's decision ────────────────────────────────────────

def test_ros2_without_either_threshold_is_refused_naming_both():
    limits, why = resolve_stall_limits("ros2", None, None)
    assert limits is None and "--stall-window" in why and "--stall-min-progress" in why


@pytest.mark.parametrize("window, min_progress, named", [
    (2.0, None, "--stall-min-progress"), (None, 0.1, "--stall-window")])
def test_ros2_with_one_threshold_is_refused_naming_the_other(window, min_progress, named):
    limits, why = resolve_stall_limits("ros2", window, min_progress)
    assert limits is None and why.startswith(f"--binding ros2 needs {named}")


def test_ros2_with_an_invalid_threshold_is_refused():
    limits, why = resolve_stall_limits("ros2", 2.0, -0.1)
    assert limits is None and "min_progress_m" in why


def test_ros2_with_both_thresholds_runs_armed():
    limits, why = resolve_stall_limits("ros2", 2.0, 0.1)
    assert why == "" and limits == StallLimits(window_s=2.0, min_progress_m=0.1)


def test_the_simulated_binding_needs_neither():
    assert resolve_stall_limits("simulated", None, None) == (None, "")


def test_half_a_configuration_is_refused_on_any_binding():
    limits, why = resolve_stall_limits("simulated", 2.0, None)
    assert limits is None and "--stall-min-progress is required with --stall-window" in why
