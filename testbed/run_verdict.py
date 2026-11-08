#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Was this run fit to quote? Gates on the MEASUREMENT, not only on the instrument.

WHY THIS EXISTS

`tf_liveness` asserts the instrument was healthy. It passed a run whose samples were
14.1x the median of the same command on the same tree twenty minutes earlier:

    /tf start 56.5 Hz, end 42.5 Hz (ratio 0.75)   -> floor 20.0 ok, ratio 0.6 ok
    3 of 6 runs skipped
    samples 1.098 / 8.424 / 11.328 s
    exit 0

Both TF conditions were satisfied by a plainly unusable run, because TF rate is a PROXY
for host contention and the proxy's thresholds sit below the contention that ruins a
measurement. The tempting fix is to raise the floor and tighten the ratio. That is
threshold-guessing from one observation, and a floor set close to the design rate turns
ordinary jitter into failed runs -- which trains people to re-run until green, the exact
behaviour the standing rule exists to prevent.

So this module does not touch the TF thresholds. It catches the failure by its EFFECT,
on two grounds that need no proxy:

  1. KINEMATIC FLOOR. `speed_at_cancel / decel` is a hard lower bound the plant cannot
     beat -- it is the time a body at that speed takes to reach zero at that
     deceleration. A sample far above its own floor is not a slow stop; it is a broken
     run. 11.328 s against a 0.360 s floor is 31.5x, and is absurd on its face without
     knowing anything about TF.

     The multiple is not tuned to taste, but the margin is narrower than it first
     looked and that is worth stating plainly. Healthy samples recorded on this rig span
     1.4x - 2.2x their own floor (the 2.2x is the run of 2026-09-02, taken after this
     gate landed). The degraded run's samples sit at 3.1x, 21.1x and 31.5x.

     So the limit does NOT cleanly separate every degraded sample from every healthy
     one: the degraded run's best sample, 3.1x, is below 4x and this gate alone would
     let it through. That run is still refused, on the skip rate. The two gates are
     complementary rather than redundant, and the kinematic gate should be read as
     catching samples that are physically impossible, not as a fine discriminator near
     the boundary. If healthy samples keep drifting upward, the right response is to
     find out why the stops are getting slower, not to raise this number.

  2. SKIP RATE. Three of six runs recorded `peak seen 0.000 m/s`: under a live, "ready"
     Nav2 stack the robot did not move at all in half the attempts. `CRUISE_FLOOR`
     correctly refused to sample them, but nothing aggregated that refusal into the
     verdict. A run that discards half its attempts and then reports a distribution over
     the survivors is reporting a distribution over whatever the contention left behind.

Neither gate can make a passing run pass more easily. Both can only turn a silent pass
into a loud failure.

This module has NO ROS dependency on purpose: the verdict is a pure function of numbers,
so it can be tested on a laptop against the two recorded runs without
bringing up Docker. See test_run_verdict.py -- a guard that only ever fails is as useless
as one that never does, so both cases are exercised.
"""

from __future__ import annotations

# A sample this many times its own kinematic floor is not a measurement of a stop.
# Observed healthy: 1.4x - 1.95x. Observed degraded: 3.1x, 21.1x, 31.5x.
KINEMATIC_MULTIPLE = 4.0

# Half the attempts discarded means the stack was not fit to measure, whatever the
# survivors say.
MAX_SKIP_FRACTION = 0.5

# The r2-robot compose service runs `fake_robot_node.py --decel 1.0`. If that changes,
# this default and docker-compose.yml must change together -- the measurement script
# cannot see the plant's flags, and a wrong decel here makes the kinematic floor wrong
# in the permissive direction.
DEFAULT_DECEL = 1.0


class RunNotQuotable(AssertionError):
    """The run completed, but its numbers must not be quoted."""


def kinematic_floor(speed_at_cancel: float, decel: float = DEFAULT_DECEL) -> float:
    """The shortest physically possible stop from this speed at this deceleration."""
    if decel <= 0:
        raise ValueError("decel must be positive")
    return speed_at_cancel / decel


def assess(samples, attempted, decel: float = DEFAULT_DECEL,
           kinematic_multiple: float = KINEMATIC_MULTIPLE,
           max_skip_fraction: float = MAX_SKIP_FRACTION):
    """Return a list of reasons this run must not be quoted. Empty list == quotable.

    samples   -- [(latency_s, speed_at_cancel_mps), ...] for runs that produced a number
    attempted -- how many runs were attempted, including skips and no-confirmed-rest
    """
    problems = []

    if attempted <= 0:
        return ["no runs were attempted"]

    skipped = attempted - len(samples)
    skip_fraction = skipped / attempted
    if skip_fraction >= max_skip_fraction:
        problems.append(
            f"{skipped} of {attempted} runs produced no sample ({skip_fraction:.0%}); "
            f"at or above the {max_skip_fraction:.0%} limit the stack was not fit to "
            f"measure, and a distribution over the survivors is a distribution over "
            f"whatever was left behind")

    absurd = []
    for latency, speed in samples:
        floor = kinematic_floor(speed, decel)
        if floor > 0 and latency > kinematic_multiple * floor:
            absurd.append((latency, speed, floor, latency / floor))
    if absurd:
        detail = "; ".join(
            f"{lat:.3f}s at {spd:.3f} m/s is {mult:.1f}x its {floor:.3f}s kinematic floor"
            for lat, spd, floor, mult in absurd)
        problems.append(
            f"{len(absurd)} sample(s) exceed {kinematic_multiple:.0f}x the physical "
            f"minimum stop time at decel {decel:.2f} m/s^2 -- {detail}. A stop cannot "
            f"take that long; this is the host, not the robot")

    return problems


def summary(samples, attempted, decel: float = DEFAULT_DECEL) -> str:
    """One line, printed ALWAYS -- beside the numbers, not only on failure."""
    if attempted <= 0:
        return "measurement: no runs attempted"
    skipped = attempted - len(samples)
    worst = 0.0
    for latency, speed in samples:
        floor = kinematic_floor(speed, decel)
        if floor > 0:
            worst = max(worst, latency / floor)
    return (f"measurement: {len(samples)}/{attempted} sampled "
            f"({skipped} skipped, {skipped / attempted:.0%}), "
            f"worst sample {worst:.1f}x its kinematic floor "
            f"(decel {decel:.2f} m/s^2, limit {KINEMATIC_MULTIPLE:.0f}x)")
