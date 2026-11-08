#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The speed a stopping interval was measured at -- read, gated on, and printed.

WHY THIS EXISTS
----------------
A stopping-distance number without the speed it was measured at is not a measurement.

`r2_smoke.py` printed `cancel -> confirmed rest: 0.302s` and INSTRUCTED the reader to
quote it as the real-world bound on the e-stop path. Two runs gave 0.301 s and 0.302 s,
which reads as excellent reproducibility. It was ~2.3x optimistic: both samples were
taken 0.22 s after goal dispatch, with the robot still on its acceleration ramp at
roughly 0.2 m/s. Repeating a biased measurement does not remove the bias -- here it
disguised it, because neither run disclosed the speed.

`r1_smoke.py` had the same hole against the R1 number: it triggers its halt on 0.15 m of
DISPLACEMENT, so the speed at halt is wherever the acceleration ramp happened to be, and
it printed neither the speed nor the ramp. Two runs of the identical flag produced
1.009 s and 0.966 s and nothing in the output could attribute the difference.

`r2_cancel_latency.py` already had the right shape. This module is that shape, factored
out, so the three scripts share ONE definition of "under way" and one way of printing it
rather than three copies that drift.

WHAT THE SHAPE IS
-----------------
1. Read the speed from odometry at the moment the halt is commanded.
2. Prefer to halt from CRUISE, not from the acceleration ramp -- so the number is a
   property of the stopping path and not of when the trigger happened to fire.
3. Print the speed beside the interval, ALWAYS, and say plainly when a sample was taken
   below the cruise floor and is therefore a lower bound.

Point 3 is the one that cannot be skipped. A gate can be argued about; an undisclosed
measurement condition cannot be recovered after the fact.
"""

from __future__ import annotations

import time

# nav2_params.yaml sets FollowPath.desired_linear_vel = 0.5 m/s and fake_robot_node.py
# cruises at --cruise, default 0.5 m/s. Requiring most of that keeps a sample off the
# acceleration ramp and off in-place rotation: is_stopped() is false whenever EITHER linear
# or angular speed exceeds its epsilon, and RegulatedPurePursuit turns in place before it
# drives, so waiting on `not is_stopped()` fires during rotation and measures how long a
# spin takes to damp out. Measured that way, 3 of 5 R2 samples had 0.000 m/s LINEAR speed at cancel.
#
# NOT widened when a run skips. A floor that is lowered until every run produces a sample
# returns a biased set instead of an honest gap -- which is the e-stop number's own
# failure, not a fix for it.
CRUISE_FLOOR = 0.35  # m/s


def speed(binding) -> float:
    """Linear speed in m/s from the binding's latest odom twist.

    Reads the cached reading directly. This is a measurement harness and the public
    accessors expose only the boolean is_stopped(); a harness that could see nothing but
    that boolean is exactly how the e-stop number came to be published unqualified.
    """
    s = getattr(binding, "_speed", None)
    return 0.0 if s is None else s[0]


def wait_for_cruise(binding, timeout: float, floor: float = CRUISE_FLOOR,
                    poll: float = 0.02) -> tuple[bool, float]:
    """Block until linear speed reaches `floor`.

    Returns (reached, peak): the HIGHEST linear speed seen during the watch. When
    `reached`, that is the first sample at or above `floor`; on a timeout it is the
    maximum over the whole watch, which may be seconds before it ends. Either way it is
    NOT the speed at any later moment, so a caller that prints a speed "at halt" or "at
    cancel" must read speed() at that moment and pass this value as `peak` to label().

    This docstring used to promise `speed_now` while the timeout path returned the peak,
    and r1_smoke printed that peak as "speed at halt" -- on the MentorPi, whose driver
    caps linear speed below `floor`, on every run. r2_cancel_latency already
    printed it as "peak seen", which is what it is.

    `reached` False means the sample that follows is not from cruise: the caller must
    either skip it or label it a lower bound. It must not quietly treat it as equivalent.
    """
    end = time.monotonic() + timeout
    peak = 0.0
    while time.monotonic() < end:
        v = speed(binding)
        peak = max(peak, v)
        if v >= floor:
            return True, v
        time.sleep(poll)
    return False, peak


def label(v: float, reached: bool, moment: str = "halt",
          floor: float = CRUISE_FLOOR, peak: float | None = None) -> str:
    """The words that go beside the number. Never just the number.

    `v` must be the speed read AT `moment`. `moment` names what it was sampled at --
    "halt" under R1's commanded stop, "cancel" under a live Nav2 controller. They are
    different paths and a reader should not have to infer which one produced a figure.

    `peak` is wait_for_cruise()'s second value. When the watch timed out it is printed,
    NAMED as the peak, beside `v`: it says how close the robot came to cruise, which the
    instant alone does not.

    The verdict is decided by `v`, the number the words sit beside, not by `reached`: a
    robot can cross the floor during the watch and be below it at the halt, and the
    label must not print "(>= floor)" beside a number that is not.

    Below the floor after a timed-out watch, the label names NO motion phase. It used to
    say "SPIN-UP", which asserts the robot was still accelerating; the harness does not
    look at the trend, and on the MentorPi the robot was at its own driver-capped cruise,
    0.2000 m/s flat for 160 ms before the halt. What the harness does know --
    the instant, the peak, and that the floor was not reached -- is what it prints.
    """
    missed = "" if reached else (
        ", cruise watch timed out" if peak is None else
        f", peak {peak:.3f} m/s while waiting for cruise, which timed out")
    if v >= floor:
        return f"speed at {moment} {v:.3f} m/s (>= {floor:.2f} m/s cruise floor{missed})"
    if reached:
        return (f"speed at {moment} {v:.3f} m/s -- below the {floor:.2f} m/s cruise floor, "
                f"which it had reached before the {moment}: this interval is a LOWER BOUND, "
                f"not the e-stop bound")
    return (f"speed at {moment} {v:.3f} m/s{missed} -- below the {floor:.2f} m/s cruise "
            f"floor: this interval is a LOWER BOUND, not the e-stop bound")
