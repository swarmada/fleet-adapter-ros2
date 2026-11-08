#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R2 cancel -> confirmed-rest latency, measured N times against a live Nav2 stack.

This is r2_smoke.py assertion 4, run repeatedly and reported as a distribution. One
sample is not a bound, and this figure is the real-world bound on the e-stop path
under a real controller -- the number `safety.md` currently estimates rather than
measures.

WHAT IS IN THE PATH (and is not, in R1):

    adapter.command_stop()
      |-- cancel_goal_async()  -> controller_server stops emitting
      |                            -> velocity_smoother ramps /cmd_vel_smoothed down
      |                            -> collision_monitor republishes to /cmd_vel
      `-- publish(Twist())     -> a SINGLE zero Twist on /cmd_vel
                                  NOTE: collision_monitor publishes /cmd_vel at ~10 Hz
                                  continuously, so this one message is overwritten
                                  within ~100 ms. Under a real Nav2 stack the halt
                                  comes from the goal cancel, NOT from this publish.

So the measured interval includes Nav2's controller loop, the velocity_smoother's
deceleration ramp, and collision_monitor's republish -- none of which exist in R1,
where the adapter is the only publisher on /cmd_vel and the fake robot's own --decel
sets the number.

Rest is CONFIRMED from odom (is_stopped), never inferred -- same discipline as C5.3.

THIS IS AN INSTRUMENT, AND ITS start() IS APPARATUS
----------------------------------------------------
It starts the binding itself, on purpose, and that is left as it is.

The line that matters is whether a file SCORES the adapter or MEASURES it. A harness
scores: it passes or fails the adapter, and one that starts the binding on the adapter's
behalf is scoring a configuration the adapter cannot produce -- which is exactly how
the inert adapter survived a conformance run, and why `r1_smoke.py` and `r2_smoke.py`
bring up
through `adapter.bring_up_robot()` instead.

This file asserts nothing about the adapter. It reports a distribution, and everything it
CAN fail on is a property of the apparatus rather than of the adapter: the readiness gate
(`r2_ready`), the TF rate at both ends (`tf_liveness`), the kinematic floor, and the
skip count. The subject of the measurement is the cancel -> confirmed-rest path through
`Nav2Binding` and Nav2's controller pipeline, so the transport being up is not a step
borrowed from the adapter -- it is the thing being measured, and there is no measurement
without it.

Two things keep that honest, and both are load-bearing:

  * `binding.start(); binding.spawn()` below is the same pair, in the same order, that
    `adapter.bring_up_robot()` performs. The apparatus is one the adapter reproduces. If
    that ever stops being true, this file is measuring something else and must be moved
    onto `bring_up_robot()`.
  * An un-started binding cannot produce a sample at all. The loop times from
    `command_stop()` until `is_stopped()` is True, and `is_stopped()` reads `_speed`,
    which only `on_odom` sets, which only a subscription created by `start()` delivers --
    so it would return False forever and the run would report NO CONFIRMED REST rather
    than a fast number. That is why this file's samples are themselves evidence the
    binding was up, and why it could not have hidden an inert adapter the way R1 did.

WHEN THE MEASUREMENT MAY BEGIN
-------------------------------
This script used to sleep `--settle` seconds and start measuring. It no longer does. It
blocks on `r2_ready.wait_until_ready()`, which asks Nav2 to actually plan a path and
checks that the newest scan is transformable into the local costmap's frame at its own
stamp. A fixed sleep is a guess about the host it was tuned on -- and the harness was
shipping a Makefile that overrode this file's own 20 s guess down to 5 s, so the
documented route to this number produced NO SAMPLES on a cold stack.

`--settle` is still accepted and now means an EXTRA hold after readiness, default 0. It
is not a substitute for the gate and cannot be used as one.

WHAT EVERY RUN RECORDS
-----------------------
  * the testbed digest -- which apparatus produced these numbers (testbed_provenance.py)
  * how long readiness took, and what was refused on the way
  * the TF rate at both ends of the window (tf_liveness.py)
  * the SPEED at every cancel, beside every latency

Run inside the testbed:
    python3 testbed/r2_cancel_latency.py --runs 6
"""

from __future__ import annotations

import argparse
import statistics
import time

try:
    from fleet_adapter_ros2.nav2_binding import Nav2Binding
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: cannot import Nav2Binding ({exc})")
    raise SystemExit(2) from exc

# Testbed-local helpers (noqa: E402 -- they follow the adapter import guard above).
from r2_ready import NotReady, wait_until_ready  # noqa: E402
from speed_gate import CRUISE_FLOOR, wait_for_cruise  # noqa: E402
from testbed_provenance import changed_since, stamp  # noqa: E402
from run_verdict import DEFAULT_DECEL, assess  # noqa: E402
from run_verdict import summary as verdict_summary  # noqa: E402
from tf_liveness import TfDegraded, TfLiveness  # noqa: E402

ROBOT = "robot-conformance-1"
READY_TIMEOUT = 180.0  # deadline for the Nav2 readiness gate, not a settle
DRIVE_WATCH = 45.0   # time allowed to get under way before a cancel is meaningful
STOP_WATCH = 15.0    # generous: we are MEASURING this, not asserting a budget
# Alternating open-floor goals. From make_map.py the racking occupies x in [-6,-2] and
# [2,6], y in [-4,4]; the corridor at x=0 is clear, so these never plan into a rack.
GOALS = [(0.0, 6.0), (0.0, -6.0)]


def _wait(pred, timeout: float, poll: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(poll)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT,
                    help="deadline for the Nav2 readiness gate, seconds")
    ap.add_argument("--decel", type=float, default=DEFAULT_DECEL,
                    help="the plant's deceleration in m/s^2, used for the kinematic "
                         "floor. MUST match the --decel the r2-robot "
                         "compose service passes to fake_robot_node.py; this script "
                         "cannot see the plant's flags, and a wrong value here makes "
                         "the floor wrong in the permissive direction.")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="EXTRA hold after readiness, seconds. Default 0. This is not a "
                         "substitute for the gate; see the module docstring.")
    args = ap.parse_args()

    # First line of every run: which apparatus this is. A number whose testbed cannot
    # be identified cannot be re-derived, and re-derivation is the rule.
    stamp_at_start = stamp()
    print(f"testbed: {stamp_at_start}")

    binding = Nav2Binding()
    # Deliberately this file's own bring-up, and NOT `adapter.bring_up_robot()`: this is an
    # instrument and the transport is its apparatus, not a step borrowed from the adapter
    # (see the module docstring). Same two calls in the same order the adapter makes, so
    # the configuration measured is one the adapter reproduces.
    binding.start()
    binding.spawn(ROBOT)
    # The instrument is measured alongside the measurement. TF output was observed
    # decaying over a run, which stops the robot via collision_monitor and aborts goals --
    # neither of which is visible in a latency number that merely completed.
    tf = TfLiveness.attach(binding)

    # NOT a sleep. Nav2 is asked to plan, and the scan is checked against the TF cache;
    # see r2_ready.py. A failure here is a stack that never became usable,
    # which is a different and more honest outcome than six skipped runs.
    print("waiting for Nav2 to become READY (planner round-trip + scan into costmap) ...")
    try:
        ready = wait_until_ready(args.ready_timeout)
    except NotReady as exc:
        print(f"\nNOT READY: {exc}")
        print("No measurement taken. The stack never became able to plan, so there is "
              "nothing to report -- not even a skipped run.")
        return 2
    if args.settle > 0:
        print(f"holding a further {args.settle:.0f}s (--settle) ...")
        time.sleep(args.settle)
    print(f"TF at start: {tf.sample_start():.1f} Hz")

    samples: list[tuple[float, float]] = []   # (latency_s, speed_at_cancel)
    for i in range(args.runs):
        goal = GOALS[i % len(GOALS)]
        print(f"\nrun {i + 1}/{args.runs}: goal {goal}")
        binding.command_move(ROBOT, *goal)

        # "Under way" must mean DRIVING, not merely "not stopped".
        #
        # is_stopped() is false whenever EITHER linear or angular speed exceeds its
        # epsilon, and RegulatedPurePursuit turns in place before it drives. Waiting on
        # `not is_stopped()` therefore fires during rotation, and cancelling then
        # measures how long a spin takes to damp out -- not the braking distance of a
        # driving robot, which is what the e-stop bound means. Measured this way, 3 of 5
        # samples had 0.000 m/s linear speed at the moment of cancel.
        reached, v = wait_for_cruise(binding, DRIVE_WATCH)
        if not reached:
            print(f"  SKIP: never reached {CRUISE_FLOOR} m/s linear "
                  f"(peak seen {v:.3f} m/s)")
            binding.command_stop(ROBOT)
            time.sleep(2.0)
            continue

        t_cancel = time.monotonic()
        binding.command_stop(ROBOT)
        if binding.is_stopped(ROBOT):
            # Rest reported in the same instant as the command is inference, not
            # confirmation. Record it rather than silently accepting it.
            print("  WARNING: is_stopped() true immediately after command_stop")
        rested = _wait(lambda: binding.is_stopped(ROBOT), STOP_WATCH)
        dt = time.monotonic() - t_cancel
        if not rested:
            print(f"  NO CONFIRMED REST within {STOP_WATCH:.0f}s (speed at cancel {v:.3f} m/s)")
            continue
        samples.append((dt, v))
        print(f"  cancel -> confirmed rest: {dt:.3f}s   (speed at cancel {v:.3f} m/s)")
        time.sleep(2.0)

    tf.sample_end()
    print("\n" + "=" * 62)
    # Printed ALWAYS, next to every number, not only when something failed. A reader
    # must never have to wonder what the instrument was doing.
    print(f"instrument: {tf.summary()}")
    print(f"readiness:  {ready.summary()}")
    print(f"testbed:    {stamp()}")
    drift = changed_since(stamp_at_start)
    if drift:
        print(drift)
    print(f"{verdict_summary(samples, args.runs, args.decel)}")
    if not samples:
        print("NO SAMPLES -- the robot never came to confirmed rest after a cancel.")
        tf.assert_healthy()
        return 1
    lat = sorted(s[0] for s in samples)
    spd = [s[1] for s in samples]
    print(f"cancel -> confirmed rest over {len(lat)} run(s):")
    print(f"  min    {min(lat):.3f}s")
    print(f"  median {statistics.median(lat):.3f}s")
    print(f"  max    {max(lat):.3f}s")
    print(f"  all    {[round(x, 3) for x in lat]}")
    print(f"  speed at cancel: {[round(x, 3) for x in spd]} m/s")
    print("=" * 62)
    # LAST, and after the numbers are printed: the numbers stay on screen -- a reader
    # must be able to see what was rejected -- but the run exits non-zero and every
    # failing condition is named, not just the first one found.
    #
    # TWO independent verdicts, because they catch different things:
    #   * tf_liveness  -- was the INSTRUMENT healthy? A proxy for host contention.
    #   * run_verdict  -- was the MEASUREMENT physically possible, and did enough of
    #                     the attempts produce anything? Catches by effect, so it does
    #                     not depend on the proxy's thresholds being right.
    problems = []
    try:
        tf.assert_healthy()
    except TfDegraded as exc:
        problems.append(str(exc))
    problems.extend(assess(samples, args.runs, args.decel))
    if problems:
        print()
        print("THIS RUN MUST NOT BE QUOTED -- " + str(len(problems)) + " condition(s) failed:")
        for n, problem in enumerate(problems, 1):
            print(f"  {n}. {problem}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
