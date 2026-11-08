#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R1 Step 5 probe — what does the adapter DO on goal rejection / ABORTED / stall?

`r1_smoke.py` asserts the happy path, so under fault injection it fails at "the robot
moved" and stops. That says the harness noticed; it does not say what the ADAPTER did,
which is the question Step 5 of the runbook actually asks:

    "the adapter must reach FAILED, not hang"

This question was unanswerable while the spin thread died on the first /diagnostics
message — both the goal-response and result futures resolve on that
executor. With the executor alive it is answerable, so this asks it directly: dispatch one
goal, then watch task_state() until it leaves RUNNING, with a bounded wait.

Prints one line per phase and exits 0 whatever the outcome — this is an OBSERVATION, not
an assertion. The runbook says to write down what the adapter does; a probe that failed
the run would just be r1_smoke again.

AN INSTRUMENT, and its start() stays. The test is whether a file SCORES the
adapter or MEASURES it: a harness passes or fails the adapter, so a harness that starts the
binding on the adapter's behalf is scoring a configuration the adapter cannot produce —
which is how the inert adapter survived a conformance run, and why `r1_smoke.py` and
`r2_smoke.py`
now bring up through `adapter.bring_up_robot()`. This file has no assertion to corrupt. It
exits 0 on every outcome by design, including the hang, and the R1/R2 results table
records its Step 5 row as measured HERE precisely because `r1_smoke.py` cannot answer the
question without failing the run. Driving the binding directly is the apparatus.

That is a line, not a licence: the first `_check` added to this file makes it a harness,
and the bring-up has to move to `adapter.bring_up_robot()` in the same commit.

    python3 /ws/testbed/r1_step5_probe.py [--watch 30] [--stall-window S --stall-min-progress M]

With both --stall-* values the binding's progress watchdog is armed, as the adapter arms it;
without them it is not, and the probe says which. The values are the plant's: pass the ones
you are claiming for the robot under test, and they are printed with the result.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.watchdog import StallLimits

ROBOT = "robot-conformance-1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--stall-window", type=float, default=None)
    ap.add_argument("--stall-min-progress", type=float, default=None)
    # The goal is the map pose a run sheet certified for this floor. The defaults keep the
    # probe's original goal, so a sheet that passes nothing drives exactly where it always did.
    ap.add_argument("--goal-x", type=float, default=3.0, help="goal x in the map frame, metres")
    ap.add_argument("--goal-y", type=float, default=0.0, help="goal y in the map frame, metres")
    ap.add_argument("--goal-yaw", type=float, default=0.0, help="goal yaw in the map frame, radians")
    args = ap.parse_args()

    limits = None
    if args.stall_window is not None or args.stall_min_progress is not None:
        limits = StallLimits(window_s=args.stall_window, min_progress_m=args.stall_min_progress)
    print(f"progress watchdog: {'armed -- ' + limits.describe() if limits else 'NOT armed'}")
    b = Nav2Binding(stall_limits=limits)
    # This probe's own bring-up, deliberately: it observes the binding and scores nothing
    # (see the module docstring). Same two calls in the same order `bring_up_robot()` makes.
    b.start()
    b.spawn(ROBOT)
    time.sleep(args.settle)

    # No `hasattr` guard: `RobotBinding` DECLARES transport_alive(), so a
    # binding without it is a broken binding and this probe should say so loudly rather
    # than print "n/a" and carry on. `transport_fault()` below keeps its guard because it
    # is Nav2Binding's own diagnostic and is deliberately NOT on the Protocol — the guard
    # marks a genuine optional extra rather than papering over a contract member.
    print(f"transport_alive after settle: {b.transport_alive()}")
    print(f"pose={b.pose(ROBOT)}  battery={b.battery_percent(ROBOT)}%  "
          f"hardware={[(h.component_name, h.status) for h in b.latest_hardware(ROBOT)]}")

    print(f"state before dispatch: {b.task_state(ROBOT)}")
    t0 = time.monotonic()
    print(f"goal: map ({args.goal_x:+.3f}, {args.goal_y:+.3f}) yaw {args.goal_yaw:+.4f} rad")
    b.command_move(ROBOT, args.goal_x, args.goal_y, args.goal_yaw)

    terminal, last = None, None
    end = t0 + args.watch
    while time.monotonic() < end:
        st = b.task_state(ROBOT)
        if st != last:
            print(f"  t+{time.monotonic() - t0:6.2f}s  state={st}  "
                  f"pose={b.pose(ROBOT)}  progress={b.task_progress(ROBOT)}%")
            last = st
        if st != task_map.TASK_RUNNING:
            terminal = st
            break
        time.sleep(0.05)

    dt = time.monotonic() - t0
    if terminal is None:
        print(f"RESULT: NO TERMINAL STATE within {args.watch:.0f}s — still "
              f"{b.task_state(ROBOT)} (this is the 'hang' the runbook asks about)")
    else:
        print(f"RESULT: terminal after {dt:.3f}s, state={terminal}")
        reason = b.task_failure(ROBOT)
        if reason:
            print(f"        reason: {reason}")
    print(f"at end: pose={b.pose(ROBOT)}  progress={b.task_progress(ROBOT)}%  "
          f"is_stopped={b.is_stopped(ROBOT)}")

    fault = b.transport_fault() if hasattr(b, "transport_fault") else None
    print(f"transport_fault at end: {fault!r}")
    sys.stdout.flush()
    os._exit(0)   # same reason as r1_smoke._exit: daemon spin thread vs interpreter teardown


if __name__ == "__main__":
    main()
