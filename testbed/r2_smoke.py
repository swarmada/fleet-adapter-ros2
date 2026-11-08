#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R2 smoke test - the adapter against a REAL Nav2 stack.

R1 proved the transport: start(), dispatch, cancel, halt, confirmed stop. Those
all pass against a fake action server, so R1 proves nothing about how the adapter
behaves when the thing on the other end has opinions.

R2 is where Nav2's opinions arrive. Every assertion below is a semantic that a
fake server cannot produce:

  1. Nav2's own action server is discoverable and accepts a goal
  2. a reachable goal completes, and the robot ends up where it was sent
  3. distance_remaining feedback actually decreases (the progressPct baseline in
     nav2_binding.py assumes monotonicity - if Nav2 replans, it may not)
  4. an UNREACHABLE goal terminates. Nav2 either rejects it or ABORTs after
     recovery; the adapter must reach a terminal state rather than hang forever
  5. a cancel issued while the controller is actively driving produces a
     confirmed stop, and the latency is measured rather than assumed

Assertion 4 is the important one. A hang here is a real defect that no simulated
binding could ever have surfaced.

THIS IS A HARNESS, AND BRING-UP IS THE ADAPTER'S
--------------------------------------------------
This file SCORES the adapter: every line is PASS/FAIL, a failure aborts the run, and a
clean run prints `ALL R2 ASSERTIONS PASSED` and nominates `adapters/REGISTRY.md`'s
standing for retirement. That is what makes it a harness rather than an instrument, and
the distinction is not vocabulary. An instrument drives the binding on purpose because it
is measuring the binding, so bringing the transport up is part of its apparatus
(`r2_cancel_latency.py`, `r2_transport_check.py`). A harness scores the ADAPTER, so a
harness that starts the binding on the adapter's behalf is scoring a configuration the
adapter cannot produce.

It used to call ``binding.start()`` itself. That is precisely how a severity-1 defect survived a
conformance run in R1: the shipped adapter never called ``start()``, so ``--binding ros2``
constructed a binding that talked to nothing, while the harness started it by hand and
reported a full sheet of passing assertions against a transport the adapter had never
opened. R2 called it from the same line. **A harness that supplies the step
under test is not testing it.**

Bring-up therefore goes through ``adapter.bring_up_robot()`` -- the same call
``run_control_stream()`` makes on every real run -- and the transport is ASSERTED live
afterwards rather than assumed. If the lifecycle is dropped from the adapter again, R2
goes red instead of green.

THIS IS A SMOKE TEST. IT DOES NOT PRODUCE THE E-STOP NUMBER
------------------------------------------------------------
Assertion 5 used to print its single latency with a NOTE instructing the reader to quote
it as "the real-world bound on the e-stop path". That instruction was wrong three times
over and the note has been deleted:

  * n=1. `r2_cancel_latency.py --runs 20` on the same stack, minutes later, gave
    min 0.590 / median 0.708 / max 1.097 s. The quoted 0.302 s was 2.3x optimistic
    against the median and 3.6x against the max.
  * The sample was taken during SPIN-UP. The cancel landed 0.22 s after dispatch with
    the robot at roughly 0.2 m/s. It stopped quickly because it was barely moving.
  * No speed was printed, so two runs at 0.301 s and 0.302 s read as convergence rather
    than as two samples of the same near-stationary condition. Repeating a biased
    measurement does not remove the bias; here it disguised it.

This file now waits for CRUISE before cancelling and prints the speed beside the
interval, and labels the sample a LOWER BOUND when the floor was not reached. The figure
to quote is the distribution from `r2_cancel_latency.py`, with its speed range and the
controller frequency -- never a minimum, and never one sample.

Run (inside the testbed; readiness is gated, not slept -- see r2_ready.py):
    python3 testbed/r2_smoke.py
"""

from __future__ import annotations

import time

try:
    from fleet_adapter_ros2 import task as task_map
    from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter
    from fleet_adapter_ros2.nav2_binding import Nav2Binding
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: cannot import Nav2Binding ({exc})")
    raise SystemExit(2) from exc

# Testbed-local helpers (noqa: E402 -- they follow the adapter import guard above).
from r2_ready import NotReady, wait_until_ready  # noqa: E402
from speed_gate import label, speed, wait_for_cruise  # noqa: E402
from testbed_provenance import changed_since, stamp  # noqa: E402
from tf_liveness import TfLiveness  # noqa: E402

ROBOT = "robot-conformance-1"
READY_TIMEOUT = 180.0   # deadline for the Nav2 readiness gate. NOT a settle
CRUISE_WATCH = 30.0     # time allowed to reach the cruise floor before the cancel
GOAL_WATCH = 60.0
TERMINAL_WATCH = 90.0
STOP_WATCH = 8.0
# nav2_params.yaml -> controller_server.controller_frequency. Named in the output because
# any figure quoted for this path has to name it: the excess over the kinematic floor is
# ~3.3 controller ticks and the spread is where in a tick the cancel lands.
CONTROLLER_HZ = 10.0

# From make_map.py: racking occupies roughly x in [-6,-2] and [2,6], y in [-4,4].
REACHABLE = (0.0, -6.0)     # open floor below the racks
# OFF-MAP, and deliberately so. The global costmap is a rolling 40 m window, so this pose is
# outside its bounds and the planner refuses it outright:
#   planner_server: GridBased plugin failed to plan from (0.00, 0.00) to (50.00, 50.00):
#                   "Goal Coordinates of(50.000000, 50.000000) was outside bounds"
# -> ACTION_STATE_FAILED in ~2 s, with the robot never moving.
#
# NEITHER in-map candidate works, which is why this is off-map rather than
# "deeper into the racking":
#   (4.0, 0.0)  FREE (0)       -> the planner plans toward it indefinitely and CORRECTLY,
#                                 because a lidar cannot see behind the rack face and
#                                 track_unknown_space is false. RUNNING at 90 s.
#   (2.0, 0.0)  INSCRIBED(253) -> the controller drives up to the face and the goal checker
#                                 accepts. SUCCEEDED at 202 s.
# "Unreachable" is not a property of a pose on this map. It is a property of being outside it.
UNREACHABLE = (50.0, 50.0)


class Failure(AssertionError):
    pass


def _check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  - ' + detail) if detail and not ok else ''}")
    if not ok:
        raise Failure(label)


def _wait(pred, timeout: float, poll: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(poll)
    return False


def _dist(binding, target) -> float:
    p = binding.pose(ROBOT)
    return ((p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2) ** 0.5


def main() -> int:
    print("R2 smoke - fleet-adapter-ros2 against a live Nav2 stack\n")
    # Which apparatus this is. First line, before anything can be quoted.
    stamp_at_start = stamp()
    print(f"testbed: {stamp_at_start}\n")
    binding = Nav2Binding()

    # Bring-up is the ADAPTER's, not this file's (see the module docstring).
    # This harness does NOT call binding.start(). `bring_up_robot()` is the same method
    # `run_control_stream()` calls, which is the same method `main()` reaches on every
    # real run, so there is one bring-up implementation and this file does not own a copy
    # of it. pb / pb_grpc are None: `bring_up_robot()` touches neither, and R2 stands up
    # no control plane -- Nav2 is the thing under test here, not gRPC.
    adapter = fleet_adapter_ros2Adapter(None, None, binding, ROBOT)
    try:
        adapter.bring_up_robot()   # start() + spawn(), in that order, as the adapter does
    except Exception as exc:
        print(f"  FAIL  adapter.bring_up_robot() raised: {exc!r}")
        raise Failure("adapter bring-up")
    # Asserted, not assumed. `bring_up_robot()` returning proves no exception escaped, NOT
    # that a node exists -- on a binding whose start() had been dropped it would return
    # just as quietly, and every assertion below would then run against a binding that
    # talks to nothing. transport_alive() is True only while the rclpy executor is running.
    _check("the adapter's bring-up left the ROS 2 transport live", binding.transport_alive(),
           "no rclpy executor thread is running after bring_up_robot() - the adapter is "
           "not driving RobotBinding.start(), which is the inert adapter returning")
    # Measure the INSTRUMENT alongside the run. TF output was observed decaying over a
    # stack's life; collision_monitor then stops the robot and goals abort with an empty
    # error_msg, none of which is the adapter's doing and none of which announces itself
    # in an assertion that merely passed.
    tf = TfLiveness.attach(binding)

    # NOT a sleep. Nav2 is asked to plan a real path and the newest scan is
    # checked against the TF cache; `active` on the lifecycle means neither of those.
    print("1. waiting for Nav2 to become READY (planner round-trip + scan into costmap)")
    try:
        ready = wait_until_ready(READY_TIMEOUT)
    except NotReady as exc:
        print(f"  FAIL  Nav2 never became ready - {exc}")
        raise Failure("nav2 readiness")
    # `ready.summary()` is not re-printed here: wait_until_ready already printed it, and
    # it appears again in the footer beside the numbers.
    print(f"   TF at start: {tf.sample_start():.1f} Hz")
    _check("pose() is populated from live /odom", binding.pose(ROBOT) is not None)

    # ---- 2. reachable goal ------------------------------------------------
    print(f"\n2. reachable goal {REACHABLE} - real planner, real costmap")
    start = binding.pose(ROBOT)
    binding.command_move(ROBOT, *REACHABLE)
    moved = _wait(lambda: _dist(binding, start[:2]) > 0.3, GOAL_WATCH)
    _check("Nav2 accepted the goal and the controller drove the robot", moved,
           "no motion - check `ros2 action list` shows /navigate_to_pose and "
           "that TF map->odom->base_link is complete")

    arrived = _wait(lambda: _dist(binding, REACHABLE) < 0.6, TERMINAL_WATCH)
    _check("the robot reached the goal pose", arrived,
           f"stopped {_dist(binding, REACHABLE):.2f} m short at {binding.pose(ROBOT)}")

    settled = _wait(lambda: binding.is_stopped(ROBOT), STOP_WATCH)
    _check("the robot came to confirmed rest at the goal", settled)

    # ---- 3. unreachable goal ---------------------------------------------
    # The one that matters. Nav2 will refuse or give up; the adapter must not hang.
    print(f"\n3. UNREACHABLE goal {UNREACHABLE} - outside the map bounds")
    before_pose = binding.pose(ROBOT)   # for the did-not-move assertion below
    t0 = time.monotonic()
    try:
        binding.command_move(ROBOT, *UNREACHABLE)
    except Exception as exc:
        _check("command_move did not raise on a rejected goal", False, repr(exc))

    # Assert on the GOAL's terminal action status, never on the pose.
    #
    # This block used to wait on `is_stopped(ROBOT) and _dist(before) < 8.0`. BOTH
    # clauses are already true the instant it is reached: assertion 2 leaves the robot
    # at confirmed rest, and it has not moved since `before` was sampled. `_wait`
    # returned on its first poll, `dt` printed as ~0.0s, and the assertion the
    # docstring calls "the important one" passed without Nav2 doing anything at all.
    # The follow-up was no better -- the robot sits ~7 m from UNREACHABLE at the
    # assertion-2 goal, so "did not drive into the racking" was true before the goal
    # was ever sent. A check satisfied by state that predates what it claims to test
    # is worth less than no check, because it gets quoted as evidence.
    #
    # `task_state()` reads the live NavigateToPose GoalStatus, and `command_move`
    # resets it to in-flight SYNCHRONOUSLY before dispatching. Any terminal state
    # observed below therefore belongs to THIS goal by construction -- stale state
    # cannot satisfy it.
    def terminal() -> bool:
        return binding.task_state(ROBOT) != task_map.TASK_RUNNING

    done = _wait(terminal, TERMINAL_WATCH)
    dt = time.monotonic() - t0
    state = binding.task_state(ROBOT)
    _check("the adapter reached a terminal state instead of hanging", done,
           f"still {state} after {TERMINAL_WATCH:.0f}s - THIS IS A REAL DEFECT: "
           "Nav2 rejected or aborted and the binding never noticed")
    print(f"        terminal after {dt:.1f}s, state={state}")
    _check("the terminal state is a failure, not a success",
           state in (task_map.TASK_FAILED, task_map.TASK_CANCELLED),
           f"Nav2 reported {state} for a goal outside the map bounds - the planner should "
           "have refused it outright ('was outside bounds'), so a success here means "
           "something accepted a target it cannot have planned")
    # The robot must not have moved AT ALL. An off-map goal is refused before planning, so
    # any motion means a goal was dispatched that should have been declined.
    #
    # Distance from UNREACHABLE itself is ~70 m here and can never fail, which is exactly
    # why it is not what is asserted -- the previous version of this line was a check
    # incapable of failing even before the target moved off-map.
    moved = _dist(binding, before_pose[:2])
    _check("the robot did not move toward an off-map goal", moved < 0.3,
           f"moved {moved:.2f} m toward a goal the planner refused")

    # ---- 4. cancel under an active controller -----------------------------
    print("\n4. cancel while Nav2's controller is actively driving")
    binding.command_move(ROBOT, 0.0, 6.0)
    driving = _wait(lambda: not binding.is_stopped(ROBOT), GOAL_WATCH)
    _check("the robot is under way before the cancel", driving)

    # Cancel from CRUISE, not from the acceleration ramp. "Not stopped" is
    # true during in-place rotation and during spin-up; cancelling then measures how
    # long a nearly stationary robot takes to stop, which is not the e-stop path.
    #
    # Failing to reach the floor is NOT an assertion failure here -- this is a smoke
    # test of the cancel path, not of the robot's top speed. The sample is taken anyway
    # and LABELLED a lower bound, which is the one thing the previous version did not do.
    at_cruise, v_peak = wait_for_cruise(binding, CRUISE_WATCH)

    # The speed AT the cancel, read from the latest odom immediately before command_stop()
    # -- not wait_for_cruise()'s second value, which on a timeout is the PEAK over the
    # watch. Both are printed, each under its own
    # name, exactly as r1_smoke prints the speed at the halt.
    v_cancel = speed(binding)
    t_cancel = time.monotonic()
    binding.command_stop(ROBOT)
    _check("is_stopped() is NOT immediately True (confirmed, never inferred)",
           not binding.is_stopped(ROBOT))
    rested = _wait(lambda: binding.is_stopped(ROBOT), STOP_WATCH)
    latency = time.monotonic() - t_cancel
    _check("confirmed stop after a live cancel", rested,
           f"still moving {STOP_WATCH:.0f}s after cancel - the e-stop path's worst case")
    print(f"        cancel -> confirmed rest: {latency:.3f}s")
    print(f"        {label(v_cancel, at_cruise, 'cancel', peak=v_peak)}")
    print( "        This is ONE sample from a smoke test. It is NOT the e-stop bound and")
    print( "        must not be quoted as one: the spread on this path is a property of")
    print(f"        the {CONTROLLER_HZ:.0f} Hz controller loop, not noise, and two samples at the same")
    print( "        speed have differed by 1.9x. For a figure, run:")
    print( "            make -f Makefile.r2 r2-latency        # 6 runs, min/median/max")
    print( "        and quote the distribution with its speed range and the controller")
    print( "        frequency.")

    tf.sample_end()
    # Printed ALWAYS, beside the numbers, not only on failure.
    print(f"\ninstrument: {tf.summary()}")
    print(f"readiness:  {ready.summary()}")
    print(f"testbed:    {stamp()}")
    drift = changed_since(stamp_at_start)
    if drift:
        print(drift)
    # Before the success banner: a run taken through a degraded instrument has not
    # demonstrated anything, however many assertions passed.
    tf.assert_healthy()

    print("\nALL R2 ASSERTIONS PASSED")
    print("The ROS 2 adapter has been exercised against a live Nav2 runtime.")
    print("adapters/REGISTRY.md's 'no live ROS 2/Nav2 runtime yet' can now be retired.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure:
        print("\nR2 FAILED - see the first FAIL above.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
