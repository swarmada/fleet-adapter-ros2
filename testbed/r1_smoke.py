#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R1 smoke test - executes the four blocks that have never run.

This drives ``Nav2Binding`` with no Swarmada control plane and no gRPC. That is
deliberate: the four uncovered blocks are all in the ROS 2 transport, so standing up a
control plane to reach them would only add failure modes that have nothing to do with
what R1 is proving.

BRING-UP IS THE ADAPTER'S, NOT THIS FILE'S
-------------------------------------------
The one thing this file does NOT drive directly is the binding's lifecycle. It used to
call ``binding.start()`` itself, and that is how a severity-1 defect survived a conformance run: the
shipped adapter never called ``start()``, so ``--binding ros2`` constructed a binding
that talked to nothing, while this harness started the binding by hand and reported
eight passing assertions against a transport the adapter had never opened. The harness
was compensating for the defect it existed to catch.

Bring-up therefore goes through ``adapter.bring_up_robot()`` -- the same call
``run_control_stream()`` makes on every real run. This file asserts the transport is
live afterwards rather than assuming it; if the lifecycle is dropped from the adapter
again, R1 goes red instead of green.

Covers, in order:

  A. adapter.bring_up_robot() node, 3 subscriptions, ActionClient, publisher, executor
  B. nav2_binding.py:122  goal dispatch    a real NavigateToPose goal over DDS
  C. nav2_binding.py:170  zero-Twist halt  the /cmd_vel publish on command_stop
  D. nav2_binding.py:158  goal cancel      the live cancel on the action client

and asserts the property that matters most:

  E. C5.3 confirmed stop - is_stopped() only becomes True AFTER the odometry the
     fake robot publishes shows real zero velocity. Never inferred from the
     command, a timer, or silence.

THE INTERVAL IS REPORTED WITH THE SPEED IT WAS MEASURED AT
-----------------------------------------------------------
R1's headline number is the confirmed-stop deceleration interval, and this file used to
print it as a bare duration. The duration depends on how fast the robot was travelling
when the halt was commanded, and this file neither controlled that speed nor recorded it:
the halt is triggered by 0.15 m of DISPLACEMENT, so the speed at halt was wherever the
acceleration ramp happened to be. Two runs of `--decel 0.3` gave 1.009 s and 0.966 s -- a
4.3% spread on a figure quoted to three decimal places, and NOTHING IN THE OUTPUT COULD
ATTRIBUTE IT. Two runs that differ because the robot was moving at different speeds look
identical to two runs that differ because the adapter changed.

That is the e-stop number's own failure one severity down, and that number shows how it
ends: a latency published from n=1 at an undisclosed speed, ~2.3x optimistic, believed
precisely because it had been measured.

So this file now waits for the same CRUISE_FLOOR that `r2_cancel_latency.py` uses before
commanding the halt, and prints the speed beside the interval. R1 and R2 measure
comparably as a result -- and, deliberately, R1 intervals recorded BEFORE this change are
NOT directly comparable to ones recorded after: they were taken from wherever the ramp
was at 0.15 m, which under `--decel 0.3` is about 0.30 m/s, below the 0.35 m/s floor.

A stopping-distance number without the speed it was measured at is not a measurement.

THE INTERVAL COMES FROM ODOMETRY STAMPS, AND SAYS HOW FINE IT IS
------------------------------------------------------------------
The interval used to be `time.monotonic()` around a 50 ms polling loop. `not immediate`
guarantees that loop sleeps at least once, so 0.051 s was the smallest non-zero value it
could emit -- and on the first hardware run it emitted exactly that. One tick is +/-100%,
where the simulated ~0.51 s series was ten ticks and +/-5%: the harness crossed from
adequate to inadequate on moving to hardware and nothing in its output said so.

The interval is now taken between robot-clock `header.stamp`s of the odometry itself:

  S_pre   the last odom message received BEFORE command_stop()
  S_post  the first odom message received AFTER command_stop() returned
  S_rest  the first odom message on which is_stopped() answered True

The halt has NO stamp. It is on this process's clock and the stamps are on the robot's,
and those are never subtracted from each other. The halt is BRACKETED instead: it came
after S_pre was received, and before S_post was sent (up to transport latency). So

    S_rest - S_post  <=  halt -> confirmed rest  <  S_rest - S_pre

and the bracket width is printed on the same line as the interval, as the speed is. Every
odom message carries a receipt count (Nav2Binding.odom_stamp), and each observation reads
it on both sides of is_stopped(), so the harness can PROVE it saw consecutive messages at
rest rather than assume it; when it did not, the line says how many it missed. If stamps
are missing or do not increase in receipt order, no interval is derived at all and the
line prints a BOUND on this process's clock instead. That is the honest outcome, not a
failure.

Exit code 0 = every assertion held. Non-zero = the first failure, named.

Run (inside the testbed container, with fake_robot_node.py already running):

    python3 testbed/r1_smoke.py
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from dataclasses import dataclass

# Import from the adapter under test. PYTHONPATH must include the adapter checkout.
try:
    from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter
    from fleet_adapter_ros2.nav2_binding import Nav2Binding
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: cannot import the adapter - is it on PYTHONPATH? ({exc})")
    raise SystemExit(2) from exc

# Testbed-local helpers (noqa: E402 -- they follow the adapter import guard above).
import fake_robot_args  # noqa: E402
from speed_gate import label, speed, wait_for_cruise  # noqa: E402
from testbed_provenance import changed_since, stamp  # noqa: E402

ROBOT = "robot-conformance-1"
SETTLE = 3.0        # s to wait for DDS discovery + first odom
MOVE_WATCH = 8.0    # s to watch for motion after a goal
# s to reach the cruise floor before the halt. fake_robot_node.py cruises at --cruise
# (default 0.5 m/s) and accelerates at --accel (which defaults to --decel), so the slowest
# flag the runbook's Step 5 uses, --decel 0.3, reaches 0.35 m/s at about t+1.2 s. 12 s leaves room
# for a much slower ramp without turning a real stall into a timeout.
CRUISE_WATCH = 12.0
STOP_WATCH = 5.0    # s to watch for confirmed rest after a halt
POLL = 0.05
# s between observations while watching for rest. NOT the interval's resolution -- that is
# the odometry stamps' -- but it must sit well under the odom period (33 ms on the MentorPi,
# 50 ms from fake_robot_node.py) so consecutive messages are seen, which the receipt count
# then proves rather than assumes.
STAMP_POLL = 0.005


# ── the fault gate ───────────────────────────────────────────────────────────
# This harness asserts the happy path. That is fine until someone selects a fault
# case, because then "the assertions that ran, passed" and "the fault was injected
# and handled" print the same line. Two of the four documented fault cases cannot
# fire within the window this file drives (the goal is cancelled ~0.46 s in, after
# ~0.16 m of a 3.0 m goal), so `ALL R1 ASSERTIONS PASSED` has been reported for a
# fault that never arrived -- and two rows of the R1/R2 results table were written
# from it.
#
# The rule: a harness that cannot score the case it was handed must report that it
# cannot, never that it passed.

#: Fault flags `fake_robot_node.py` accepts, and whether THIS file can score them.
#: `reason` is printed verbatim when it cannot -- the operator is told which fault
#: was configured, why it is unscoreable here, and what to run instead.
FAULT_CASES: dict[str, dict[str, object]] = {
    "--reject-goals": {
        "scoreable": False,
        "reason": (
            "the goal is rejected at acceptance, so assertion B ('the robot moved') "
            "fails as a HARNESS failure rather than an adapter report. The adapter's "
            "actual behaviour (it reaches ACTION_STATE_FAILED in ~0.098 s) is only "
            "observable with the probe."
        ),
    },
    "--goal-outcome": {
        "scoreable": False,
        "reason": (
            "ABORTED fires in fake_robot_node.py's completion branch, when the 3.0 m "
            "goal pose is REACHED (~6 s at 0.5 m/s). This file cancels unconditionally "
            "at ~0.46 s / ~0.16 m and does not parameterise the goal distance, so the "
            "completion branch is structurally unreachable from here."
        ),
    },
    "--stall-after": {
        "scoreable": False,
        "reason": (
            "the stall is scheduled on goal time (default 3.0 s); this file has already "
            "cancelled by ~0.46 s. Even inside the window the assertion cannot pass: a "
            "stalled robot has not moved, and what the adapter DOES about it -- its "
            "progress watchdog cancels the goal and reports ACTION_STATE_FAILED with the "
            "reason -- is a terminal state this file does not watch for."
        ),
    },
    # Continuous parameters, not fault injections: they change the condition every
    # assertion already runs under, so the happy path genuinely scores them. Each must be
    # a flag fake_robot_node.py IMPLEMENTS -- `--cruise` was listed here for a node that
    # had none and silently dropped it; tests pin this table to its parser.
    "--decel": {"scoreable": True, "reason": ""},
    "--accel": {"scoreable": True, "reason": ""},
    "--cruise": {"scoreable": True, "reason": ""},
}

PROBE = "testbed/r1_step5_probe.py"


def configured_faults(robot_args: str) -> list[str]:
    """Fault flags present in ROBOT_ARGS, in the order they appear.

    Pure and ROS-free so it can be tested without a runtime -- which is the point:
    the gate must be provable without the stack the gate exists to protect.
    """
    # `--stall-after=3.0` is the same flag as `--stall-after 3.0`; matching whole tokens
    # only would let the `=` spelling past the gate while the robot applied it.
    return [f for f in (tok.split("=", 1)[0] for tok in robot_args.split())
            if f in FAULT_CASES]


def unscoreable(robot_args: str) -> list[str]:
    """The configured faults this harness cannot score. Empty = safe to run."""
    return [f for f in configured_faults(robot_args)
            if not FAULT_CASES[f]["scoreable"]]


def fault_gate(robot_args: str) -> int:
    """Refuse the run if it could only report a pass it has not earned.

    Returns 0 to proceed, or a non-zero exit code. Deliberately runs BEFORE the adapter's
    bring-up (and so before any ROS 2 node exists): a harness that cannot score its case
    should not consume robot time to find that out, and on a real robot that time is the
    scarce thing.
    """
    faults = configured_faults(robot_args)
    print(f"fault gate: ROBOT_ARGS={robot_args!r}")
    if not robot_args.strip():
        print("  NOTE  ROBOT_ARGS is empty in this container. If the robot was started")
        print("        with a fault flag, this harness CANNOT SEE IT and cannot gate on")
        print("        it -- pass ROBOT_ARGS through to the r1-smoke service too.")
        return 0
    refused = robot_refuses(robot_args)
    if refused:
        return refused
    if not faults:
        print("  no fault case configured - the happy path is the whole claim")
        return 0
    print(f"  configured: {' '.join(faults)}")
    bad = unscoreable(robot_args)
    if not bad:
        print("  all configured cases are scoreable by this harness")
        return 0
    print()
    for f in bad:
        print(f"  FAIL  fault case {f} is configured and THIS HARNESS CANNOT SCORE IT")
        print(f"        {FAULT_CASES[f]['reason']}")
    print()
    print("  REFUSING TO RUN. Every assertion below would have passed without the")
    print("  fault ever arriving, and the run would have printed ALL R1 ASSERTIONS")
    print("  PASSED - which is the false pass this gate exists for, and which has")
    print("  already put two false rows into the R1/R2 results table.")
    print(f"  Run this case through {PROBE} instead (observation-only).")
    print("\nR1 NOT RUN - unscoreable fault case (fault gate)")
    return 3


def robot_refuses(robot_args: str) -> int:
    """Ask fake_robot_node.py's own parser whether it implements ROBOT_ARGS.

    The node refuses what it does not implement, so a bad flag stops the run
    either way; asking here says so in THIS output, before bring-up, rather than as a
    robot container that exited 2 while the harness waited for odometry.
    """
    try:
        fake_robot_args.parse(shlex.split(robot_args))
    except (fake_robot_args.ArgsError, ValueError) as exc:  # ValueError: shlex quoting
        print(f"  FAIL  fake_robot_node.py does not implement ROBOT_ARGS: {exc}")
        print("        A run would not have the condition ROBOT_ARGS names.")
        print("\nR1 NOT RUN - ROBOT_ARGS names a condition the robot does not "
              "implement")
        return 3
    return 0


class Failure(AssertionError):
    pass


def _check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
        return
    print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
    raise Failure(label)


def _wait_until(predicate, timeout: float, poll: float = POLL) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


# ── the interval, from odometry stamps ───────────────────────────────────────
# An observation is (receipt count, stamp_ns, is_stopped()) for ONE odom message. The
# functions below are pure so the derivation is testable without ROS 2.

def _observe(binding) -> tuple[int, int | None, bool]:
    """Read is_stopped() so its answer is attributable to one odom message: the receipt
    count is read on both sides, and the read is retried if a message landed between."""
    for _ in range(1000):
        count, stamp_ns = binding.odom_stamp(ROBOT)
        stopped = binding.is_stopped(ROBOT)
        if binding.odom_stamp(ROBOT)[0] == count:
            return count, stamp_ns, stopped
    raise RuntimeError("odom kept arriving between three consecutive reads")


def _observe_halt(binding) -> tuple[tuple[int, int | None, bool], float]:
    """The last observation before command_stop() AND the linear speed carried by that
    SAME odom message -- the speed at the halt, as the label beside the interval says.

    It used to be wait_for_cruise()'s second value, which on a timeout is the PEAK over
    the 12 s watch. On the MentorPi, which never reaches the floor, that is
    every run. Read here, the speed and the bracket's lower edge S_pre are one message."""
    for _ in range(1000):
        o = _observe(binding)
        v = speed(binding)
        if binding.odom_stamp(ROBOT)[0] == o[0]:
            return o, v
    raise RuntimeError("odom kept arriving between the halt observation and its speed")


def _watch_for_rest(binding, obs: list, timeout: float,
                    poll: float = STAMP_POLL) -> tuple[bool, float]:
    """Append each changed observation to `obs` until is_stopped() is True.
    Returns (rested, time.monotonic() at the observation that ended the watch)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        o = _observe(binding)
        if o != obs[-1]:
            obs.append(o)
        if o[2]:
            return True, time.monotonic()
        time.sleep(poll)
    return False, time.monotonic()


def undecided_by_odom(obs: list) -> list[int]:
    """Receipt counts on which is_stopped() gave BOTH answers -- i.e. it changed its mind
    with no new odom message, which only a binding inferring rest from time or from the
    command can do (C5.3). Empty on a correct binding."""
    first: dict[int, bool] = {}
    bad = []
    for count, _, stopped in obs:
        if first.setdefault(count, stopped) != stopped and count not in bad:
            bad.append(count)
    return bad


@dataclass(frozen=True)
class StopInterval:
    """halt -> confirmed rest, in s, on the robot's odometry clock. `lo`..`hi` IS the
    result: the halt is bracketed, not stamped. Non-empty `unresolved` means no interval
    could be derived and says why."""
    lo: float = 0.0
    hi: float = 0.0
    bracket: float = 0.0   # S_post - S_pre: the halt's bracket, and so the interval's width
    spacing: float = 0.0   # last-motion stamp -> first-rest stamp
    missed: int = 0        # odom messages between those two that were never observed
    unresolved: str = ""


def stop_interval(obs: list, count_after_halt: int) -> StopInterval:
    """Derive the interval from observations. obs[0] is the last one before
    command_stop(); `count_after_halt` is the receipt count read as it returned."""
    pre = obs[0]
    rest = next((o for o in obs if o[2]), None)
    if rest is None:
        return StopInterval(unresolved="no odom message showing rest was observed")
    seen = obs[:obs.index(rest) + 1]
    stamps = {c: s for c, s, _ in seen}   # one stamp per message
    if any(s is None or s <= 0 for s in stamps.values()):
        return StopInterval(unresolved="the odometry carries no header.stamp")
    ordered = [stamps[c] for c in sorted(stamps)]
    if any(b <= a for a, b in zip(ordered, ordered[1:])):
        return StopInterval(unresolved="odom stamps do not increase in receipt order")
    post = next((o for o in seen if o[0] > count_after_halt), None)
    motion = [o for o in seen if not o[2] and o[0] < rest[0]]
    if post is None or not motion:
        return StopInterval(unresolved="no odom message was seen between the halt and rest")
    ns = 1e-9
    return StopInterval(lo=(rest[1] - post[1]) * ns, hi=(rest[1] - pre[1]) * ns,
                        bracket=(post[1] - pre[1]) * ns,
                        spacing=(rest[1] - motion[-1][1]) * ns,
                        missed=rest[0] - motion[-1][0] - 1)


def resolution(iv: StopInterval) -> str:
    """The words that go beside the interval, as speed_gate.label() does for speed."""
    seen = ("consecutive messages" if iv.missed == 0
            else f"{iv.missed} message(s) between them NOT observed")
    return (f"[odom stamps; the halt is unstamped, bracketed within {iv.bracket * 1e3:.0f} ms; "
            f"last motion -> first rest {iv.spacing * 1e3:.0f} ms apart, {seen}]")


def main() -> int:
    print("R1 smoke - fleet-adapter-ros2 against a live ROS 2 runtime\n")
    # Which apparatus this is. First line, before anything can be quoted.
    stamp_at_start = stamp()
    print(f"testbed: {stamp_at_start}\n")

    # Before the robot is touched: can this harness score the case it was handed?
    # (A harness that cannot must say so, not print a pass.)
    gate = fault_gate(os.environ.get("ROBOT_ARGS", ""))
    if gate:
        return gate
    print()

    binding = Nav2Binding()

    # ---- A. bring-up THROUGH THE ADAPTER ---------------------------------
    print("A. adapter bring-up - node, subscriptions, action client, cmd_vel publisher")
    # This harness does NOT call binding.start(). It used to, and that is the whole
    # reason the inert adapter survived a conformance run: `start()` creates the rclpy
    # node, the three subscriptions and the action client, the shipped adapter never
    # called it, and this file called it on the adapter's behalf -- so R1 reported eight
    # green assertions against a transport `--binding ros2` had never opened. A harness
    # that supplies the step under test is not testing it.
    #
    # So bring-up goes through the adapter's own path, `bring_up_robot()` -- the same
    # method `run_control_stream()` calls, which is the same method `main()` reaches on
    # every real run. There is one bring-up implementation and this file does not own a
    # copy of it. If the lifecycle is ever dropped from it again, this run goes red for
    # the same reason the shipped adapter would be inert, instead of staying green and
    # concealing it.
    # pb / pb_grpc are None: bring_up_robot() touches neither, and R1 deliberately
    # stands up no control plane (see the module docstring).
    adapter = fleet_adapter_ros2Adapter(None, None, binding, ROBOT)
    try:
        adapter.bring_up_robot()   # start() + spawn(), in that order, as the adapter does
    except Exception as exc:
        print(f"  FAIL  adapter.bring_up_robot() raised: {exc!r}")
        return 1
    _check("adapter.bring_up_robot() returned without raising", True)
    # Bring-up is only claimed if the transport it was supposed to create is actually
    # there. `bring_up_robot()` returning proves no exception escaped, NOT that a node
    # exists -- on a binding whose start() had been dropped it would return just as
    # quietly. transport_alive() is True only while the rclpy executor thread is running.
    _check("the adapter's bring-up left the ROS 2 transport live",
           binding.transport_alive(),
           "no rclpy executor thread is running after bring_up_robot() - the adapter "
           "constructed the binding without starting it")
    print(f"  waiting {SETTLE:.0f}s for DDS discovery and first /odom ...")
    time.sleep(SETTLE)

    p0 = binding.pose(ROBOT)
    batt = binding.battery_percent(ROBOT)
    _check("pose() is populated from live /odom", p0 is not None, f"pose={p0}")
    _check("battery_percent() is populated from live /battery_state",
           isinstance(batt, int) and 0 < batt <= 100, f"battery={batt}")
    print(f"        pose={p0}  battery={batt}%")

    # ---- B. live goal dispatch -------------------------------------------
    print("\nB. goal dispatch - a real NavigateToPose goal over DDS")
    binding.command_move(ROBOT, 3.0, 0.0)
    moved = _wait_until(
        lambda: abs(binding.pose(ROBOT)[0] - p0[0]) > 0.15
        or abs(binding.pose(ROBOT)[1] - p0[1]) > 0.15,
        MOVE_WATCH,
    )
    _check("the robot moved after the goal was dispatched", moved,
           f"pose still {binding.pose(ROBOT)} after {MOVE_WATCH:.0f}s")
    print(f"        pose={binding.pose(ROBOT)}")

    _check("is_stopped() is False while moving", not binding.is_stopped(ROBOT))

    # ---- C + D. halt and cancel ------------------------------------------
    print("\nC+D. command_stop() - zero-Twist on /cmd_vel AND live goal cancel")

    # Halt from CRUISE, and record the speed either way. The assertion
    # below is about the adapter's confirmed-stop path; failing to reach cruise is not
    # an adapter fault, so it does not fail the smoke -- but the interval is then
    # LABELLED a lower bound rather than reported as if it were comparable.
    at_cruise, v_peak = wait_for_cruise(binding, CRUISE_WATCH)

    # The last odom message before the halt: the lower edge of its bracket, and the
    # speed that message carries -- the speed AT the halt, not the watch's peak. Both
    # are printed, each under its own name.
    pre, v_halt = _observe_halt(binding)
    obs = [pre]
    t_halt = time.monotonic()
    binding.command_stop(ROBOT)
    count_after_halt = binding.odom_stamp(ROBOT)[0]

    # ---- E. the property that matters ------------------------------------
    # is_stopped MUST NOT flip the instant we command the stop. The fake robot
    # decelerates over a real interval, so a binding that inferred rest from the
    # command would be caught here.
    immediate = binding.is_stopped(ROBOT)
    _check("is_stopped() is NOT immediately True after command_stop (not inferred)",
           not immediate,
           "the binding reported rest before any odometry could confirm it - "
           "this is exactly the C5.3 'confirmed, never inferred' violation")

    rested, t_seen = _watch_for_rest(binding, obs, STOP_WATCH)
    _check("is_stopped() became True from live odometry", rested,
           f"still moving {STOP_WATCH:.0f}s after the halt")
    # Re-pointed from `dt > 0.02`, which `not immediate` above entailed and so could never
    # fail. This one fails on a binding that confirms rest from time passing
    # after the command -- which `not immediate` cannot catch, because such a binding is
    # also False at first. Caught whenever the flip is not masked by a message landing in
    # the same STAMP_POLL step.
    flipped = undecided_by_odom(obs)
    _check("rest was confirmed by a new odom message, not by time passing (C5.3)",
           not flipped,
           f"is_stopped() changed its answer on odom message(s) {flipped} with no new "
           f"message arriving")
    # The speed AND the resolution go on the SAME line as the interval. A reader must not
    # have to find either elsewhere, and there must be no version of this output that
    # omits them.
    iv = stop_interval(obs, count_after_halt)
    if iv.unresolved:
        # Both readings are time.monotonic(): one clock, so this subtraction is sound.
        print(f"        confirmed rest within {t_seen - t_halt:.3f}s of the halt on this "
              f"process's clock -- A BOUND, NOT AN INTERVAL: {iv.unresolved}, "
              f"{label(v_halt, at_cruise, peak=v_peak)}")
    else:
        print(f"        confirmed rest {iv.lo:.3f}-{iv.hi:.3f}s after the halt "
              f"{resolution(iv)}, {label(v_halt, at_cruise, peak=v_peak)}")
        print(f"        cross-check on this process's clock: rest first observed "
              f"{t_seen - t_halt:.3f}s after command_stop() -- adds transport latency and "
              f"a {STAMP_POLL * 1e3:.0f} ms observation step; not the interval")

    p_rest = binding.pose(ROBOT)
    time.sleep(1.0)
    p_still = binding.pose(ROBOT)
    _check("the robot stays at rest (the halt held)",
           abs(p_still[0] - p_rest[0]) < 0.02 and abs(p_still[1] - p_rest[1]) < 0.02,
           f"{p_rest} -> {p_still}")

    print("\nALL R1 ASSERTIONS PASSED")
    print("Executed: adapter bring-up - goal dispatch - zero-Twist halt - goal cancel")
    print("Confirmed: C5.3 stop read from live odometry, not inferred.")
    print(f"testbed: {stamp()}")
    drift = changed_since(stamp_at_start)
    if drift:
        print(drift)
    return 0


def _exit(code: int) -> None:
    """Leave the process without running interpreter teardown.

    Nav2Binding spins rclpy on a DAEMON thread. It now exposes `stop()` to join it
    (part of the RobotBinding lifecycle, and registered via atexit by `start()`), so
    the crash below is no longer the only outcome available -- but this harness still
    exits hard, because an exit code that reflects the assertions and NOTHING ELSE is
    worth more here than a tidy teardown, and `stop()`'s 5 s thread join is a new way
    for a green run to go red.

    What the crash was, for the record:
    on a normal exit CPython tears down the rclpy C++ state while that thread is
    still inside its wait set, and the C++ side calls std::terminate ->
    "terminate called without an active exception", exit 133 -- AFTER every
    assertion has already passed. That turns a green run red.

    A test harness owns its process, so exiting hard is the honest fix: the exit
    code then reflects the assertions and nothing else.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        _exit(main())
    except Failure:
        print("\nR1 FAILED - see the first FAIL above.")
        _exit(1)
    except KeyboardInterrupt:
        _exit(130)
    except Exception as exc:                      # noqa: BLE001 - report, do not mask
        print(f"\nR1 ERRORED: {exc!r}")
        _exit(2)
