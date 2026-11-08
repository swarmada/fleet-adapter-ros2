#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Measure how long THIS navigation server takes to answer a goal, so the bound is not guessed.

`--binding ros2` refuses to start without `--goal-response-bound`: how long a goal sent to the
navigation server may go unanswered before the adapter abandons it -- stops the robot, cancels
the goal by its id and reports it FAILED. Too short and good goals are killed; too long and
every stop commanded inside the window waits on a goal that may still be accepted and driven.
It is a property of the server and its transport, so it is MEASURED on them, on the machine and
the network the run will use.

WHAT IT DOES. N times: read the robot's current pose, send a NavigateToPose goal AT that pose,
and time the interval from `send_goal_async` to the goal response callback. Then cancel, whether
the server accepted or refused. It prints median / p99 / max and the bound its rule proposes.

TWO POPULATIONS, AND THE BOUND IS DERIVED ON THE WORSE. The loop above measures an IDLE
server, which is not the case the bound is for. The conformance harness PRE-EMPTS: it sends a
goal while another is still active, and a server that answers an idle goal in milliseconds can
take far longer to answer one that first has to terminate the goal it is running. On the floor
on 2026-09-21 an idle server answered in 4.3 ms at worst and a bound of 0.25 s was derived from
it; the pre-empting harness then had five goals go unanswered past that bound in one run, every
one of them a goal that needed the robot to turn around first. A bound measured on the easy
case is not a bound.

`--preempt` therefore measures BOTH and derives the bound on the worse of them:

  goal 1  a pose ~0.5 m ahead in the robot's OWN frame, so the server has something to be busy
          WITH -- a goal at the robot's own pose is inside the goal tolerance and finishes at
          once, which occupies nothing.
  goal 2  sent while goal 1 is still active; ITS answer is what is timed.

THIS ONE DRIVES THE ROBOT. Goal 1 is a real goal to a point the robot is not at, and the robot
starts towards it. The drive is bounded in code, not by intention: goal 1 is cancelled within
`--drive-budget` seconds (default 1.0) of being accepted, whatever happens to goal 2, so at an
0.20 m/s cap a sample costs at most 0.2 m and in practice a few centimetres, because the base
is still ramping up when the cancel goes out. Keep `--preempt-n` small (default 10), run it on
clear floor with the operator at the switch, and say on the run sheet that the robot moves.

WHY THE GOAL IS THE ROBOT'S OWN POSE. This probe runs on a real robot with an operator beside
it. A goal already inside the controller's `xy_goal_tolerance` is reached the moment it is
accepted, so there is nothing for the controller to drive towards; and the cancel goes out on
the acceptance callback regardless. It is still a COMMANDED goal -- this is not an observe-only
tool, and it belongs after the footprint has been verified live, with the operator at the
switch. If the pose cannot be read, it refuses rather than sending a goal to the origin.

THE RULE, stated rather than tuned afterwards:

    bound = max(1.5 * max_observed, floor)     and REFUSE if that exceeds the ceiling

  * 1.5x the MAXIMUM, not a percentile: the bound exists for the tail, and the sample is small.
  * a floor (default 0.25 s), because a handful of fast answers on a quiet machine is not
    evidence that the tail is fast; the bound must not collapse to milliseconds.
  * a ceiling (default 1.0 s), which is half the adapter's 2.0 s confirmed-stop budget: a stop
    commanded while a goal is unanswered must still have a second of budget left after the goal
    is abandoned. A server that cannot answer inside that is a server this adapter cannot make
    a bounded promise about, so the probe REFUSES and says so rather than proposing a bound
    that spends the whole stop budget.

AN INSTRUMENT, not a harness. It measures the navigation server and scores nothing about the
adapter -- it never constructs a Nav2Binding, so there is no bring-up here to route through
`adapter.bring_up_robot()`. The only `start()` in this file is an rclpy spin thread.

    ros2 run ... (any sourced ROS 2 env):
      python3 goal_response_probe.py --n 20 [--action /navigate_to_pose] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

DEFAULT_FLOOR_S = 0.25
DEFAULT_CEILING_S = 1.0
STOP_BUDGET_S = 2.0   # the adapter's confirmed-stop budget; the ceiling is half of it
# How long the pre-empt mode's FIRST goal -- the one that occupies the server -- may drive
# before it is cancelled, counted from the moment the server accepts it. A SAFETY limit, not a
# tuning knob: at the 0.20 m/s cap this floor runs at it bounds one sample at 0.2 m.
DRIVE_BUDGET_S = 1.0
# How far ahead, in the robot's own frame, the pre-empt mode places that goal. Far enough to be
# outside any goal tolerance this stack uses (0.20 m), near enough that the robot is still
# ramping up when the cancel arrives.
AHEAD_M = 0.5
# How long to wait for the spin thread to come out of `spin_once()` at teardown. The thread
# ticks at 0.05 s, so this is ~100 ticks: long enough that a normal exit never reaches it, short
# enough that a wedged spinner does not hang a probe that has already printed its numbers.
SPIN_JOIN_S = 5.0


def summarise(samples_s) -> dict:
    """median / p99 / max / n over the answer times, in seconds. Refuses an empty sample."""
    xs = sorted(float(s) for s in samples_s)
    if not xs:
        raise ValueError("no goal was answered, so there is nothing to summarise: a bound "
                         "cannot be derived from a server that answered nothing")
    for x in xs:
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"an answer time must be a finite number >= 0, got {x!r}")
    # Nearest-rank p99: with a small sample this is the maximum, and saying so is better than
    # an interpolation that invents a number between two measurements.
    rank = max(1, math.ceil(0.99 * len(xs)))
    return {"n": len(xs), "median_s": statistics.median(xs), "p99_s": xs[rank - 1],
            "max_s": xs[-1], "min_s": xs[0]}


def worse(*stats: dict) -> dict:
    """The distribution with the slowest answer in it -- the one a bound must be derived on.

    Not a merge of the samples: two populations pooled would let a hundred fast idle answers
    bury ten slow pre-empted ones in the median and in anything derived from a percentile. The
    rule reads the maximum, and the maximum belongs to whichever population owns it.
    """
    candidates = [st for st in stats if st]
    if not candidates:
        raise ValueError("no distribution to choose between")
    return max(candidates, key=lambda st: st["max_s"])


def propose_bound(stats: dict, *, factor: float = 1.5, floor_s: float = DEFAULT_FLOOR_S,
                  ceiling_s: float = DEFAULT_CEILING_S, lost: int = 0,
                  population: str = "idle") -> dict:
    """The bound this rule proposes, or a refusal with the reason. Never a silent fallback.

    `population` names the distribution `stats` came from and is recorded with the proposal, so
    a reader of the JSON -- and `adapter_cmd.py`, which refuses a bound derived on the easy
    case when the hard one was measured -- can tell which case the number is a bound on.
    """
    head = {"from": population, "max_s": stats["max_s"]}
    if lost:
        return {**head, "ok": False, "bound_s": None, "why": (
            f"{lost} of {stats['n'] + lost} goals were never answered at all. A server that "
            f"loses answers has no measurable response time, and a bound derived from the ones "
            f"that arrived would be a bound on the wrong population.")}
    want = max(factor * stats["max_s"], floor_s)
    if want > ceiling_s:
        return {**head, "ok": False, "bound_s": None, "why": (
            f"{factor:g} x the slowest answer ({stats['max_s']:.3f} s) is {want:.3f} s, above "
            f"the {ceiling_s:g} s ceiling -- half the adapter's {STOP_BUDGET_S:g} s "
            f"confirmed-stop budget. A bound that large leaves a stop commanded inside the "
            f"window less than a second to resolve after the goal is abandoned. This server is "
            f"too slow, or too loaded, for this adapter to promise a bounded stop on. Do not "
            f"raise the ceiling to make the number fit.")}
    # Two decimals: the flag is written by hand onto a run sheet and read back off a terminal.
    # Rounded UP so the bound is never below its own rule -- but the product is rounded to
    # 9 places first, or 1.5 x 0.400 = 0.6000000000000001 becomes 0.61.
    bound = math.ceil(round(want, 9) * 100.0) / 100.0
    return {**head, "ok": True, "bound_s": bound, "why": (
        f"max({factor:g} x max {stats['max_s']:.3f} s, floor {floor_s:g} s) = {want:.3f} s, "
        f"rounded up to {bound:g} s; under the {ceiling_s:g} s ceiling")}


def _distribution(label: str, stats: dict, *, refused: int = 0, lost: int = 0,
                  abandoned: int = 0) -> list:
    return [
        f"{label}: n={stats['n']} answered"
        + (f", {refused} refused by the server" if refused else "")
        + (f", {lost} NEVER ANSWERED" if lost else "")
        + (f", {abandoned} sample(s) dropped: the occupying goal was not taken" if abandoned
           else ""),
        f"  median {stats['median_s'] * 1000:.1f} ms   p99 {stats['p99_s'] * 1000:.1f} ms   "
        f"max {stats['max_s'] * 1000:.1f} ms   min {stats['min_s'] * 1000:.1f} ms",
    ]


def _proposal_lines(proposal: dict) -> list:
    if proposal["ok"]:
        return [f"  --goal-response-bound {proposal['bound_s']:g}", f"  rule: {proposal['why']}"]
    return ["  REFUSED: no bound is proposed.", f"  {proposal['why']}"]


def render(stats: dict, proposal: dict, *, refused: int = 0, lost: int = 0) -> str:
    return "\n".join(_distribution("goal response", stats, refused=refused, lost=lost)
                     + _proposal_lines(proposal))


def render_both(idle: dict, preempt: dict, proposal: dict) -> str:
    """Both distributions, then the bound -- which is derived on ONE of them and says which.

    Printed in this order on purpose: a reader who sees only the last two lines still reads
    which population the number is a bound on, and a reader who sees the whole block can check
    that the slower distribution is the one it came from.
    """
    lines = _distribution("goal response, IDLE server", idle["stats"],
                          refused=idle.get("refused", 0), lost=idle.get("lost", 0))
    lines += _distribution("goal response, PRE-EMPTING a live goal", preempt["stats"],
                           refused=preempt.get("refused", 0), lost=preempt.get("lost", 0),
                           abandoned=preempt.get("abandoned", 0))
    lines.append(f"the bound is derived on the {proposal['from'].upper()} distribution, the "
                 f"slower of the two (max {proposal['max_s'] * 1000:.1f} ms)")
    return "\n".join(lines + _proposal_lines(proposal))


# ── the ROS 2 side ────────────────────────────────────────────────────────────


def run_probe(client, pose_of, n: int, *, cancel=None, clock=time.monotonic,
              sleep=time.sleep, settle_s: float = 0.2, timeout_s: float = 5.0):
    """Send `n` goals, time each answer, cancel each at once. Transport-agnostic.

    `client.send(goal, on_answer)` must call `on_answer(handle_or_None)` when the server
    answers; `pose_of()` returns the robot's pose or raises. `cancel(handle)` cancels an
    accepted goal. Returns (answer times, refused count, lost count).

    `clock` and `sleep` are injected together, and every wait goes through `sleep`, so a test
    can drive the clock: a lost-answer path that only terminates because real time passes
    cannot be tested for at all, and it is the path that matters most.
    """
    times, refused, lost = [], 0, 0
    for _ in range(n):
        pose = pose_of()            # raises if the pose cannot be read -- never a bare origin
        answer_s, handle = _time_answer(client, pose, clock, sleep, timeout_s)
        if answer_s is None:
            lost += 1
            continue
        times.append(answer_s)
        if _accepted(handle):
            if cancel is not None:
                cancel(handle)      # accepted goals are cancelled AT ONCE, before they drive
        else:
            refused += 1
        sleep(settle_s)
    return times, refused, lost


def _accepted(handle) -> bool:
    return handle is not None and getattr(handle, "accepted", True)


def _time_answer(client, pose, clock, sleep, timeout_s):
    """Send ONE goal and time the server's answer: ``(seconds, handle)``, or ``(None, None)``
    if nothing came within `timeout_s`. A lost answer is not a slow one and gets no time."""
    answered = []
    t0 = clock()
    client.send(pose, lambda handle: answered.append((clock(), handle)))
    deadline = t0 + timeout_s
    # Never sleeps PAST the deadline. The pre-empt mode clips this wait to what is left of the
    # drive budget, so a poll that overshot by its own tick would be the robot driving past a
    # limit stated in metres.
    while not answered and (left := deadline - clock()) > 0:
        sleep(min(0.005, left))
    if not answered:
        return None, None
    t1, handle = answered[0]
    return t1 - t0, handle


def run_preempt_probe(client, pose_of, ahead_of, n, *, cancel=None, clock=time.monotonic,
                      sleep=time.sleep, settle_s: float = 0.5, timeout_s: float = 5.0,
                      busy_s: float = 0.3, drive_budget_s: float = DRIVE_BUDGET_S):
    """Time the answer to a goal sent while the server is ALREADY RUNNING ONE.

    One sample: send goal 1 to `ahead_of(pose)` -- a point the robot is not at, so the server
    has to actually take it up; wait `busy_s` for the controller to start on it; send goal 2 at
    the robot's pose and time THAT answer. Returns (answer times, refused, lost, abandoned).

    THE SAFETY PROPERTY THIS FUNCTION CARRIES: goal 1 is cancelled within `drive_budget_s` of
    being accepted, on every path out of the sample -- the answer arriving, the answer being
    lost, the pose read raising. The wait for goal 2 is clipped to what is left of that budget,
    so a slow answer costs the sample, never more driving. Goal 1 is cancelled even when goal 2
    pre-empted it already: a cancel on a goal the server has finished with is a no-op, and
    depending on the pre-emption to have ended it would make the limit a belief about the
    server rather than something this loop does.

    `abandoned` counts samples where the server refused or lost goal 1. Those measure an idle
    server, not a pre-empted one, so they are NOT counted as answers: a sample that did not
    pre-empt anything must not enter the pre-empt distribution.
    """
    times, refused, lost, abandoned = [], 0, 0, 0
    for _ in range(n):
        pose = pose_of()            # raises if the pose cannot be read -- never a bare origin
        _, busy = _time_answer(client, ahead_of(pose), clock, sleep, timeout_s)
        accepted_at = clock()
        if not _accepted(busy):
            abandoned += 1
            if busy is not None and cancel is not None:
                cancel(busy)
            sleep(settle_s)
            continue
        try:
            sleep(min(busy_s, drive_budget_s))   # let the controller take goal 1 up
            left = drive_budget_s - (clock() - accepted_at)
            answer_s, handle = _time_answer(
                client, pose_of(), clock, sleep, max(0.0, min(timeout_s, left)))
        finally:
            if cancel is not None:
                cancel(busy)        # ALWAYS, and inside the drive budget
        if answer_s is None:
            lost += 1
        else:
            times.append(answer_s)
            if _accepted(handle):
                if cancel is not None:
                    cancel(handle)
            else:
                refused += 1
        sleep(settle_s)
    return times, refused, lost, abandoned


def _ros_main(args) -> int:  # pragma: no cover - requires a live ROS 2 + Nav2 runtime
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import Odometry
    from rclpy.action import ActionClient
    from rclpy.node import Node

    rclpy.init()
    node = Node("swarmada_goal_response_probe")
    latest = {}
    node.create_subscription(Odometry, args.odom, lambda m: latest.update(msg=m), 10)
    client = ActionClient(node, NavigateToPose, args.action)

    def spin(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    if not client.wait_for_server(timeout_sec=args.wait):
        print(f"REFUSED: no action server on {args.action} within {args.wait:g} s",
              file=sys.stderr)
        return 2
    spin(1.0)
    if "msg" not in latest:
        print(f"REFUSED: no message on {args.odom}, so the robot's pose cannot be read and a "
              f"goal would go to the origin. This probe never sends a goal it did not place.",
              file=sys.stderr)
        return 2

    def pose_of():
        m = latest["msg"]
        p = PoseStamped()
        p.header.frame_id = args.frame
        p.header.stamp = node.get_clock().now().to_msg()
        p.pose = m.pose.pose
        return p

    class _Client:
        def send(self, pose, on_answer):
            goal = NavigateToPose.Goal()
            goal.pose = pose
            fut = client.send_goal_async(goal)
            fut.add_done_callback(lambda f: on_answer(f.result()))

    def ahead_of(pose):
        """`pose` translated `--ahead` metres along the robot's OWN heading, same orientation.

        The heading is read from the pose's quaternion rather than assumed, so the goal is
        ahead of the robot whichever way it is facing -- a fixed offset in the map frame would
        be behind it half the time, and a goal behind this robot needs the ~3.45 s in-place
        turn that is the whole reason this mode exists.
        """
        q = pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p2 = PoseStamped()
        p2.header.frame_id = pose.header.frame_id
        p2.header.stamp = node.get_clock().now().to_msg()
        p2.pose.position.x = pose.pose.position.x + args.ahead * math.cos(yaw)
        p2.pose.position.y = pose.pose.position.y + args.ahead * math.sin(yaw)
        p2.pose.position.z = pose.pose.position.z
        p2.pose.orientation = q
        return p2

    def _cancel(handle):
        handle.cancel_goal_async()

    # rclpy delivers the answer on the executor, so the probe's wait loop has to spin it.
    import threading
    stop = threading.Event()
    spinner = threading.Thread(
        target=lambda: [rclpy.spin_once(node, timeout_sec=0.05)
                        for _ in iter(lambda: not stop.is_set(), False)],
        daemon=True)
    spinner.start()
    pre = None
    try:
        times, refused, lost = run_probe(_Client(), pose_of, args.n, cancel=_cancel,
                                         settle_s=args.settle, timeout_s=args.timeout)
        if args.preempt:
            print(f"PRE-EMPT MODE: {args.preempt_n} samples, each driving the robot towards a "
                  f"point {args.ahead:g} m ahead of it for at most {args.drive_budget:g} s. "
                  f"Clear floor, operator at the switch.", file=sys.stderr)
            pt, pr, pl, pa = run_preempt_probe(
                _Client(), pose_of, ahead_of, args.preempt_n, cancel=_cancel,
                settle_s=args.settle, timeout_s=args.timeout, busy_s=args.busy,
                drive_budget_s=args.drive_budget)
            pre = {"samples_s": pt, "refused": pr, "lost": pl, "abandoned": pa}
    finally:
        shutdown_spin(stop, spinner, node, rclpy)

    if not times:
        print(f"REFUSED: none of {args.n} goals was answered within {args.timeout:g} s.",
              file=sys.stderr)
        return 2
    stats = summarise(times)
    out = {"samples_s": times, "stats": stats, "refused": refused, "lost": lost,
           "action": args.action}
    if pre is None:
        proposal = propose_bound(stats, factor=args.factor, floor_s=args.floor,
                                 ceiling_s=args.ceiling, lost=lost)
        print(render(stats, proposal, refused=refused, lost=lost))
    elif not pre["samples_s"]:
        # The pre-empt population is EMPTY, and the idle one must not stand in for it: that
        # substitution is the whole defect this mode exists to stop.
        print(f"REFUSED: no bound is proposed.\n  the pre-empt mode was asked for and none of "
              f"its {args.preempt_n} samples produced an answer to a pre-empting goal "
              f"({pre['lost']} lost, {pre['abandoned']} never occupied the server). The idle "
              f"distribution is not a bound on the pre-empting case and is not offered as one.")
        out["preempt"] = {**pre, "stats": None}
        proposal = {"ok": False, "bound_s": None, "from": "pre-empt", "max_s": None,
                    "why": "the pre-empt mode produced no answer to a pre-empting goal"}
        out["proposal"] = proposal
        _write_json(args.json, out)
        return 3
    else:
        pre["stats"] = summarise(pre["samples_s"])
        picked = worse(stats, pre["stats"])
        proposal = propose_bound(
            picked, factor=args.factor, floor_s=args.floor, ceiling_s=args.ceiling,
            lost=lost + pre["lost"],
            population="pre-empt" if picked is pre["stats"] else "idle")
        out["preempt"] = pre
        print(render_both({"stats": stats, "refused": refused, "lost": lost}, pre, proposal))
    out["proposal"] = proposal
    _write_json(args.json, out)
    return 0 if proposal["ok"] else 3


def shutdown_spin(stop, spinner, node, rclpy_module, join_s: float = SPIN_JOIN_S) -> bool:
    """Take the spin thread down BEFORE the node and the context it spins on. Returns joined?

    The order is the whole point. `stop.set()` only asks: the spinner may already be inside
    `rclpy.spin_once()`, which blocks for up to its timeout, and it may enter one more after the
    flag is read. Destroying the node or shutting the context down in that window leaves the
    spinner calling into freed rcl state, and rcl answers that from C++ -- in a daemon thread,
    with no Python handler above it, which ends the process on `terminate called without an
    active exception` however cleanly the measuring part finished.

    So the flag is set, the thread is JOINED, and only a spinner that is provably out of
    `spin_once()` is followed by `destroy_node()`. A spinner still alive after `join_s` is wedged
    inside rcl, and tearing the node out from under it is exactly the thing that aborts: the
    teardown is skipped and said out loud instead, leaving the process to exit on its own.
    """
    stop.set()
    spinner.join(join_s)
    if spinner.is_alive():
        print(f"the spin thread did not come out of spin_once() within {join_s:g} s; leaving the "
              f"node and the context for process exit rather than freeing them underneath it",
              file=sys.stderr)
        return False
    node.destroy_node()
    rclpy_module.shutdown()
    return True


def _write_json(path: str, payload: dict) -> None:  # pragma: no cover - ROS-side plumbing
    if not path:
        return
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {path}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--action", default="/navigate_to_pose")
    ap.add_argument("--odom", default="/odom")
    ap.add_argument("--frame", default="map")
    ap.add_argument("--wait", type=float, default=20.0, help="seconds to wait for the server")
    ap.add_argument("--settle", type=float, default=0.2, help="pause between goals")
    ap.add_argument("--timeout", type=float, default=5.0, help="an answer later than this is lost")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR_S)
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_S)
    ap.add_argument("--json", default="")
    ap.add_argument("--preempt", action="store_true",
                    help="ALSO measure the answer to a goal sent while another is active -- "
                         "the case the conformance harness creates. THIS DRIVES THE ROBOT.")
    ap.add_argument("--preempt-n", type=int, default=10,
                    help="pre-empt samples (default 10; each one drives the robot)")
    ap.add_argument("--ahead", type=float, default=AHEAD_M,
                    help="metres ahead, in the robot's own frame, to place the occupying goal")
    ap.add_argument("--busy", type=float, default=0.3,
                    help="seconds to let the controller take the occupying goal up")
    ap.add_argument("--drive-budget", type=float, default=DRIVE_BUDGET_S,
                    help="seconds the occupying goal may drive before it is cancelled")
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error("--n must be at least 1")
    if args.preempt_n < 1:
        ap.error("--preempt-n must be at least 1")
    if args.ahead <= 0:
        ap.error("--ahead must be positive: a goal at the robot's own pose occupies nothing")
    if args.drive_budget <= 0:
        ap.error("--drive-budget must be positive")
    if args.busy >= args.drive_budget:
        ap.error("--busy must be less than --drive-budget, or the occupying goal is cancelled "
                 "before the pre-empting goal is ever sent and nothing is measured")
    return _ros_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
