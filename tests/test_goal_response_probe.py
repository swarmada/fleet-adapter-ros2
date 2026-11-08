# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The goal-response probe: it measures, and it refuses rather than proposing a number.

`--goal-response-bound` has no default because it is the navigation server's property, so the
probe that measures it must not have one either. Three properties carry this file:

1. **A bound is proposed only from answers that arrived.** A lost answer is not a slow one --
   it is a different population -- and a server that loses answers gets a refusal, not a
   bound derived from the survivors.
2. **The ceiling is a refusal, not a clamp.** A server slower than half the adapter's
   confirmed-stop budget cannot be given a bound that fits; clamping to the ceiling would
   silently turn "too slow to promise a bounded stop on" into a number on a run sheet.
3. **The probe cancels every accepted goal, at once.** It runs on a real robot with a person
   beside it; a goal it forgets to cancel is a goal the controller goes on following.
4. **The bound is derived on the harder of the two cases.** The idle loop is not the case the
   bound is for -- the conformance harness sends goals while another is active -- so the
   pre-empt mode measures that too, and the rule is applied to whichever distribution owns the
   slower answer. The occupying goal it sends DRIVES the robot, and the drive is bounded by
   this loop on every path out of a sample, not by what the server happens to do.

The send/answer/cancel loop is driven against a stand-in action client that answers on the
schedule each test states, and against the testbed's fake robot the same loop will meet on the
desk: accept-then-answer, refuse, and never-answer.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "testbed"))

from goal_response_probe import (  # noqa: E402
    DRIVE_BUDGET_S,
    propose_bound,
    render,
    render_both,
    run_preempt_probe,
    run_probe,
    shutdown_spin,
    summarise,
    worse,
)


class _Clock:
    """A clock the test advances, so an 'answer time' is exact rather than approximately fast."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class _Client:
    """A stand-in action server. `answers` is one entry per goal: a delay and a handle."""

    def __init__(self, answers, clock: _Clock) -> None:
        self.answers = list(answers)
        self.clock = clock
        self.sent = []

    def send(self, pose, on_answer) -> None:
        self.sent.append(pose)
        delay, handle = self.answers.pop(0)
        if delay is None:
            return                      # never answered
        self.clock.t += delay
        on_answer(handle)


class _Handle:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.cancels = 0


def _pose_of():
    return "the robot's own pose"


def _run(answers, n=None, **kw):
    clock = _Clock()
    client = _Client(answers, clock)
    cancelled = []
    times, refused, lost = run_probe(
        client, _pose_of, len(answers) if n is None else n,
        cancel=lambda h: (cancelled.append(h), setattr(h, "cancels", h.cancels + 1)),
        clock=clock, sleep=clock.sleep, settle_s=0.0, timeout_s=0.01, **kw)
    return times, refused, lost, cancelled, client


# ── the loop ─────────────────────────────────────────────────────────────────────────────

def test_every_accepted_goal_is_cancelled_at_once():
    handles = [_Handle() for _ in range(4)]
    times, refused, lost, cancelled, client = _run([(0.004, h) for h in handles])
    assert len(client.sent) == 4 and len(times) == 4
    assert cancelled == handles and all(h.cancels == 1 for h in handles)
    assert refused == 0 and lost == 0


def test_a_refused_goal_is_counted_and_not_cancelled():
    times, refused, lost, cancelled, _ = _run([(0.003, _Handle(accepted=False))])
    assert times and refused == 1 and lost == 0 and cancelled == []


def test_a_goal_that_is_never_answered_is_lost_not_slow():
    times, refused, lost, cancelled, _ = _run([(0.005, _Handle()), (None, None)])
    assert len(times) == 1 and lost == 1 and cancelled and len(cancelled) == 1


def test_the_pose_is_read_for_every_goal_and_never_defaulted():
    calls = []

    def pose_of():
        calls.append(1)
        return "pose"
    clock = _Clock()
    run_probe(_Client([(0.002, _Handle())] * 3, clock), pose_of, 3, clock=clock,
              sleep=clock.sleep, settle_s=0.0)
    assert len(calls) == 3


def test_a_pose_that_cannot_be_read_stops_the_probe_rather_than_sending_a_goal():
    clock = _Clock()
    client = _Client([(0.002, _Handle())], clock)

    def pose_of():
        raise RuntimeError("no odometry")
    with pytest.raises(RuntimeError, match="no odometry"):
        run_probe(client, pose_of, 1, clock=clock, sleep=clock.sleep, settle_s=0.0)
    assert client.sent == [], "a goal was sent without a pose to place it at"


# ── the statistics ───────────────────────────────────────────────────────────────────────

def test_the_summary_is_of_the_answers_that_arrived():
    s = summarise([0.010, 0.004, 0.006, 0.030])
    assert s["n"] == 4 and s["min_s"] == 0.004 and s["max_s"] == 0.030
    assert s["median_s"] == pytest.approx(0.008)


def test_p99_on_a_small_sample_is_the_maximum_and_does_not_interpolate():
    s = summarise([0.001] * 19 + [0.500])
    assert s["p99_s"] == 0.500, "an interpolated p99 would invent a value between two answers"


def test_a_sample_with_nothing_in_it_is_refused():
    with pytest.raises(ValueError, match="nothing to summarise"):
        summarise([])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.001])
def test_an_answer_time_that_is_not_a_finite_non_negative_number_is_refused(bad):
    with pytest.raises(ValueError, match="finite number"):
        summarise([0.01, bad])


# ── the rule ─────────────────────────────────────────────────────────────────────────────

def test_the_bound_is_one_and_a_half_times_the_slowest_answer():
    p = propose_bound(summarise([0.004, 0.011, 0.400]))
    assert p["ok"] and p["bound_s"] == 0.6      # 1.5 * 0.400, rounded up to 2 dp
    assert "1.5 x max" in p["why"]


def test_a_fast_server_does_not_collapse_the_bound_below_the_floor():
    p = propose_bound(summarise([0.004, 0.005, 0.006]))
    assert p["ok"] and p["bound_s"] == 0.25, "a handful of fast answers is not a fast tail"


def test_a_server_too_slow_for_the_stop_budget_is_refused_not_clamped():
    p = propose_bound(summarise([0.010, 0.900]))
    assert p["ok"] is False and p["bound_s"] is None
    assert "ceiling" in p["why"] and "Do not raise the ceiling" in p["why"]


def test_a_server_that_lost_an_answer_gets_no_bound_at_all():
    p = propose_bound(summarise([0.004, 0.006]), lost=1)
    assert p["ok"] is False and "never answered at all" in p["why"]


def test_the_bound_is_rounded_UP_so_it_is_never_below_its_own_rule():
    p = propose_bound(summarise([0.271]))            # 1.5 * 0.271 = 0.4065
    assert p["ok"] and p["bound_s"] == 0.41 and p["bound_s"] >= 1.5 * 0.271


def test_the_rendered_block_names_the_flag_and_the_rule():
    stats = summarise([0.004, 0.011, 0.400])
    text = render(stats, propose_bound(stats), refused=0, lost=0)
    assert "--goal-response-bound 0.6" in text and "max 400.0 ms" in text


def test_a_refusal_renders_as_a_refusal_and_proposes_nothing():
    stats = summarise([0.9])
    text = render(stats, propose_bound(stats))
    assert "REFUSED" in text and "--goal-response-bound " not in text


# ── against the testbed's fake robot, without ROS ─────────────────────────────────────────

def test_the_loop_against_the_fake_robots_answer_shape():
    """`fake_robot_node.py` accepts a goal and answers it on the executor's next pass, and
    `--reject-goals` makes it refuse one. Both shapes go through this loop unchanged."""
    clock = _Clock()
    accepted = _Handle()
    times, refused, lost = run_probe(
        _Client([(0.004, accepted), (0.004, _Handle(accepted=False))], clock),
        _pose_of, 2, cancel=lambda h: setattr(h, "cancels", h.cancels + 1),
        clock=clock, sleep=clock.sleep, settle_s=0.0)
    assert len(times) == 2 and refused == 1 and lost == 0
    assert accepted.cancels == 1
    s = summarise(times)
    assert math.isclose(s["max_s"], 0.004, rel_tol=1e-9)
    assert propose_bound(s)["bound_s"] == 0.25


# ── the pre-empt loop ────────────────────────────────────────────────────────────────────

class _PreemptClient:
    """A stand-in server that answers each goal after a stated delay, and remembers the order.

    `answers` is read in send order, so a test states the occupying goal's answer and the
    pre-empting goal's answer separately -- which is the whole point: they are different
    populations and this fixture must be able to make them differ.
    """

    def __init__(self, answers, clock: _Clock) -> None:
        self.answers = list(answers)
        self.clock = clock
        self.sent = []

    def send(self, pose, on_answer) -> None:
        self.sent.append((pose, self.clock.t))
        delay, handle = self.answers.pop(0)
        if delay is None:
            return                      # never answered
        self.clock.t += delay
        on_answer(handle)


def _ahead(pose):
    return f"{pose}+0.5m"


def _run_preempt(answers, n=1, **kw):
    clock = _Clock()
    client = _PreemptClient(answers, clock)
    cancels = []
    kw = {"settle_s": 0.0, "timeout_s": 1.0, "busy_s": 0.1, **kw}
    times, refused, lost, abandoned = run_preempt_probe(
        client, _pose_of, _ahead, n,
        cancel=lambda h: (cancels.append((h, clock.t)),
                          setattr(h, "cancels", h.cancels + 1)),
        clock=clock, sleep=clock.sleep, **kw)
    return times, refused, lost, abandoned, cancels, client, clock


def test_the_time_recorded_is_the_second_goals_answer_not_the_first():
    busy, second = _Handle(), _Handle()
    times, refused, lost, abandoned, _, client, _ = _run_preempt(
        [(0.004, busy), (0.320, second)])
    assert [p for p, _ in client.sent] == ["the robot's own pose+0.5m", "the robot's own pose"]
    assert times == [pytest.approx(0.320)], "the occupying goal's answer is not it"
    assert (refused, lost, abandoned) == (0, 0, 0)


def test_the_occupying_goal_is_cancelled_inside_the_drive_budget_on_every_path():
    for label, answers in (("answered", [(0.004, _Handle()), (0.200, _Handle())]),
                           ("lost", [(0.004, _Handle()), (None, None)]),
                           ("refused", [(0.004, _Handle()),
                                        (0.010, _Handle(accepted=False))])):
        busy = answers[0][1]
        _, _, _, _, cancels, _, clock = _run_preempt(answers)
        at = [t for h, t in cancels if h is busy]
        assert at, f"{label}: the occupying goal was never cancelled"
        # accepted at t=1000.004, so the budget runs out at 1000.004 + DRIVE_BUDGET_S
        assert at[0] <= 1000.004 + DRIVE_BUDGET_S + 1e-9, (
            f"{label}: the robot drove past the budget ({at[0] - 1000.004:.3f} s)")
        assert busy.cancels == 1


def test_a_pre_empting_goal_that_is_never_answered_is_lost_and_costs_no_extra_driving():
    times, refused, lost, abandoned, cancels, _, _ = _run_preempt(
        [(0.004, _Handle()), (None, None)])
    assert times == [] and lost == 1 and refused == 0 and abandoned == 0
    assert len(cancels) == 1, "only the occupying goal had a handle to cancel"


def test_a_sample_whose_occupying_goal_was_refused_is_abandoned_not_counted():
    """It measured an idle server. Counting it would put an easy-case answer in the hard-case
    distribution, which is the substitution this whole mode exists to prevent."""
    times, refused, lost, abandoned, _, client, _ = _run_preempt(
        [(0.004, _Handle(accepted=False))])
    assert (times, refused, lost, abandoned) == ([], 0, 0, 1)
    assert len(client.sent) == 1, "no pre-empting goal is sent at a server that took nothing"


def test_the_pre_empting_goal_is_cancelled_too():
    busy, second = _Handle(), _Handle()
    _, _, _, _, cancels, _, _ = _run_preempt([(0.004, busy), (0.010, second)])
    assert second.cancels == 1 and busy.cancels == 1


def test_the_wait_for_the_pre_empting_answer_is_clipped_to_what_is_left_of_the_budget():
    # The occupying goal is accepted, 0.1 s of busy time passes, and the answer never comes:
    # the wait may run out the remaining budget and not a tick more.
    _, _, lost, _, cancels, _, clock = _run_preempt(
        [(0.004, _Handle()), (None, None)], timeout_s=30.0)
    assert lost == 1
    assert clock.t <= 1000.004 + DRIVE_BUDGET_S + 0.01, (
        "the wait for a lost answer outlasted the drive budget")


def test_every_sample_reads_the_pose_again_so_the_goal_follows_the_robot():
    """The robot has moved by the time the pre-empting goal is placed -- it has been driving
    towards the occupying goal, which is the point -- so the pose is re-read, not reused."""
    poses = iter(["p0", "p1", "p2", "p3"])
    clock = _Clock()
    client = _PreemptClient([(0.004, _Handle()), (0.010, _Handle())], clock)
    run_preempt_probe(client, lambda: next(poses), _ahead, 1, cancel=lambda h: None,
                      clock=clock, sleep=clock.sleep, settle_s=0.0, timeout_s=1.0, busy_s=0.1)
    assert [p for p, _ in client.sent] == ["p0+0.5m", "p1"]


def test_a_pose_that_cannot_be_read_sends_no_occupying_goal():
    clock = _Clock()
    client = _PreemptClient([(0.004, _Handle())], clock)

    def pose_of():
        raise RuntimeError("no odometry")
    with pytest.raises(RuntimeError, match="no odometry"):
        run_preempt_probe(client, pose_of, _ahead, 1, clock=clock, sleep=clock.sleep,
                          settle_s=0.0, busy_s=0.1)
    assert client.sent == []


def test_the_loop_against_the_fake_robots_pre_emption_shape():
    """`fake_robot_node.py` ends a goal overtaken by a newer one ABORTED, as Nav2 preempts it,
    and answers the newer goal on the executor's next pass. Two accepted answers, the second
    timed, and both handles cancelled -- the cancel on the overtaken goal is a no-op there and
    this loop sends it anyway rather than trusting the pre-emption to have ended it."""
    busy, second = _Handle(), _Handle()
    times, refused, lost, abandoned, cancels, _, _ = _run_preempt(
        [(0.004, busy), (0.006, second)])
    assert times == [pytest.approx(0.006)] and (refused, lost, abandoned) == (0, 0, 0)
    assert busy.cancels == 1 and second.cancels == 1


# ── choosing the population the bound is derived on ──────────────────────────────────────

def test_the_worse_distribution_is_the_one_with_the_slowest_answer():
    idle = summarise([0.004, 0.004, 0.005])
    preempt = summarise([0.010, 0.400])
    assert worse(idle, preempt) is preempt
    assert worse(preempt, idle) is preempt


def test_the_populations_are_chosen_between_and_never_pooled():
    """Pooling would let many fast idle answers bury a few slow pre-empted ones. The rule
    reads the maximum, so the maximum's own distribution is what it is applied to."""
    idle = summarise([0.004] * 50)
    preempt = summarise([0.300, 0.310])
    picked = worse(idle, preempt)
    assert picked["n"] == 2 and picked["max_s"] == 0.310
    assert propose_bound(picked, population="pre-empt")["bound_s"] == 0.47


def test_an_idle_measurement_alone_would_have_proposed_the_floor():
    """The 2026-09-21 shape in miniature: 4 ms idle answers propose the 0.25 s floor, and the
    pre-empting harness then goes unanswered past it."""
    idle = summarise([0.0043, 0.0041, 0.0039])
    assert propose_bound(idle)["bound_s"] == 0.25
    preempt = summarise([0.0043, 0.260, 0.510])
    p = propose_bound(worse(idle, preempt), population="pre-empt")
    assert p["ok"] and p["bound_s"] == 0.77 and p["from"] == "pre-empt"


def test_a_proposal_says_which_population_it_was_derived_on():
    p = propose_bound(summarise([0.004]), population="pre-empt")
    assert p["from"] == "pre-empt" and p["max_s"] == 0.004
    refusal = propose_bound(summarise([0.900]), population="pre-empt")
    assert refusal["ok"] is False and refusal["from"] == "pre-empt"


def test_a_pre_empting_server_above_the_ceiling_is_refused_not_clamped():
    p = propose_bound(worse(summarise([0.004]), summarise([0.800])), population="pre-empt")
    assert p["ok"] is False and "Do not raise the ceiling" in p["why"]


def test_both_distributions_are_printed_with_the_one_the_bound_came_from():
    idle = summarise([0.004, 0.005])
    pre = summarise([0.011, 0.400])
    proposal = propose_bound(worse(idle, pre), population="pre-empt")
    text = render_both({"stats": idle, "refused": 0, "lost": 0},
                       {"stats": pre, "refused": 0, "lost": 0, "abandoned": 1}, proposal)
    assert "IDLE server" in text and "PRE-EMPTING a live goal" in text
    assert "max 5.0 ms" in text and "max 400.0 ms" in text
    assert "derived on the PRE-EMPT distribution" in text
    assert "--goal-response-bound 0.6" in text
    assert "1 sample(s) dropped" in text


# ── exiting after the numbers are written ────────────────────────────────────────────────
#
# On hardware on 2026-09-21 the idle run printed its distribution, wrote its JSON, and then
# died on `terminate called without an active exception` / `Aborted`. Every number was intact
# and the exit code was 134, so a runbook step that reads the exit code called a good
# measurement a failure.
#
# The cause was the teardown order, not the measuring. `_ros_main` runs a daemon spin thread so
# rclpy can deliver goal answers, and at teardown it set the stop flag and went straight on to
# `destroy_node()` and `rclpy.shutdown()`. Setting the flag only ASKS: the spinner ticks at
# 0.05 s and may be inside `spin_once()`, or one check past the flag, when the node is freed
# underneath it. rcl reports that from C++ on a thread with no Python handler above it, which
# ends the process whatever the measuring part returned.
#
# `shutdown_spin` is that order made explicit and given a seam. These tests drive it with a
# real `threading.Thread` and a stand-in node, so the race is reproduced and fixed here without
# a ROS runtime. It does NOT reproduce against `testbed/fake_robot_node.py`: that is a ROS 2
# node and needs rclpy, which is not installed on this machine, so the fake robot cannot be
# brought up to run the probe end to end. The abort is in rclpy's own teardown rather than in
# anything the robot does, and the spinner-versus-node ordering it turns on is exactly what is
# reproduced below.


class _SpinNode:
    """A node that records the thing rcl aborts on: a spin after the node was destroyed."""

    def __init__(self, tick_s: float = 0.01) -> None:
        self.tick_s = tick_s
        self.destroyed = False
        self.spins_after_destroy = 0
        self.spins = 0

    def spin_once(self) -> None:
        # The real `spin_once` blocks in rcl for up to its timeout. Sleeping here is what gives
        # the main thread a window to destroy the node mid-spin, which is the race itself.
        time.sleep(self.tick_s)
        self.spins += 1
        if self.destroyed:
            self.spins_after_destroy += 1

    def destroy_node(self) -> None:
        self.destroyed = True


class _Rclpy:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _spinner_for(node, stop):
    return threading.Thread(
        target=lambda: [node.spin_once() for _ in iter(lambda: not stop.is_set(), False)],
        daemon=True)


def test_the_spin_thread_is_joined_before_the_node_it_spins_on_is_destroyed():
    """The abort's shape: no spin may land after `destroy_node()`.

    Without the join the stop flag is set and the node is freed while the spinner is still
    inside its tick, and `spins_after_destroy` is non-zero -- which on a real runtime is rcl
    touching freed state from a daemon thread, and the process aborting after it had already
    printed everything it measured.
    """
    node, stop = _SpinNode(), threading.Event()
    spinner = _spinner_for(node, stop)
    spinner.start()
    time.sleep(0.05)                      # let it get into the loop

    assert shutdown_spin(stop, spinner, node, _Rclpy()) is True
    assert spinner.is_alive() is False
    assert node.spins_after_destroy == 0
    assert node.spins > 0                 # it really was spinning, so the join meant something


def test_the_old_order_is_what_aborted():
    """The same node and thread with the flag set and the node freed straight after.

    This is the branch head's teardown, kept as a test so the fix is not silently undone: it
    reproduces the post-destroy spin that the joined order above does not have.
    """
    node, stop = _SpinNode(), threading.Event()
    spinner = _spinner_for(node, stop)
    spinner.start()
    time.sleep(0.05)

    stop.set()
    node.destroy_node()                   # no join -- the spinner is still inside its tick
    spinner.join(5.0)

    assert node.spins_after_destroy >= 1


def test_a_wedged_spinner_is_left_alone_rather_than_freed_underneath():
    """A spinner stuck in rcl is the one case where destroying the node is certainly wrong.

    The teardown is skipped and said on stderr; the probe has its numbers and the process is
    about to exit anyway, so a leaked context costs nothing and freeing it costs the exit code.
    """
    node, stop = _SpinNode(tick_s=0.2), threading.Event()
    spinner = _spinner_for(node, stop)
    spinner.start()
    rclpy_module = _Rclpy()

    assert shutdown_spin(stop, spinner, node, rclpy_module, join_s=0.01) is False
    assert node.destroyed is False
    assert rclpy_module.shutdown_calls == 0


def test_the_process_exits_0_after_the_numbers_are_written():
    """The whole symptom, end to end, as a real process and a real exit code.

    `tests/ros_stub/` is the smallest ROS layer `_ros_main` will run against, and it carries
    the one behaviour that made this a bad exit rather than a bad measurement: a teardown that
    lands while a spin thread is inside `spin_once()` is a fault reported from rcl, and it ends
    the process at context destruction -- after everything was printed and written. That stub
    returns 134 for the branch head's ordering and 0 for the joined one, so this asserts the
    exit code AND that the JSON is on disk, which is the pair the hardware run got wrong.

    The testbed's `fake_robot_node.py` would be the natural robot to run this against, and on a
    sourced ROS 2 machine it is enough. It cannot serve here: it is an rclpy node, rclpy is not
    installed on this machine, and the abort lives in rclpy's teardown rather than in anything
    the robot does or answers.
    """
    probe = Path(__file__).resolve().parent.parent / "testbed" / "goal_response_probe.py"
    stub = Path(__file__).resolve().parent / "ros_stub"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "goal_response.json"
        run = subprocess.run(
            [sys.executable, str(probe), "--n", "3", "--settle", "0", "--wait", "1",
             "--json", str(out)],
            cwd=tmp, env={**os.environ, "PYTHONPATH": str(stub)},
            capture_output=True, text=True, timeout=120)

        assert "terminate called" not in run.stderr
        assert run.returncode == 0, f"exit {run.returncode}\n{run.stdout}\n{run.stderr}"
        assert out.exists(), "the probe returned 0 without writing its JSON"
        written = json.loads(out.read_text())
        assert written["stats"]["n"] == 3
        assert written["proposal"]["ok"] is True
        assert "--goal-response-bound" in run.stdout
