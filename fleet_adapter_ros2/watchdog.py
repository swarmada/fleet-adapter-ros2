# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The progress watchdog: a goal that stops getting closer is a stalled goal.

Without it, the task state is a pure function of the last NavigateToPose GoalStatus, and a
goal the navigation stack never terminates (a wedged controller, a robot boxed in by a
costmap it cannot leave, a dropped /cmd_vel consumer) is reported RUNNING for as long as the
process lives. The control plane is told the robot is working; it is neither working nor
schedulable, and nothing reports a fault.

PROGRESS IS MEASURED, NOT INFERRED FROM THE GOAL'S STATUS. The measurement is the
NavigateToPose feedback's ``distance_remaining``: the navigation stack's own distance from
the robot's estimated pose to the goal. The best (smallest) value seen must shrink by at least
``min_progress_m`` within every ``window_s``, or the goal is declared stalled.

WHAT THIS CANNOT SEE. The distance is computed from the robot's estimated pose, and that
estimate rests on odometry. On a platform whose odometry echoes the velocity command rather
than measuring the wheels or the body -- a command-echo platform -- a robot that is commanded
but
not moving -- blocked, slipping, lifted -- reports a pose that advances, so the distance to
go shrinks and this watchdog is FOOLED. It is the same independence question as C5.3's
confirmed stop: a check can only be as independent of the command as the channel it reads.
On such a platform the watchdog catches a stack that stops commanding, and not a robot that
stops moving under a command; `plant.yaml`'s `motion_channel.odometry` says which platform
this is.

THE THRESHOLDS ARE PROPERTIES OF THE PLANT, NOT OF THIS CODE, so there is no default here
(ADR-0067). How long a robot may legitimately make no progress depends on how it turns (a
turn in place, at the start of a goal or to the final heading at its end, gains no distance),
how fast it recovers, and how its planner detours; how much counts as progress depends on the
noise in its pose estimate. A default would be a number measured on no robot, and a wrong one
fails silently in one direction or the other: too short cancels good goals, too long reports a
stalled robot as working.

NOT A STALL:

* **Before the first feedback.** Nothing has been measured, so there is nothing to judge
  against. The wait is bounded all the same, by one window from dispatch: a goal that never
  produces feedback is exactly the silence this exists to catch. The first feedback then
  starts the first measured window, so a goal is declared stalled at most two windows after
  dispatch.
* **While paused.** The watchdog is disarmed on pause and re-armed, with a fresh window and a
  fresh wait for feedback, on resume. A goal the robot is holding on purpose is not stalled.
* **Once the goal has ended or been stopped.** Disarmed.

Pure and clock-injected, so every rule above is testable without ROS and without waiting.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


def _positive_finite(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v > 0)


@dataclass(frozen=True)
class StallLimits:
    """The two thresholds, both required, both the plant's (see the module docstring)."""

    window_s: float        # a goal must make min_progress_m of progress within this
    min_progress_m: float  # the least reduction in distance to go that counts as progress

    def __post_init__(self) -> None:
        for name, v in (("window_s", self.window_s), ("min_progress_m", self.min_progress_m)):
            if not _positive_finite(v):
                raise ValueError(f"{name} must be a finite number greater than 0, got {v!r}")

    def describe(self) -> str:
        return (f"a goal whose distance to go does not shrink by {self.min_progress_m:g} m "
                f"within {self.window_s:g} s is cancelled and reported FAILED")


class ProgressWatchdog:
    """One goal's progress, judged against :class:`StallLimits`.

    ``arm()`` on dispatch and on resume, ``disarm()`` on pause, stop and a terminal result,
    ``observe()`` on every feedback, ``check()`` periodically. ``check()`` returns the stall
    reason once, and disarms itself when it does."""

    def __init__(self, limits: StallLimits,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits
        self._clock = clock
        self._active = False
        self._armed_at = 0.0
        self._first_d: float | None = None   # distance at the first feedback
        self._best: float | None = None      # smallest distance to go seen so far
        self._anchor_d: float | None = None  # distance at the start of the current window
        self._anchor_t = 0.0                 # when the current window started

    @property
    def armed(self) -> bool:
        return self._active

    def arm(self) -> None:
        self._active = True
        self._armed_at = self._clock()
        self._first_d = self._best = self._anchor_d = None

    def disarm(self) -> None:
        self._active = False

    def observe(self, distance_remaining: float) -> None:
        """A feedback's distance to go. The window restarts whenever the best distance has
        improved by min_progress_m since the window began."""
        if not self._active or not isinstance(distance_remaining, (int, float)) \
                or not math.isfinite(distance_remaining) or distance_remaining < 0:
            return
        now = self._clock()
        if self._anchor_d is None:           # the first measurement starts the first window
            self._first_d = self._best = self._anchor_d = float(distance_remaining)
            self._anchor_t = now
            return
        self._best = min(self._best, float(distance_remaining))
        if self._anchor_d - self._best >= self.limits.min_progress_m:
            self._anchor_d, self._anchor_t = self._best, now

    def check(self) -> str:
        """The stall reason, or "" while the goal is progressing (or not armed)."""
        if not self._active:
            return ""
        now = self._clock()
        lim = self.limits
        since_dispatch = now - self._armed_at
        if self._anchor_d is None:
            if since_dispatch < lim.window_s:
                return ""
            self._active = False
            return (f"stalled: no progress feedback arrived within {since_dispatch:.1f} s of "
                    f"dispatch (limit {lim.window_s:g} s); nothing shows the robot getting "
                    f"closer to the goal")
        in_window = now - self._anchor_t
        if in_window < lim.window_s:
            return ""
        self._active = False
        return (f"stalled: distance to go improved {self._anchor_d - self._best:.3f} m in the "
                f"last {in_window:.1f} s, less than the {lim.min_progress_m:g} m required "
                f"within {lim.window_s:g} s ({self._best:.2f} m still to go; "
                f"{self._first_d - self._best:.2f} m made good in {since_dispatch:.1f} s "
                f"since dispatch)")


def resolve_stall_limits(binding: str, window_s: float | None,
                         min_progress_m: float | None) -> tuple[StallLimits | None, str]:
    """The adapter's start-up decision: ``(limits, "")`` to run, or ``(None, why)`` to refuse.

    ``--binding ros2`` needs both values: without them the adapter refuses to start rather
    than guess. One without the other is refused on any binding. The simulated binding needs
    neither, and runs unarmed (its robot has no goal feedback to measure)."""
    given = {"--stall-window": window_s, "--stall-min-progress": min_progress_m}
    missing = [flag for flag, v in given.items() if v is None]
    if binding == "ros2" and missing:
        return None, (
            f"--binding ros2 needs {' and '.join(missing)}: the progress watchdog's thresholds "
            f"(--stall-window seconds, --stall-min-progress metres). They are properties of the "
            f"robot and its navigation stack -- how long it may make no progress while it turns "
            f"or recovers, and how far is more than its pose noise -- so this adapter does not "
            f"assume them. Without a watchdog a goal the stack never terminates is reported "
            f"RUNNING forever.")
    if len(missing) == 1:
        return None, f"{missing[0]} is required with {next(f for f in given if f not in missing)}"
    if not missing:
        try:
            return StallLimits(window_s=window_s, min_progress_m=min_progress_m), ""
        except ValueError as exc:
            return None, f"--stall-window / --stall-min-progress: {exc}"
    return None, ""


def resolve_response_bound(binding: str, bound_s: float | None) -> tuple[float | None, str]:
    """The adapter's start-up decision on ``--goal-response-bound``: ``(bound, "")`` to run, or
    ``(None, why)`` to refuse.

    How long a goal sent to the navigation server may go unanswered before it is abandoned
    (Nav2Binding `_watch_answers`). Like the stall thresholds, no default: how long an answer
    legitimately takes is the server's and its transport's, measured on them. Too short fails
    good goals; too long holds every stop in that window unconfirmed. ``--binding ros2`` needs
    it; the simulated binding sends no goals and needs none."""
    if bound_s is None:
        if binding == "ros2":
            return None, (
                "--binding ros2 needs --goal-response-bound: how long, in seconds, a goal sent "
                "to the navigation server may go unanswered before it is abandoned (stopped, "
                "cancelled by its id, and FAILED). It is a property of the navigation server "
                "and its transport, measured on them, so this adapter does not assume it. "
                "Without it a goal whose answer is lost is never cancelled, and every later "
                "stop waits on it.")
        return None, ""
    if not _positive_finite(bound_s):
        return None, f"--goal-response-bound must be a finite number greater than 0, got {bound_s!r}"
    return float(bound_s), ""


# ── The binding's rest evidence ───────────────────────────────────────────────
#
# WHAT CONFIRMS REST IS A PROPERTY OF THE PLATFORM, NOT OF THIS ADAPTER, so there is no
# default here either. CONFORMANCE.md C5.3 forbids reporting a stop that was inferred, and
# ADR-0071 states the precondition: a STOPPED may be reported only from a motion channel
# INDEPENDENT OF THE COMMAND PATH -- one whose reading would differ if the drive did not do
# what it was told (a wheel slipped, a motor stalled, someone pushed the robot).
#
# `Nav2Binding` confirms rest from the odometry twist. On some bases that twist is the
# commanded velocity, clamped, filtered or integrated -- the drive controller publishes back
# what it was asked for, with no encoder anywhere in the path. On such a base the twist reads
# zero the instant a stop is commanded, whatever the wheels are doing, so "rest confirmed from
# odom" is an echo of the stop command: exactly the inference C5.3 forbids, and it is
# undetectable from the wire. Nothing in the ROS graph distinguishes the two cases -- the
# topic, the message type and the frame are identical -- so the adapter cannot read this and
# must be told.
ODOMETRY_MEASURED = "measured"
ODOMETRY_COMMAND_DERIVED = "command-derived"
ODOMETRY_SOURCES = (ODOMETRY_MEASURED, ODOMETRY_COMMAND_DERIVED)


def resolve_odometry_source(binding: str, source: str | None) -> tuple[str | None, str]:
    """The adapter's start-up decision on ``--odometry-source``: ``(source, "")`` to run, or
    ``(None, why)`` to refuse.

    Whether the odometry this binding confirms rest from MEASURES the robot, or reproduces the
    velocity command. ``--binding ros2`` needs it; the simulated binding's odometry is its own
    plant model's state, which is a measurement of the only robot there is."""
    if source is None:
        if binding == "ros2":
            return None, (
                "--binding ros2 needs --odometry-source: whether the odometry this binding "
                "confirms rest from is 'measured' (a sensor that observes the robot's motion, "
                "so it would read differently if the drive did not do what it was commanded) "
                "or 'command-derived' (the velocity command, clamped, filtered or integrated, "
                "published back as odometry). Nothing in the ROS graph tells the two apart, so "
                "this adapter does not assume it. Declared 'command-derived', no stop is ever "
                "reported confirmed from odom, because on such a base the twist reads zero the "
                "instant a stop is commanded whether or not the robot is still moving, and "
                "CONFORMANCE.md C5.3 forbids reporting that as a confirmation.")
        return None, ""
    if source not in ODOMETRY_SOURCES:
        return None, (f"--odometry-source must be one of {', '.join(ODOMETRY_SOURCES)}, "
                      f"got {source!r}")
    return source, ""


def rest_evidence_gap(source: str | None) -> str:
    """Why a stop cannot be reported confirmed on this binding's evidence, or ``""``.

    One sentence, carried verbatim into every unconfirmed stop report (the estop's FAILED,
    the lease self-stop's FAILED, the cancel's unacknowledged answer), so an operator reading
    the report is told what the platform cannot do rather than left to infer a fault."""
    if source == ODOMETRY_MEASURED:
        return ""
    if source == ODOMETRY_COMMAND_DERIVED:
        return ("this binding's odometry is declared command-derived (--odometry-source "
                "command-derived), so it reads zero the instant a stop is commanded whether "
                "or not the robot is still moving: nothing here can confirm rest, and "
                "reporting a stop from it would be the inference C5.3 forbids")
    return ("this binding was given no --odometry-source, so whether its odometry measures "
            "the robot or reproduces the velocity command is undeclared, and rest cannot be "
            "confirmed from a channel whose independence from the command is unknown")
