#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`fake_robot_node.py`'s command line: every flag it implements, and a refusal for the rest.

WHY THIS EXISTS
----------------
`fake_robot_node.py` parsed with `ap.parse_known_args()` and handed the leftovers to
`rclpy.init()`, which ignores anything outside a `--ros-args` section. `r1_smoke.py`'s
fault gate listed `--cruise` as a scoreable condition, and the node declared no
`--cruise`. So `ROBOT_ARGS="--cruise 0.2" make r1` -- the obvious way to simulate the
MentorPi's 0.2 m/s cap -- passed the gate, ran the plant at 0.5 m/s, and printed a result
under a condition that never applied. Nothing said the flag was dropped.

Two things changed, and the second is the one that would have caught it at the time:

1. `--cruise` is implemented. It sets the built-in controller's cruise speed, so a
   capped platform is simulable in R1.
2. Anything that is not implemented is REFUSED, before a node exists. Only a
   `--ros-args ... [--]` section goes to rclpy. Outside it, an unknown flag, an
   abbreviation (`--stall 3` used to reach `--stall-after` while the gate, which matches
   whole tokens, saw nothing), and a flag that `--passive` switches off all exit 2 with
   the reason. The last is the same defect in PASSIVE mode: there is no built-in
   controller or action server to apply `--cruise`, `--reject-goals`, `--goal-outcome`
   or `--stall-after` to, so accepting them would print a run under a condition that
   never applied.

This module is ROS-free so the refusal is testable without a runtime, and so
`r1_smoke.py`'s gate can ask the robot's own parser before bring-up instead of keeping a
second list of flags that drifts from the first.
"""

from __future__ import annotations

import argparse

CRUISE_SPEED = 0.5  # m/s -- the default; nav2_params.yaml's desired_linear_vel is the same

#: Flags that act on the built-in controller or the NavigateToPose server, both of which
#: `--passive` switches off. Accepting one alongside `--passive` would be a silent no-op.
ACTIVE_ONLY = ("--cruise", "--goal-outcome", "--reject-goals", "--stall-after")


class ArgsError(ValueError):
    """The command line names something the node does not implement."""


class _Parser(argparse.ArgumentParser):
    # argparse exits from inside parse_args(); raise instead, so the gate in r1_smoke can
    # ask this parser a question without it ending the harness.
    def error(self, message: str):
        raise ArgsError(message)


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: a prefix would reach a flag under a name the gate cannot see.
    ap = _Parser(prog="fake_robot_node.py", allow_abbrev=False,
                 description="A fake ROS 2 robot for exercising fleet-adapter-ros2.")
    ap.add_argument("--goal-outcome", choices=["succeed", "abort"], default="succeed",
                    help="terminal status once the goal pose is reached")
    ap.add_argument("--reject-goals", action="store_true",
                    help="reject every goal at acceptance -- the adapter must map this to FAILED")
    ap.add_argument("--decel", type=float, default=2.0,
                    help="m/s^2; lower widens the confirmed-stop observation window")
    ap.add_argument("--accel", type=float, default=None,
                    help="m/s^2 acceleration limit, INDEPENDENT of --decel. Defaults to "
                         "--decel, which reproduces the previous single-rate behaviour "
                         "exactly. Set it HIGHER than --decel to get a robot that reaches "
                         "speed quickly and halts slowly -- the only configuration that "
                         "distinguishes a CONFIRMED stop from an announced one, because "
                         "--decel alone also caps the speed ever reached.")
    ap.add_argument("--cruise", type=float, default=CRUISE_SPEED,
                    help=f"m/s the built-in controller drives at (default {CRUISE_SPEED}). "
                         "Below r1_smoke's 0.35 m/s cruise floor it simulates a platform "
                         "whose driver caps linear speed, as the MentorPi's does at 0.2. "
                         "ACTIVE mode only.")
    ap.add_argument("--stall-after", type=float, default=0.0,
                    help="seconds; stop making progress mid-goal without terminating it")
    ap.add_argument("--passive", action="store_true",
                    help="R2 mode: no NavigateToPose action server and no built-in controller. "
                         "Nav2 owns the action and drives via /cmd_vel; this node is a pure plant.")
    return ap


def declared_flags() -> set[str]:
    """Every option string the node implements."""
    return {s for a in build_parser()._actions for s in a.option_strings} - {"-h", "--help"}


def split_ros_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """(ours, rclpy's). rclpy's are a `--ros-args` section, ended by `--` or the end."""
    ours: list[str] = []
    ros: list[str] = []
    in_ros = False
    for tok in argv:
        if tok == "--ros-args":
            in_ros = True
        elif in_ros and tok == "--":
            ros.append(tok)
            in_ros = False
            continue
        (ros if in_ros else ours).append(tok)
    return ours, ros


def _given(tokens: list[str], flag: str) -> bool:
    return any(t == flag or t.startswith(flag + "=") for t in tokens)


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse the node's command line or raise ArgsError saying what was refused."""
    ours, ros = split_ros_args(argv)
    args = build_parser().parse_args(ours)
    if args.accel is None:
        args.accel = args.decel  # single-rate default: behaviour identical to before
    if args.cruise <= 0.0:
        raise ArgsError(f"--cruise must be > 0 m/s, got {args.cruise}")
    if args.passive:
        dead = [f for f in ACTIVE_ONLY if _given(ours, f)]
        if dead:
            raise ArgsError(
                f"{' '.join(dead)} with --passive: --passive switches off the built-in "
                "controller and the NavigateToPose server these act on, so the run would "
                "not have the condition it names")
    return args, ros
