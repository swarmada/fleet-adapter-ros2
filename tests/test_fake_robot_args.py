# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""fake_robot_node must refuse an argument it does not implement.

`fake_robot_node.py` parsed with `parse_known_args()` and handed the leftovers to rclpy,
which ignores anything outside `--ros-args`. `r1_smoke`'s gate called `--cruise`
scoreable; the node had no `--cruise`. So `ROBOT_ARGS="--cruise 0.2"` ran the plant at
0.5 m/s and printed a result under a condition that never applied.

`--cruise` is now implemented, and these tests pin that it takes effect. The rest pin
the part that would have caught it at the time: anything the node does not implement is
refused, with the reason, before a node exists.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

_TESTBED = pathlib.Path(__file__).resolve().parents[1] / "testbed"
if str(_TESTBED) not in sys.path:
    sys.path.insert(0, str(_TESTBED))

import fake_robot_args as fra  # noqa: E402


# ── --cruise is implemented ──────────────────────────────────────────────────

def test_cruise_is_parsed_and_defaults_to_the_old_constant() -> None:
    args, ros = fra.parse(["--cruise", "0.2", "--decel", "1.0"])
    assert args.cruise == 0.2 and ros == []
    assert fra.parse(["--decel", "1.0"])[0].cruise == fra.CRUISE_SPEED == 0.5


def test_the_controller_drives_at_cruise_not_at_a_constant() -> None:
    """The flag has to reach the physics, not just the namespace: the controller's
    straight-line branch returns args.cruise, and no CRUISE_SPEED constant is left in the
    node to drift from it."""
    src = (_TESTBED / "fake_robot_node.py").read_text()
    tree = ast.parse(src)
    ctrl = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_controller")
    returns = [ast.unparse(r.value) for r in ast.walk(ctrl) if isinstance(r, ast.Return)]
    assert "(self._args.cruise, 0.0)" in returns
    assert "CRUISE_SPEED" not in src


@pytest.mark.parametrize("bad", [["--cruise", "0"], ["--cruise", "-0.2"]])
def test_a_cruise_that_cannot_drive_is_refused(bad) -> None:
    with pytest.raises(fra.ArgsError, match="--cruise must be > 0"):
        fra.parse(bad)


# ── anything not implemented is refused ─────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["--decel", "1.0", "--max-speed", "0.2"],     # a flag the node never had
    ["--decel", "1.0", "--crusie", "0.2"],        # a typo of one it has
    ["--decel", "1.0", "0.2"],                    # a stray operand
])
def test_an_unimplemented_argument_is_refused_not_passed_to_rclpy(argv) -> None:
    """The shape of it: before the fix every one of these reached rclpy, was
    ignored, and the run went ahead under the defaults."""
    with pytest.raises(fra.ArgsError, match="unrecognized arguments"):
        fra.parse(argv)


@pytest.mark.parametrize("argv", [["--stall", "3.0"], ["--cru", "0.2"], ["--reject"]])
def test_an_abbreviation_is_refused(argv) -> None:
    """argparse accepts unique prefixes by default. `--stall 3.0` then reached
    `--stall-after` while r1_smoke's gate, matching whole tokens, saw no fault at all."""
    with pytest.raises(fra.ArgsError):
        fra.parse(argv)


@pytest.mark.parametrize("flag", fra.ACTIVE_ONLY)
def test_passive_refuses_every_flag_it_switches_off(flag) -> None:
    """--passive has no built-in controller or action server: a flag acting on either
    would be accepted and do nothing, which is the same defect in R2's plant."""
    operand = {"--cruise": ["0.2"], "--goal-outcome": ["abort"],
               "--stall-after": ["3.0"], "--reject-goals": []}[flag]
    with pytest.raises(fra.ArgsError, match="with --passive"):
        fra.parse(["--decel", "1.0", "--passive", flag, *operand])
    with pytest.raises(fra.ArgsError, match="with --passive"):
        fra.parse(["--passive", f"{flag}={operand[0]}"] if operand else ["--passive", flag])


def test_r2s_own_command_line_is_accepted() -> None:
    """docker-compose.yml's r2-robot: `--decel 1.0 --passive`."""
    args, ros = fra.parse(["--decel", "1.0", "--passive"])
    assert args.passive and args.decel == args.accel == 1.0 and ros == []


# ── only a --ros-args section reaches rclpy ──────────────────────────────────

def test_ros_args_go_to_rclpy_and_are_not_ours_to_refuse() -> None:
    args, ros = fra.parse(["--decel", "1.0", "--ros-args", "-r", "__node:=fr",
                           "--log-level", "debug"])
    assert args.decel == 1.0
    assert ros == ["--ros-args", "-r", "__node:=fr", "--log-level", "debug"]


def test_a_double_dash_ends_the_ros_section_and_what_follows_is_parsed() -> None:
    args, ros = fra.parse(["--ros-args", "-r", "a:=b", "--", "--cruise", "0.2"])
    assert ros == ["--ros-args", "-r", "a:=b", "--"] and args.cruise == 0.2
    with pytest.raises(fra.ArgsError, match="unrecognized"):
        fra.parse(["--ros-args", "-r", "a:=b", "--", "--max-speed", "0.2"])


# ── the node uses this parser, and only this parser ──────────────────────────

def test_the_node_no_longer_parses_known_args() -> None:
    """parse_known_args() is what let an unimplemented flag through. The node must call
    fake_robot_args.parse() and must not build a second parser of its own."""
    tree = ast.parse((_TESTBED / "fake_robot_node.py").read_text())
    called = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "parse" in called
    assert not {c for c in called if c.endswith("parse_known_args")}
    assert "argparse.ArgumentParser" not in called
