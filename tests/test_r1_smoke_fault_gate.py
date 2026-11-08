# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`r1_smoke` must not report a pass for a fault that never fired.

Two of the four documented R1 fault cases cannot fire inside the window
`r1_smoke.py` drives: it cancels the 3.0 m goal unconditionally at ~0.46 s /
~0.16 m, while `--goal-outcome abort` fires in the robot's *completion* branch
and `--stall-after` is scheduled on goal time. The file asserted the happy path
and only the happy path, so when the fault did not arrive the happy path held
and it printed ``ALL R1 ASSERTIONS PASSED`` — which is how two rows of
`R1_R2_RESULTS_20260902.md` came to record a deceleration figure against a fault
that was never present.

These tests pin the gate, not the assertions. The gate is pure and runs *before*
``Nav2Binding.start()``, so it is provable with no ROS 2 install — which is the
point: the check that protects the stack must not need the stack.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# r1_smoke lives in testbed/ and imports its neighbours (speed_gate,
# testbed_provenance) by bare name, as it does inside the container.
_TESTBED = pathlib.Path(__file__).resolve().parents[1] / "testbed"
if str(_TESTBED) not in sys.path:
    sys.path.insert(0, str(_TESTBED))

r1_smoke = pytest.importorskip("r1_smoke")
import fake_robot_args  # noqa: E402


# ── what counts as a configured fault ────────────────────────────────────────

@pytest.mark.parametrize("args,expected", [
    ("", []),
    ("--decel 1.0", ["--decel"]),
    ("--decel 1.0 --reject-goals", ["--decel", "--reject-goals"]),
    ("--decel 1.0 --goal-outcome abort", ["--decel", "--goal-outcome"]),
    ("--decel 1.0 --stall-after 3.0", ["--decel", "--stall-after"]),
    # The flag's VALUE must never be mistaken for a flag: `abort` and `3.0` are
    # operands, and counting them would make the gate's report wrong about what
    # was configured even when its verdict happened to be right.
    ("--goal-outcome abort --stall-after 3.0",
     ["--goal-outcome", "--stall-after"]),
])
def test_configured_faults_reads_robot_args(args, expected) -> None:
    assert r1_smoke.configured_faults(args) == expected


# ── the three that must be refused ───────────────────────────────────────────

@pytest.mark.parametrize("fault", ["--reject-goals", "--goal-outcome abort",
                                   "--stall-after 3.0", "--stall-after=3.0"])
def test_the_unfireable_faults_are_refused(fault, capsys) -> None:
    """Each fault injection is unscoreable by this harness, and says so. The operands
    are ones the robot accepts, so this is the gate's own refusal and not the robot's
    parser refusing an operand -- and the `=` spelling is the same flag."""
    args = f"--decel 1.0 {fault}"
    flag = fault.split()[0].split("=")[0]
    assert r1_smoke.unscoreable(args) == [flag]
    assert r1_smoke.fault_gate(args) != 0, (
        f"{fault} was configured and the gate let the run proceed; it would have "
        f"printed ALL R1 ASSERTIONS PASSED without the fault arriving"
    )
    assert "CANNOT SCORE IT" in capsys.readouterr().out


def test_a_refused_run_never_prints_the_pass_line(capsys) -> None:
    """The exact regression: the pass line must not appear, and the operator must
    be told which fault was configured and what to run instead."""
    code = r1_smoke.fault_gate("--decel 1.0 --goal-outcome abort")
    out = capsys.readouterr().out
    assert code != 0
    assert "ALL R1 ASSERTIONS PASSED" not in out
    assert "--goal-outcome" in out            # which fault
    assert "CANNOT SCORE IT" in out           # and that it cannot be scored
    assert r1_smoke.PROBE in out              # and what to run instead


# ── and the cases that must still run ────────────────────────────────────────

@pytest.mark.parametrize("args", ["", "--decel 1.0", "--decel 0.3", "--accel 0.5 --cruise 0.4",
                                  "--cruise 0.2 --decel 1.0"])
def test_scoreable_conditions_are_not_refused(args) -> None:
    """A gate that refused everything would be as useless as one that refused
    nothing. `--decel` and friends change the condition every assertion already
    runs under, so the happy path genuinely scores them."""
    assert r1_smoke.unscoreable(args) == []
    assert r1_smoke.fault_gate(args) == 0


def test_every_fault_case_declares_a_reason_when_it_cannot_score() -> None:
    """An unscoreable verdict with no reason is an unexplained refusal, which is
    the failure mode this item is about pointed the other way."""
    for flag, case in r1_smoke.FAULT_CASES.items():
        if not case["scoreable"]:
            assert str(case["reason"]).strip(), f"{flag} refuses without saying why"


# ── the gate may only vouch for what the robot implements ───────────────────


def test_every_fault_case_is_a_flag_the_robot_implements() -> None:
    """The test that would have caught it when `--cruise` was added to the table:
    the gate called it scoreable, and fake_robot_node.py had no such flag."""
    assert set(r1_smoke.FAULT_CASES) <= fake_robot_args.declared_flags()


def test_every_robot_flag_is_classified_by_the_gate() -> None:
    """A new robot flag must be put in FAULT_CASES -- scoreable or not, with a reason --
    rather than reach R1 unclassified. --passive is R2's plant mode, not an R1 condition."""
    assert fake_robot_args.declared_flags() - set(r1_smoke.FAULT_CASES) == {"--passive"}


@pytest.mark.parametrize("args", [
    "--decel 1.0 --max-speed 0.2",   # not a flag the robot has
    "--decel 1.0 --stall 3.0",       # an abbreviation the gate would not have matched
    "--decel 1.0 --cruise 0",        # a cruise the plant cannot drive at
    "--passive --cruise 0.2",        # a flag --passive switches off
])
def test_the_gate_refuses_what_the_robot_would_refuse(args, capsys) -> None:
    code = r1_smoke.fault_gate(args)
    out = capsys.readouterr().out
    assert code != 0
    assert "ALL R1 ASSERTIONS PASSED" not in out
    assert "fake_robot_node.py does not implement ROBOT_ARGS" in out
    assert "R1 NOT RUN" in out
