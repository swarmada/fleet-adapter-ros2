# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A harness may not bring the binding up on the adapter's behalf.

`r1_smoke.py` called `binding.start()` itself. The shipped adapter never called it, so
`--binding ros2` constructed a binding that talked to nothing while the harness started
one by hand and reported eight passing assertions against a transport the adapter had
never opened. **That is how a severity-1 defect survived a conformance run**, and the defect is not
"a missing call" — it is a harness supplying the step it exists to test.

The distinction this file enforces:

  INSTRUMENT  drives the binding directly on purpose. It is MEASURING the binding, so
              bringing the transport up is part of its apparatus and there is no
              measurement without it. It scores nothing about the adapter.
  HARNESS     SCORES the adapter — it passes, fails, or nominates a registry standing.
              A harness that starts the binding is scoring a configuration the adapter
              cannot produce, so its bring-up must go through `adapter.bring_up_robot()`,
              the same call every real run makes.

A comment saying which is which decays; this test does not. Any NEW `start()` call in
`testbed/` fails here until it is classified, and any classified instrument that grows an
assertion has to move its bring-up in the same commit.

Pure AST over the source text: no rclpy, no import of the testbed modules, no container.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_TESTBED = pathlib.Path(__file__).resolve().parents[1] / "testbed"

# The instruments, each with the argument that makes it one. Keys are filenames; the
# value is what the file measures, and is the thing to re-read before adding a name here.
INSTRUMENTS = {
    "r2_cancel_latency.py":
        "measures the cancel -> confirmed-rest distribution through the binding and "
        "Nav2's controller pipeline; it asserts nothing about the adapter, and an "
        "un-started binding cannot produce a sample at all",
    "r2_transport_check.py":
        "one-shot diagnosis of the binding's TRANSPORT; no assertion, no exit code, no "
        "verdict — without start() every reading below it is a tautology",
    "r1_step5_probe.py":
        "observation-only probe; exits 0 on every outcome by design, including the hang "
        "r1_smoke cannot report without failing the run",
    "goal_response_probe.py":
        "measures the NAVIGATION SERVER's goal answer time, to derive the value "
        "--goal-response-bound has no default for; it never constructs a Nav2Binding at "
        "all, and the start() the sweep finds is an rclpy spin THREAD, not a binding -- "
        "which is why it is listed here rather than left to fail as unclassified",
}

# The harnesses, which score the adapter and must therefore use the adapter's bring-up.
HARNESSES = ("r1_smoke.py", "r2_smoke.py")


def _binding_starts(path: pathlib.Path) -> list[int]:
    """Line numbers of `<something>.start()` calls. Docstrings and comments are invisible
    to the AST, which is what makes this a test of behaviour rather than of prose."""
    tree = ast.parse(path.read_text())
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "start"]


@pytest.mark.parametrize("name", HARNESSES)
def test_a_harness_does_not_start_the_binding_itself(name: str) -> None:
    path = _TESTBED / name
    starts = _binding_starts(path)
    assert not starts, (
        f"{name} calls .start() at line(s) {starts}. It SCORES the adapter, so it must "
        f"bring up through adapter.bring_up_robot() — starting the binding itself scores "
        f"a configuration the adapter cannot produce, which is the inert adapter "
        f"returning.")


@pytest.mark.parametrize("name", HARNESSES)
def test_a_harness_brings_up_through_the_adapter(name: str) -> None:
    """The other half: not calling `start()` is not the same as bringing the robot up.
    A harness that skipped bring-up entirely would fail differently and just as quietly."""
    assert "bring_up_robot()" in (_TESTBED / name).read_text(), (
        f"{name} must call adapter.bring_up_robot() — the same method run_control_stream() "
        f"calls on every real run, so there is exactly one bring-up implementation.")


def test_every_start_in_the_testbed_is_a_classified_instrument() -> None:
    """The sweep. A new file that starts the binding is unclassified until someone decides
    which it is, and an unclassified one is the state r1_smoke shipped in for months."""
    found = {p.name: _binding_starts(p) for p in sorted(_TESTBED.glob("*.py"))
             if _binding_starts(p)}
    assert set(found) == set(INSTRUMENTS), (
        f"testbed files calling .start(): {sorted(found)}. Declared instruments: "
        f"{sorted(INSTRUMENTS)}. Classify the difference — instrument (measures the "
        f"binding; add it above WITH the argument) or harness (scores the adapter; move "
        f"its bring-up to adapter.bring_up_robot()).")


@pytest.mark.parametrize("name", sorted(INSTRUMENTS))
def test_an_instrument_says_in_the_file_why_it_is_one(name: str) -> None:
    """The classification has to be readable where the `start()` is, not only here. The
    next person to touch these files is reading them, not this test."""
    text = (_TESTBED / name).read_text()
    assert "INSTRUMENT" in text or "instrument" in text, (
        f"{name} keeps its own bring-up but does not say why. Record the argument in the "
        f"module docstring: {INSTRUMENTS[name]}.")
