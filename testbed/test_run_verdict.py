#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The degraded run must FAIL and the healthy run must still PASS.

Both cases are exercised deliberately. A guard that only ever fails is as useless as one
that never does, and four separate defects in this repository were nothing but a check
that passed without having run.

The two runs below are the pair that produced this verdict, transcribed from the run logs
-- same command, same tree, differing only in host load. No ROS or Docker needed:
the verdict is a pure function of the numbers.

    python3 testbed/test_run_verdict.py     (or: pytest testbed/test_run_verdict.py)
"""

from __future__ import annotations

from run_verdict import KINEMATIC_MULTIPLE, assess, kinematic_floor, summary

# --- the degraded run: load avg 20.56, testbed 724efae185e0 --------------------------
# /tf start 56.5 Hz, end 42.5 Hz (ratio 0.75) -- BOTH TF conditions passed.
DEGRADED_SAMPLES = [(1.098, 0.350), (11.328, 0.360), (8.424, 0.400)]
DEGRADED_ATTEMPTED = 6          # 3 sampled, 3 skipped at "peak seen 0.000 m/s"

# --- the healthy run: load avg 11.39, testbed 9df1d1dea89e ---------------------------
# /tf start 59.0 Hz, end 60.0 Hz (ratio 1.02). 5 of 6 sampled, 0.539 - 0.701 s.
HEALTHY_SAMPLES = [(0.539, 0.383), (0.596, 0.362), (0.701, 0.383),
                   (0.612, 0.374), (0.588, 0.368)]
HEALTHY_ATTEMPTED = 6

# --- the run taken on the new readiness gate, for a second healthy case --------------
TODAY_SAMPLES = [(0.601, 0.375), (0.611, 0.379), (0.607, 0.396),
                 (0.598, 0.381), (0.599, 0.366), (0.593, 0.395)]
TODAY_ATTEMPTED = 6

# --- the run of 2026-09-02, taken through all three gates. Its worst sample is
# 2.2x its floor -- the highest healthy value on record, so it is the one that keeps
# test_the_threshold_sits_in_an_empty_region honest.
SEPT_02_SAMPLES = [(0.710, 0.380), (0.694, 0.360), (0.559, 0.396),
              (0.815, 0.377), (0.733, 0.378), (0.609, 0.376)]
SEPT_02_ATTEMPTED = 6


def test_the_2026_09_02_run_is_accepted():
    problems = assess(SEPT_02_SAMPLES, SEPT_02_ATTEMPTED)
    assert not problems, f"the 2026-09-02 run was refused: {problems}"
    print("  2026-09-02 run accepted (6/6 sampled, 0.559-0.815 s)")


def test_degraded_run_is_refused():
    problems = assess(DEGRADED_SAMPLES, DEGRADED_ATTEMPTED)
    assert problems, "the degraded run was accepted -- the defect is not fixed"
    joined = " ".join(problems)
    assert "kinematic floor" in joined, "the absurd samples were not caught"
    assert "no sample" in joined, "the 50% skip rate was not caught"
    print(f"  degraded run refused on {len(problems)} ground(s):")
    for p in problems:
        print(f"    - {p}")


def test_healthy_run_is_accepted():
    problems = assess(HEALTHY_SAMPLES, HEALTHY_ATTEMPTED)
    assert not problems, f"a healthy run was refused: {problems}"
    print("  healthy run accepted (5/6 sampled, 0.539-0.701 s)")


def test_todays_run_is_accepted():
    problems = assess(TODAY_SAMPLES, TODAY_ATTEMPTED)
    assert not problems, f"today's run was refused: {problems}"
    print("  the readiness-gate run accepted (6/6 sampled, 0.593-0.611 s)")


def test_the_threshold_sits_in_an_empty_region():
    """The multiple is a line across a gap, not a guess about where slow begins."""
    healthy = [lat / kinematic_floor(spd)
               for lat, spd in HEALTHY_SAMPLES + TODAY_SAMPLES + SEPT_02_SAMPLES]
    degraded = [lat / kinematic_floor(spd) for lat, spd in DEGRADED_SAMPLES]
    assert max(healthy) < KINEMATIC_MULTIPLE < max(degraded)
    print(f"  healthy worst {max(healthy):.2f}x < limit {KINEMATIC_MULTIPLE:.0f}x "
          f"< degraded worst {max(degraded):.1f}x")


def test_each_gate_fires_on_its_own():
    """Neither gate is carried by the other."""
    # Absurd latencies, but every run sampled -> only the kinematic gate fires.
    only_kinematic = assess([(9.0, 0.4)] * 6, 6)
    assert len(only_kinematic) == 1 and "kinematic floor" in only_kinematic[0]
    # Sane latencies, but half the runs skipped -> only the skip gate fires.
    only_skip = assess([(0.6, 0.38)] * 3, 6)
    assert len(only_skip) == 1 and "no sample" in only_skip[0]
    print("  both gates fire independently")


def test_summary_is_printable_for_both():
    assert "worst sample" in summary(HEALTHY_SAMPLES, HEALTHY_ATTEMPTED)
    assert "worst sample" in summary(DEGRADED_SAMPLES, DEGRADED_ATTEMPTED)
    print(f"  degraded: {summary(DEGRADED_SAMPLES, DEGRADED_ATTEMPTED)}")
    print(f"  healthy:  {summary(HEALTHY_SAMPLES, HEALTHY_ATTEMPTED)}")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"{name}:")
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL: {exc}")
    print()
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
