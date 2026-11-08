# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A veto-only calibration record: T for the scan channel when it may only veto rest.

When the operator confirms the handover's rest, the scan channel stays on as a veto: any moving
and any unknown keep the gate shut, and it never confirms. Its T still comes from the robot, from
stationary recordings only: s0 is the largest S over at least two stationary runs of at least
MIN_VETO_RUN_S each, and T = 1.5 x s0. The record says ``claim: veto-only`` and makes no motion
claim: no motion run, no miss count, no time to rest.

The two claims never stand in for each other. The operator source takes a veto-only record only,
and the scan source (which lets the channel confirm rest) never takes one: a record measured
without a motion run is no evidence that the channel sees motion.
"""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys

import pytest
from scan_room import Lidar, Plant, record
from test_scan_rest_record import _write_run, good_record

from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2.scan_rest import CalibrationRefused

REPO = pathlib.Path(__file__).resolve().parents[1]


def good_veto() -> dict:
    return {
        "format": sr.RECORD_FORMAT, "claim": sr.VETO_CLAIM, "definition": dict(sr.CONSTANTS),
        "s0_m": 0.011, "threshold_m": 0.0165, "calibration_runs": ["still-1", "still-2"],
        "held_out_not_shown_fraction": 0.0034,
        "runs": [
            {"name": "still-1", "kind": "A-cal", "duration_s": 119.9, "scans": 1200,
             "s_max_m": 0.006, "windows": 1194, "not_shown": 4},
            {"name": "still-2", "kind": "A-cal", "duration_s": 119.8, "scans": 1199,
             "s_max_m": 0.011, "windows": 1193, "not_shown": 4},
            {"name": "held-1", "kind": "A-held", "duration_s": 119.5, "scans": 1190,
             "s_max_m": 0.011, "windows": 1185, "not_shown": 4},
        ]}


def _refused(doc, *words) -> str:
    with pytest.raises(CalibrationRefused) as e:
        sr.check_veto_record(doc)
    for w in words:
        assert w in str(e.value), str(e.value)
    return str(e.value)


def _with(doc, path, value):
    doc = copy.deepcopy(doc)
    *head, last = path
    node = doc
    for k in head:
        node = node[k]
    if value is _DROP:
        del node[last]
    else:
        node[last] = value
    return doc


_DROP = object()


def test_a_good_veto_record_gives_T_and_says_it_is_a_veto_only() -> None:
    t, summary = sr.check_veto_record(good_veto())
    assert t == 0.0165
    assert "veto-only" in summary and "never confirms rest" in summary
    assert "0.0165" in summary and "0.0110" in summary


# ── what a veto-only record must show ────────────────────────────────────────

@pytest.mark.parametrize("doc,words", [
    ({"format": "nope"}, ("not a scan-rest-calibration/1 record",)),
    ([], ("not a scan-rest-calibration/1 record",)),
    (_with(good_veto(), ["definition", "lag_s"], 0.3), ("different definition",)),
    (_with(good_veto(), ["threshold_m"], 0.02), ("is not 1.5 x s0",)),
    (_with(good_veto(), ["s0_m"], 0.006), ("is not the largest S",)),
    (_with(good_veto(), ["s0_m"], float("nan")), ("no s0_m or threshold_m",)),
    (_with(good_veto(), ["threshold_m"], True), ("no s0_m or threshold_m",)),
    (_with(good_veto(), ["runs"], "x"), ("no runs",)),
])
def test_a_veto_record_that_does_not_show_its_T_is_refused(doc, words) -> None:
    _refused(doc, *words)


def test_two_stationary_runs_are_needed() -> None:
    doc = _with(good_veto(), ["runs", 1, "kind"], "A-held")
    _refused(doc, "1 stationary calibration runs; 2 needed")


@pytest.mark.parametrize("duration", [99.9, 0.0, float("nan"), float("inf"), "120", True, None])
def test_each_stationary_run_must_last_at_least_the_minimum(duration) -> None:
    doc = good_veto()
    if duration is None:
        del doc["runs"][0]["duration_s"]
    else:
        doc["runs"][0]["duration_s"] = duration
    _refused(doc, "still-1", f"{sr.MIN_VETO_RUN_S:g} s")


def test_exactly_the_minimum_is_enough() -> None:
    doc = _with(good_veto(), ["runs", 0, "duration_s"], sr.MIN_VETO_RUN_S)
    assert sr.check_veto_record(doc)[0] == 0.0165


@pytest.mark.parametrize("kind", ["B", "C", "A-adv", "other"])
def test_a_veto_record_makes_no_motion_claim_and_holds_no_motion_run(kind) -> None:
    doc = good_veto()
    doc["runs"].append({"name": "drive-1", "kind": kind, "duration_s": 120.0,
                        "motion_windows": 95, "misses": 0})
    _refused(doc, "no motion claim", "drive-1")


@pytest.mark.parametrize("key", ["motion_windows", "misses", "time_to_rest_s",
                                 "time_to_rest_after_hold_s", "measured_speed"])
def test_a_stationary_run_carrying_a_motion_field_is_refused(key) -> None:
    doc = _with(good_veto(), ["runs", 2, key], 0)
    _refused(doc, "no motion claim", key)


# ── the two claims never stand in for each other ─────────────────────────────

@pytest.mark.parametrize("claim", [_DROP, "measured", "", None, "veto"])
def test_the_operator_source_never_takes_a_record_that_is_not_veto_only(claim) -> None:
    doc = _with(good_veto(), ["claim"], claim)
    _refused(doc, "not a veto-only record")


def test_a_measured_record_is_not_taken_as_a_veto_only_one() -> None:
    sr.check_record(good_record())                       # it is a good measured record
    _refused(good_record(), "not a veto-only record")


def test_a_veto_only_record_is_never_evidence_that_the_channel_is_measured() -> None:
    with pytest.raises(CalibrationRefused, match="veto-only"):
        sr.check_record(good_veto())
    # Not even with the motion runs of a measured record written into it.
    doc = good_record()
    doc["claim"] = sr.VETO_CLAIM
    with pytest.raises(CalibrationRefused, match="veto-only"):
        sr.check_record(doc)


def test_load_veto_record_reads_a_file_and_names_its_md5(tmp_path) -> None:
    p = tmp_path / "veto.json"
    p.write_text(json.dumps(good_veto()))
    t, summary, md5 = sr.load_veto_record(str(p))
    assert t == 0.0165 and "veto-only" in summary and len(md5) == 32
    with pytest.raises(CalibrationRefused, match="veto-only"):
        sr.load_record(str(p))
    with pytest.raises(CalibrationRefused, match="no calibration record"):
        sr.load_veto_record(str(tmp_path / "absent.json"))
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(CalibrationRefused, match="not JSON"):
        sr.load_veto_record(str(tmp_path / "bad.json"))


# ── calibrate --veto-only, from stationary recordings ────────────────────────

STILL = [(0.0, 0.0, 0.0), (200.0, 0.0, 0.0)]


def _still_run(np, tmp_path, name, seconds, seed, x=0.0, y=0.0, yaw=0.0):
    scans = record(Plant(x=x, y=y, yaw=yaw), Lidar(seed=seed, beams=240), seconds)
    _write_run(np, tmp_path / f"{name}.npz", scans, STILL)
    return scans


def test_calibrate_veto_only_writes_a_record_the_operator_source_accepts(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    _still_run(np, tmp_path, "still-1", 101.0, 21)
    _still_run(np, tmp_path, "still-2", 101.0, 22, x=0.6, y=0.4, yaw=1.0)
    _still_run(np, tmp_path, "held-1", 30.0, 23, x=-0.5, y=0.8, yaw=2.0)
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"name": "still-1", "kind": "A-cal", "file": "still-1.npz"},
        {"name": "still-2", "kind": "A-cal", "file": "still-2.npz"},
        {"name": "held-1", "kind": "A-held", "file": "held-1.npz"}]}))
    out = tmp_path / "veto.json"
    res = subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.scan_rest", "calibrate",
                          "--veto-only", str(tmp_path / "manifest.json"), "--out", str(out)],
                         cwd=REPO, capture_output=True, text=True, timeout=600, check=True)
    assert "the operator source would accept it as a veto-only record" in res.stdout
    rec = json.loads(out.read_text())
    assert rec["claim"] == sr.VETO_CLAIM
    t, _, _ = sr.load_veto_record(str(out))
    assert t == pytest.approx(1.5 * rec["s0_m"])
    cal = [r for r in rec["runs"] if r["kind"] == "A-cal"]
    assert rec["s0_m"] == max(r["s_max_m"] for r in cal)
    assert all(r["duration_s"] >= sr.MIN_VETO_RUN_S for r in cal)
    assert all(len(r["md5"]) == 32 for r in rec["runs"])
    held = next(r for r in rec["runs"] if r["kind"] == "A-held")
    assert held["windows"] > 0 and "not_shown" in held
    for r in rec["runs"]:
        assert not set(r) & set(sr.MOTION_FIELDS)
    with pytest.raises(CalibrationRefused, match="veto-only"):
        sr.load_record(str(out))


def test_calibrate_veto_only_says_refuse_for_a_run_shorter_than_the_minimum(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    _still_run(np, tmp_path, "still-1", 101.0, 31)
    _still_run(np, tmp_path, "still-2", 40.0, 32, x=0.6, y=0.4, yaw=1.0)
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"name": "still-1", "kind": "A-cal", "file": "still-1.npz"},
        {"name": "still-2", "kind": "A-cal", "file": "still-2.npz"}]}))
    out = tmp_path / "veto.json"
    res = subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.scan_rest", "calibrate",
                          "--veto-only", str(tmp_path / "manifest.json"), "--out", str(out)],
                         cwd=REPO, capture_output=True, text=True, timeout=600, check=True)
    assert "would REFUSE it" in res.stdout and "still-2" in res.stdout
    with pytest.raises(CalibrationRefused, match="shorter than"):
        sr.load_veto_record(str(out))


@pytest.mark.parametrize("kind", ["B", "C", "A-adv"])
def test_calibrate_veto_only_never_reads_a_motion_recording(tmp_path, kind) -> None:
    """The stationary recordings may set a veto's T; a motion recording never feeds it."""
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"name": "still-1", "kind": "A-cal", "file": "still-1.npz"},
        {"name": "still-2", "kind": "A-cal", "file": "still-2.npz"},
        {"name": "drive-1", "kind": kind, "file": "drive-1.npz"}]}))
    out = tmp_path / "veto.json"
    res = subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.scan_rest", "calibrate",
                          "--veto-only", str(tmp_path / "manifest.json"), "--out", str(out)],
                         cwd=REPO, capture_output=True, text=True, timeout=120, check=False)
    assert res.returncode != 0 and "drive-1" in res.stderr and "stationary" in res.stderr
    assert not out.exists()


def test_without_veto_only_calibrate_is_unchanged_and_writes_no_claim(tmp_path) -> None:
    """The measured record keeps its shape: no claim field, so a record written before reads
    as it did."""
    np = pytest.importorskip("numpy")
    _still_run(np, tmp_path, "still-1", 8.0, 41)
    _still_run(np, tmp_path, "still-2", 8.0, 42, x=0.6, y=0.4, yaw=1.0)
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"name": "still-1", "kind": "A-cal", "file": "still-1.npz"},
        {"name": "still-2", "kind": "A-cal", "file": "still-2.npz"}]}))
    rec = sr.calibrate(str(tmp_path / "manifest.json"))
    assert "claim" not in rec
    with pytest.raises(CalibrationRefused, match="no motion"):
        sr.check_record(rec)
    with pytest.raises(CalibrationRefused, match="not a veto-only record"):
        sr.check_veto_record(rec)
