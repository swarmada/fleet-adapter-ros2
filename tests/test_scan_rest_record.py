# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The scan channel is declared measured only by a calibration record, and the replay over a
recording computes the same S_t as an implementation written separately.

The record is what the Air's session measures: s0 and T from two stationary runs, and per run the
counts that show the channel sees motion. A record that was measured with another definition, whose
T is not 1.5 x s0, whose motion runs are short, or with any miss, is no evidence: it is refused, and
the refusal names what is missing.

The parity check: ``tests/fixtures/scan_rest_golden.json`` holds a recording (scans of 449, 450 and
451 beams, NaN, inf and out-of-cap ranges, a gap, a scan that cannot pair) and the S_t a second
implementation computes from it (`_second_series`, numpy, all-pairs by bearing; not the module's
code). The module must agree with it to 1e-6, here without numpy, and the replay command must too
on the same recording written as ``.npz``.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import subprocess
import sys

import pytest
from scan_room import Lidar, Plant, record

from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2.scan_rest import CalibrationRefused, Scan

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "scan_rest_golden.json"
REPO = pathlib.Path(__file__).resolve().parents[1]


def good_record() -> dict:
    return {
        "format": sr.RECORD_FORMAT, "definition": dict(sr.CONSTANTS),
        "s0_m": 0.011, "threshold_m": 0.0165, "calibration_runs": ["still-1", "still-2"],
        "held_out_not_shown_fraction": 0.02,
        "runs": [
            {"name": "still-1", "kind": "A-cal", "s_max_m": 0.011, "windows": 1190, "not_shown": 0},
            {"name": "still-2", "kind": "A-cal", "s_max_m": 0.0104, "windows": 1190, "not_shown": 0},
            {"name": "held-1", "kind": "A-held", "windows": 1190, "not_shown": 20},
            {"name": "drive-1", "kind": "B", "motion_windows": 95, "misses": 0},
            {"name": "drive-2", "kind": "B", "motion_windows": 20, "misses": 0},
            {"name": "stop-1", "kind": "C", "time_to_rest_s": 1.7},
        ]}


def test_a_good_record_gives_T() -> None:
    t, summary = sr.check_record(good_record())
    assert t == 0.0165 and "0 misses" in summary


def _refused(doc, *words) -> None:
    with pytest.raises(CalibrationRefused) as e:
        sr.check_record(doc)
    for w in words:
        assert w in str(e.value), str(e.value)


def test_not_a_record_is_refused() -> None:
    _refused({"format": "something-else"}, "not a scan-rest-calibration/1 record")
    _refused([], "not a")


@pytest.mark.parametrize("key,value", [("lag_s", 0.4), ("window_s", 2.0), ("min_pairs", 100),
                                       ("cap_m", 8.0), ("t_factor", 2.0), ("pairing", "index")])
def test_a_record_for_another_definition_is_refused(key, value) -> None:
    doc = good_record()
    doc["definition"][key] = value
    _refused(doc, "different definition")


def test_T_not_one_and_a_half_s0_is_refused() -> None:
    doc = good_record()
    doc["threshold_m"] = 0.02
    _refused(doc, "is not 1.5 x s0")


def test_s0_not_the_largest_calibration_S_is_refused() -> None:
    doc = good_record()
    doc["s0_m"], doc["threshold_m"] = 0.0104, 0.0156
    _refused(doc, "not the largest S")


def test_fewer_than_two_calibration_runs_is_refused() -> None:
    doc = good_record()
    doc["runs"] = [r for r in doc["runs"] if r["name"] != "still-2"]
    _refused(doc, "1 stationary calibration runs")


def test_no_s0_or_T_is_refused() -> None:
    for k in ("s0_m", "threshold_m"):
        doc = good_record()
        del doc[k]
        _refused(doc, "no s0_m or threshold_m")


@pytest.mark.parametrize("run", ["drive-1", "held-1", "stop-1"])
def test_any_miss_in_any_run_is_refused(run) -> None:
    doc = good_record()
    next(r for r in doc["runs"] if r["name"] == run)["misses"] = 1
    _refused(doc, f"run {run!r} has 1 misses")


def test_a_motion_run_with_fewer_than_20_windows_is_refused() -> None:
    doc = good_record()
    doc["runs"][4]["motion_windows"] = 19
    _refused(doc, "'drive-2' has 19 motion windows, fewer than 20")


def test_a_motion_run_with_no_miss_count_is_refused() -> None:
    doc = good_record()
    del doc["runs"][3]["misses"]
    _refused(doc, "'drive-1' records no miss count")


def test_no_motion_run_is_refused() -> None:
    doc = good_record()
    doc["runs"] = [r for r in doc["runs"] if r["kind"] != "B"]
    _refused(doc, "no motion (B) run")


def test_a_missing_or_unreadable_file_is_refused(tmp_path) -> None:
    with pytest.raises(CalibrationRefused, match="no calibration record"):
        sr.load_record(str(tmp_path / "absent.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(CalibrationRefused, match="not JSON"):
        sr.load_record(str(bad))
    good = tmp_path / "good.json"
    good.write_text(json.dumps(good_record()))
    t, _, md5 = sr.load_record(str(good))
    assert t == 0.0165 and len(md5) == 32


# ── the parity check ─────────────────────────────────────────────────────────

def _golden_scans(doc) -> list[Scan]:
    return [Scan(s["t_s"], [math.nan if r == "nan" else math.inf if r == "inf" else r
                            for r in s["ranges"]],
                 s["range_min"], s["range_max"], s["angle_min"], s["angle_increment"])
            for s in doc["scans"]]


def test_the_module_matches_the_golden_S_to_1e_6() -> None:
    doc = json.loads(GOLDEN.read_text())
    got = sr.series(_golden_scans(doc))
    want = doc["S_m"]
    assert len(got) == len(want)
    assert sum(1 for w in want if w is not None) >= 10 and want.count(None) >= 5
    for (g, _), w in zip(got, want):
        assert (g is None) == (w is None)
        if w is not None:
            assert abs(g - w) <= 1e-6


def _second_series(t, ranges, nb, amin, ainc, rmin, rmax):
    """A second implementation of S_t, written from the definition, not from the module: all
    pairs by circular bearing distance at once, argmin (first on a tie), numpy throughout."""
    np = pytest.importorskip("numpy")
    out = []
    for i in range(len(t)):
        d = np.abs(t[:i] - (t[i] - sr.LAG_S))
        if not len(d) or d.min() > sr.LAG_TOL_S:
            out.append(None)
            continue
        k = int(np.argmin(d))
        bi = np.mod(amin[i] + np.arange(nb[i]) * ainc[i], 2 * np.pi)
        bk = np.mod(amin[k] + np.arange(nb[k]) * ainc[k], 2 * np.pi)
        dist = np.abs(bi[:, None] - bk[None, :])
        dist = np.minimum(dist, 2 * np.pi - dist)
        j = np.argmin(dist, axis=1)
        close = dist[np.arange(nb[i]), j] <= ainc[k] / 2
        ri = ranges[i, :nb[i]].astype(np.float64)
        rk = ranges[k, j].astype(np.float64)
        ok_i = np.isfinite(ri) & (ri >= rmin[i]) & (ri <= min(rmax[i], sr.CAP_M))
        ok_k = np.isfinite(rk) & (rk >= rmin[k]) & (rk <= min(rmax[k], sr.CAP_M))
        m = close & ok_i & ok_k
        out.append(float(np.median(np.abs(ri[m] - rk[m]))) if m.sum() >= sr.MIN_PAIRS else None)
    return out


def _as_arrays(doc):
    np = pytest.importorskip("numpy")
    scans = doc["scans"]
    bmax = max(len(s["ranges"]) for s in scans)
    ranges = np.full((len(scans), bmax), np.nan, dtype=np.float32)
    for i, s in enumerate(scans):
        ranges[i, :len(s["ranges"])] = [float(r) for r in s["ranges"]]

    def col(k, dt=np.float64):
        return np.array([s[k] for s in scans], dtype=dt)
    return {"t_s": col("t_s"), "ranges": ranges, "n_beams": col("n_beams", np.int32),
            "angle_min": col("angle_min"), "angle_increment": col("angle_increment"),
            "range_min": col("range_min"), "range_max": col("range_max")}


def test_the_golden_S_is_the_second_implementations() -> None:
    doc = json.loads(GOLDEN.read_text())
    a = _as_arrays(doc)
    want = _second_series(a["t_s"], a["ranges"], a["n_beams"], a["angle_min"],
                          a["angle_increment"], a["range_min"], a["range_max"])
    for w, g in zip(want, doc["S_m"]):
        assert (w is None) == (g is None) and (w is None or abs(w - g) <= 1e-12)


def test_the_replay_command_prints_the_same_S_over_the_recording(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    doc = json.loads(GOLDEN.read_text())
    run = tmp_path / "golden.npz"
    np.savez(run, **_as_arrays(doc), cmd_t_s=np.array([0.0]), cmd_vx=np.array([0.0]),
             cmd_wz=np.array([0.0]))
    out = subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.scan_rest", "replay",
                          str(run), "--threshold-m", "0.02"], cwd=REPO, capture_output=True,
                         text=True, timeout=120, check=True).stdout
    rows = [line.split("\t") for line in out.splitlines() if not line.startswith("#")]
    assert len(rows) == len(doc["S_m"])
    for row, w in zip(rows, doc["S_m"]):
        if w is None:
            assert row[2].startswith("undefined")
        else:
            assert abs(float(row[2]) - w) <= 1e-6
        assert row[3].split(" ")[0] in ("at_rest", "moving", "unknown")


def make_golden() -> dict:
    """The golden recording (run this module to rewrite the fixture)."""
    np = pytest.importorskip("numpy")
    lidar = Lidar(seed=91)
    plant = Plant()
    layouts = [(449, 0.0065), (450, 0.0), (451, 0.0087), (450, 0.0002)]
    scans = []
    for i in range(34):
        n, a0 = layouts[i % 4]
        if 10 <= i < 18:
            plant.true = (0.1, 0.0, 0.0)
        elif 18 <= i < 24:
            plant.true = (0.0, 0.0, 0.3)
        else:
            plant.true = (0.0, 0.0, 0.0)
        t = round(0.1 * i + (0.4 if i >= 28 else 0.0) + 0.003 * ((i * 7) % 5), 6)  # a gap at 28
        s = lidar.cast(plant, t, beams=n, angle_min=a0)
        plant.step(0.1)
        r = [float(np.float32(x)) for x in s.ranges]
        if i == 26:
            r = [math.nan] * n                                       # a NaN-only scan
        if i == 31:
            r = r[:100]                                              # a scan that cannot pair
            n = 100
        r = [7.5 if (k % 37 == 0 and math.isfinite(x)) else x for k, x in enumerate(r)]
        scans.append({"t_s": t, "n_beams": n, "angle_min": s.angle_min,
                      "angle_increment": 2 * math.pi / (len(s.ranges)),
                      "range_min": s.range_min, "range_max": s.range_max,
                      "ranges": ["nan" if math.isnan(x) else "inf" if math.isinf(x) else x
                                 for x in r]})
    doc = {"note": "synthetic; S_m from tests/test_scan_rest_record.py _second_series",
           "scans": scans}
    a = _as_arrays(doc)
    doc["S_m"] = _second_series(a["t_s"], a["ranges"], a["n_beams"], a["angle_min"],
                                a["angle_increment"], a["range_min"], a["range_max"])
    return doc


# ── calibrate, end to end, on synthetic recordings ───────────────────────────

def _write_run(np, path, scans, cmd):
    bmax = max(len(s.ranges) for s in scans)
    ranges = np.full((len(scans), bmax), np.nan, dtype=np.float32)
    for i, s in enumerate(scans):
        ranges[i, :len(s.ranges)] = s.ranges
    np.savez(path, t_s=np.array([s.stamp for s in scans]), ranges=ranges,
             n_beams=np.array([len(s.ranges) for s in scans]),
             angle_min=np.array([s.angle_min for s in scans]),
             angle_increment=np.array([s.angle_increment for s in scans]),
             range_min=np.array([s.range_min for s in scans]),
             range_max=np.array([s.range_max for s in scans]),
             cmd_t_s=np.array([c[0] for c in cmd]), cmd_vx=np.array([c[1] for c in cmd]),
             cmd_wz=np.array([c[2] for c in cmd]))


def test_calibrate_writes_a_record_the_adapter_accepts(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    still = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    runs = []

    def add(name, kind, scans, cmd, **extra):
        _write_run(np, tmp_path / f"{name}.npz", scans, cmd)
        runs.append({"name": name, "kind": kind, "file": f"{name}.npz", **extra})

    add("still-1", "A-cal", record(Plant(), Lidar(seed=1), 8.0), still)
    add("still-2", "A-cal", record(Plant(x=0.6, y=0.4, yaw=1.0), Lidar(seed=2), 8.0), still)
    add("held-1", "A-held", record(Plant(x=-0.5, y=0.8, yaw=2.0), Lidar(seed=3), 6.0), still)
    p = Plant(true=(0.1, 0.0, 0.0))
    add("drive-1", "B", record(p, Lidar(seed=4), 8.0), [(0.0, 0.1, 0.0), (7.9, 0.1, 0.0),
                                                   (8.0, 0.0, 0.0)], measured_speed=0.1)
    c = Plant(true=(0.2, 0.0, 0.0))
    scans = record(c, Lidar(seed=5), 2.0)
    c.true = (0.0, 0.0, 0.0)
    scans += record(c, Lidar(seed=6), 3.0, t0=2.0)
    add("stop-1", "C", scans, [(0.0, 0.2, 0.0), (1.99, 0.2, 0.0), (2.0, 0.0, 0.0)])
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    out = tmp_path / "record.json"
    res = subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.scan_rest", "calibrate",
                          str(tmp_path / "manifest.json"), "--out", str(out)], cwd=REPO,
                         capture_output=True, text=True, timeout=300, check=True)
    assert "the adapter would accept it" in res.stdout
    t, _, _ = sr.load_record(str(out))
    rec = json.loads(out.read_text())
    assert t == pytest.approx(1.5 * rec["s0_m"])
    b1 = next(r for r in rec["runs"] if r["name"] == "drive-1")
    assert b1["motion_windows"] >= 20 and b1["misses"] == 0
    c1 = next(r for r in rec["runs"] if r["name"] == "stop-1")
    assert 1.0 <= c1["time_to_rest_after_hold_s"] <= 2.0
    # The same recordings with a run that misses: the record is written, and refused.
    bad = copy.deepcopy(rec)
    bad["runs"][3]["misses"] = 2
    with pytest.raises(CalibrationRefused, match="misses"):
        sr.check_record(bad)


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(make_golden(), indent=None) + "\n")
    print(f"wrote {GOLDEN}")
