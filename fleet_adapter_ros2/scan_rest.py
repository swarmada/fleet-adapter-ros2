# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A rest channel measured by the lidar itself: how much the scan changes from one scan to the
one half a second before it. For the navigate-handover's rest wait only; never for a stop.

WHAT IT READS, AND WHAT IT NEVER READS. Scans: a stamp, the ranges, and each scan's own layout
(``angle_min``, ``angle_increment``, ``range_min``, ``range_max``; the beam count is the length of
the ranges, and it is not fixed). It takes no velocity command, no setpoint, no odometry, no pose
and no velocity, so nothing the drive was told to do enters its answer (ADR-0071 1: the channel
is independent of the command path because the command is not one of its inputs). If the drive
did not do what it was told -- a wheel slipped, the base was pushed, a motor stalled -- the room
in the scan moves or does not, and so does the answer.

THE DEFINITION (the claim; every number below is a module constant, not a flag, because changing
one changes what "at rest" means):

* **Pairs, by bearing.** A beam's bearing is ``angle_min + i * angle_increment`` (mod 2 pi). For a
  scan at t, the earlier scan is the one nearest to t - LAG_S (none within LAG_TOL_S: undefined).
  Each beam of the scan at t is paired with the earlier scan's nearest beam by bearing; the pair
  counts only when the bearings differ by at most half the earlier scan's increment and both
  ranges are finite and inside [range_min, min(range_max, CAP_M)] of their own scan.
* **S_t** (metres) is the median over counted pairs of |r_t - r_earlier|; undefined below
  MIN_PAIRS counted pairs.
* **Rest is shown** at t iff the scans in the last WINDOW_S number at least MIN_SCANS, have no gap
  above GAP_S between them, and S_u is defined and below the threshold for every one of them.
* **The threshold** T = T_FACTOR x s0, s0 the largest defined S_u over the stationary calibration
  runs. It comes only from a calibration record (`load_record`); there is no default.

THE ANSWER: `AT_REST`, `MOVING` or `UNKNOWN` with a reason. Everything that is not a shown rest is
"not shown": no scan, a stale last scan, a gap, too few scans, an undefined S, a stamp that is
missing or goes backwards. Refute wins: one defined S_u at or above T in the window is `MOVING`,
whatever any other source says.

TWO RULES ON TOP, NEVER INSTEAD (`RestChannel`):

* **The window post-dates the hold.** The window used lies entirely after the base was commanded
  to hold, so braking is in it. Time alone never confirms rest: this is a precondition on the
  scans, not a substitute for them.
* **A structure guard** (the aperture problem). A scan difference cannot see motion along a flat
  wall: in a corridor whose ends are beyond the cap, translating along the axis changes no range.
  So every scan in the window must show structure: valid beams over at least MIN_COVER_DEG of
  bearing, at least MIN_INSIDE_FRAC of them inside the cap, and no single wall direction (local
  line direction mod 180 degrees, in any WALL_BIN_DEG-wide arc) holding more than MAX_WALL_FRAC of
  them. Otherwise the answer is `UNKNOWN` (no structure). This is a conservative guard, not a
  proof that motion in every room is visible.

Pure: no ROS, no numpy in the core. ``python -m fleet_adapter_ros2.scan_rest replay|calibrate``
reads recordings (``.npz``, numpy needed there only).
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

# ── the definition (the maintainer's, 2026-10-05, amended to pair beams by bearing) ─────────
LAG_S = 0.5            # the earlier scan is the one nearest to t - LAG_S
LAG_TOL_S = 0.06       # ... and none within this of t - LAG_S leaves S_t undefined
WINDOW_S = 1.0         # rest is judged over the scans of the last WINDOW_S
GAP_S = 0.3            # a gap between the window's scans above this: not shown
MIN_SCANS = 5          # fewer scans than this in the window: not shown
MIN_PAIRS = 120        # fewer counted pairs than this: S_t undefined
CAP_M = 6.0            # a range counts only up to min(range_max, CAP_M)
T_FACTOR = 1.5         # T = T_FACTOR x s0
STALE_S = 0.3          # the last scan older than this when asked: not shown

# ── the structure guard (this adapter's, on top of the definition) ──────────────────────────
MIN_COVER_DEG = 270.0     # valid beams must cover this much bearing (10-degree sectors)
COVER_SECTOR_DEG = 10.0
MIN_INSIDE_FRAC = 0.40    # of the valid beams, at least this share inside the cap
MAX_WALL_FRAC = 0.70      # no single wall direction may hold more than this share of them
WALL_BIN_DEG = 20.0       # a "single direction" is any arc this wide, at every 1-degree offset
_LINE_K = 8               # neighbours each side considered for a beam's local line
_LINE_RADIUS_M = 0.25     # ... within this distance of it
_LINE_MIN_POINTS = 5
_LINE_MAX_FLATNESS = 0.1  # smaller / larger eigenvalue: above this it is not a line

# How late a scan's stamp may be against the moment it is placed after the hold, on the
# adapter's clock: one sweep (0.1 s at 10 Hz) plus the shortest delivery delay. See RestChannel.
HOLD_MARGIN_S = 0.2

CONSTANTS = {
    "lag_s": LAG_S, "lag_tol_s": LAG_TOL_S, "window_s": WINDOW_S, "gap_s": GAP_S,
    "min_scans": MIN_SCANS, "min_pairs": MIN_PAIRS, "cap_m": CAP_M, "t_factor": T_FACTOR,
    "pairing": "bearing",
}

AT_REST, MOVING, UNKNOWN = "at_rest", "moving", "unknown"

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Scan:
    """One scan as published: its stamp (s, one clock), ranges (m; NaN and inf kept) and its own
    layout. The beam count is ``len(ranges)``."""
    stamp: float
    ranges: Sequence[float]
    range_min: float
    range_max: float
    angle_min: float
    angle_increment: float


@dataclass(frozen=True)
class Verdict:
    state: str
    reason: str = ""

    @property
    def at_rest(self) -> bool:
        return self.state == AT_REST

    def __str__(self) -> str:
        return self.state if not self.reason else f"{self.state} ({self.reason})"


class CalibrationRefused(ValueError):
    """The calibration record is missing, unreadable or does not show what it must."""


# ── the statistic ───────────────────────────────────────────────────────────────────────────

def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _counts(r: float, lo: float, hi: float) -> bool:
    return math.isfinite(r) and lo <= r <= hi


def _wrap(a: float) -> float:
    """``a`` mod 2 pi, in [0, 2 pi)."""
    a = math.fmod(a, _TWO_PI)
    return a + _TWO_PI if a < 0 else a


def _bearing_gap(a: float, b: float) -> float:
    d = abs(_wrap(a) - _wrap(b))
    return min(d, _TWO_PI - d)


def _nearest(earlier: Scan, bearing: float) -> tuple[int, float] | None:
    """The earlier scan's beam nearest to ``bearing``, and how far off it is; ties go to the
    lower index. None when the earlier scan has no beams or no usable layout."""
    n, a0, inc = len(earlier.ranges), earlier.angle_min, earlier.angle_increment
    if n == 0 or not _finite(a0, inc) or inc <= 0:
        return None
    k = _wrap(bearing - a0) / inc
    lo = math.floor(k)
    cands = {min(max(lo, 0), n - 1), min(max(lo + 1, 0), n - 1), 0, n - 1}
    best = None
    for j in sorted(cands):
        d = _bearing_gap(bearing, a0 + j * inc)
        if best is None or d < best[1]:
            best = (j, d)
    return best


def pair_diffs(scan: Scan, earlier: Scan) -> list[float]:
    """|r_t - r_earlier| over the counted pairs, by bearing (the definition's pairing)."""
    if not _finite(scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
                   earlier.range_min, earlier.range_max, earlier.angle_increment):
        return []
    hi_t, hi_e = min(scan.range_max, CAP_M), min(earlier.range_max, CAP_M)
    half = earlier.angle_increment / 2.0
    out = []
    for i, r in enumerate(scan.ranges):
        r = float(r)
        if not _counts(r, scan.range_min, hi_t):
            continue
        near = _nearest(earlier, scan.angle_min + i * scan.angle_increment)
        if near is None or near[1] > half:
            continue
        re = float(earlier.ranges[near[0]])
        if _counts(re, earlier.range_min, hi_e):
            out.append(abs(r - re))
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def statistic(scan: Scan, earlier: Scan | None) -> tuple[float | None, str]:
    """S_t and, when it is undefined, why."""
    if earlier is None:
        return None, f"no scan within {LAG_TOL_S:g} s of {LAG_S:g} s before"
    diffs = pair_diffs(scan, earlier)
    if len(diffs) < MIN_PAIRS:
        return None, f"{len(diffs)} beams pair by bearing inside the cap, fewer than {MIN_PAIRS}"
    return _median(diffs), ""


def lag_index(stamps: Sequence[float], i: int) -> int | None:
    """The index of the scan nearest to stamps[i] - LAG_S, among scans before i, or None when
    none is within LAG_TOL_S. Ties go to the earlier scan. ``stamps`` strictly increasing."""
    target = stamps[i] - LAG_S
    j = bisect.bisect_left(stamps, target, 0, i)
    best = None
    for k in (j - 1, j):
        if 0 <= k < i:
            d = abs(stamps[k] - target)
            if d <= LAG_TOL_S and (best is None or d < best[1]):
                best = (k, d)
    return None if best is None else best[0]


def series(scans: Sequence[Scan]) -> list[tuple[float | None, str]]:
    """S_t for every scan of a recording (stamps strictly increasing)."""
    stamps = [s.stamp for s in scans]
    out = []
    for i, s in enumerate(scans):
        k = lag_index(stamps, i)
        out.append(statistic(s, None if k is None else scans[k]))
    return out


def window_bounds(stamps: Sequence[float], i: int) -> int:
    """The first index of the window that ends at scan i: stamps in [t - WINDOW_S, t]."""
    return bisect.bisect_left(stamps, stamps[i] - WINDOW_S, 0, i + 1)


def rest_shown(stamps: Sequence[float], s: Sequence[tuple[float | None, str]], i: int,
               threshold_m: float) -> Verdict:
    """The definition's verdict at scan i, from the S series: AT_REST only when shown; MOVING when
    a defined S_u in the window is at or above T; otherwise UNKNOWN with the reason."""
    j = window_bounds(stamps, i)
    vals = s[j:i + 1]
    hit = [v for v, _ in vals if v is not None and v >= threshold_m]
    if hit:
        return Verdict(MOVING, f"S {max(hit):.4f} m >= T {threshold_m:.4f} m")
    if i - j + 1 < MIN_SCANS:
        return Verdict(UNKNOWN, f"{i - j + 1} scans in the last {WINDOW_S:g} s, fewer than "
                                f"{MIN_SCANS}")
    gaps = [stamps[k + 1] - stamps[k] for k in range(j, i)]
    if gaps and max(gaps) > GAP_S:
        return Verdict(UNKNOWN, f"a {max(gaps):.2f} s gap between scans, above {GAP_S:g} s")
    for v, why in vals:
        if v is None:
            return Verdict(UNKNOWN, f"S undefined: {why}")
    return Verdict(AT_REST)


# ── the structure guard ─────────────────────────────────────────────────────────────────────

def _points(scan: Scan) -> list[tuple[int, float, float, float] | None]:
    """Per beam: (index, bearing, x, y) when valid (finite, inside [range_min, range_max])."""
    out: list[tuple[int, float, float, float] | None] = []
    for i, r in enumerate(scan.ranges):
        r = float(r)
        if not _counts(r, scan.range_min, scan.range_max):
            out.append(None)
            continue
        b = scan.angle_min + i * scan.angle_increment
        out.append((i, b, r * math.cos(b), r * math.sin(b)))
    return out


def _line_direction(pts, i: int) -> float | None:
    """The local line direction at beam i, in degrees mod 180, or None if it is not on a line."""
    _, _, x0, y0 = pts[i]
    near = []
    n = len(pts)
    for k in range(-_LINE_K, _LINE_K + 1):
        p = pts[(i + k) % n]
        if p is not None and math.hypot(p[2] - x0, p[3] - y0) <= _LINE_RADIUS_M:
            near.append(p)
    if len(near) < _LINE_MIN_POINTS:
        return None
    mx = sum(p[2] for p in near) / len(near)
    my = sum(p[3] for p in near) / len(near)
    sxx = sum((p[2] - mx) ** 2 for p in near)
    syy = sum((p[3] - my) ** 2 for p in near)
    sxy = sum((p[2] - mx) * (p[3] - my) for p in near)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    big, small = tr / 2.0 + disc, tr / 2.0 - disc
    if big <= 0 or small / big > _LINE_MAX_FLATNESS:
        return None
    return math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy)) % 180.0


def structure_gap(scan: Scan) -> str:
    """Why this scan shows too little structure for a scan difference to see motion, or ""."""
    pts = _points(scan)
    valid = [p for p in pts if p is not None]
    if not valid:
        return "no valid beam"
    sectors = {int(_wrap(p[1]) // math.radians(COVER_SECTOR_DEG)) for p in valid}
    cover = len(sectors) * COVER_SECTOR_DEG
    if cover < MIN_COVER_DEG:
        return f"valid beams cover {cover:g} degrees of bearing, fewer than {MIN_COVER_DEG:g}"
    cap = min(scan.range_max, CAP_M)
    inside = sum(1 for p in valid if math.hypot(p[2], p[3]) <= cap)
    if inside < MIN_INSIDE_FRAC * len(valid):
        return (f"{inside} of {len(valid)} valid beams inside {cap:g} m, fewer than "
                f"{MIN_INSIDE_FRAC:.0%}")
    hist = [0] * 180
    for i, p in enumerate(pts):
        if p is not None:
            d = _line_direction(pts, i)
            if d is not None:
                hist[int(d) % 180] += 1
    w = int(WALL_BIN_DEG)
    worst = max(sum(hist[(a + k) % 180] for k in range(w)) for a in range(180))
    if worst > MAX_WALL_FRAC * len(valid):
        return (f"{worst} of {len(valid)} valid beams lie along one wall direction, more than "
                f"{MAX_WALL_FRAC:.0%}: motion along it would not show")
    return ""


# ── the live channel ────────────────────────────────────────────────────────────────────────

@dataclass
class _Entry:
    scan: Scan
    recv: float
    s: float | None
    why: str
    structure: str


class RestChannel:
    """The live channel: scans in, a `Verdict` out. Thread-safe.

    Two clocks. The statistic and the window run on the scans' own stamps. The question ("now")
    and the hold come from the adapter's clock. A scan's stamp is placed on the adapter's clock
    by the smallest delivery delay seen over the recent scans (receipt minus stamp), which can
    place a scan later than it was taken by that smallest delay; so a scan counts as after the
    hold only when it is at least HOLD_MARGIN_S after it as placed. A last scan older than
    STALE_S, by its placed stamp or by its receipt, is stale."""

    _KEEP_S = LAG_S + LAG_TOL_S + WINDOW_S + 0.5
    _DELAY_SCANS = 50

    def __init__(self, threshold_m: float) -> None:
        if not (_finite(threshold_m) and not isinstance(threshold_m, bool) and threshold_m > 0):
            raise ValueError(f"threshold_m must be a finite number above 0 from a calibration "
                             f"record, got {threshold_m!r}")
        self.threshold_m = float(threshold_m)
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._delays: list[float] = []
        self._broken = ""   # why the sequence was cut (a missing or backwards stamp)

    def add(self, scan: Scan, recv: float) -> None:
        """One scan, received at ``recv`` on the adapter's clock."""
        with self._lock:
            if not _finite(scan.stamp):
                self._entries, self._broken = [], "a scan arrived with no usable stamp"
                return
            if self._entries and scan.stamp <= self._entries[-1].scan.stamp:
                self._entries, self._broken = [], "a scan's stamp did not increase"
            stamps = [e.scan.stamp for e in self._entries] + [scan.stamp]
            k = lag_index(stamps, len(stamps) - 1)
            s, why = statistic(scan, None if k is None else self._entries[k].scan)
            # The guard costs more than the statistic; a scan whose S is undefined fails anyway.
            structure = structure_gap(scan) if s is not None else ""
            self._entries.append(_Entry(scan, recv, s, why, structure))
            cut = scan.stamp - self._KEEP_S
            while self._entries and self._entries[0].scan.stamp < cut:
                self._entries.pop(0)
            self._delays = (self._delays + [recv - scan.stamp])[-self._DELAY_SCANS:]

    def verdict(self, now: float, hold_at: float | None) -> Verdict:
        """At rest, moving or unknown, asked at ``now``, for a hold commanded at ``hold_at`` (both
        on the adapter's clock; None: no hold, so never at rest)."""
        with self._lock:
            if not self._entries:
                return Verdict(UNKNOWN, self._broken or "no scan yet")
            offset = min(self._delays)
            last = self._entries[-1]
            age = max(now - offset - last.scan.stamp, now - last.recv)
            if age > STALE_S:
                return Verdict(UNKNOWN, f"the last scan is {age:.2f} s old, above {STALE_S:g} s")
            stamps = [e.scan.stamp for e in self._entries]
            s = [(e.s, e.why) for e in self._entries]
            i = len(self._entries) - 1
            v = rest_shown(stamps, s, i, self.threshold_m)
            if v.state == MOVING:
                return v
            t = last.scan.stamp
            if hold_at is None:
                return Verdict(UNKNOWN, "the base was not commanded to hold")
            if t - WINDOW_S + offset < hold_at + HOLD_MARGIN_S:
                return Verdict(UNKNOWN, "the scan window does not yet lie entirely after the hold")
            if v.state != AT_REST:
                if self._broken and len(self._entries) < MIN_SCANS + 1:
                    return Verdict(UNKNOWN, self._broken)
                return v
            for e in self._entries[window_bounds(stamps, i):]:
                if e.structure:
                    return Verdict(UNKNOWN, f"no structure: {e.structure}")
            self._broken = ""
            return Verdict(AT_REST)

    def window_stats(self) -> tuple[int, float | None]:
        """The last window's scan count and its largest defined S (None: none defined), for a
        message; never a verdict."""
        with self._lock:
            if not self._entries:
                return 0, None
            stamps = [e.scan.stamp for e in self._entries]
            win = self._entries[window_bounds(stamps, len(stamps) - 1):]
            vals = [e.s for e in win if e.s is not None]
            return len(win), (max(vals) if vals else None)


# ── the calibration record ──────────────────────────────────────────────────────────────────

RECORD_FORMAT = "scan-rest-calibration/1"
MIN_MOTION_WINDOWS = 20

# A veto-only record: T for a channel that may only veto rest, never confirm it. From stationary
# runs alone, each at least MIN_VETO_RUN_S long; it makes no motion claim.
VETO_CLAIM = "veto-only"
MIN_VETO_RUN_S = 100.0
VETO_KINDS = ("A-cal", "A-held")
MOTION_FIELDS = ("motion_windows", "misses", "time_to_rest_s", "time_to_rest_after_hold_s",
                 "stop_s", "measured_speed")


def _number(v) -> bool:
    return _finite(v) and not isinstance(v, bool)


def _check_t(doc, cal: list) -> tuple[float, float]:
    """s0 and T of a record: s0 the largest S of its calibration runs, T = T_FACTOR x s0."""
    smax = [r.get("s_max_m") for r in cal]
    if not all(_number(v) and v > 0 for v in smax):
        raise CalibrationRefused("a calibration run has no largest S (s_max_m)")
    s0, t = doc.get("s0_m"), doc.get("threshold_m")
    if not (_number(s0) and _number(t)):
        raise CalibrationRefused("the record has no s0_m or threshold_m")
    if not math.isclose(s0, max(smax), rel_tol=1e-9, abs_tol=1e-12):
        raise CalibrationRefused(f"s0_m {s0} is not the largest S of the calibration runs "
                                 f"({max(smax)})")
    if not math.isclose(t, T_FACTOR * s0, rel_tol=1e-6, abs_tol=1e-9):
        raise CalibrationRefused(f"threshold_m {t} is not {T_FACTOR:g} x s0 ({T_FACTOR * s0})")
    return float(s0), float(t)


def _check_shape(doc) -> list:
    """The runs of a record of this format and this definition, or `CalibrationRefused`."""
    if not isinstance(doc, dict) or doc.get("format") != RECORD_FORMAT:
        raise CalibrationRefused(f"not a {RECORD_FORMAT} record")
    if doc.get("definition") != CONSTANTS:
        raise CalibrationRefused(
            f"the record was measured with a different definition ({doc.get('definition')!r}); "
            f"this adapter's is {CONSTANTS!r}: a record for another definition is no evidence "
            f"for this one")
    runs = doc.get("runs")
    if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
        raise CalibrationRefused("the record has no runs")
    return runs


def check_veto_record(doc) -> tuple[float, str]:
    """``(T, summary)`` from a veto-only record, or `CalibrationRefused` naming what is missing.

    The record says ``claim: veto-only``: its T may only veto rest. Refused unless its definition
    is this module's, it holds stationary runs only (no motion run and no motion field: it makes
    no motion claim), at least two of them calibration runs of at least MIN_VETO_RUN_S each, s0 is
    their largest S and T is T_FACTOR x s0. A record without that claim -- a measured one
    included -- is refused: the two claims never stand in for each other."""
    if isinstance(doc, dict) and doc.get("format") == RECORD_FORMAT \
            and doc.get("claim") != VETO_CLAIM:
        raise CalibrationRefused(
            f"not a veto-only record (claim {doc.get('claim')!r}): the operator source takes "
            f"only a record whose T comes from stationary runs with no motion claim, and a "
            f"record that claims more is not used as one")
    runs = _check_shape(doc)
    for r in runs:
        if r.get("kind") not in VETO_KINDS:
            raise CalibrationRefused(f"a veto-only record makes no motion claim: run "
                                     f"{r.get('name')!r} is kind {r.get('kind')!r}")
        extra = [k for k in MOTION_FIELDS if k in r]
        if extra:
            raise CalibrationRefused(f"a veto-only record makes no motion claim: run "
                                     f"{r.get('name')!r} carries {', '.join(extra)}")
    cal = [r for r in runs if r.get("kind") == "A-cal"]
    if len(cal) < 2:
        raise CalibrationRefused(f"{len(cal)} stationary calibration runs; 2 needed")
    for r in cal:
        d = r.get("duration_s")
        if not (_number(d) and d >= MIN_VETO_RUN_S):
            raise CalibrationRefused(f"stationary run {r.get('name')!r} lasts {d!r} s, shorter "
                                     f"than {MIN_VETO_RUN_S:g} s (or not recorded)")
    s0, t = _check_t(doc, cal)
    return t, (f"veto-only: T {t:.4f} m (s0 {s0:.4f} m, the largest S of {len(cal)} stationary "
               f"runs of {min(r['duration_s'] for r in cal):.0f} s or more); no motion claim: "
               f"the scan only vetoes, it never confirms rest")


def check_record(doc) -> tuple[float, str]:
    """``(T, summary)`` from a calibration record, or `CalibrationRefused` naming what is missing.

    The record is what a hardware session measures on the robot, written by ``calibrate``: the
    definition it was measured with, s0 and T, and per run its kind and counts. Refused unless
    its definition is this module's, T is T_FACTOR x s0 and s0 is the largest S over its
    calibration runs, there are at least two calibration runs, and every motion (B) run has at
    least MIN_MOTION_WINDOWS motion windows and no miss. Any miss in any run refuses it. A
    veto-only record is refused: it makes no motion claim, so it is no evidence of one."""
    if isinstance(doc, dict) and doc.get("claim", "measured") != "measured":
        raise CalibrationRefused(
            f"the record claims {doc.get('claim')!r}, not measured: a veto-only record is no "
            f"evidence the scan channel is measured")
    runs = _check_shape(doc)
    cal = [r for r in runs if r.get("kind") == "A-cal"]
    if len(cal) < 2:
        raise CalibrationRefused(f"{len(cal)} stationary calibration runs; 2 needed")
    s0, t = _check_t(doc, cal)
    for r in runs:
        if r.get("misses", 0) != 0:
            raise CalibrationRefused(f"run {r.get('name')!r} has {r.get('misses')} misses: rest "
                                     f"was shown while the robot moved")
    motion = [r for r in runs if r.get("kind") == "B"]
    if not motion:
        raise CalibrationRefused("no motion (B) run: nothing shows the channel sees motion")
    for r in motion:
        w = r.get("motion_windows")
        if not isinstance(w, int) or isinstance(w, bool) or w < MIN_MOTION_WINDOWS:
            raise CalibrationRefused(f"motion run {r.get('name')!r} has {w!r} motion windows, "
                                     f"fewer than {MIN_MOTION_WINDOWS}")
        if r.get("misses") != 0:
            raise CalibrationRefused(f"motion run {r.get('name')!r} records no miss count")
    held = doc.get("held_out_not_shown_fraction")
    c_times = [r.get("time_to_rest_s") for r in runs if r.get("kind") == "C"]
    summary = (f"T {t:.4f} m (s0 {s0:.4f} m), {len(motion)} motion runs with 0 misses, "
               f"held-out not-shown fraction {held if held is not None else 'not recorded'}, "
               f"C-run times to rest {c_times or 'not recorded'}")
    return float(t), summary


def _read_record(path: str):
    """The record at ``path`` and its md5, or `CalibrationRefused`."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise CalibrationRefused(f"no calibration record at the given path ({exc.strerror})") \
            from None
    try:
        doc = json.loads(raw)
    except ValueError:
        raise CalibrationRefused("the calibration record is not JSON") from None
    return doc, hashlib.md5(raw).hexdigest()


def load_record(path: str) -> tuple[float, str, str]:
    """``(T, summary, md5)`` from the measured record at ``path``, or `CalibrationRefused`."""
    doc, md5 = _read_record(path)
    t, summary = check_record(doc)
    return t, summary, md5


def load_veto_record(path: str) -> tuple[float, str, str]:
    """``(T, summary, md5)`` from the veto-only record at ``path``, or `CalibrationRefused`."""
    doc, md5 = _read_record(path)
    t, summary = check_veto_record(doc)
    return t, summary, md5


# ── recordings: replay and calibrate (numpy here only) ─────────────────────────────────────

def read_recording(path: str) -> tuple[list[Scan], dict]:
    """The scans of one ``.npz`` recording, and its labels (never read by the statistic).

    Per scan: ``t_s``, ``ranges`` (N x Bmax, NaN-padded), and ``n_beams``, ``angle_min``,
    ``angle_increment``, ``range_min``, ``range_max`` -- each per scan (N), or one value for all.
    Without ``n_beams`` a scan has Bmax beams."""
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["t_s"], dtype=np.float64)
        ranges = np.asarray(z["ranges"])
        n = len(t)

        def per_scan(key, default=None):
            if key not in z.files:
                if default is None:
                    raise ValueError(f"{path}: no {key}")
                return [default] * n
            v = np.asarray(z[key])
            return [v.item()] * n if v.ndim == 0 else list(v)
        nb = per_scan("n_beams", ranges.shape[1])
        amin, ainc = per_scan("angle_min"), per_scan("angle_increment")
        rmin, rmax = per_scan("range_min"), per_scan("range_max")
        labels = {k: np.asarray(z[k], dtype=np.float64) for k in ("cmd_t_s", "cmd_vx", "cmd_wz")
                  if k in z.files}
    scans = [Scan(float(t[i]), [float(x) for x in ranges[i, :int(nb[i])]], float(rmin[i]),
                  float(rmax[i]), float(amin[i]), float(ainc[i])) for i in range(n)]
    return scans, labels


def _check_increasing(scans: Sequence[Scan], path: str) -> None:
    for a, b in itertools.pairwise(scans):
        if not b.stamp > a.stamp:
            raise ValueError(f"{path}: stamps do not increase at {b.stamp}")


def _nonzero_span(labels: dict) -> tuple[float, float, float | None] | None:
    """(first non-zero command, last non-zero command, first zero after the first non-zero)."""
    t, vx, wz = labels.get("cmd_t_s"), labels.get("cmd_vx"), labels.get("cmd_wz")
    if t is None or vx is None or wz is None:
        return None
    moving = [i for i in range(len(t)) if vx[i] != 0 or wz[i] != 0]
    if not moving:
        return None
    stop = next((float(t[i]) for i in range(moving[0], len(t)) if vx[i] == 0 and wz[i] == 0),
                None)
    return float(t[moving[0]]), float(t[moving[-1]]), stop


def evaluate(scans: Sequence[Scan], threshold_m: float):
    s = series(scans)
    stamps = [x.stamp for x in scans]
    return s, [rest_shown(stamps, s, i, threshold_m) for i in range(len(scans))]


def calibrate(manifest_path: str) -> dict:
    """The calibration record from a manifest of recordings and their labels.

    The manifest (JSON): ``{"runs": [{"name", "kind", "file", "measured_speed"?, "note"?}]}``,
    kinds A-cal (two calibration runs), A-held (held out), A-adv (adversarial, reported, not
    graded), B (motion), C (a stop); files relative to the manifest. s0 and T come from the A-cal
    runs alone; the others are held out."""
    import os
    with open(manifest_path) as f:
        man = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    loaded = []
    for r in man["runs"]:
        p = os.path.join(base, r["file"])
        scans, labels = read_recording(p)
        _check_increasing(scans, p)
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        loaded.append((r, scans, labels, md5))
    smax = {}
    for r, scans, _, _ in loaded:
        if r["kind"] == "A-cal":
            vals = [v for v, _ in series(scans) if v is not None]
            if not vals:
                raise ValueError(f"calibration run {r['name']}: no defined S")
            smax[r["name"]] = max(vals)
    if len(smax) < 2:
        raise ValueError("the manifest names fewer than two A-cal runs")
    s0 = max(smax.values())
    t = T_FACTOR * s0
    out_runs, held_w, held_ns = [], 0, 0
    for r, scans, labels, md5 in loaded:
        s, verdicts = evaluate(scans, t)
        stamps = [x.stamp for x in scans]
        vals = [v for v, _ in s if v is not None]
        row = {"name": r["name"], "kind": r["kind"], "file": r["file"], "md5": md5,
               "scans": len(scans), "s_defined": len(vals),
               "s_median_m": _median(vals) if vals else None, "s_max_m": max(vals) if vals else None}
        for k in ("measured_speed", "note"):
            if k in r:
                row[k] = r[k]
        if len(stamps) > 1:
            row["scan_rate_hz"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        full = [i for i in range(len(scans)) if stamps[i] - WINDOW_S >= stamps[0]]
        if r["kind"] in ("A-held", "A-adv", "A-cal"):
            ns = sum(1 for i in full if not verdicts[i].at_rest)
            row["windows"], row["not_shown"] = len(full), ns
            if r["kind"] == "A-held":
                held_w, held_ns = held_w + len(full), held_ns + ns
        elif r["kind"] == "B":
            span = _nonzero_span(labels)
            if span is None:
                raise ValueError(f"motion run {r['name']}: its labels show no non-zero command")
            lo, hi = span[0] + 1.5, span[1] - 1.5
            idx = [i for i in range(len(scans)) if stamps[i] - WINDOW_S >= lo and stamps[i] <= hi]
            row["motion_windows"] = len(idx)
            row["misses"] = sum(1 for i in idx if verdicts[i].at_rest)
        elif r["kind"] == "C":
            span = _nonzero_span(labels)
            stop = None if span is None else span[2]
            row["stop_s"] = stop
            first = next((stamps[i] for i in range(len(scans))
                          if stop is not None and stamps[i] >= stop and verdicts[i].at_rest), None)
            after = next((stamps[i] for i in range(len(scans))
                          if stop is not None and stamps[i] - WINDOW_S >= stop
                          and verdicts[i].at_rest), None)
            row["time_to_rest_s"] = None if first is None else first - stop
            row["time_to_rest_after_hold_s"] = None if after is None else after - stop
        else:
            raise ValueError(f"run {r['name']}: unknown kind {r['kind']!r}")
        out_runs.append(row)
    return {"format": RECORD_FORMAT, "definition": dict(CONSTANTS), "s0_m": s0, "threshold_m": t,
            "calibration_runs": sorted(smax), "runs": out_runs,
            "held_out_not_shown_fraction": (held_ns / held_w) if held_w else None}


def calibrate_veto(manifest_path: str) -> dict:
    """The veto-only record from a manifest of stationary recordings (the manifest as for
    `calibrate`; kinds A-cal and A-held only). s0 and T come from the A-cal runs; an A-held run
    is reported (windows, not shown), never used for T. A manifest naming any other kind is
    refused before a single recording is read: a motion recording never sets a veto's T."""
    import os
    with open(manifest_path) as f:
        man = json.load(f)
    for r in man["runs"]:
        if r.get("kind") not in VETO_KINDS:
            raise ValueError(f"run {r.get('name')!r} is kind {r.get('kind')!r}: --veto-only reads "
                             f"stationary recordings only ({', '.join(VETO_KINDS)})")
    base = os.path.dirname(os.path.abspath(manifest_path))
    loaded = []
    for r in man["runs"]:
        p = os.path.join(base, r["file"])
        scans, _ = read_recording(p)                # the labels are not read
        _check_increasing(scans, p)
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        loaded.append((r, scans, md5))
    smax = {}
    for r, scans, _ in loaded:
        if r["kind"] == "A-cal":
            vals = [v for v, _ in series(scans) if v is not None]
            if not vals:
                raise ValueError(f"calibration run {r['name']}: no defined S")
            smax[r["name"]] = max(vals)
    if len(smax) < 2:
        raise ValueError("the manifest names fewer than two A-cal runs")
    s0 = max(smax.values())
    t = T_FACTOR * s0
    out_runs, held_w, held_ns = [], 0, 0
    for r, scans, md5 in loaded:
        s, verdicts = evaluate(scans, t)
        stamps = [x.stamp for x in scans]
        vals = [v for v, _ in s if v is not None]
        full = [i for i in range(len(scans)) if stamps[i] - WINDOW_S >= stamps[0]]
        ns = sum(1 for i in full if not verdicts[i].at_rest)
        row = {"name": r["name"], "kind": r["kind"], "file": r["file"], "md5": md5,
               "scans": len(scans), "duration_s": (stamps[-1] - stamps[0]) if scans else 0.0,
               "s_defined": len(vals), "s_median_m": _median(vals) if vals else None,
               "s_max_m": max(vals) if vals else None, "windows": len(full), "not_shown": ns}
        if "note" in r:
            row["note"] = r["note"]
        if r["kind"] == "A-held":
            held_w, held_ns = held_w + len(full), held_ns + ns
        out_runs.append(row)
    return {"format": RECORD_FORMAT, "claim": VETO_CLAIM, "definition": dict(CONSTANTS),
            "s0_m": s0, "threshold_m": t, "calibration_runs": sorted(smax), "runs": out_runs,
            "held_out_not_shown_fraction": (held_ns / held_w) if held_w else None}


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m fleet_adapter_ros2.scan_rest",
                                 description="The lidar rest channel over recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="print S_t per scan and each window's verdict")
    rp.add_argument("run", help="a .npz recording")
    rp.add_argument("--threshold-m", type=float, default=None, metavar="T",
                    help="judge the windows against T (without it: S_t only)")
    cp = sub.add_parser("calibrate", help="write the calibration record from a manifest")
    cp.add_argument("manifest")
    cp.add_argument("--out", required=True, help="where to write the record (JSON)")
    cp.add_argument("--veto-only", action="store_true",
                    help="write a veto-only record from stationary recordings alone (A-cal, "
                         "A-held): T for a scan that may only veto rest, never confirm it")
    args = ap.parse_args(argv)
    if args.cmd == "replay":
        scans, _ = read_recording(args.run)
        _check_increasing(scans, args.run)
        if args.threshold_m is not None:
            RestChannel(args.threshold_m)          # the same refusal as the adapter's
            s, verdicts = evaluate(scans, args.threshold_m)
        else:
            s, verdicts = series(scans), None
        print("# i\tt_s\tS_m\tverdict\tstructure")
        for i, sc in enumerate(scans):
            v, why = s[i]
            col = f"{v:.9f}" if v is not None else f"undefined: {why}"
            ver = str(verdicts[i]) if verdicts is not None else "-"
            st = (structure_gap(sc) or "ok") if verdicts is not None else "-"
            print(f"{i}\t{sc.stamp:.6f}\t{col}\t{ver}\t{st}")
        return 0
    if args.veto_only:
        import sys
        try:
            rec = calibrate_veto(args.manifest)
        except ValueError as exc:
            print(f"calibrate --veto-only: {exc}", file=sys.stderr)
            return 2
    else:
        rec = calibrate(args.manifest)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    check, who = ((check_veto_record, "the operator source") if args.veto_only
                  else (check_record, "the adapter"))
    what = " as a veto-only record" if args.veto_only else ""
    try:
        _, summary = check(rec)
        print(f"record written; {who} would accept it{what}: {summary}")
    except CalibrationRefused as exc:
        print(f"record written; {who} would REFUSE it: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
