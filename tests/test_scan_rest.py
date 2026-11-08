# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The lidar rest channel: at rest only when the scan shows it, moving when the scan moves, and
"not shown" -- unknown, with the reason -- for everything else.

Every scan here is ray-cast from the plant's TRUE pose (`scan_room`). The plant's command is kept
beside it and never reaches the channel, which has no argument that could take it. So the two
counterfactuals are the ADR-0071 test itself: told to stop but drifting reads `moving`; told to
drive but stalled reads `at_rest`.

T comes the way a record makes it: T_FACTOR x the largest S over two stationary runs at two spots,
with the noise the lidar states.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib

import pytest
from scan_room import CORRIDOR, Lidar, Person, Plant, record

from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2.scan_rest import AT_REST, MOVING, UNKNOWN, RestChannel, Scan

LATENCY_S = 0.01           # delivery delay of every scan in these runs


@pytest.fixture(scope="module")
def T() -> float:
    """T_FACTOR x s0, s0 the largest S over two stationary runs at two spots."""
    s0 = 0.0
    for seed, (x, y, yaw) in ((11, (0.0, 0.0, 0.0)), (12, (0.6, 0.4, 1.0))):
        scans = record(Plant(x=x, y=y, yaw=yaw), Lidar(seed=seed), 12.0)
        s0 = max(s0, max(v for v, _ in sr.series(scans) if v is not None))
    return sr.T_FACTOR * s0


def feed(scans, T, *, hold_at=0.0, start_after=1.6):
    """Feed a channel, asking after every scan once the window can be full; the verdicts."""
    ch = RestChannel(T)
    out = []
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
        if s.stamp - scans[0].stamp >= start_after:
            out.append(ch.verdict(s.stamp + LATENCY_S, hold_at))
    return out


def states(verdicts) -> set:
    return {v.state for v in verdicts}


# ── the red cases: the robot and the room ─────────────────────────────────────

def test_robot_still_is_at_rest(T) -> None:
    v = feed(record(Plant(), Lidar(seed=3), 5.0), T, hold_at=-1.0)
    assert states(v) == {AT_REST}, [str(x) for x in v if not x.at_rest][:3]


def test_translating_at_005_mps_is_moving(T) -> None:
    v = feed(record(Plant(true=(0.05, 0.0, 0.0)), Lidar(seed=4), 5.0), T, hold_at=-1.0)
    assert states(v) == {MOVING}


@pytest.mark.parametrize("vel", [(-0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.035, 0.035, 0.0)])
def test_translating_slowly_in_any_direction_is_moving(T, vel) -> None:
    v = feed(record(Plant(true=vel), Lidar(seed=5), 4.0), T, hold_at=-1.0)
    assert states(v) == {MOVING}


def test_spinning_at_015_radps_is_moving(T) -> None:
    v = feed(record(Plant(true=(0.0, 0.0, 0.15)), Lidar(seed=6), 5.0), T, hold_at=-1.0)
    assert states(v) == {MOVING}


def _person():
    # 0.6 m from the robot, walking across it at 1 m/s.
    return Person(start=(0.6, -0.9), vel=(0.0, 1.0))


def test_a_person_crossing_a_fifth_of_the_beams_leaves_a_still_robot_at_rest(T) -> None:
    lidar = Lidar(seed=7)
    with_p = record(Plant(), lidar, 2.0, people=[_person()])
    bare = record(Plant(), Lidar(seed=7, noise_m=0.0), 2.0)
    occluded = [i for i, (a, b) in enumerate(zip(with_p[15].ranges, bare[15].ranges))
                if abs(a - b) > 0.05]
    lag = [i for i, (a, b) in enumerate(zip(with_p[10].ranges, bare[10].ranges))
           if abs(a - b) > 0.05]
    share = len(set(occluded) | set(lag)) / len(bare[15].ranges)
    assert 0.15 <= share <= 0.25                       # the person changes about 20 % of beams
    v = feed(with_p, T, hold_at=-1.0, start_after=1.5)
    assert states(v) == {AT_REST}                      # the median is robust to them


def test_a_person_crossing_while_the_robot_moves_is_moving(T) -> None:
    v = feed(record(Plant(true=(0.05, 0.0, 0.0)), Lidar(seed=8), 2.0, people=[_person()]), T,
             hold_at=-1.0, start_after=1.5)
    assert states(v) == {MOVING}


def test_dropped_beams_still_tell_rest_from_motion(T) -> None:
    still = feed(record(Plant(), Lidar(seed=9, drop=0.3), 4.0), T, hold_at=-1.0)
    moving = feed(record(Plant(true=(0.05, 0.0, 0.0)), Lidar(seed=9, drop=0.3), 4.0), T,
                  hold_at=-1.0)
    assert states(still) == {AT_REST} and states(moving) == {MOVING}


def test_at_the_stated_noise_held_out_rest_is_below_T_and_every_slow_drift_window_moves(T) -> None:
    """T sits above the stationary S of a run it was not made from. At 0.05 m/s a single scan's
    S is NOT always above it at this noise (25 mm in 0.5 s against +-10 mm per beam): the window
    reads moving because one S at or above T in it is enough. Every window of a 20 s drift does."""
    held = [v for v, _ in sr.series(record(Plant(x=-0.5, y=0.8, yaw=2.0), Lidar(seed=21), 8.0))]
    assert max(v for v in held if v is not None) < T
    drift = record(Plant(true=(0.05, 0, 0)), Lidar(seed=22), 20.0)
    s, verdicts = sr.evaluate(drift, T)
    assert all(v.state == MOVING for v in verdicts[16:])
    below = sum(1 for v, _ in s[5:] if v is not None and v < T)
    print(f"0.05 m/s drift at {T=:.4f} m: {below} of {len(s) - 5} scans have S below T")


# ── the counterfactuals (ADR-0071 1): only the true motion counts ──────────────

def test_commanded_stopped_but_drifting_is_moving(T) -> None:
    plant = Plant(cmd=(0.0, 0.0, 0.0), true=(0.08, 0.0, 0.0))      # a slipping or pushed base
    assert states(feed(record(plant, Lidar(seed=13), 4.0), T, hold_at=-1.0)) == {MOVING}


def test_commanded_to_move_but_stalled_is_at_rest(T) -> None:
    plant = Plant(cmd=(0.2, 0.0, 0.0), true=(0.0, 0.0, 0.0))       # a stalled motor
    assert states(feed(record(plant, Lidar(seed=14), 4.0), T, hold_at=-1.0)) == {AT_REST}


def test_the_channel_takes_no_command_no_odometry_no_pose() -> None:
    src = pathlib.Path(sr.__file__).read_text()
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    for banned in ("geometry_msgs", "nav_msgs", "rclpy", "sensor_msgs", "fleet_adapter_ros2"):
        assert not any(m == banned or m.startswith(banned + ".") for m in imported), banned
    assert "Twist" not in src and "Odometry" not in src
    for fn in (RestChannel.add, RestChannel.verdict, sr.statistic, sr.series, sr.rest_shown):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"cmd", "command", "setpoint", "odom", "odometry", "pose", "velocity",
                             "twist", "speed"}, fn.__name__
    assert set(inspect.signature(RestChannel.add).parameters) == {"self", "scan", "recv"}
    assert {f for f in Scan.__dataclass_fields__} == {
        "stamp", "ranges", "range_min", "range_max", "angle_min", "angle_increment"}


# ── refute wins ─────────────────────────────────────────────────────────────────

def test_one_scan_at_or_above_T_in_the_window_is_moving_whatever_the_rest_say() -> None:
    stamps = [i * 0.1 for i in range(16)]
    s = [(0.001, "")] * 16
    s[12] = (0.5, "")
    assert sr.rest_shown(stamps, s, 15, 0.02).state == MOVING
    s[12] = (0.02, "")                                    # at T is not below T
    assert sr.rest_shown(stamps, s, 15, 0.02).state == MOVING
    s[12] = (None, "undefined")
    s[13] = (0.03, "")                                    # an undefined S does not hide a moving one
    assert sr.rest_shown(stamps, s, 15, 0.02).state == MOVING


# ── the beam count is not fixed: pairs by bearing (the amended definition) ──────────

def test_449_450_and_451_beams_give_the_same_S_as_a_fixed_layout(T) -> None:
    def varying(i):
        return (449, 450, 451)[i % 3], 0.0065 * (i % 2)

    for true in ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)):
        fixed = [v for v, _ in sr.series(record(Plant(true=true), Lidar(seed=31), 3.0))][6:]
        vary = sr.series(record(Plant(true=true), Lidar(seed=31), 3.0, layout=varying))[6:]
        assert all(v is not None for v, _ in vary), vary[:2]
        for a, (b, _) in zip(fixed, vary):
            assert abs(a - b) < 0.004                      # the same, to the noise
    still = feed(record(Plant(), Lidar(seed=32), 4.0, layout=varying), T, hold_at=-1.0)
    moving = feed(record(Plant(true=(0.05, 0, 0)), Lidar(seed=32), 4.0, layout=varying), T,
                  hold_at=-1.0)
    assert states(still) == {AT_REST} and states(moving) == {MOVING}


def test_a_layout_that_cannot_pair_120_beams_is_unknown(T) -> None:
    # Scans alternate between the two halves of the turn: a scan and the one 0.5 s before it
    # (five scans back, the other half) share almost no bearing, so almost nothing pairs.
    full = record(Plant(), Lidar(seed=33), 3.0)
    inc = full[0].angle_increment
    scans = [Scan(f.stamp, list(f.ranges[:225]) if i % 2 == 0 else list(f.ranges[225:]),
                  f.range_min, f.range_max, 0.0 if i % 2 == 0 else 225 * inc, inc)
             for i, f in enumerate(full)]
    s = sr.series(scans)
    assert all(v is None for v, _ in s[5:]) and "fewer than 120" in s[-1][1]
    v = feed(scans, T, hold_at=-1.0)
    assert states(v) == {UNKNOWN} and "fewer than 120" in v[-1].reason
    few = record(Plant(), Lidar(seed=34, beams=100), 3.0)           # 100 beams in all
    assert states(feed(few, T, hold_at=-1.0)) == {UNKNOWN}


def test_pairing_is_by_bearing_across_the_turn_and_never_by_index() -> None:
    inc = 2 * math.pi / 360
    a = Scan(0.0, [1.0] * 360, 0.05, 12.0, 0.0, inc)
    # The same room, published starting 10 beams later: by index every pair would differ.
    ranges = [1.0 + 0.001 * i for i in range(360)]
    b0 = Scan(0.0, ranges, 0.05, 12.0, 0.0, inc)
    b1 = Scan(0.5, ranges[10:] + ranges[:10], 0.05, 12.0, 10 * inc, inc)
    v, _ = sr.statistic(b1, b0)
    assert v == pytest.approx(0.0, abs=1e-9)
    assert sr.statistic(Scan(0.5, [1.0] * 360, 0.05, 12.0, -math.pi, inc), a)[0] == 0.0


# ── fail closed ──────────────────────────────────────────────────────────────

def test_a_scan_gap_is_not_shown(T) -> None:
    scans = record(Plant(), Lidar(seed=41), 4.0)
    cut = [s for s in scans if not 2.4 < s.stamp < 2.85]               # 0.5 s without a scan
    ch = RestChannel(T)
    for s in cut:
        ch.add(s, s.stamp + LATENCY_S)
        if 2.85 <= s.stamp <= 3.3:
            v = ch.verdict(s.stamp + LATENCY_S, -1.0)
            assert v.state == UNKNOWN, (s.stamp, str(v))
    assert "gap" in str(RestChannel_verdict_at(cut, T, 2.95))


def RestChannel_verdict_at(scans, T, t):
    ch = RestChannel(T)
    for s in scans:
        if s.stamp <= t:
            ch.add(s, s.stamp + LATENCY_S)
    return ch.verdict(max(s.stamp for s in scans if s.stamp <= t) + LATENCY_S, -1.0)


def test_a_stale_last_scan_is_not_shown(T) -> None:
    scans = record(Plant(), Lidar(seed=42), 3.0)
    ch = RestChannel(T)
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
    last = scans[-1].stamp + LATENCY_S
    assert ch.verdict(last, -1.0).state == AT_REST
    v = ch.verdict(last + 0.35, -1.0)
    assert v.state == UNKNOWN and "old" in v.reason


def test_a_scan_delivered_late_is_stale_by_its_receipt_too(T) -> None:
    scans = record(Plant(), Lidar(seed=43), 3.0)
    ch = RestChannel(T)
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
    v = ch.verdict(scans[-1].stamp + LATENCY_S + 0.31, -1.0)
    assert v.state == UNKNOWN


@pytest.mark.parametrize("bad", [math.nan, math.inf, None])
def test_a_missing_stamp_cuts_the_sequence(T, bad) -> None:
    scans = record(Plant(), Lidar(seed=44), 3.0)
    ch = RestChannel(T)
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
    last = scans[-1]
    ch.add(Scan(bad, last.ranges, 0.05, 12.0, 0.0, last.angle_increment), last.stamp + 0.1)
    v = ch.verdict(last.stamp + 0.1, -1.0)
    assert v.state == UNKNOWN and "stamp" in v.reason


def test_a_stamp_that_goes_backwards_is_not_shown(T) -> None:
    scans = record(Plant(), Lidar(seed=45), 3.0)
    ch = RestChannel(T)
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
    back = scans[-5]
    ch.add(back, scans[-1].stamp + 0.1)
    assert ch.verdict(scans[-1].stamp + 0.1, -1.0).state == UNKNOWN


def test_nan_only_scans_are_not_shown(T) -> None:
    scans = [Scan(i * 0.1, [math.nan] * 450, 0.05, 12.0, 0.0, 2 * math.pi / 450)
             for i in range(30)]
    v = feed(scans, T, hold_at=-1.0)
    assert states(v) == {UNKNOWN} and "fewer than 120" in v[-1].reason


def test_no_scan_and_no_threshold() -> None:
    assert RestChannel(0.02).verdict(0.0, -1.0).state == UNKNOWN
    for bad in (0, -0.01, math.nan, math.inf, None, True, "0.02"):
        with pytest.raises(ValueError):
            RestChannel(bad)


# ── the window must post-date the hold; time alone never confirms rest ─────────

def test_a_window_that_does_not_lie_after_the_hold_is_not_shown(T) -> None:
    scans = record(Plant(), Lidar(seed=51), 4.0)
    ch = RestChannel(T)
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
    now = scans[-1].stamp + LATENCY_S
    hold = now - 0.9                                     # the hold is inside the last second
    v = ch.verdict(now, hold)
    assert v.state == UNKNOWN and "hold" in v.reason
    assert ch.verdict(now, None).state == UNKNOWN
    assert ch.verdict(now, now - 1.5).state == AT_REST


def test_time_since_the_hold_never_confirms_rest_by_itself(T) -> None:
    v = feed(record(Plant(true=(0.05, 0, 0)), Lidar(seed=52), 6.0), T, hold_at=-100.0)
    assert states(v) == {MOVING}
    nothing = RestChannel(T)
    assert nothing.verdict(1000.0, 0.0).state == UNKNOWN


def test_braking_after_the_hold_keeps_it_from_rest_until_the_scans_show_rest(T) -> None:
    plant, lidar = Plant(true=(0.2, 0.0, 0.0)), Lidar(seed=53)
    scans = record(plant, lidar, 2.0)                      # driving at 0.2 m/s
    plant.true = (0.1, 0.0, 0.0)                           # held at 2.0; braking for 0.3 s
    scans += record(plant, lidar, 0.3, t0=2.0)
    plant.true = (0.0, 0.0, 0.0)
    scans += record(plant, lidar, 3.0, t0=2.3)
    ch, first = RestChannel(T), None
    for s in scans:
        ch.add(s, s.stamp + LATENCY_S)
        v = ch.verdict(s.stamp + LATENCY_S, 2.0)
        if s.stamp < 2.3 + sr.LAG_S:
            assert v.state != AT_REST
        if v.at_rest and first is None:
            first = s.stamp
    # Rest is shown only once a whole window lies after the hold (plus its margin), and no S in
    # it compares against a scan taken while the base drove at 0.2 m/s. (The last 0.3 s of
    # braking, 0.1 m/s, moves the scan less than T over the lag: rest may be shown with it in.)
    assert first is not None and first >= 2.0 + sr.HOLD_MARGIN_S + sr.WINDOW_S - LATENCY_S
    assert first >= 2.0 + sr.LAG_S + sr.WINDOW_S - 0.05


# ── the aperture problem: a corridor ─────────────────────────────────────────

def _corridor(true, yaw=0.0, seed=61):
    return record(Plant(true=true, yaw=yaw), Lidar(walls=list(CORRIDOR), seed=seed), 4.0)


def test_in_a_corridor_the_definition_alone_cannot_see_motion_along_it(T) -> None:
    """The finding the guard exists for: translating along a corridor whose ends are beyond the
    cap changes no range, and the bare definition shows rest."""
    scans = _corridor((0.05, 0.0, 0.0))
    _, verdicts = sr.evaluate(scans, T)
    assert all(v.at_rest for v in verdicts[16:])


@pytest.mark.parametrize("yaw_deg", [0.0, 20.0, 37.0, 90.0])
def test_in_a_corridor_the_channel_says_unknown_no_structure(T, yaw_deg) -> None:
    v = feed(_corridor((0.05, 0.0, 0.0), yaw=math.radians(yaw_deg)), T, hold_at=-1.0)
    # yaw turns the robot; it translates along its own x, so along the corridor only at 0 and
    # crosswise at 90, where the scan sees it.
    if yaw_deg == 90.0:
        assert states(v) == {MOVING}
    else:
        assert states(v) <= {UNKNOWN, MOVING} and AT_REST not in states(v)
        assert any("no structure" in x.reason for x in v) or states(v) == {MOVING}


def test_the_corridor_still_robot_is_unknown_not_at_rest(T) -> None:
    v = feed(_corridor((0.0, 0.0, 0.0)), T, hold_at=-1.0)
    assert states(v) == {UNKNOWN} and all("no structure" in x.reason for x in v)


def test_the_furnished_room_has_structure() -> None:
    for x, y, yaw in ((0.0, 0.0, 0.0), (0.6, 0.4, 1.0), (-0.5, 0.8, 2.0)):
        scan = Lidar(seed=71).cast(Plant(x=x, y=y, yaw=yaw), 0.0)
        assert sr.structure_gap(scan) == ""


def test_too_little_bearing_or_too_far_is_no_structure() -> None:
    scan = Lidar(seed=72).cast(Plant(), 0.0)
    half = Scan(0.0, [r if i < 180 else math.inf for i, r in enumerate(scan.ranges)], 0.05,
                12.0, 0.0, scan.angle_increment)
    assert "cover" in sr.structure_gap(half)
    far = Scan(0.0, [8.0 + 0.01 * (i % 7) for i in range(450)], 0.05, 12.0, 0.0,
               scan.angle_increment)
    assert "inside" in sr.structure_gap(far)


def test_the_constants_are_the_definition() -> None:
    assert (sr.LAG_S, sr.LAG_TOL_S, sr.WINDOW_S, sr.GAP_S, sr.MIN_SCANS, sr.MIN_PAIRS, sr.CAP_M,
            sr.T_FACTOR, sr.STALE_S) == (0.5, 0.06, 1.0, 0.3, 5, 120, 6.0, 1.5, 0.3)
    assert (sr.MIN_COVER_DEG, sr.MIN_INSIDE_FRAC, sr.MAX_WALL_FRAC, sr.WALL_BIN_DEG) == (
        270.0, 0.40, 0.70, 20.0)
