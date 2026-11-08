# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view contract (fleet_adapter_ros2/mapview/CONTRACT.md), its sources and its viewer
state. Synthetic data only: no map, run or log from a real site is read or shipped here."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import sources as s

REPO = Path(__file__).resolve().parents[1]
R, F = "sim-a", "map@test"


def grid(h=7, w=11, seed=1):
    rng = random.Random(seed)
    return [[rng.choice((-1, 0, 100, 37)) for _ in range(w)] for _ in range(h)]


def snap(g, seq=0, frame=F, stamp=1.0):
    return c.message("MapSnapshot", R, frame, stamp, seq=seq, **c.encode_grid(g, 0.05, (-1.0, -2.0)))


def pose(x=0.0, y=0.0, frame=F, stamp=2.0, yaw=0.0):
    return c.message("PoseSample", R, frame, stamp, x=x, y=y, yaw_deg=yaw, source="slam")


def status(phase="started", frame=F, stamp=1.0, source="guided:t", **kw):
    return c.message("SessionStatus", R, frame, stamp, source=source, phase=phase, text="x",
                     area_free_m2=None, **kw)


# -- grids -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("h,w", [(1, 1), (7, 11), (225, 155)])
def test_grid_round_trip_keeps_every_value_and_the_row_order(h, w):
    g = grid(h, w)
    g[0][0] = 100        # bottom-left
    g[-1][-1] = -1       # top-right
    back = c.decode_grid(c.encode_grid(g, 0.05, (0, 0)))
    assert back == g


def test_a_flat_row_major_grid_encodes_as_its_rows():
    g = grid(3, 4)
    flat = [v for row in g for v in row]
    assert c.encode_flat(flat, 4, 3, 0.05, (0, 0)) == c.encode_grid(g, 0.05, (0, 0))


def test_a_changed_cell_fails_the_digest():
    f = c.encode_grid(grid(), 0.05, (0, 0))
    f["sha256"] = "0" * 64
    with pytest.raises(c.ContractError, match="sha256"):
        c.decode_grid(f)


def test_a_size_that_does_not_match_the_png_is_refused():
    f = c.encode_grid(grid(), 0.05, (0, 0))
    f["width"] += 1
    with pytest.raises(c.ContractError, match="size"):
        c.decode_grid(f)


def test_values_outside_the_occupancy_range_are_refused():
    with pytest.raises(c.ContractError):
        c.encode_grid([[101]], 0.05, (0, 0))


def test_a_grid_larger_than_the_contract_allows_is_refused_before_it_is_decoded():
    f = c.encode_grid([[0]], 0.05, (0, 0))
    f["width"], f["height"] = 5000, 5000
    with pytest.raises(c.ContractError, match="cells"):
        c.decode_grid(f)


def test_a_resolution_that_is_not_positive_is_refused():
    with pytest.raises(c.ContractError, match="resolution"):
        c.encode_grid([[0]], 0.0, (0, 0))


# -- messages --------------------------------------------------------------------------------------

def test_every_kind_round_trips_through_a_json_line():
    msgs = [snap(grid()), pose(1.0, -0.5, yaw=90.0),
            c.message("Overlays", R, F, 3.0, keepouts=[],
                      zones=[{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                      target={"label": "A", "x": 1.0, "y": 2.0}),
            status()]
    for m in msgs:
        assert c.loads(c.dumps(m)) == json.loads(json.dumps(m))


def test_a_missing_field_or_another_version_is_refused():
    m = pose()
    bad = dict(m)
    del bad["yaw_deg"]
    with pytest.raises(c.ContractError, match="yaw_deg"):
        c.check(bad)
    with pytest.raises(c.UnknownVersion, match="contractVersion"):
        c.check({**m, "contractVersion": "mapview/9"})
    with pytest.raises(c.ContractError, match="unknown kind"):
        c.check({**m, "kind": "DriveCommand"})


def test_an_unknown_version_is_refused_before_any_other_field_is_read():
    # A later version may rename or drop every other field: nothing else is looked at.
    with pytest.raises(c.UnknownVersion):
        c.check({"contractVersion": "mapview/1"})
    with pytest.raises(c.UnknownVersion):
        c.check({"kind": "PoseSample"})


def test_extra_fields_are_ignored():
    m = c.message("PoseSample", R, F, 2.0, x=1.0, y=0.0, yaw_deg=0.0, source="slam", future_field=1)
    assert c.loads(c.dumps(m))["future_field"] == 1


def test_the_contract_has_no_command_message():
    assert set(c.REQUIRED) == {"MapSnapshot", "PoseSample", "Overlays", "SessionStatus"}


@pytest.mark.parametrize("robot", ["192.0.2.20", "robot.local", "aa:bb:cc:dd:ee:ff",
                                   "fe80::1", "Robot A", ""])
def test_a_robot_that_looks_like_a_network_name_or_address_is_refused(robot):
    with pytest.raises(c.ContractError, match="robot"):
        c.message("PoseSample", robot, F, 2.0, x=0.0, y=0.0, yaw_deg=0.0, source="slam")


@pytest.mark.parametrize("field", ["x", "y", "yaw_deg"])
def test_a_number_that_is_not_finite_is_refused(field):
    m = pose()
    m[field] = float("nan")
    with pytest.raises(c.ContractError, match=field):
        c.check(m)
    m[field] = "1.0"
    with pytest.raises(c.ContractError, match=field):
        c.check(m)


def test_a_phase_the_contract_does_not_name_is_refused():
    with pytest.raises(c.ContractError, match="phase"):
        status(phase="exploring")


def test_a_zone_needs_three_points_and_a_target_needs_a_position():
    with pytest.raises(c.ContractError, match="polygon"):
        c.message("Overlays", R, F, 3.0, keepouts=[], zones=[{"name": "z", "polygon": [[0, 0], [1, 0]]}],
                  target=None)
    with pytest.raises(c.ContractError, match="target"):
        c.message("Overlays", R, F, 3.0, keepouts=[], zones=[], target={"label": "A", "x": 1.0})


def test_the_optional_session_fields_are_typed_when_present():
    assert status(simulated=True, progress_pct=40)["simulated"] is True
    with pytest.raises(c.ContractError, match="simulated"):
        status(simulated="yes")
    with pytest.raises(c.ContractError, match="progress_pct"):
        status(progress_pct=140)


@pytest.mark.parametrize("source,driver", [("guided:r13", "operator"), ("explore-floor", "robot"),
                                           ("replay:r12", "replay")])
def test_who_drives_is_read_from_the_source_and_a_guided_session_is_operator_driven(source, driver):
    assert c.driver_of(source) == driver
    text = c.describe_source(source, simulated=False)
    if driver == "operator":
        assert "operator" in text and "autonom" not in text.lower()


# -- the viewer's state ----------------------------------------------------------------------------

def test_the_viewer_locks_to_the_first_frame_and_counts_refusals():
    st = c.ViewState()
    assert st.accept(snap(grid(), frame="map@r1"))
    assert not st.accept(pose(frame="map@r2"))
    assert st.refused["frame"] == 1 and st.last_pose is None


def test_a_session_started_in_another_frame_replaces_the_view_and_is_never_drawn_with_it():
    st = c.ViewState()
    st.accept(status(frame="map@s1"))
    st.accept(snap(grid(), frame="map@s1"))
    st.accept(pose(1.0, 1.0, frame="map@s1"))
    assert st.accept(status(frame="map@s2", stamp=9.0))
    assert st.frame == "map@s2" and st.map is None and st.poses == [] and st.last_pose is None
    assert not st.accept(pose(frame="map@s1", stamp=10.0))
    assert st.refused["frame"] == 1


def test_a_bad_message_is_counted_not_raised():
    st = c.ViewState()
    f = snap(grid())
    f["sha256"] = "0" * 64
    assert not st.accept(f)
    assert st.refused["contract"] == 1 and st.map is None


def test_an_unknown_version_is_counted_apart_and_named():
    st = c.ViewState()
    assert not st.accept({**pose(), "contractVersion": "mapview/1"})
    assert st.refused == {"frame": 0, "contract": 0, "version": 1}
    assert st.versions_seen == {"mapview/1"}


def test_an_older_snapshot_does_not_replace_a_newer_one():
    st = c.ViewState()
    st.accept(snap(grid(seed=1), seq=5))
    st.accept(snap(grid(seed=2), seq=4))
    assert st.map[3] == 5


def test_the_path_keeps_a_pose_only_when_the_robot_moved():
    st = c.ViewState(path_step=0.02)
    for x in (0.0, 0.005, 0.01, 0.03, 0.031):
        st.accept(pose(x, stamp=x))
    assert [round(p[1], 3) for p in st.poses] == [0.0, 0.03]
    assert st.last_pose[1] == pytest.approx(0.031)


def test_staleness_is_judged_by_when_a_message_arrived_not_by_its_stamp():
    st = c.ViewState()
    st.accept(status(), now=100.0)
    st.accept(snap(grid(), stamp=1.0e9), now=100.0)       # a robot clock far from ours
    st.accept(pose(stamp=1.0e9), now=100.0)
    assert st.staleness(now=101.0) == {"pose": "fresh", "map": "fresh", "ended": False}
    assert st.staleness(now=100.0 + c.POSE_STALE_S + 0.1)["pose"] == "stale"
    assert st.staleness(now=100.0 + c.POSE_LOST_S + 0.1)["pose"] == "lost"
    assert st.staleness(now=100.0 + c.MAP_STALE_S + 0.1)["map"] == "stale"


def test_nothing_is_stale_once_the_session_has_stopped():
    st = c.ViewState()
    st.accept(status(), now=0.0)
    st.accept(pose(), now=0.0)
    st.accept(status(phase="stopped", stamp=2.0), now=1.0)
    assert st.staleness(now=500.0) == {"pose": None, "map": None, "ended": True}


# -- sources -----------------------------------------------------------------------------------------

TF_ECHO = """At time 1791002953.289595000
- Translation: [0.149, 0.436, 0.000]
- Rotation: in Quaternion (xyzw) [0.000, 0.000, 0.311, 0.950]
- Rotation: in RPY (radian) [0.000, -0.000, 0.633]
- Rotation: in RPY (degree) [0.000, -0.000, 36.263]
- Matrix:
  0.806 -0.592  0.000  0.149
"""


def test_tf2_echo_gives_one_pose_from_the_degree_line_only():
    p, out = s.Tf2EchoParser(), []
    for line in TF_ECHO.splitlines():
        r = p.feed(line)
        if r:
            out.append(r)
    assert out == [(pytest.approx(1791002953.289595), 0.149, 0.436, 36.263)]


def test_tf2_echo_unpadded_nanoseconds_are_read_as_tf2_echo_means_them():
    p = s.Tf2EchoParser()
    p.feed("At time 100.26014595")
    p.feed("- Translation: [1.0, 2.0, 0.0]")
    assert p.feed("- Rotation: in RPY (degree) [0, 0, 5]")[0] == pytest.approx(100.026014595)


MAP_ECHO = """header:
  stamp:
    sec: 1791002953
    nanosec: 500000000
  frame_id: map
info:
  map_load_time:
    sec: 0
    nanosec: 0
  resolution: 0.05
  width: 3
  height: 2
  origin:
    position:
      x: -6.36
      y: -2.92
      z: 0.0
data:
- -1
- 0
- 100
- 0
- 0
- -1
"""


def test_a_map_echo_message_parses_to_a_grid_with_row_zero_at_the_origin():
    g, res, o, stamp = s.parse_map_echo(MAP_ECHO)
    assert g == [[-1, 0, 100], [0, 0, -1]]
    assert (res, o) == (0.05, (-6.36, -2.92)) and stamp == pytest.approx(1791002953.5)
    assert c.decode_grid(c.encode_grid(g, res, o)) == g


def test_a_truncated_map_echo_is_none():
    assert s.parse_map_echo(MAP_ECHO.replace("- -1\n", "", 1)) is None


# -- a saved operator-driven run, replayed (a synthetic run: the shape of one, none of its data) ------

def _synthetic_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "mask").mkdir(parents=True)
    w, h = 6, 4
    # map_server PGM, top row first: walls round a free room, one unknown cell.
    rows = [[0] * w] + [[0] + [254] * (w - 2) + [0] for _ in range(h - 2)] + [[0] * w]
    rows[1][1] = 205
    (run / "room.pgm").write_bytes(b"P5\n%d %d\n255\n" % (w, h) + bytes(v for r in rows for v in r))
    (run / "room.yaml").write_text("image: room.pgm\nresolution: 0.5\norigin: [-1.0, -1.0, 0.0]\n"
                                   "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    ko = [[0 if (x, y) == (4, 1) else 254 for x in range(w)] for y in range(h)]
    (run / "mask" / "keepout.pgm").write_bytes(b"P5\n%d %d\n255\n" % (w, h)
                                               + bytes(v for r in ko for v in r))
    (run / "mask" / "keepout.yaml").write_text((run / "room.yaml").read_text()
                                               .replace("room.pgm", "keepout.pgm"))
    (run / "zones.yaml").write_text(
        "kind: FleetZone\nmetadata: {name: zone-1}\nspec:\n  physicalBounds:\n    polygon:\n"
        "    - {x: 0.0, y: 0.0}\n    - {x: 1.0, y: 0.0}\n    - {x: 1.0, y: 0.5}\n")
    with open(run / "guide-log.jsonl", "w") as f:
        for i in range(5):
            f.write(json.dumps({"kind": "pose", "t": 100.0 + i, "x": 0.1 * i, "y": 0.0, "yaw": 10.0 * i})
                    + "\n")
            f.write(json.dumps({"kind": "note", "t": 100.5 + i}) + "\n")
    return run


def test_a_saved_map_reads_as_occupancy_with_row_zero_at_the_bottom(tmp_path):
    run = _synthetic_run(tmp_path)
    g, res, o = s.load_saved_map(run / "room.yaml")
    assert (res, o) == (0.5, (-1.0, -1.0)) and len(g) == 4 and len(g[0]) == 6
    assert g[0] == [100] * 6 and g[-1] == [100] * 6          # walls top and bottom
    assert g[2][1] == -1                                      # the PGM's row 1 is the grid's row 2
    assert s.free_area(g, res) == pytest.approx(7 * 0.25)


def test_a_saved_run_replays_through_the_contract_to_the_same_map_and_last_pose(tmp_path):
    run = _synthetic_run(tmp_path)
    msgs = list(s.convert_guided_run(run, run / "room.yaml", "sim-a", "map@replay", run / "zones.yaml"))
    st = c.ViewState()
    for m in msgs:
        assert st.accept(c.loads(c.dumps(m)))
    g, _, _ = s.load_saved_map(run / "room.yaml")
    assert st.map[0] == g
    assert st.last_pose[1:] == pytest.approx((0.4, 0.0, 40.0))
    assert len(st.overlays["zones"]) == 1 and len(st.keepout_grids) == 1
    assert st.status["phase"] == "stopped" and st.status["source"].startswith("replay:")
    assert st.refused == {"frame": 0, "contract": 0, "version": 0}


# -- the simulated source (this repository's SimulatedExploreStack, no robot, no ROS) ---------------

def test_the_simulated_source_writes_a_growing_map_and_a_moving_robot(tmp_path):
    out = tmp_path / "sim.jsonl"
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "fleet_adapter_ros2.mapview.sim_source", "--out", str(out),
                    "--speed", "60", "--seconds", "30", "--map-every", "5"],
                   check=True, env=env, capture_output=True, timeout=120)
    st, known, poses = c.ViewState(), [], set()
    for line in out.read_text().splitlines():
        m = c.loads(line)
        assert st.accept(m)
        if m["kind"] == "MapSnapshot":
            known.append(sum(v != -1 for row in st.map[0] for v in row))
        if m["kind"] == "PoseSample":
            poses.add((round(m["x"], 1), round(m["y"], 1)))
    assert len(known) >= 3 and known == sorted(known) and known[-1] > known[0]   # the map only grows
    assert len(poses) > 10                                                     # the robot moved
    assert st.overlays["target"]["label"] == "B"                                # A was reached
    assert st.status["phase"] == "stopped" and st.status["simulated"] is True
    assert st.refused == {"frame": 0, "contract": 0, "version": 0}


# -- the record tool: read-only through its runner ---------------------------------------------------

def test_record_sends_its_runner_only_the_two_echoes_and_their_end(tmp_path):
    from fleet_adapter_ros2.mapview import contract_tool as tool

    calls = tmp_path / "calls.txt"
    runner = tmp_path / "runner"
    # A stand-in for the robot side: it logs every line it is asked to run, and answers the two
    # echoes with canned output, then ends (so the recording ends: both streams closed).
    runner.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$1\" >> {calls}\n"
        "case \"$1\" in\n"
        f"  *'topic echo /map'*) cat {tmp_path / 'map.txt'} ;;\n"
        f"  *tf2_echo*) cat {tmp_path / 'tf.txt'} ;;\n"
        "esac\n")
    runner.chmod(0o755)
    (tmp_path / "map.txt").write_text(MAP_ECHO + "---\n")
    (tmp_path / "tf.txt").write_text(TF_ECHO)
    run = _synthetic_run(tmp_path)
    (run / "targets.txt").write_text("A 1.0 2.0\n")
    out = tmp_path / "rec.jsonl"
    tool.main(["record", "--runner", str(runner), "--run-dir", str(run), "--session", "s1",
               "--out", str(out), "--robot", "sim-a"])
    sent = calls.read_text().splitlines()
    assert sorted(sent[:2]) == sorted([tool.MAP_CMD, tool.POSE_CMD]) and sent[2:] == [tool.END_CMD]
    st = c.ViewState()
    kinds = []
    for line in out.read_text().splitlines():
        m = c.loads(line)
        kinds.append(m["kind"])
        assert st.accept(m)
    assert {"MapSnapshot", "PoseSample", "Overlays"} <= set(kinds)
    assert st.status["phase"] == "stopped" and c.driver_of(st.status["source"]) == "operator"
    assert st.overlays["target"]["label"] == "A" and st.keepout_grids


# -- the desktop viewer --------------------------------------------------------------------------------

def test_the_desktop_viewer_draws_a_file_and_names_an_unknown_version(tmp_path):
    pytest.importorskip("matplotlib")
    from fleet_adapter_ros2.mapview import contract_view as view

    src = tmp_path / "v.jsonl"
    lines = [c.dumps(status(source="guided:s1")), c.dumps(snap(grid())), c.dumps(pose(0.1, 0.2)),
             json.dumps({**pose(), "contractVersion": "mapview/1"}), "not json"]
    src.write_text("\n".join(lines) + "\n")
    png = tmp_path / "v.png"
    view.main(["--source", str(src), "--png", str(png)])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    st = c.ViewState()
    view.Feed(src, "follow", 1.0).take(st)
    assert st.refused == {"frame": 0, "contract": 1, "version": 1}
    head = view.title(st, 5.0, st.arrived["PoseSample"])
    assert "an operator drives the robot" in head and "mapview/1" in head


# -- what the contract path may import ---------------------------------------------------------------

def test_the_contract_and_its_readers_import_nothing_from_the_adapter():
    """The contract, its sources, the record tool and the viewers are what another fleet's
    writer or a viewer would take: none of them may need the adapter."""
    import ast

    pkg = REPO / "fleet_adapter_ros2" / "mapview"
    for name in ("contract.py", "sources.py", "contract_tool.py", "contract_view.py"):
        tree = ast.parse((pkg / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            else:
                continue
            if mod.startswith("fleet_adapter"):
                assert mod.startswith("fleet_adapter_ros2.mapview"), f"{name} imports {mod}"
