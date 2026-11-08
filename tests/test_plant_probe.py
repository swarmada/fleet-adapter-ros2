# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The plant probe: a failed read is ``unknown``, never a number; and the parsers read the
real robot's files.

The defect this guards against is a harness that carries a plant constant it did not read:
``max_velocity: 0.5`` because a lookup failed looks exactly like ``max_velocity: 0.5``
because the robot said so. So the central tests here make every read fail, or answer
``NOT_SET``, or time out, and require the document to say ``unknown`` with the reason.

The parser tests run on synthetic fixtures: two transcripts and three Nav2 parameter files,
generated from what a physical robot reported and carrying its numbers
(``tests/fixtures/plant/``, provenance and md5s in its README).
"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import re
import sys
import types

import pytest

_PROBE = pathlib.Path(__file__).resolve().parents[1] / "testbed" / "probe"
if str(_PROBE) not in sys.path:
    sys.path.insert(0, str(_PROBE))

import plant_fields as pf  # noqa: E402
import plant_guided as pg  # noqa: E402
import plant_probe as pp  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "plant"


def _opts(**kw):
    base = dict(window=1.0, discovery=0.0, declare_kind=None, reference_utc=None, param_dir=[])
    base.update(kw)
    return types.SimpleNamespace(**base)


# ── readers ─────────────────────────────────────────────────────────────────


class FailingReader:
    """Every read fails. Nothing may come back as a number."""

    def env(self, name):
        return None

    def rmw(self):
        raise pp.ReadFailed("rmw read failed (test)")

    def domain_id(self):
        raise pp.ReadFailed("domain read failed (test)")

    def nodes(self):
        raise pp.ReadFailed("graph read failed (test)")

    def topics(self):
        raise pp.ReadFailed("graph read failed (test)")

    def publishers(self, topic):
        raise pp.ReadFailed("graph read failed (test)")

    def subscriber_count(self, topic):
        raise pp.ReadFailed("graph read failed (test)")

    def get_params(self, node, names, timeout=2.0):
        raise pp.ReadFailed(f"{node}/get_parameters did not answer (test)")

    def sample(self, plan, window_s, latched=()):
        raise pp.ReadFailed("sampling failed (test)")

    def clock(self):
        raise pp.ReadFailed("clock read failed (test)")

    def own_publishers(self):
        raise pp.ReadFailed("own publishers read failed (test)")

    def ament_prefixes(self):
        raise pp.ReadFailed("no AMENT_PREFIX_PATH (test)")


class PV:
    """Stands in for rcl_interfaces/msg/ParameterValue, zero-filled like the real one."""

    def __init__(self, type_=0, **slots):
        self.type = type_
        self.bool_value = False
        self.integer_value = 0
        self.double_value = 0.0
        self.string_value = ""
        self.byte_array_value = []
        self.bool_array_value = []
        self.integer_array_value = []
        self.double_array_value = []
        self.string_array_value = []
        for k, v in slots.items():
            setattr(self, k, v)


def odom_recs(n=40, hz=20.0):
    return [{"recv": 100.0 + i / hz, "wall": 1.7e9 + i / hz, "stamp": 1.7e9 + i / hz,
             "x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0} for i in range(n)]


class GraphReader(FailingReader):
    """A reader that answers the graph; parameters and samples are per test."""

    def __init__(self, nodes, topics, pubs, samples=None, params=None):
        self._nodes, self._topics, self._pubs = nodes, topics, pubs
        self._samples = samples or {}
        self._params = params or {}

    def env(self, name):
        return {"ROS_DISTRO": "jazzy"}.get(name)

    def rmw(self):
        return "rmw_fastrtps_cpp"

    def domain_id(self):
        return 42

    def nodes(self):
        return self._nodes

    def topics(self):
        return self._topics

    def publishers(self, topic):
        return self._pubs.get(topic, [])

    def subscriber_count(self, topic):
        return 0

    def get_params(self, node, names, timeout=2.0):
        p = self._params.get(node)
        if isinstance(p, Exception):
            raise p
        if p is None:
            raise pp.ReadFailed(f"{node}/get_parameters did not answer (test)")
        return {n: p.get(n, PV()) for n in names}

    def sample(self, plan, window_s, latched=()):
        return {t: list(self._samples.get(t, [])) for t in plan}

    def clock(self):
        return {"ros_time_s": 1.7e9, "wall_s": 1.7e9 + 0.001, "use_sim_time": False}

    def own_publishers(self):
        return ["/parameter_events"]

    def ament_prefixes(self):
        return []


# ── the rule: no field ever holds a default ─────────────────────────────────


def test_field_constructors_refuse_a_value_without_a_source():
    with pytest.raises(pf.FieldError):
        pf.read(0.5, "")
    with pytest.raises(pf.FieldError):
        pf.read(None, "param: /velocity_smoother max_velocity")
    with pytest.raises(pf.FieldError):
        pf.read("unknown", "anything")
    with pytest.raises(pf.FieldError):
        pf.unknown("")
    with pytest.raises(pf.FieldError):
        pf.operator(0.17, "")


def test_a_bare_number_anywhere_in_the_document_is_refused():
    doc = {"schema": pf.SCHEMA, "nav2": {"velocity_smoother": {"max_velocity": 0.5}}}
    with pytest.raises(pf.FieldError, match="bare number 0.5"):
        pf.assert_no_defaults(doc)
    doc = {"nav2": {"max_velocity": {"value": 0.5}}}
    with pytest.raises(pf.FieldError):
        pf.assert_no_defaults(doc)


def test_writer_refuses_a_defaulted_field(tmp_path):
    doc = {"schema": pf.SCHEMA, "nav2": {"max_velocity": {"value": 0.5, "kind": "read",
                                                          "source": ""}}}
    with pytest.raises(pf.FieldError):
        pf.write_plant(doc, str(tmp_path))
    assert not (tmp_path / "plant.yaml").exists()


def test_every_read_failing_yields_unknown_everywhere_and_no_number():
    doc = pp.build_plant(FailingReader(), _opts())
    n = pf.assert_no_defaults(doc)
    assert n > 10
    known = [(p, f) for p, f in pf.iter_fields(doc) if not pf.is_unknown(f)]
    assert known == [], f"a failed read produced a value: {known}"
    ms = doc["nav2"]["velocity_smoother"]["max_velocity"]
    assert ms["value"] == "unknown" and "not in the graph" in ms["reason"]
    assert doc["battery"]["voltage"]["value"] == "unknown"
    assert doc["plant"]["kind"]["value"] == "unknown"


def test_param_service_timeout_is_unknown_with_the_timeout_as_reason():
    r = GraphReader(nodes=["/velocity_smoother"], topics=[], pubs={},
                    params={"/velocity_smoother": pp.ReadFailed(
                        "/velocity_smoother/get_parameters timed out after 2.0 s")})
    doc = pp.build_plant(r, _opts())
    f = doc["nav2"]["velocity_smoother"]["max_decel"]
    assert f == {"value": "unknown",
                 "reason": "/velocity_smoother/get_parameters timed out after 2.0 s"}


def test_param_not_set_is_unknown_not_the_zero_the_message_carries():
    # The real ParameterValue for an undeclared parameter is type NOT_SET with every slot
    # zero-filled. Reading .double_value would give 0.0 -- a number nobody measured.
    pv = PV(0)
    assert pv.double_value == 0.0 and pv.double_array_value == []
    with pytest.raises(pf.ParamNotSet):
        pf.param_value(pv)
    r = GraphReader(nodes=["/velocity_smoother"], topics=[], pubs={},
                    params={"/velocity_smoother": {}})
    doc = pp.build_plant(r, _opts())
    for k in pp.SMOOTHER_PARAMS:
        f = doc["nav2"]["velocity_smoother"][k]
        assert f["value"] == "unknown" and "NOT_SET" in f["reason"], (k, f)


def test_live_parameter_is_read_with_its_node_and_name_as_source():
    live = {"max_velocity": PV(8, double_array_value=[0.35, 0.0, 1.0]),
            "max_decel": PV(8, double_array_value=[-2.5, 0.0, -3.2]),
            "feedback": PV(4, string_value="OPEN_LOOP")}
    r = GraphReader(nodes=["/velocity_smoother"], topics=[], pubs={},
                    params={"/velocity_smoother": live})
    doc = pp.build_plant(r, _opts())
    vs = doc["nav2"]["velocity_smoother"]
    assert vs["max_velocity"] == {"value": [0.35, 0.0, 1.0], "kind": "read",
                                  "source": "param: /velocity_smoother max_velocity"}
    assert vs["max_decel"]["value"] == [-2.5, 0.0, -3.2]
    assert vs["min_velocity"]["value"] == "unknown"      # not answered -> NOT_SET -> unknown
    pf.assert_no_defaults(doc)


def test_a_topic_with_publishers_and_no_messages_has_no_rate():
    r = GraphReader(nodes=["/robot"], topics=[("/odom", [pp.T_ODOM])],
                    pubs={"/odom": ["/robot"]}, samples={"/odom": []})
    doc = pp.build_plant(r, _opts())
    f = doc["rates"]["/odom"]
    assert f["value"] == "unknown" and "0 message(s)" in f["reason"]


def test_battery_with_no_publisher_says_so():
    r = GraphReader(nodes=["/robot"], topics=[("/battery_state", [pp.T_BATT])],
                    pubs={"/battery_state": []})
    doc = pp.build_plant(r, _opts())
    assert doc["battery"]["voltage"] == {"value": "unknown",
                                         "reason": "/battery_state has 0 publishers"}


def test_battery_nan_is_the_messages_own_not_measured_and_stays_unknown():
    recs = [{"recv": 1.0 + i, "wall": 1.0 + i, "stamp": None, "voltage": float("nan"),
             "percentage": 0.5, "present": True} for i in range(5)]
    r = GraphReader(nodes=["/robot"], topics=[("/battery_state", [pp.T_BATT])],
                    pubs={"/battery_state": ["/robot"]}, samples={"/battery_state": recs})
    doc = pp.build_plant(r, _opts())
    assert doc["battery"]["voltage"]["value"] == "unknown"
    assert doc["battery"]["percentage"]["value"] == 0.5


# ── simulated plant labelling ───────────────────────────────────────────────


def fake_robot_reader():
    return GraphReader(nodes=["/fake_robot"],
                       topics=[("/odom", [pp.T_ODOM]), ("/cmd_vel", [pp.T_TWIST])],
                       pubs={"/odom": ["/fake_robot"]}, samples={"/odom": odom_recs()})


def test_fake_robot_is_labelled_simulated_with_the_marker_as_source():
    doc = pp.build_plant(fake_robot_reader(), _opts())
    k = doc["plant"]["kind"]
    assert k["value"] == "simulated" and "/fake_robot" in k["source"]
    assert doc["rates"]["/odom"]["value"]["hz"] == pytest.approx(20.0, rel=1e-3)


def test_operator_cannot_declare_hardware_over_a_simulator_marker():
    doc = pp.build_plant(fake_robot_reader(), _opts(declare_kind="hardware"))
    k = doc["plant"]["kind"]
    assert k["value"] == "unknown" and "conflict" in k["reason"]


def test_hardware_is_only_ever_an_operator_statement():
    r = GraphReader(nodes=["/odom_publisher"], topics=[], pubs={})
    assert pp.build_plant(r, _opts())["plant"]["kind"]["value"] == "unknown"
    k = pp.build_plant(r, _opts(declare_kind="hardware"))["plant"]["kind"]
    assert k["value"] == "hardware" and k["kind"] == "operator"


def test_summary_ends_with_needs_a_guided_step():
    doc = pp.build_plant(fake_robot_reader(), _opts())
    text = pp.summary(doc)
    tail = text[text.rindex("Needs a guided step:"):]
    assert "motion_channel.odometry./odom" in tail
    assert "not-testable-here" in tail
    assert "SIMULATED PLANT" in text


# ── parsers against the robot's numbers ────────────────────────────────────

yaml = pytest.importorskip("yaml")


def _load(name):
    return yaml.safe_load((FIX / name).read_text())


def test_fixture_bytes_are_the_pinned_ones():
    want = {"synthetic_nav2_params.yaml": "810cffe5b9a6df72d90894c5c28daac0",
            "synthetic_nav2_controller_teb.yaml": "f053863df6d544ad7bfe6e35ac21aa2a",
            "synthetic_nav2_controller_dwb.yaml": "9202a4b43ca93fc8db83ee1f2bdbb405",
            "synthetic_lidar_mount.txt": "dd4963a88ac215bda88fd833bf3f898f",
            "synthetic_velocity_limits.txt": "41cfc6370cbfddc163034d6afbc7bee7"}
    for name, md5 in want.items():
        assert pf.md5_text((FIX / name).read_bytes()) == md5, name


def test_params_recover_the_velocity_cap_and_decel():
    got = pf.parse_nav2_param_doc(_load("synthetic_nav2_params.yaml"))
    vs = got["velocity_smoother"]
    assert vs["max_velocity"] == [0.35, 0.0, 1.0]
    assert vs["max_decel"] == [-2.5, 0.0, -3.2]
    assert vs["feedback"] == "OPEN_LOOP"


def test_file_cap_equals_the_live_param_dump_taken_with_nav2_up():
    text = (FIX / "synthetic_velocity_limits.txt").read_text()
    arrays = [[float(x) for x in m.split(",")]
              for m in re.findall(r"Double values are: array\('d', \[([^\]]*)\]\)", text)]
    assert arrays[0] == [0.35, 0.0, 1.0] and arrays[1] == [-2.5, 0.0, -3.2]
    vs = pf.parse_nav2_param_doc(_load("synthetic_nav2_params.yaml"))["velocity_smoother"]
    assert (vs["max_velocity"], vs["max_decel"]) == (arrays[0], arrays[1])


def test_params_recover_the_planner_radius_and_no_footprint():
    got = pf.parse_nav2_param_doc(_load("synthetic_nav2_params.yaml"))
    for cm in ("local_costmap", "global_costmap"):
        assert got[cm]["robot_radius"] == 0.05
        assert "footprint" not in got[cm]            # absent stays absent, never filled
        assert got[cm]["inflation_layer.inflation_radius"] == 0.25
        assert got[cm]["robot_base_frame"] == "base_footprint"
    assert pf.radius_in_force("[]", 0.05)[:2] == ("robot_radius", 0.05)
    assert pf.radius_in_force(None, 0.05)[0] is None


def test_both_controller_files_name_their_planner():
    teb = pf.parse_nav2_param_doc(_load("synthetic_nav2_controller_teb.yaml"))["controller_server"]
    fp = teb["plugins"]["FollowPath"]
    assert fp["plugin"] == "teb_local_planner::TebLocalPlannerROS" and fp["family"] == "TEB"
    assert fp["footprint_model"] == {"type": "circular", "radius": 0.01}
    assert fp["min_obstacle_dist"] == 0.25 and fp["max_vel_x"] == 0.4
    assert teb["controller_frequency"] == 10.0
    dwb = pf.parse_nav2_param_doc(_load("synthetic_nav2_controller_dwb.yaml"))["controller_server"]
    assert dwb["plugins"]["FollowPath"]["family"] == "DWB"


def test_file_candidates_are_found_and_labelled_as_files():
    cands, err = pp._file_candidates([], [str(FIX)])
    assert err is None
    by = {pathlib.Path(c["file"]).name: c for c in cands}
    assert set(by) == {"synthetic_nav2_params.yaml", "synthetic_nav2_controller_teb.yaml",
                       "synthetic_nav2_controller_dwb.yaml"}
    f = by["synthetic_nav2_params.yaml"]["values"]["velocity_smoother.max_velocity"]
    assert f["value"] == [0.35, 0.0, 1.0] and f["source"].startswith("file: ")
    assert "810cffe5b9a6df72d90894c5c28daac0" in f["source"]


def _urdf_from_dump():
    text = (FIX / "synthetic_lidar_mount.txt").read_text()
    line = next(ln for ln in text.splitlines() if "String value is: " in ln)
    return line.split("String value is: ", 1)[1]


def _tf2_echo_translations():
    text = (FIX / "synthetic_lidar_mount.txt").read_text()
    out = {}
    for base in ("base_footprint", "base_link"):
        i = text.index(f"tf2_echo {base} lidar_frame")
        m = re.search(r"Translation: \[([-\d., ]+)\]", text[i:])
        out[base] = [float(x) for x in m.group(1).split(",")]
    return out


def test_urdf_dump_recovers_the_lidar_pose_and_agrees_with_tf2_echo():
    joints = pf.urdf_joints(_urdf_from_dump())
    assert joints["lidar_frame"]["parent"] == "base_link"
    edges = pf.urdf_edges(joints)
    echo = _tf2_echo_translations()
    for base, z in (("base_link", 0.092501), ("base_footprint", 0.162501)):
        t, q, frames = pf.chain_pose(edges, base, "lidar_frame")
        assert t == pytest.approx((-0.012242, -8.533e-05, z), abs=1e-9), base
        assert pf.pose_dict(t, q)["yaw"] == 0.0
        assert frames[0] == base and frames[-1] == "lidar_frame"
        # tf2_echo prints 3 decimals: the URDF and the live TF agree to that resolution.
        assert list(t) == pytest.approx(echo[base], abs=6e-4), base
    t, q, _ = pf.chain_pose(edges, "base_link", "imu_link")
    assert pf.pose_dict(t, q)["yaw"] == pytest.approx(-1.57, abs=1e-6)


def test_chain_pose_inverts_when_the_target_is_above_the_base():
    edges = pf.urdf_edges(pf.urdf_joints(_urdf_from_dump()))
    t, _, frames = pf.chain_pose(edges, "lidar_frame", "base_footprint")
    assert t == pytest.approx((0.012242, 8.533e-05, -0.162501), abs=1e-9)
    assert pf.chain_pose(edges, "base_link", "no_such_frame") is None


def test_yaml_writer_round_trips_through_pyyaml(tmp_path):
    doc = pp.build_plant(fake_robot_reader(), _opts())
    paths = pf.write_plant(doc, str(tmp_path))
    from_yaml = yaml.safe_load(pathlib.Path(paths["yaml"]).read_text())
    from_json = json.loads(pathlib.Path(paths["json"]).read_text())
    assert from_yaml == from_json


# ── the guided steps ────────────────────────────────────────────────────────


def test_guided_on_a_simulated_plant_is_not_testable_here_and_invents_no_push(tmp_path):
    doc = pp.build_plant(fake_robot_reader(), _opts())
    pf.write_plant(doc, str(tmp_path))
    assert pg.main(["--plant", str(tmp_path / "plant.json")]) == 0
    out = json.loads((tmp_path / "plant.json").read_text())
    for path, f in pf.iter_fields(out["motion_channel"]):
        assert f["value"] == "unknown" and f["status"] == "not-testable-here", path
        assert f["unlock"]
    assert out["geometry"]["physical_width_m"]["status"] == "not-testable-here"
    assert out["guided"]["ran"] == "no — simulated plant"
    pf.assert_no_defaults(out)


def _win(values, t0, key_fn, hz=20.0):
    return [dict(key_fn(v), recv=t0 + i / hz) for i, v in enumerate(values)]


def _bounds():
    return {"rest": (0.0, 5.0), "hand": (10.0, 15.0), "push": (20.0, 26.0),
            "after": (30.0, 35.0)}


def _odom(xs_by_window):
    recs = []
    for w, (t0, _) in _bounds().items():
        recs += _win(xs_by_window[w], t0, lambda x: {"x": x, "y": 0.0, "vx": 0.0, "vy": 0.0})
    return pg.split_windows(recs, _bounds())


def _imu(push_spike):
    recs = []
    for w, (t0, _) in _bounds().items():
        spike = push_spike if w == "push" else 0.0
        vals = [9.46 + (0.01 if i % 2 else -0.01) + (spike if i == 20 else 0.0)
                for i in range(100)]
        recs += _win(vals, t0, lambda a: {"ax": 0.0, "ay": 0.0, "az": a,
                                          "wx": 0.0, "wy": 0.0, "wz": 0.0})
    return pg.split_windows(recs, _bounds())


def test_odometry_that_does_not_move_under_a_witnessed_push_is_command_derived():
    motion = pg.odom_motion(_odom({w: [0.0] * 40 for w in _bounds()}))
    imu = pg.imu_refutation(_imu(push_spike=1.5))
    assert imu["refutes_in_push"] is True and imu["refutes_in_hand"] is False
    wit = pg.push_witnesses({"/imu": imu}, {}, None)
    f = pg.classify_odometry("/odom", motion, wit, commands=[])
    assert f["value"] == "command-derived" and f["kind"] == "measured"


def test_odometry_that_moves_under_a_push_is_measured():
    xs = {"rest": [0.0] * 40, "hand": [0.0] * 40, "push": [0.05] * 40, "after": [0.1] * 40}
    f = pg.classify_odometry("/odom", pg.odom_motion(_odom(xs)), [], commands=[])
    assert f["value"] == "measured"


def test_still_odometry_without_a_push_witness_is_unknown_not_command_derived():
    motion = pg.odom_motion(_odom({w: [0.0] * 40 for w in _bounds()}))
    imu = pg.imu_refutation(_imu(push_spike=0.0))
    wit = pg.push_witnesses({"/imu": imu}, {}, None)
    assert wit == []
    f = pg.classify_odometry("/odom", motion, wit, commands=[])
    assert f["value"] == "unknown" and "not-pushed" in f["reason"]


def test_a_velocity_command_during_the_test_voids_it():
    motion = pg.odom_motion(_odom({w: [0.0] * 40 for w in _bounds()}))
    f = pg.classify_odometry("/odom", motion, ["x"], commands=[{"vx": 0.1, "vy": 0.0,
                                                                "wz": 0.0}])
    assert f["value"] == "unknown" and "void" in f["reason"]


def test_imu_is_reported_as_refuting_never_as_confirming():
    imu = pg.imu_refutation(_imu(push_spike=1.5))
    f = pg.imu_field("/imu", imu)
    assert f["value"] == "refutes rest during the push"
    assert "cannot confirm" in f["source"]


def test_imu_with_no_push_window_messages_is_unknown_not_did_not_refute():
    by = _imu(push_spike=1.5)
    by["push"] = []
    imu = pg.imu_refutation(by)
    assert imu["refutes_in_push"] is None
    assert pg.imu_field("/imu", imu)["value"] == "unknown"


def _scan(ranges_by_window):
    recs = []
    for w, (t0, _) in _bounds().items():
        recs += _win(ranges_by_window[w], t0,
                     lambda r: {"ranges": r, "range_min": 0.05, "range_max": 12.0}, hz=10.0)
    return pg.split_windows(recs, _bounds())


def test_scan_difference_reports_numbers_and_the_hand_on_control():
    base = [1.0] * 100
    noisy = [[1.0 + (0.002 if (i + j) % 2 else -0.002) for i in range(100)] for j in range(30)]
    push = [[1.0 + 0.01 * j] * 100 for j in range(30)]
    hand = [[1.0] * 90 + [0.3 + 0.001 * j] * 10 for j in range(30)]
    r = pg.scan_distributions(_scan({"rest": noisy, "hand": hand, "push": push,
                                     "after": [base] * 30}))
    assert r["push_separates_from_rest"] is True
    assert r["push_separates_from_hand_on"] is True
    assert r["windows"]["rest"]["mean_abs_m"]["max"] == pytest.approx(0.004)
    f = pg.scan_field("/scan_raw", r)
    assert "not claimed as a rest-confirming channel" in f["source"]


def test_parse_mm_refuses_what_is_not_a_length():
    assert pg.parse_mm("170") == 0.17
    assert pg.parse_mm(" 150 mm ") == 0.15
    for bad in ("", "0", "-5", "abc", "nan", "99999"):
        assert pg.parse_mm(bad) is None, bad


# ── observe only: the scripts cannot command anything ───────────────────────


@pytest.mark.parametrize("name", ["plant_probe.py", "plant_guided.py", "plant_fields.py"])
def test_probe_scripts_create_no_publisher_goal_or_parameter_set(name):
    src = (_PROBE / name).read_text()
    tree = ast.parse(src)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    forbidden_calls = {"create_publisher", "send_goal", "send_goal_async", "set_parameters",
                       "set_parameters_atomically", "publish"}
    assert not (called & forbidden_calls), called & forbidden_calls
    assert not ({"SetParameters", "SetParametersAtomically", "ActionClient"} & imported)


def test_probe_sources_parse_as_python_3_10():
    for name in ("plant_probe.py", "plant_guided.py", "plant_fields.py"):
        ast.parse((_PROBE / name).read_text(), feature_version=(3, 10))


def test_rate_stats_needs_three_samples_and_reports_jitter():
    assert pf.rate_stats([1.0, 2.0]) is None
    s = pf.rate_stats([0.0, 0.1, 0.2, 0.35])
    assert s["n"] == 4 and s["period_ms"]["max"] == pytest.approx(150.0)
    assert s["period_ms"]["sd"] > 0
    assert math.isclose(s["hz"], 3 / 0.35, rel_tol=1e-3)


def test_a_graph_shaped_like_the_captured_robot_recovers_pose_and_lists_file_candidates():
    """The captured robot's shape: URDF on /robot_description, /scan_raw in lidar_frame,
    static TF, a raw-numeric battery topic, /battery_state with no publisher, and Nav2 not
    running — so every live Nav2 field is unknown and the parameter files are candidates."""
    urdf = _urdf_from_dump()
    joints = pf.urdf_joints(urdf)

    def tf_rec(child):
        j = joints[child]
        return (j["parent"], child, tuple(j["xyz"]), pf.quat_from_rpy(*j["rpy"]))

    tf_static = [{"recv": 1.0, "wall": 1.0, "stamp": None,
                  "transforms": [tf_rec("base_link"), tf_rec("lidar_frame"),
                                 tf_rec("imu_link")]}]
    scans = [{"recv": 1.0 + i / 10, "wall": 1.0 + i / 10, "stamp": 1.0 + i / 10,
              "frame": "lidar_frame", "n": 450, "range_min": 0.05, "range_max": 12.0,
              "angle_increment": 0.014, "ranges": [1.0] * 450} for i in range(100)]
    batt_raw = [{"recv": 1.0 + i, "wall": 1.0 + i, "stamp": None, "data": 7354 + i % 3}
                for i in range(10)]
    topics = [("/scan_raw", [pp.T_SCAN]), ("/tf_static", [pp.T_TF]),
              ("/robot_description", [pp.T_STRING]), ("/battery_state", [pp.T_BATT]),
              ("/ros_robot_controller/battery", ["std_msgs/msg/UInt16"]),
              ("/odom_rf2o", [pp.T_ODOM])]
    pubs = {"/scan_raw": ["/MS200"], "/tf_static": ["/robot_state_publisher"],
            "/robot_description": ["/robot_state_publisher"],
            "/ros_robot_controller/battery": ["/ros_robot_controller"]}
    r = GraphReader(nodes=["/MS200", "/robot_state_publisher", "/ros_robot_controller"],
                    topics=topics, pubs=pubs,
                    samples={"/scan_raw": scans, "/tf_static": tf_static,
                             "/robot_description": [{"recv": 1.0, "wall": 1.0, "stamp": None,
                                                     "data": urdf}],
                             "/ros_robot_controller/battery": batt_raw})
    doc = pp.build_plant(r, _opts(declare_kind="hardware", param_dir=[str(FIX)]))
    pf.assert_no_defaults(doc)

    assert doc["plant"]["kind"]["value"] == "hardware"
    assert doc["plant"]["kind"]["kind"] == "operator"
    for sect, prefix in (("tf", "tf: "), ("urdf", "urdf joints: ")):
        f = doc[sect]["lidar_pose"]["lidar_frame_in_base_footprint"]
        assert f["source"].startswith(prefix)
        assert (f["value"]["x"], f["value"]["z"]) == pytest.approx((-0.012242, 0.162501))
        assert doc[sect]["lidar_pose"]["lidar_frame_in_base_link"]["value"]["z"] == \
            pytest.approx(0.092501)
    assert doc["rates"]["/scan"]["reason"] == ("/scan is not in the graph; same-type topics "
                                               "present: /scan_raw")
    assert doc["rates"]["/scan_raw"]["value"]["hz"] == pytest.approx(10.0, rel=1e-3)
    assert doc["rates"]["/odom_rf2o"] == {"value": "unknown",
                                          "reason": "/odom_rf2o has 0 publishers"}
    assert doc["battery"]["voltage"]["reason"] == "/battery_state has 0 publishers"
    raw = doc["battery"]["other_battery_topics"][0]["raw"]
    assert raw["value"] == 7355 and "not converted" in raw["units"]
    vs = doc["nav2"]["velocity_smoother"]["max_velocity"]
    assert vs["value"] == "unknown" and "not in the graph" in vs["reason"]
    names = {pathlib.Path(c["file"]).name for c in doc["nav2"]["file_candidates"]}
    assert "synthetic_nav2_params.yaml" in names
    needs = [p for p, _ in pp.guided_needs(doc)]
    assert "motion_channel.scan_difference./scan_raw" in needs
    assert "geometry.physical_width_m" in needs
    assert "geometry.lidar_height_m" not in needs      # TF and URDF both answered it


def _hardware_doc_with(topics):
    """A probe document for a declared-hardware plant whose rates delivered on ``topics``."""
    rates = {t: pf.measured({"hz": 10.0}, "measured: test", type=tn) for t, tn in topics.items()}
    return {"schema": pf.SCHEMA, "probe": {"tool": "plant_probe.py"},
            "plant": {"kind": pf.operator("hardware", "--declare-kind hardware"),
                      "simulator_markers": pf.unknown("none found")},
            "rates": rates,
            "tf": {"lidar_pose": pf.unknown("no TF")}, "urdf": {"lidar_pose": pf.unknown("x")},
            "geometry": {"physical_width_m": pf.unknown("not from the stack")}}


def test_assemble_on_hardware_writes_a_document_the_writer_accepts(tmp_path):
    """The live path, minus rclpy: windows in, a writable document out."""
    doc = _hardware_doc_with({"/odom": pg.T_ODOM, "/imu": pg.T_IMU})
    odom = [r for w in _odom({w: [0.0] * 40 for w in _bounds()}).values() for r in w]
    imu = [r for w in _imu(push_spike=1.5).values() for r in w]
    answers = {"distance_cm": "9", "physical_width_m": "170", "lidar_height_m": "abc"}
    doc = pg.assemble(doc, {"/odom": odom, "/imu": imu, "/cmd_vel": []}, _bounds(),
                      lambda prompt, key: answers.get(key))
    doc["guided"] = {"tool": "plant_guided.py", "windows_s": {"rest": 5.001},
                     "own_publishers": [], "cmd_vel_messages_during_test": 0}
    pf.write_plant(doc, str(tmp_path))            # would raise on a defaulted field
    mc = doc["motion_channel"]
    assert mc["odometry"]["/odom"]["value"] == "command-derived"
    assert mc["imu_refutes_rest"]["/imu"]["refutes"] is True
    assert mc["operator_push_distance_cm"]["value"] == 9.0
    assert mc["scan_difference"]["value"] == "unknown"      # no scan topic delivered
    geo = doc["geometry"]
    assert geo["physical_width_m"]["value"] == 0.17 and geo["physical_width_m"]["kind"] == \
        "operator"
    assert geo["lidar_height_m"] == {"value": "unknown",
                                     "reason": "operator typed 'abc', not a length in mm"}


def test_assemble_with_nobody_at_a_terminal_is_unknown_not_zero():
    doc = _hardware_doc_with({"/odom": pg.T_ODOM})
    odom = [r for w in _odom({w: [0.0] * 40 for w in _bounds()}).values() for r in w]
    doc = pg.assemble(doc, {"/odom": odom}, _bounds(), lambda prompt, key: None)
    pf.assert_no_defaults(doc)
    assert doc["motion_channel"]["operator_push_distance_cm"]["reason"] == \
        "no operator at a terminal"
    assert doc["geometry"]["physical_width_m"]["reason"] == "no operator at a terminal"
    assert doc["motion_channel"]["odometry"]["/odom"]["value"] == "unknown"   # no witness


def test_declared_hardware_over_a_simulator_still_gets_no_guided_push(tmp_path):
    doc = pp.build_plant(fake_robot_reader(), _opts(declare_kind="hardware"))
    assert doc["plant"]["kind"]["value"] == "unknown"                   # the conflict
    assert doc["plant"]["simulator_markers"]["value"]
    pf.write_plant(doc, str(tmp_path))
    assert pg.main(["--plant", str(tmp_path / "plant.json")]) == 0
    out = json.loads((tmp_path / "plant.json").read_text())
    for path, f in pf.iter_fields(out["motion_channel"]):
        assert f.get("status") == "not-testable-here", path
