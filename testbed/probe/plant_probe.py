#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Read what a running ROS 2 robot stack can say about itself, and write a draft plant.yaml.

OBSERVE ONLY. This process creates no publisher of its own, sends no goal and sets no
parameter. It subscribes, asks the graph, calls ``GetParameters`` (a read), and reads
files. ``probe.own_publishers`` in the output lists what rclpy itself opened for the
probe's node, so the claim is checked on every run rather than asserted here.

Every field is read with its source, or ``unknown`` with the reason (``plant_fields``).
A failed read never becomes a number.

Runs with ``rclpy`` + the standard library on Python >= 3.10 (ROS 2 Humble and later).
Nothing is installed and nothing is written outside ``--out``.

    python3 plant_probe.py --out /tmp/probe/out                  # ~30 s
    python3 plant_probe.py --out OUT --declare-kind hardware     # operator says: real robot
    python3 plant_probe.py --out OUT --reference-utc "$(python3 -c 'import time;print(time.time())')"

Exit 0 when the draft was written (however many fields are unknown); 2 on bad arguments;
1 when the document could not be written.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plant_fields as pf  # noqa: E402
from plant_fields import derived, measured, read, unknown  # noqa: E402

VERSION = "0.1.0"

T_ODOM = "nav_msgs/msg/Odometry"
T_SCAN = "sensor_msgs/msg/LaserScan"
T_IMU = "sensor_msgs/msg/Imu"
T_BATT = "sensor_msgs/msg/BatteryState"
T_JOINTS = "sensor_msgs/msg/JointState"
T_DIAG = "diagnostic_msgs/msg/DiagnosticArray"
T_TF = "tf2_msgs/msg/TFMessage"
T_TWIST = "geometry_msgs/msg/Twist"
T_STRING = "std_msgs/msg/String"
T_CLOCK = "rosgraph_msgs/msg/Clock"
NUMERIC_STD = {f"std_msgs/msg/{t}" for t in (
    "UInt8", "UInt16", "UInt32", "UInt64", "Int8", "Int16", "Int32", "Int64",
    "Float32", "Float64")}

RATE_TYPES = (T_ODOM, T_SCAN, T_IMU, T_BATT, T_JOINTS)
# Rows that always appear, present or not, so their absence is stated rather than silent.
NAMED_ROWS = {"/odom": T_ODOM, "/scan": T_SCAN, "/battery_state": T_BATT,
              "/joint_states": T_JOINTS}
BASE_FRAMES = ("base_link", "base_footprint")

# Simulator markers. A match labels the plant simulated, with the marker as the source.
SIM_NODE_PATTERNS = (
    (re.compile(r"^/fake_robot$"), "the Swarmada testbed's simulated plant (fake_robot_node.py)"),
    (re.compile(r"gazebo|/gz_|ros_gz|ign_bridge|/ignition"), "a Gazebo node"),
    (re.compile(r"isaac", re.I), "an Isaac Sim node"),
    (re.compile(r"webots|stage_ros|flatland", re.I), "a simulator node"),
)

NAV2_NODES = {
    "velocity_smoother": "/velocity_smoother",
    "controller_server": "/controller_server",
    "local_costmap": "/local_costmap/local_costmap",
    "global_costmap": "/global_costmap/global_costmap",
}
SMOOTHER_PARAMS = ("max_velocity", "min_velocity", "max_accel", "max_decel", "feedback",
                   "odom_topic")
COSTMAP_PARAMS = ("robot_radius", "footprint", "footprint_padding", "robot_base_frame",
                  "inflation_layer.inflation_radius")
CONTROLLER_SUB = ("plugin", "footprint_model.type", "footprint_model.radius",
                  "min_obstacle_dist", "max_vel_x")

# Where parameter FILES are looked for: every ament prefix's share/*/{config,params}.
FILE_GLOBS = ("share/*/config/*.yaml", "share/*/params/*.yaml")
FILE_MAX_BYTES = 512 * 1024


class ReadFailed(Exception):
    """A read that did not produce an answer. The message becomes the field's reason."""


class _NoPublisher:
    """Stands in for rclpy's /parameter_events publisher on the probe's node: sends nothing."""

    def publish(self, msg) -> None:
        return None

    def destroy(self) -> bool:
        return True


# ── message summaries (duck-typed so they are testable without ROS) ─────────


def _stamp(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    s = h.stamp
    return s.sec + s.nanosec * 1e-9


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def summarize(type_name: str, msg) -> dict:
    """The few numbers the probe and the guided steps need from one message."""
    out = {"stamp": _stamp(msg)}
    if type_name == T_ODOM:
        p = msg.pose.pose
        tw = msg.twist.twist
        out.update(x=p.position.x, y=p.position.y, yaw=_yaw(p.orientation),
                   vx=tw.linear.x, vy=tw.linear.y, wz=tw.angular.z,
                   frame=msg.header.frame_id, child=msg.child_frame_id)
    elif type_name == T_SCAN:
        out.update(frame=msg.header.frame_id, n=len(msg.ranges),
                   range_min=msg.range_min, range_max=msg.range_max,
                   angle_increment=msg.angle_increment, ranges=list(msg.ranges))
    elif type_name == T_IMU:
        a, w = msg.linear_acceleration, msg.angular_velocity
        out.update(frame=msg.header.frame_id, ax=a.x, ay=a.y, az=a.z, wx=w.x, wy=w.y, wz=w.z)
    elif type_name == T_BATT:
        out.update(voltage=msg.voltage, percentage=msg.percentage, present=msg.present)
    elif type_name == T_JOINTS:
        out.update(names=list(msg.name))
    elif type_name == T_DIAG:
        sim = []
        hw = []
        for st in msg.status:
            hw.append(st.hardware_id)
            for kv in st.values:
                if kv.key.lower() == "simulated" and kv.value.lower() in ("true", "1", "yes"):
                    sim.append(st.name)
        out.update(hardware_ids=sorted(set(hw)), simulated_status=sim)
    elif type_name == T_TF:
        out["transforms"] = [
            (t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/"),
             (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z),
             (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
              t.transform.rotation.w))
            for t in msg.transforms]
    elif type_name == T_TWIST:
        out.update(vx=msg.linear.x, vy=msg.linear.y, wz=msg.angular.z)
    elif type_name == T_STRING:
        out["data"] = msg.data
    elif type_name in NUMERIC_STD:
        out["data"] = msg.data
    return out


# ── the rclpy reader ────────────────────────────────────────────────────────


class RclpyReader:
    """Every live read the probe makes. Each method raises ReadFailed with a reason."""

    def __init__(self, node_name: str = "swarmada_plant_probe") -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        self._rclpy = rclpy
        rclpy.init(args=None)

        class _SilentNode(Node):
            """A node with no publisher at all.

            rclpy opens /parameter_events for every node and announces the node's own
            ``use_sim_time`` there as soon as it is created (Humble: one publish() call,
            Jazzy: two, counted by wrapping Publisher.publish). Whether a listener receives
            it depends on discovery timing, so a silent listener proves nothing. The probe
            declares and sets nothing, so that publisher is never created;
            own_publishers() checks the result live.
            """

            def create_publisher(self, msg_type, topic, *a, **kw):
                if topic == "/parameter_events":
                    return _NoPublisher()
                raise RuntimeError(f"the probe does not publish ({topic})")

        # No /rosout publisher and no parameter services either.
        self.node = _SilentNode(node_name, enable_rosout=False, start_parameter_services=False)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def close(self) -> None:
        try:
            self.executor.shutdown()
            self.node.destroy_node()
        finally:
            self._rclpy.try_shutdown() if hasattr(self._rclpy, "try_shutdown") \
                else self._rclpy.shutdown()

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=min(0.05, max(0.0, end - time.monotonic())))

    # graph
    def env(self, name: str):
        return os.environ.get(name)

    def rmw(self) -> str:
        return self._rclpy.get_rmw_implementation_identifier()

    def domain_id(self) -> int:
        ctx = self.node.context
        if not hasattr(ctx, "get_domain_id"):
            raise ReadFailed("rclpy Context has no get_domain_id() in this distro")
        return int(ctx.get_domain_id())

    def nodes(self) -> list:
        """Every node in the graph except this probe's own."""
        me = self.node.get_fully_qualified_name()
        out = []
        for name, ns in self.node.get_node_names_and_namespaces():
            out.append((ns.rstrip("/") + "/" + name) if ns != "/" else "/" + name)
        return sorted(set(out) - {me})

    def topics(self) -> list:
        return sorted((n, list(t)) for n, t in self.node.get_topic_names_and_types())

    def publishers(self, topic: str) -> list:
        infos = self.node.get_publishers_info_by_topic(topic)
        return sorted({(i.node_namespace.rstrip("/") + "/" + i.node_name) for i in infos})

    def subscriber_count(self, topic: str) -> int:
        return int(self.node.count_subscribers(topic))

    def own_publishers(self) -> list:
        name = self.node.get_name()
        ns = self.node.get_namespace()
        return sorted(t for t, _ in self.node.get_publisher_names_and_types_by_node(name, ns))

    # parameters (GetParameters is a read; SetParameters is never created)
    def get_params(self, node_fqn: str, names: list, timeout: float = 2.0) -> dict:
        from rcl_interfaces.srv import GetParameters
        cli = self.node.create_client(GetParameters, node_fqn.rstrip("/") + "/get_parameters")
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                raise ReadFailed(f"{node_fqn}/get_parameters did not answer within {timeout} s")
            req = GetParameters.Request()
            req.names = list(names)
            fut = cli.call_async(req)
            end = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < end:
                self.executor.spin_once(timeout_sec=0.05)
            if not fut.done():
                raise ReadFailed(f"{node_fqn}/get_parameters timed out after {timeout} s")
            res = fut.result()
            if res is None or len(res.values) != len(names):
                raise ReadFailed(f"{node_fqn}/get_parameters returned "
                                 f"{0 if res is None else len(res.values)} values for "
                                 f"{len(names)} names")
            return dict(zip(names, res.values))
        finally:
            self.node.destroy_client(cli)

    # topics
    def sample(self, plan: dict, window_s: float, latched: tuple = ()) -> dict:
        """Subscribe to every topic in ``plan`` ({topic: type}) for ``window_s``.

        Returns {topic: [record]} where a record is summarize()'s dict plus ``recv``
        (monotonic) and ``wall`` (time.time()) at arrival. Topics in ``latched`` are
        subscribed reliable + transient-local (tf_static, robot_description); every
        other topic best-effort, which matches both reliable and best-effort publishers.
        """
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy, qos_profile_sensor_data)
        from rosidl_runtime_py.utilities import get_message
        got = {t: [] for t in plan}
        subs = []
        for topic, type_name in plan.items():
            try:
                cls = get_message(type_name)
            except Exception as e:  # noqa: BLE001 — an unimportable type is a reason
                got[topic] = ReadFailed(f"message type {type_name} not importable here: {e}")
                continue
            if topic in latched:
                qos = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            else:
                qos = qos_profile_sensor_data

            def cb(msg, _t=topic, _n=type_name):
                rec = summarize(_n, msg)
                rec["recv"] = time.monotonic()
                rec["wall"] = time.time()
                got[_t].append(rec)
            subs.append(self.node.create_subscription(cls, topic, cb, qos))
        try:
            self.spin_for(window_s)
        finally:
            for s in subs:
                self.node.destroy_subscription(s)
        return got

    def clock(self) -> dict:
        ros_ns = self.node.get_clock().now().nanoseconds
        wall = time.time()
        return {"ros_time_s": ros_ns * 1e-9, "wall_s": wall,
                "use_sim_time": bool(self.node.get_parameter("use_sim_time").value)}

    def ament_prefixes(self) -> list:
        return [p for p in (os.environ.get("AMENT_PREFIX_PATH") or "").split(":") if p]


# ── building the document ───────────────────────────────────────────────────


def _try(fn, *a, **kw):
    """(value, None) or (None, reason). Any exception is a reason, never a value."""
    try:
        return fn(*a, **kw), None
    except ReadFailed as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001 — the reason is recorded, not swallowed
        return None, f"{type(e).__name__}: {e}"


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}Z"


def simulator_markers(nodes, diag_records, clock_pubs) -> list:
    markers = []
    for n in nodes or []:
        for pat, what in SIM_NODE_PATTERNS:
            if pat.search(n):
                markers.append(f"graph: node {n} is {what}")
    for rec in diag_records or []:
        for name in rec.get("simulated_status", []):
            markers.append(f"topic /diagnostics: status {name!r} declares simulated=true")
    if clock_pubs:
        markers.append(f"graph: /clock has publisher(s) {', '.join(clock_pubs)} (simulated "
                       "or replayed time)")
    return sorted(set(markers))


def plant_kind(markers, declared) -> dict:
    if declared == "hardware" and markers:
        return unknown("conflict: the operator declared hardware, but the graph shows "
                       + "; ".join(markers))
    if markers:
        return read("simulated", "; ".join(markers))
    if declared == "hardware":
        return pf.operator("hardware", "--declare-kind hardware (no simulator marker found "
                           "in the graph, /diagnostics or /clock)")
    if declared == "simulated":
        return pf.operator("simulated", "--declare-kind simulated")
    return unknown("no simulator marker in the graph, /diagnostics or /clock, and no "
                   "--declare-kind given; the probe cannot prove a plant is hardware")


def _rate_field(topic, type_name, pubs, recs, window_s, missing_reason=None):
    if missing_reason:
        return unknown(missing_reason)
    if isinstance(recs, Exception):
        return unknown(str(recs))
    if not pubs:
        return unknown(f"{topic} has 0 publishers")
    arr = pf.rate_stats([r["recv"] for r in recs])
    if arr is None:
        return unknown(f"{len(recs)} message(s) in {window_s} s from {len(pubs)} publisher(s) "
                       f"({', '.join(pubs)}); fewer than 3, so no rate — QoS mismatch, a "
                       "silent publisher, or a rate below 0.2 Hz")
    stamps = [r["stamp"] for r in recs if r.get("stamp") is not None]
    extra = {"type": type_name, "publishers": pubs}
    st = pf.rate_stats(stamps) if stamps else None
    if st is not None:
        extra["by_header_stamp"] = st
    offs = [r["wall"] - r["stamp"] for r in recs if r.get("stamp") is not None]
    if offs:
        extra["receive_minus_stamp_s"] = {"median": round(statistics.median(offs), 6),
                                          "min": round(min(offs), 6),
                                          "max": round(max(offs), 6)}
    return measured(arr, f"measured: arrival times at the probe over a {window_s} s window "
                         "(best-effort subscription); jitter = sd/p99/max of arrival periods",
                    **extra)


def _param_fields(reader, node_fqn, names, present_nodes):
    """{name: field} for one node's parameters; a missing node is a reason, not zeros.

    The service is asked even when the node list does not show the node: node discovery
    can fail where service discovery works (mixed ROS 2 distributions do exactly that),
    so absence from the node list is recorded, not trusted.
    """
    listed = node_fqn in present_nodes
    vals, err = _try(reader.get_params, node_fqn, list(names), 2.0 if listed else 1.0)
    if err:
        why = err if listed else f"node {node_fqn} is not in the graph, and {err}"
        return {n: unknown(why) for n in names}
    out = {}
    for n in names:
        v, e = _try(pf.param_value, vals[n])
        out[n] = unknown(f"{node_fqn} {n}: {e}") if e else read(v, f"param: {node_fqn} {n}")
    return out


def _file_candidates(prefixes, extra_dirs=()):
    """Nav2-shaped parameter files on disk: what a launch COULD load, never what is."""
    try:
        import yaml
    except ImportError:
        return None, "PyYAML is not importable here, so parameter files were not parsed"
    seen = set()
    out = []
    roots = list(prefixes) + list(extra_dirs)
    for root in roots:
        pats = FILE_GLOBS if root in prefixes else ("*.yaml", "**/*.yaml")
        for pat in pats:
            for path in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    if os.path.getsize(real) > FILE_MAX_BYTES:
                        continue
                    with open(real, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if b"ros__parameters" not in data or not any(
                        k in data for k in (b"velocity_smoother", b"controller_server",
                                            b"local_costmap", b"global_costmap")):
                    continue
                try:
                    doc = yaml.safe_load(data.decode("utf-8", "replace"))
                except Exception:  # noqa: BLE001 — an unparsable file is skipped, and counted
                    out.append({"file": path, "md5": pf.md5_text(data),
                                "parse": unknown("YAML did not parse")})
                    continue
                vals = pf.parse_nav2_param_doc(doc) if isinstance(doc, dict) else {}
                if not vals:
                    continue
                src = f"file: {path} (md5 {pf.md5_text(data)})"
                out.append({"file": path, "md5": pf.md5_text(data),
                            "values": _file_fields(vals, src)})
    return out, None


def _file_fields(vals: dict, src: str) -> dict:
    out = {}
    for section, body in vals.items():
        for k, v in body.items():
            if k == "plugins":
                for pid, entry in v.items():
                    for kk, vv in entry.items():
                        out[f"{section}.{pid}.{kk}"] = read(vv, src)
            else:
                out[f"{section}.{k}"] = read(v, src)
    return out


def _pose_fields(edges, bases, lidar_frames, source_prefix):
    out = {}
    for lf in lidar_frames:
        for b in bases:
            key = f"{lf}_in_{b}"
            got = pf.chain_pose(edges, b, lf)
            if got is None:
                out[key] = unknown(f"{source_prefix}: no chain {b} -> {lf}")
            else:
                t, q, frames = got
                out[key] = read(pf.pose_dict(t, q),
                                f"{source_prefix}: {' -> '.join(frames)}", units="m, rad")
    return out


def build_plant(reader, opts) -> dict:
    """The whole document from one reader. Pure given the reader: testable with a fake."""
    t0_wall = time.time()
    t0 = time.monotonic()
    doc = {"schema": pf.SCHEMA,
           "probe": {"tool": "plant_probe.py", "version": VERSION,
                     "started_wall_utc": _utc(t0_wall), "window_s": opts.window}}

    # ROS environment
    ros = {}
    for key, env in (("distro", "ROS_DISTRO"),):
        v = reader.env(env)
        ros[key] = read(v, f"env: {env}") if v else unknown(f"env {env} is not set")
    v, e = _try(reader.rmw)
    ros["rmw"] = read(v, "rclpy.get_rmw_implementation_identifier()") if not e else unknown(e)
    envr = reader.env("RMW_IMPLEMENTATION")
    ros["rmw_env"] = (read(envr, "env: RMW_IMPLEMENTATION") if envr else
                      unknown("env RMW_IMPLEMENTATION is not set; the rmw field above is "
                              "the one in use"))
    v, e = _try(reader.domain_id)
    ros["domain_id"] = read(v, "rclpy Context.get_domain_id()") if not e else unknown(e)

    # discovery settles, then the graph is read
    if hasattr(reader, "spin_for"):
        reader.spin_for(opts.discovery)
    nodes, e = _try(reader.nodes)
    ros["nodes"] = read(nodes, "graph: get_node_names_and_namespaces(), the probe's own "
                        "node excluded", count=len(nodes)) if not e else unknown(e)
    nodes = nodes or []
    topics, e = _try(reader.topics)
    if e:
        ros["topics"] = unknown(e)
        topics = []
    else:
        rows = []
        for name, types in topics:
            pubs, pe = _try(reader.publishers, name)
            subs, se = _try(reader.subscriber_count, name)
            rows.append({"name": name, "types": types,
                         "publishers": (read(len(pubs), "graph: get_publishers_info_by_topic",
                                             nodes=pubs) if not pe else unknown(pe)),
                         "subscribers": (read(subs, "graph: count_subscribers")
                                         if not se else unknown(se))})
        ros["topics"] = rows
        published = [r["name"] for r in rows if not pf.is_unknown(r["publishers"])
                     and r["publishers"]["value"] > 0
                     and r["name"] not in ("/parameter_events", "/rosout")]
        if published and not nodes:
            ros["node_discovery"] = derived(
                "inconsistent", f"0 nodes listed while {len(published)} topic(s) have "
                "publishers: node discovery is not working here (mixed ROS 2 distributions "
                "do this), so a node's absence from the list is not evidence")
        elif published:
            ros["node_discovery"] = derived("consistent", f"{len(nodes)} node(s) listed; "
                                            f"{len(published)} topic(s) have publishers")
    doc["ros"] = ros
    types_of = {n: t for n, t in topics}

    def pubs_of(t):
        p, _ = _try(reader.publishers, t)
        return p or []

    cmd_start = pubs_of("/cmd_vel") if "/cmd_vel" in types_of else []

    # what to sample
    plan = {}
    for name, types in topics:
        tn = types[0] if types else ""
        if tn in RATE_TYPES or tn in (T_DIAG, T_TF) or (
                tn in NUMERIC_STD and "battery" in name.lower()):
            if pubs_of(name):
                plan[name] = tn
    if "/robot_description" in types_of and T_STRING in types_of["/robot_description"]:
        plan["/robot_description"] = T_STRING
    if "/cmd_vel" in types_of and T_TWIST in types_of["/cmd_vel"]:
        plan["/cmd_vel"] = T_TWIST
    latched = ("/tf_static", "/robot_description")
    samples, e = _try(reader.sample, plan, opts.window, latched)
    if e:
        samples = {t: ReadFailed(f"sampling failed: {e}") for t in plan}
    clk, clk_err = _try(reader.clock)

    # plant kind
    diag = samples.get("/diagnostics")
    clock_pubs = pubs_of("/clock") if "/clock" in types_of else []
    markers = simulator_markers(nodes, diag if isinstance(diag, list) else [], clock_pubs)
    doc["plant"] = {
        "kind": plant_kind(markers, opts.declare_kind),
        "simulator_markers": (read(markers, "graph node names, /diagnostics, /clock")
                              if markers else
                              unknown("none found in the node names, /diagnostics or /clock; "
                                      "absence of a marker does not prove hardware")),
    }

    # rates
    rates = {}
    rate_topics = sorted({n for n, t in topics if t and t[0] in RATE_TYPES})
    for row, row_type in NAMED_ROWS.items():
        if row not in types_of:
            same = sorted(n for n, t in topics if t and t[0] == row_type)
            hint = f"; same-type topics present: {', '.join(same)}" if same else ""
            rates[row] = unknown(f"{row} is not in the graph{hint}")
    if not any(t and t[0] == T_IMU for _, t in topics):
        rates["/imu"] = unknown("no sensor_msgs/msg/Imu topic is in the graph")
    for name in rate_topics:
        rates[name] = _rate_field(name, types_of[name][0], pubs_of(name),
                                  samples.get(name, []), opts.window)
    doc["rates"] = dict(sorted(rates.items()))

    # frames: TF and URDF
    tf = {}
    edges = {}
    frames = set()
    for t in ("/tf_static", "/tf"):
        recs = samples.get(t)
        if isinstance(recs, list):
            for r in recs:
                for parent, child, xyz, q in r.get("transforms", []):
                    if t == "/tf_static" or child not in edges:
                        edges[child] = (parent, (xyz, q))
                    frames.update((parent, child))
    tf["frames"] = (read(sorted(frames), "tf: /tf_static + /tf over the window")
                    if frames else unknown("no transform received on /tf_static or /tf"))
    lidar_frames = sorted({r["frame"] for n, t in topics if t and t[0] == T_SCAN
                           for r in (samples.get(n) if isinstance(samples.get(n), list) else [])
                           if r.get("frame")})
    scan_topics = sorted(n for n, t in topics if t and t[0] == T_SCAN)
    tf["lidar_frames"] = (read(lidar_frames, "topic header.frame_id: " + ", ".join(scan_topics))
                          if lidar_frames else
                          unknown("no LaserScan message received, so no lidar frame_id"
                                  if scan_topics else "no LaserScan topic in the graph"))
    bases = [b for b in BASE_FRAMES if b in frames]
    tf["base_frames"] = (read(bases, "tf: frames present among " + ", ".join(BASE_FRAMES))
                         if bases else unknown("neither base_link nor base_footprint is in TF"))
    if lidar_frames and bases:
        tf["lidar_pose"] = _pose_fields(edges, bases, lidar_frames, "tf")
    else:
        tf["lidar_pose"] = unknown("needs both a lidar frame_id and a base frame in TF")
    doc["tf"] = tf

    urdf = {}
    rd = samples.get("/robot_description")
    text = rd[-1]["data"] if isinstance(rd, list) and rd else None
    src = "topic: /robot_description (transient-local)"
    if text is None and "/robot_state_publisher" in nodes:
        vals, err = _try(reader.get_params, "/robot_state_publisher", ["robot_description"])
        if not err:
            text, err = _try(pf.param_value, vals["robot_description"])
            src = "param: /robot_state_publisher robot_description"
    if text:
        joints, err = _try(pf.urdf_joints, text)
        if err:
            urdf["description"] = unknown(f"{src} did not parse as URDF: {err}")
        else:
            urdf["description"] = read(f"{len(joints)} joints", src,
                                       md5=pf.md5_text(text.encode()))
            uedges = pf.urdf_edges(joints)
            ubases = [b for b in BASE_FRAMES if b in uedges or any(
                j["parent"] == b for j in joints.values())]
            ulidar = lidar_frames or []
            urdf["lidar_pose"] = (_pose_fields(uedges, ubases, ulidar, "urdf joints")
                                  if ulidar and ubases else
                                  unknown("no lidar frame_id read from a scan, so no URDF "
                                          "link to look up" if not ulidar else
                                          "no base_link/base_footprint link in the URDF"))
    else:
        urdf["description"] = unknown("no /robot_description message and no "
                                      "/robot_state_publisher robot_description parameter")
        urdf["lidar_pose"] = unknown("no URDF")
    doc["urdf"] = urdf

    # Nav2, live
    nav2 = {}
    nav2["velocity_smoother"] = _param_fields(reader, NAV2_NODES["velocity_smoother"],
                                              SMOOTHER_PARAMS, nodes)
    cs = NAV2_NODES["controller_server"]
    ctrl = _param_fields(reader, cs, ("controller_plugins", "controller_frequency"), nodes)
    ids = ctrl["controller_plugins"]
    if not pf.is_unknown(ids):
        for pid in ids["value"]:
            sub = _param_fields(reader, cs, [f"{pid}.{k}" for k in CONTROLLER_SUB], nodes)
            ctrl.update(sub)
            plug = sub.get(f"{pid}.plugin")
            if plug and not pf.is_unknown(plug):
                fam = pf.controller_family(plug["value"])
                ctrl[f"{pid}.family"] = (derived(fam, f"from param {cs} {pid}.plugin")
                                         if fam else
                                         unknown(f"plugin {plug['value']!r} is not a "
                                                 "controller family this probe names"))
    nav2["controller_server"] = ctrl
    for cm in ("local_costmap", "global_costmap"):
        body = _param_fields(reader, NAV2_NODES[cm], COSTMAP_PARAMS, nodes)
        fp, rr = body["footprint"], body["robot_radius"]
        what, val, why = pf.radius_in_force(None if pf.is_unknown(fp) else fp["value"],
                                            None if pf.is_unknown(rr) else rr["value"])
        body["geometry_in_force"] = (derived({"param": what, "value": val},
                                             f"{why} (Costmap2DROS rule)")
                                     if what else unknown(why))
        nav2[cm] = body
    prefixes, perr = _try(reader.ament_prefixes)
    cands, cerr = _file_candidates(prefixes or [], opts.param_dir)
    if perr and not cands:
        cerr = cerr or f"ament prefixes not readable: {perr}"
    nav2["file_candidates"] = (unknown(cerr) if cerr else
                               cands if cands else
                               unknown("no Nav2-shaped parameter file under the ament "
                                       "prefixes" + (" or --param-dir" if opts.param_dir
                                                     else "")))
    doc["nav2"] = nav2

    # battery
    batt = {}
    br = samples.get("/battery_state")
    if "/battery_state" not in types_of:
        batt["voltage"] = batt["percentage"] = unknown("/battery_state is not in the graph")
    elif not pubs_of("/battery_state"):
        batt["voltage"] = batt["percentage"] = unknown("/battery_state has 0 publishers")
    elif not isinstance(br, list) or not br:
        batt["voltage"] = batt["percentage"] = unknown(
            "/battery_state has publishers but no message arrived in the window")
    else:
        for k, unit in (("voltage", "V"), ("percentage", "fraction 0..1")):
            vs = [r[k] for r in br if isinstance(r.get(k), float) and math.isfinite(r[k])]
            batt[k] = (read(round(statistics.median(vs), 4),
                            f"topic: /battery_state {k}, median of {len(vs)}", units=unit)
                       if vs else unknown(f"/battery_state {k} is NaN (the message's own "
                                          "'not measured' value) in every message"))
    other = []
    for name, types in topics:
        if types and types[0] in NUMERIC_STD and "battery" in name.lower():
            recs = samples.get(name)
            if isinstance(recs, list) and recs:
                vals = [r["data"] for r in recs]
                other.append({"topic": name, "type": types[0],
                              "raw": read(statistics.median(vals),
                                          f"topic: {name} data, median of {len(vals)}",
                                          units="not declared by the message type; "
                                                "not converted")})
            else:
                other.append({"topic": name, "type": types[0],
                              "raw": unknown(f"{name}: no message in the window")})
    batt["other_battery_topics"] = other if other else unknown(
        "no other topic with 'battery' in its name and a numeric std_msgs type")
    doc["battery"] = batt

    # clock
    clock = {}
    if clk_err:
        clock["ros_time"] = unknown(clk_err)
    else:
        clock["ros_time_utc"] = read(_utc(clk["ros_time_s"]),
                                     "probe node clock (ROS time), after the window")
        clock["container_wall_utc"] = read(_utc(clk["wall_s"]), "time.time() in the container")
        clock["ros_minus_wall_s"] = measured(round(clk["ros_time_s"] - clk["wall_s"], 6),
                                             "the two reads above, back to back",
                                             use_sim_time=clk["use_sim_time"])
        if opts.reference_utc is not None:
            clock["container_minus_reference_s"] = measured(
                round(t0_wall - opts.reference_utc, 3),
                "container time.time() at probe start minus --reference-utc (the "
                "operator's machine, taken before ssh; includes ssh + startup latency, "
                "so it is an upper bound on |offset| + a few seconds)")
        else:
            clock["container_minus_reference_s"] = unknown(
                "no --reference-utc given; the container cannot know true time by itself")
    clock["host_ntp"] = unknown(
        "host-side: the container shares the host kernel clock but cannot read its NTP "
        "state. The operator runs `timedatectl` on the host and keeps its output beside "
        "this file")
    doc["clock"] = clock

    # what only a person can answer
    def per_topic(type_name, reason, none_reason):
        live = [n for n, f in doc["rates"].items()
                if not pf.is_unknown(f) and f.get("type") == type_name]
        return {n: unknown(reason) for n in live} if live else unknown(none_reason)

    doc["motion_channel"] = {
        "odometry": per_topic(T_ODOM, "a static probe cannot tell measured odometry from "
                              "command-derived; needs the guided push test",
                              "no odometry topic delivered messages, so there is nothing to "
                              "classify"),
        "imu_refutes_rest": per_topic(T_IMU, "needs the guided push test",
                                      "no IMU topic delivered messages"),
        "scan_difference": per_topic(T_SCAN, "needs the guided push test",
                                     "no LaserScan topic delivered messages"),
    }
    doc["geometry"] = {
        "physical_width_m": unknown("no ROS 2 interface reports the robot's physical size; "
                                    "a costmap radius is configuration, not a measurement"),
    }

    cmd_end = pubs_of("/cmd_vel") if "/cmd_vel" in types_of else []
    cmd_msgs = samples.get("/cmd_vel")
    own, oe = _try(reader.own_publishers)
    doc["probe"].update({
        "duration_s": round(time.monotonic() - t0, 2),
        "own_publishers": own if not oe else f"unknown: {oe}",
        "cmd_vel_publishers_at_start": cmd_start,
        "cmd_vel_publishers_at_end": cmd_end,
        "cmd_vel_messages_seen": (len(cmd_msgs) if isinstance(cmd_msgs, list)
                                  else "not subscribed: no /cmd_vel topic"),
    })
    return doc


# ── the human summary ───────────────────────────────────────────────────────

GUIDED = (
    ("motion_channel.odometry.", "motion-channel push test (plant_guided.py step a)"),
    ("motion_channel.imu_refutes_rest.", "motion-channel push test (plant_guided.py step a)"),
    ("motion_channel.scan_difference.", "scan-difference rest check (plant_guided.py step b)"),
    ("geometry.physical_width_m", "one tape measurement (plant_guided.py step c)"),
)
# A field whose reason says this is not something a person can supply on the guided run.
NOT_GUIDABLE = ("nothing to classify", "delivered messages")


def guided_needs(doc: dict) -> list:
    """Unknown fields a person can answer on the guided run, in step order."""
    out = []
    kind = doc.get("plant", {}).get("kind", {})
    sim = kind.get("value") == "simulated"
    for path, reason in pf.unknown_paths(doc):
        if any(s in reason for s in NOT_GUIDABLE):
            continue
        for prefix, how in GUIDED:
            if path == prefix or path.startswith(prefix):
                out.append((path, how + (" — not-testable-here: the plant is simulated"
                                         if sim else "")))
                break
    lp_tf = doc.get("tf", {}).get("lidar_pose")
    lp_urdf = doc.get("urdf", {}).get("lidar_pose")
    if all(isinstance(x, dict) and pf.is_field(x) and pf.is_unknown(x)
           for x in (lp_tf, lp_urdf)):
        out.append(("geometry.lidar_height_m",
                    "one ruler measurement, floor to lidar window centre (step c)"
                    + (" — not-testable-here: the plant is simulated" if sim else "")))
    return out


def summary(doc: dict) -> str:
    L = []
    kind = doc["plant"]["kind"]
    L.append(f"PLANT PROBE {doc['probe']['version']}  started {doc['probe']['started_wall_utc']}"
             f"  took {doc['probe'].get('duration_s')} s")
    L.append(f"plant kind: {kind['value']}  ({kind.get('source') or kind.get('reason')})")
    if kind["value"] == "simulated":
        L.append("  THIS IS A SIMULATED PLANT. Nothing below is a hardware fact.")
    r = doc["ros"]
    L.append(f"ROS: distro={r['distro']['value']} rmw={r['rmw']['value']} "
             f"domain={r['domain_id']['value']} nodes="
             f"{r['nodes'].get('count', r['nodes']['value'])}")
    nd = r.get("node_discovery")
    if nd and not pf.is_unknown(nd) and nd["value"] == "inconsistent":
        L.append(f"  WARNING node discovery: {nd['source']}")
    L.append("")
    L.append("rates (arrival, over the window):")
    for t, f in doc["rates"].items():
        if pf.is_unknown(f):
            L.append(f"  {t:32s} unknown — {f['reason']}")
        else:
            v = f["value"]
            L.append(f"  {t:32s} {v['hz']:7.2f} Hz  n={v['n']:<4d} period median "
                     f"{v['period_ms']['median']:.1f} ms p99 {v['period_ms']['p99']:.1f} "
                     f"max {v['period_ms']['max']:.1f} sd {v['period_ms']['sd']:.1f}")
    L.append("")
    L.append("frames:")
    for sect in ("tf", "urdf"):
        lp = doc[sect]["lidar_pose"]
        if pf.is_field(lp):
            L.append(f"  lidar pose ({sect}): unknown — {lp['reason']}")
        else:
            for k, f in lp.items():
                if pf.is_unknown(f):
                    L.append(f"  {k} ({sect}): unknown — {f['reason']}")
                else:
                    v = f["value"]
                    L.append(f"  {k} ({sect}): x={v['x']:+.4f} y={v['y']:+.4f} z={v['z']:+.4f}"
                             f" yaw={v['yaw']:+.4f}  [{f['source']}]")
    L.append("")
    L.append("Nav2 (live parameters):")
    for sect, keys in (("velocity_smoother", ("max_velocity", "max_decel")),
                       ("local_costmap", ("robot_radius", "footprint", "geometry_in_force")),
                       ("global_costmap", ("robot_radius", "footprint", "geometry_in_force"))):
        for k in keys:
            f = doc["nav2"][sect][k]
            L.append(f"  {sect}.{k}: " + (f"unknown — {f['reason']}" if pf.is_unknown(f)
                                           else f"{f['value']}  [{f['source']}]"))
    ctrl = doc["nav2"]["controller_server"]
    fams = [(k, f) for k, f in ctrl.items() if k.endswith(".family")]
    if fams:
        for k, f in fams:
            L.append(f"  controller {k}: " + (f"unknown — {f['reason']}" if pf.is_unknown(f)
                                               else f"{f['value']}  [{f['source']}]"))
    else:
        f = ctrl["controller_plugins"]
        L.append("  controller loaded: " + (f"unknown — {f['reason']}" if pf.is_unknown(f)
                                            else str(f["value"])))
    fc = doc["nav2"]["file_candidates"]
    if isinstance(fc, list):
        L.append(f"  file candidates (NOT what is loaded): {len(fc)}")
        for c in fc:
            vals = c.get("values", {})
            keys = [k for k in vals if k.endswith(("max_velocity", "max_decel", "robot_radius",
                                                   ".plugin", "footprint_model.radius",
                                                   "min_obstacle_dist"))]
            L.append(f"    {c['file']}")
            for k in keys:
                L.append(f"      {k} = {vals[k]['value']}")
    else:
        L.append(f"  file candidates: unknown — {fc['reason']}")
    L.append("")
    b = doc["battery"]
    for k in ("voltage", "percentage"):
        f = b[k]
        L.append(f"battery {k}: " + (f"unknown — {f['reason']}" if pf.is_unknown(f)
                                     else f"{f['value']} {f.get('units', '')}"))
    ob = b["other_battery_topics"]
    if isinstance(ob, list):
        for o in ob:
            f = o["raw"]
            L.append(f"battery-named topic {o['topic']} ({o['type']}): " +
                     (f"unknown — {f['reason']}" if pf.is_unknown(f)
                      else f"raw {f['value']} (units not declared; not converted)"))
    c = doc["clock"]
    L.append("")
    if "ros_minus_wall_s" in c:
        L.append(f"clock: container wall {c['container_wall_utc']['value']}  ROS - wall "
                 f"{c['ros_minus_wall_s']['value']} s")
    f = c.get("container_minus_reference_s")
    if f:
        L.append("clock: container - reference " + (f"unknown — {f['reason']}"
                                                     if pf.is_unknown(f) else f"{f['value']} s"))
    L.append("clock: host NTP is a HOST-side line — run `timedatectl` on the host; "
             "not read from inside the container")
    p = doc["probe"]
    L.append("")
    L.append(f"observe-only check: probe's own publishers {p['own_publishers']}; "
             f"/cmd_vel publishers start {p['cmd_vel_publishers_at_start']} end "
             f"{p['cmd_vel_publishers_at_end']}; /cmd_vel messages seen "
             f"{p['cmd_vel_messages_seen']}")
    n_unknown = len(pf.unknown_paths(doc))
    n_fields = sum(1 for _ in pf.iter_fields(doc))
    L.append(f"fields: {n_fields}, of which unknown: {n_unknown}")
    L.append("")
    L.append("Needs a guided step:")
    needs = guided_needs(doc)
    for path, how in needs:
        L.append(f"  - {path}: {how}")
    if not needs:
        L.append("  (nothing)")
    return "\n".join(L) + "\n"


# ── main ────────────────────────────────────────────────────────────────────


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for plant.json/.yaml, summary.txt")
    ap.add_argument("--window", type=float, default=10.0,
                    help="rate/TF sampling window, seconds (a window, not a plant value)")
    ap.add_argument("--discovery", type=float, default=3.0,
                    help="seconds to let DDS discovery settle before reading the graph")
    ap.add_argument("--declare-kind", choices=("hardware", "simulated"),
                    help="the operator's statement of what the plant is; recorded as "
                         "source: operator, and refused if the graph contradicts it")
    ap.add_argument("--reference-utc", type=float,
                    help="epoch seconds from the operator's machine, taken just before launch")
    ap.add_argument("--param-dir", action="append", default=[],
                    help="extra directory to search for Nav2 parameter files (repeatable)")
    args = ap.parse_args(argv)
    if not (0.5 <= args.window <= 60):
        ap.error("--window must be within [0.5, 60] s")
    return args


def main(argv=None) -> int:
    opts = parse_args(sys.argv[1:] if argv is None else argv)
    reader = RclpyReader()
    try:
        doc = build_plant(reader, opts)
    finally:
        reader.close()
    try:
        paths = pf.write_plant(doc, opts.out)
    except pf.FieldError as e:
        print(f"REFUSED TO WRITE: {e}", file=sys.stderr)
        return 1
    text = summary(doc)
    with open(os.path.join(opts.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.write(text)
    print(f"\nwrote {paths['yaml']}, {paths['json']}, {os.path.join(opts.out, 'summary.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
