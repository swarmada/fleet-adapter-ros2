# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The ROS-free half of the plant probe: fields, parsers, statistics, and the writer.

Nothing here imports rclpy, so every rule in this file is unit-testable on a desk.
``plant_probe.py`` and ``plant_guided.py`` do the ROS 2 reads and hand the raw material
to the functions below.

The one rule
------------
**No field ever holds a default.** A field is one of:

* ``read(value, source)``      — taken from the running stack (topic, node parameter,
                                 file, TF), with the exact place it came from;
* ``measured(value, source)``  — computed from observations the probe itself timed;
* ``derived(value, source)``   — computed from other fields, which ``source`` names;
* ``operator(value, prompt)``  — typed by a person, with the question they were asked;
* ``unknown(reason)``          — could not be read, with why.

``check_field`` enforces the shape and ``assert_no_defaults`` walks a whole document. The
writers call it before anything reaches disk, so a probe that tried to write
``max_velocity: 0.5`` without a source would stop instead of writing it.

Python 3.10 compatible (the robot runs 3.10): no ``match``, no PEP 695, no 3.11+ stdlib.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET

UNKNOWN = "unknown"
KINDS = ("read", "measured", "derived", "operator")

SCHEMA = "plant.yaml/draft-0"
# Top-level sections that describe the RUN (tool versions, window lengths, own-publisher
# self-checks), not the plant. Everything else is plant facts and must be fields.
RUN_METADATA = ("schema", "probe", "guided")


class FieldError(ValueError):
    """A field that is neither sourced nor an explained unknown."""


# ── field constructors ──────────────────────────────────────────────────────


def read(value, source: str, **extra) -> dict:
    return _sourced("read", value, source, extra)


def measured(value, source: str, **extra) -> dict:
    return _sourced("measured", value, source, extra)


def derived(value, source: str, **extra) -> dict:
    return _sourced("derived", value, source, extra)


def operator(value, prompt: str, **extra) -> dict:
    if not prompt or not str(prompt).strip():
        raise FieldError("operator() needs the prompt the person answered")
    return _sourced("operator", value, "operator", dict(extra, prompt=prompt))


def unknown(reason: str, **extra) -> dict:
    if not reason or not str(reason).strip():
        raise FieldError("unknown() needs the reason the value could not be read")
    out = {"value": UNKNOWN, "reason": str(reason)}
    out.update(extra)
    return out


def _sourced(kind: str, value, source: str, extra: dict) -> dict:
    if value is None:
        raise FieldError(f"{kind}() given None; use unknown(reason) (source was {source!r})")
    if value == UNKNOWN:
        raise FieldError(f"{kind}() given the literal {UNKNOWN!r}; use unknown(reason)")
    if not source or not str(source).strip():
        raise FieldError(f"{kind}() needs a source for value {value!r}")
    out = {"value": value, "kind": kind, "source": str(source)}
    out.update(extra)
    return out


def is_field(node) -> bool:
    return isinstance(node, dict) and "value" in node and ("reason" in node or "kind" in node)


def is_unknown(field: dict) -> bool:
    return field.get("value") == UNKNOWN


def check_field(path: str, field: dict) -> None:
    if field.get("value") == UNKNOWN:
        if not str(field.get("reason", "")).strip():
            raise FieldError(f"{path}: unknown without a reason")
        if "kind" in field or "source" in field:
            raise FieldError(f"{path}: an unknown carries a reason, not a source")
        return
    if field.get("value") is None:
        raise FieldError(f"{path}: value is None — a failed read must be unknown(reason)")
    if field.get("kind") not in KINDS:
        raise FieldError(f"{path}: kind {field.get('kind')!r} is not one of {KINDS}")
    if not str(field.get("source", "")).strip():
        raise FieldError(f"{path}: value {field.get('value')!r} has no source")
    if field["kind"] == "operator" and not str(field.get("prompt", "")).strip():
        raise FieldError(f"{path}: an operator value must record the prompt it answered")


def iter_fields(doc, path: str = ""):
    """Yield (path, field) for every field in a plant document, depth first."""
    if is_field(doc):
        yield path, doc
        return
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield from iter_fields(v, f"{path}.{k}" if path else str(k))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from iter_fields(v, f"{path}[{i}]")


def assert_no_defaults(doc) -> int:
    """Check every field; also refuse a bare number anywhere outside a field or ``probe``.

    Returns the number of fields checked. A bare number outside a field is how a default
    sneaks in: ``{"max_velocity": 0.5}`` has no source and no reason, so it is refused.
    """
    n = 0
    for p, f in iter_fields(doc):
        check_field(p, f)
        n += 1
    _refuse_bare_numbers(doc, "")
    return n


def _refuse_bare_numbers(node, path: str) -> None:
    if is_field(node):
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if path == "" and k in RUN_METADATA:
                continue   # the tools' own timings and self-checks, not plant facts
            _refuse_bare_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _refuse_bare_numbers(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        raise FieldError(f"{path}: bare number {node!r} with no source and no reason")


# ── ROS parameter values (rcl_interfaces/msg/ParameterValue) ────────────────
#
# A ParameterValue carries EVERY slot, and the unused ones hold zero values: an
# undeclared parameter comes back as type NOT_SET with double_value 0.0. Reading
# ``.double_value`` without switching on ``.type`` is precisely how a failed read turns
# into a number. The type constants are rcl_interfaces/msg/ParameterType's.

PARAMETER_NOT_SET = 0
_PARAM_SLOTS = {
    1: "bool_value",
    2: "integer_value",
    3: "double_value",
    4: "string_value",
    5: "byte_array_value",
    6: "bool_array_value",
    7: "integer_array_value",
    8: "double_array_value",
    9: "string_array_value",
}


class ParamNotSet(LookupError):
    """The node answered, and the parameter is not set on it."""


def param_value(pv):
    """ParameterValue -> Python value. Raises ParamNotSet for NOT_SET, never returns 0."""
    t = int(getattr(pv, "type", PARAMETER_NOT_SET))
    if t == PARAMETER_NOT_SET:
        raise ParamNotSet("parameter is not set on the node (type NOT_SET)")
    slot = _PARAM_SLOTS.get(t)
    if slot is None:
        raise ParamNotSet(f"parameter type {t} is not a known ParameterType")
    v = getattr(pv, slot)
    if t in (5, 6, 7, 8, 9):
        return list(v)
    return v


# ── Nav2 parameter files ────────────────────────────────────────────────────
#
# Used for FILE CANDIDATES only. A file on disk says what a launch *could* load; only a
# live ``GetParameters`` answer says what *is* loaded. The probe therefore never copies a
# file value into a live field, and says which one it has.

SMOOTHER_KEYS = ("max_velocity", "min_velocity", "max_accel", "max_decel", "feedback",
                 "odom_topic", "smoothing_frequency")
COSTMAP_KEYS = ("robot_radius", "footprint", "footprint_padding", "robot_base_frame")
CONTROLLER_FAMILIES = (
    ("teb_local_planner", "TEB"),
    ("dwb_core", "DWB"),
    ("nav2_mppi_controller", "MPPI"),
    ("nav2_regulated_pure_pursuit_controller", "RegulatedPurePursuit"),
    ("nav2_rotation_shim_controller", "RotationShim"),
    ("nav2_graceful_controller", "Graceful"),
)


def controller_family(plugin: str) -> str | None:
    p = (plugin or "").strip().strip('"')
    for prefix, name in CONTROLLER_FAMILIES:
        if p.startswith(prefix):
            return name
    return None


def md5_text(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _params(doc: dict, node: str) -> dict | None:
    """``doc[node]['ros__parameters']``, tolerating a namespaced key like ``/**``."""
    if not isinstance(doc, dict):
        return None
    block = doc.get(node)
    if isinstance(block, dict) and isinstance(block.get("ros__parameters"), dict):
        return block["ros__parameters"]
    return None


def parse_nav2_param_doc(doc: dict) -> dict:
    """Extract what the probe needs from one loaded Nav2 params YAML document.

    Returns only keys that are PRESENT in the file; an absent key is absent from the
    result, never filled. Shape::

        {"velocity_smoother": {...SMOOTHER_KEYS present...},
         "local_costmap":     {...COSTMAP_KEYS present...},
         "global_costmap":    {...},
         "controller_server": {"controller_plugins": [...],
                               "plugins": {"FollowPath": {"plugin": ..., "family": ...,
                                                          "footprint_model": {...},
                                                          "min_obstacle_dist": ...}},
                               "controller_frequency": ...}}
    """
    out: dict = {}

    sm = _params(doc, "velocity_smoother")
    if sm is not None:
        got = {k: sm[k] for k in SMOOTHER_KEYS if k in sm}
        if got:
            out["velocity_smoother"] = got

    for outer in ("local_costmap", "global_costmap"):
        block = doc.get(outer) if isinstance(doc, dict) else None
        inner = _params(block, outer) if isinstance(block, dict) else None
        if inner is None:
            inner = _params(doc, outer)
        if inner is not None:
            got = {k: inner[k] for k in COSTMAP_KEYS if k in inner}
            infl = inner.get("inflation_layer")
            if isinstance(infl, dict) and "inflation_radius" in infl:
                got["inflation_layer.inflation_radius"] = infl["inflation_radius"]
            if got:
                out[outer] = got

    cs = _params(doc, "controller_server")
    if cs is not None:
        got = {}
        if "controller_frequency" in cs:
            got["controller_frequency"] = cs["controller_frequency"]
        ids = cs.get("controller_plugins")
        if isinstance(ids, list):
            got["controller_plugins"] = list(ids)
            plugins = {}
            for pid in ids:
                blk = cs.get(pid)
                if not isinstance(blk, dict) or "plugin" not in blk:
                    continue
                entry = {"plugin": str(blk["plugin"]).strip('"')}
                fam = controller_family(entry["plugin"])
                if fam:
                    entry["family"] = fam
                fm = blk.get("footprint_model")
                if isinstance(fm, dict):
                    entry["footprint_model"] = {k: fm[k] for k in ("type", "radius") if k in fm}
                for k in ("min_obstacle_dist", "max_vel_x", "max_vel_theta", "acc_lim_x"):
                    if k in blk:
                        entry[k] = blk[k]
                plugins[pid] = entry
            if plugins:
                got["plugins"] = plugins
        if got:
            out["controller_server"] = got
    return out


def radius_in_force(footprint, robot_radius) -> tuple:
    """Which Costmap2DROS geometry applies. Returns (what, value, why).

    Costmap2DROS uses ``footprint`` when it is a non-empty polygon string, else
    ``robot_radius``. Either input may be None (not read); then the answer is None.
    """
    if footprint is None:
        return (None, None, "footprint not read, so which geometry is in force is not known")
    fp = str(footprint).strip()
    if fp not in ("", "[]"):
        return ("footprint", fp, "footprint is a non-empty polygon, so it overrides robot_radius")
    if robot_radius is None:
        return (None, None, "footprint is empty, so robot_radius applies, but it was not read")
    return ("robot_radius", robot_radius, "footprint is '[]', so robot_radius is in force")


# ── URDF ────────────────────────────────────────────────────────────────────


def _floats(s: str | None, n: int = 3) -> list:
    if s is None:
        return [0.0] * n          # URDF spec: an absent origin attribute IS zero
    vals = [float(x) for x in s.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers in {s!r}")
    return vals


def urdf_joints(urdf_xml: str) -> dict:
    """child link -> {"parent", "joint", "type", "xyz", "rpy"} for every joint."""
    root = ET.fromstring(urdf_xml)
    out = {}
    for j in root.iter("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        origin = j.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None)
        rpy = _floats(origin.get("rpy") if origin is not None else None)
        out[child.get("link")] = {"parent": parent.get("link"), "joint": j.get("name"),
                                  "type": j.get("type"), "xyz": xyz, "rpy": rpy}
    return out


def urdf_links(urdf_xml: str) -> list:
    return [lk.get("name") for lk in ET.fromstring(urdf_xml).iter("link")]


# ── rigid transforms (pure python, no numpy on the robot is assumed) ────────


def quat_from_rpy(r: float, p: float, y: float) -> tuple:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quat_mul(a: tuple, b: tuple) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_rotate(q: tuple, v: tuple) -> tuple:
    x, y, z, w = q
    qv = (v[0], v[1], v[2], 0.0)
    r = quat_mul(quat_mul(q, qv), (-x, -y, -z, w))
    return (r[0], r[1], r[2])


def rpy_from_quat(q: tuple) -> tuple:
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


def compose(a: tuple, b: tuple) -> tuple:
    """(t, q) of frame A->B composed with B->C gives A->C."""
    ta, qa = a
    tb, qb = b
    rt = quat_rotate(qa, tb)
    return ((ta[0] + rt[0], ta[1] + rt[1], ta[2] + rt[2]), quat_mul(qa, qb))


def chain_pose(edges: dict, base: str, target: str):
    """Pose of ``target`` in ``base`` from parent edges ``{child: (parent, (t, q))}``.

    Returns (t, q, [frames]) or None when the two frames are not connected. Handles
    ``target`` above ``base`` as well as below it.
    """
    def up(frame):
        path = [frame]
        seen = {frame}
        while frame in edges:
            frame = edges[frame][0]
            if frame in seen:
                return path   # a cycle in a broken tree: stop, do not loop
            seen.add(frame)
            path.append(frame)
        return path

    pb, pt = up(base), up(target)
    common = next((f for f in pb if f in set(pt)), None)
    if common is None:
        return None

    def from_root(path, stop):
        pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        chain = path[:path.index(stop)]
        for child in reversed(chain):
            pose = compose(pose, edges[child][1])
        return pose

    root_to_t = from_root(pt, common)
    root_to_b = from_root(pb, common)
    (tb, qb) = root_to_b
    qbi = (-qb[0], -qb[1], -qb[2], qb[3])
    tbi = quat_rotate(qbi, (-tb[0], -tb[1], -tb[2]))
    t, q = compose((tbi, qbi), root_to_t)
    frames = pb[:pb.index(common) + 1] + list(reversed(pt[:pt.index(common)]))
    return t, q, frames


def urdf_edges(joints: dict) -> dict:
    return {child: (j["parent"], (tuple(j["xyz"]), quat_from_rpy(*j["rpy"])))
            for child, j in joints.items()}


def pose_dict(t: tuple, q: tuple) -> dict:
    r, p, y = rpy_from_quat(q)
    return {"x": round(t[0], 6), "y": round(t[1], 6), "z": round(t[2], 6),
            "roll": round(r, 6), "pitch": round(p, 6), "yaw": round(y, 6)}


# ── timing statistics ───────────────────────────────────────────────────────


def _pct(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        raise ValueError("no values")
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def rate_stats(times: list) -> dict | None:
    """Rate and jitter from a list of arrival (or stamp) times in seconds.

    None when fewer than 3 samples: two points give one period, which has no jitter,
    and a rate from one period is a guess. The caller turns None into unknown(reason).
    """
    if len(times) < 3:
        return None
    ts = sorted(times)
    periods = [b - a for a, b in zip(ts, ts[1:])]
    span = ts[-1] - ts[0]
    if span <= 0:
        return None
    sp = sorted(periods)
    return {
        "hz": round((len(ts) - 1) / span, 3),
        "n": len(ts),
        "period_ms": {
            "median": round(statistics.median(sp) * 1e3, 3),
            "p99": round(_pct(sp, 0.99) * 1e3, 3),
            "max": round(sp[-1] * 1e3, 3),
            "min": round(sp[0] * 1e3, 3),
            "sd": round(statistics.pstdev(sp) * 1e3, 3),
        },
        "non_increasing": sum(1 for a, b in zip(times, times[1:]) if b <= a),
    }


def summary_stats(vals: list) -> dict | None:
    if not vals:
        return None
    s = sorted(vals)
    return {"n": len(s), "min": round(s[0], 6), "median": round(statistics.median(s), 6),
            "p90": round(_pct(s, 0.90), 6), "max": round(s[-1], 6)}


# ── scan difference (frame to frame), for the guided rest check ─────────────


def scan_pair_diff(a: list, b: list, range_min: float, range_max: float,
                   changed_m: float = 0.02) -> dict | None:
    """Frame-to-frame range difference between two consecutive scans.

    Only beams valid in BOTH frames count (finite and inside [range_min, range_max]).
    Returns the mean absolute difference and the fraction of beams that changed by more
    than ``changed_m``. This is a difference statistic, not a scan-matched velocity.
    """
    if len(a) != len(b):
        return None
    diffs = []
    for ra, rb in zip(a, b):
        if (math.isfinite(ra) and math.isfinite(rb)
                and range_min <= ra <= range_max and range_min <= rb <= range_max):
            diffs.append(abs(ra - rb))
    if not diffs:
        return None
    return {"mean_abs_m": sum(diffs) / len(diffs),
            "frac_changed": sum(1 for d in diffs if d > changed_m) / len(diffs),
            "valid": len(diffs)}


# ── writers ─────────────────────────────────────────────────────────────────

_PLAIN = re.compile(r"^[A-Za-z_./][A-Za-z0-9_./:+-]*$")
_YAML_WORDS = {"true", "false", "yes", "no", "on", "off", "null", "~", "y", "n"}


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else json.dumps(str(v))
    s = str(v)
    if _PLAIN.match(s) and s.lower() not in _YAML_WORDS:
        return s
    return json.dumps(s, ensure_ascii=False)


def to_yaml(node, indent: int = 0) -> str:
    """A small, deterministic YAML emitter (stdlib only; the robot has no PyYAML promise)."""
    pad = "  " * indent
    lines = []
    if isinstance(node, dict):
        if not node:
            return pad + "{}\n"
        for k, v in node.items():
            key = _scalar(str(k))
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{key}:\n{to_yaml(v, indent + 1)}")
            elif isinstance(v, dict):
                lines.append(f"{pad}{key}: {{}}\n")
            elif isinstance(v, list):
                lines.append(f"{pad}{key}: []\n")
            else:
                lines.append(f"{pad}{key}: {_scalar(v)}\n")
        return "".join(lines)
    if isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)) and v:
                body = to_yaml(v, indent + 1)
                first, _, rest = body.partition("\n")
                lines.append(f"{pad}- {first.lstrip()}\n{rest}")
            elif isinstance(v, dict):
                lines.append(f"{pad}- {{}}\n")
            elif isinstance(v, list):
                lines.append(f"{pad}- []\n")
            else:
                lines.append(f"{pad}- {_scalar(v)}\n")
        return "".join(lines)
    return pad + _scalar(node) + "\n"


HEADER = (
    "# plant.yaml — DRAFT, written by a probe. Not a declaration until a person reviews it.\n"
    "# Every field is either read / measured / derived / operator (with `source:`), or\n"
    "# `unknown` with `reason:`. No field holds a default. Schema: conformance/PLANT.md.\n"
)


def write_plant(doc: dict, out_dir: str) -> dict:
    """Check, then write plant.json (machine) and plant.yaml (human). Returns paths."""
    import os
    assert_no_defaults(doc)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "plant.json")
    py = os.path.join(out_dir, "plant.yaml")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    with open(py, "w", encoding="utf-8") as f:
        f.write(HEADER)
        f.write(to_yaml(doc))
    return {"json": pj, "yaml": py}


def unknown_paths(doc) -> list:
    return [(p, f["reason"]) for p, f in iter_fields(doc) if is_unknown(f)]
