# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""What writers of the map-view contract parse: the two ROS 2 command-line streams an operator
can watch without touching the robot (``tf2_echo`` and ``ros2 topic echo /map``), saved maps,
a keep-out mask, a target, FleetZones, and a saved operator-driven run for replay.

Standard library and PyYAML. Nothing here sends anything to a robot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fleet_adapter_ros2.mapview import contract as c

# -- parsers for the two ROS 2 command-line streams ----------------------------------------------------

_NUMS = re.compile(r"\[([^\]]*)\]")


def _bracket(line: str) -> list[float]:
    m = _NUMS.search(line)
    try:
        return [float(v) for v in m.group(1).split(",")] if m else []
    except ValueError:
        return []


class Tf2EchoParser:
    """Poses from ``ros2 run tf2_ros tf2_echo map base_footprint``: reads "At time",
    "- Translation:" and "- Rotation: in RPY (degree)" only (the quaternion and matrix lines are
    not poses)."""

    def __init__(self):
        self.t = None
        self.xy = None

    def feed(self, line: str):
        line = line.strip()
        if line.startswith("At time"):
            sec, _, nsec = line.split()[-1].partition(".")
            try:
                self.t = float(f"{sec}.{nsec.zfill(9)}") if nsec.isdigit() else float(sec)
            except ValueError:
                self.t = None
            self.xy = None
        elif line.startswith("- Translation:"):
            v = _bracket(line)
            self.xy = (v[0], v[1]) if len(v) >= 2 else None
        elif line.startswith("- Rotation: in RPY (degree)") and self.t is not None and self.xy:
            v = _bracket(line)
            if len(v) == 3:
                return self.t, self.xy[0], self.xy[1], v[2]
        return None


def parse_map_echo(text: str):
    """One ``ros2 topic echo /map --full-length`` message -> (rows, resolution, (ox, oy), stamp),
    row 0 at the origin as in the OccupancyGrid, or None when it is not a whole message."""
    try:
        res = float(re.search(r"resolution: ([-\d.e]+)", text).group(1))
        w = int(re.search(r"width: (\d+)", text).group(1))
        h = int(re.search(r"height: (\d+)", text).group(1))
        o = re.search(r"origin:\s*\n\s*position:\s*\n\s*x: ([-\d.e]+)\s*\n\s*y: ([-\d.e]+)", text)
        s = re.search(r"stamp:\s*\n\s*sec: (\d+)\s*\n\s*nanosec: (\d+)", text)
        data = [int(v) for v in re.findall(r"^- (-?\d+)$", text, re.MULTILINE)]
        origin = (float(o.group(1)), float(o.group(2)))
    except (AttributeError, ValueError):
        return None
    if len(data) != w * h or w < 1 or h < 1:
        return None
    stamp = int(s.group(1)) + int(s.group(2)) / 1e9 if s else None
    return [data[r * w:(r + 1) * w] for r in range(h)], res, origin, stamp


# -- saved maps (map_server yaml + pgm) ---------------------------------------------------------------

def _read_pgm(path: Path) -> tuple[int, int, bytes]:
    """A binary (P5) 8-bit PGM: (width, height, pixels top row first)."""
    b = Path(path).read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while pos < len(b) and b[pos:pos + 1].isspace():
            pos += 1
        if b[pos:pos + 1] == b"#":
            pos = b.index(b"\n", pos) + 1
            continue
        start = pos
        while pos < len(b) and not b[pos:pos + 1].isspace():
            pos += 1
        tokens.append(b[start:pos])
    if tokens[0] != b"P5" or int(tokens[3]) > 255:
        raise ValueError(f"{Path(path).name}: not an 8-bit binary PGM")
    w, h = int(tokens[1]), int(tokens[2])
    pixels = b[pos + 1:pos + 1 + w * h]
    if len(pixels) != w * h:
        raise ValueError(f"{Path(path).name}: {len(pixels)} pixels for {w} x {h}")
    return w, h, pixels


def load_saved_map(yaml_path: Path):
    """A saved map as rows of occupancy (-1 unknown, 0 free, 100 occupied; row 0 = the bottom),
    its resolution and origin, read with the thresholds its yaml declares."""
    with open(yaml_path) as f:
        m = yaml.safe_load(f)
    w, h, px = _read_pgm(Path(yaml_path).parent / m["image"])
    free, occ, negate = float(m["free_thresh"]), float(m["occupied_thresh"]), bool(m.get("negate"))
    rows = []
    for r in range(h - 1, -1, -1):          # the image's top row is the grid's last
        row = []
        for v in px[r * w:(r + 1) * w]:
            p = v / 255.0 if negate else (255 - v) / 255.0
            row.append(100 if p > occ else 0 if p < free else -1)
        rows.append(row)
    return rows, float(m["resolution"]), (float(m["origin"][0]), float(m["origin"][1]))


def keepout_overlay(run_dir: Path) -> list[dict]:
    """The keep-out mask a run built (``mask/keepout.yaml``) as an Overlays entry, or []."""
    y = Path(run_dir) / "mask" / "keepout.yaml"
    if not y.exists():
        return []
    g, res, o = load_saved_map(y)
    ko = [[100 if v == 100 else 0 for v in row] for row in g]
    return [{"name": "keep-out", "grid": c.encode_grid(ko, res, o)}]


def read_target(run_dir: Path):
    """The last ``<label> <x> <y>`` line of ``targets.txt``: where the operator is asked to go."""
    try:
        rows = [ln.split() for ln in (Path(run_dir) / "targets.txt").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        lab, x, y = rows[-1][0], float(rows[-1][1]), float(rows[-1][2])
        return {"label": lab[:32], "x": x, "y": y}
    except (OSError, IndexError, ValueError):
        return None


def zones_from_file(path: Path) -> list[dict]:
    """The FleetZones in a YAML file, as Overlays zones."""
    out = []
    try:
        with open(path) as f:
            for d in yaml.safe_load_all(f):
                if d and d.get("kind") == "FleetZone":
                    pts = d["spec"]["physicalBounds"]["polygon"]
                    out.append({"name": d["metadata"]["name"],
                                "polygon": [[p["x"], p["y"]] if isinstance(p, dict) else [p[0], p[1]]
                                            for p in pts]})
    except OSError:
        pass
    return out


def free_area(rows, res: float) -> float:
    return float(sum(v == 0 for row in rows for v in row)) * res * res


# -- a saved operator-driven run -> contract lines (for replay) ---------------------------------------

def convert_guided_run(run_dir: Path, map_yaml: Path, robot: str, frame: str,
                       zones: Path | None = None):
    """Contract messages for a saved guided run: Overlays, one MapSnapshot (the saved map), the
    guide log's poses, and SessionStatus at start and end. Yields dicts in stamp order. The
    source is ``replay:<run>``: what is shown is a stored run, played back."""
    with open(Path(run_dir) / "guide-log.jsonl") as f:
        log = [json.loads(ln) for ln in f if ln.strip()]
    poses = [ln for ln in log if ln.get("kind") == "pose"]
    if not poses:
        raise c.ContractError("no poses in the guide log")
    g, res, o = load_saved_map(map_yaml)
    t0, t1 = poses[0]["t"], poses[-1]["t"]
    label = f"replay:{Path(run_dir).name}"
    yield c.message("SessionStatus", robot, frame, t0, source=label, phase="started",
                    text="replay of a saved operator-driven run", area_free_m2=None)
    yield c.message("Overlays", robot, frame, t0, keepouts=keepout_overlay(run_dir),
                    zones=zones_from_file(zones) if zones else [], target=None)
    yield c.message("MapSnapshot", robot, frame, t0, seq=0, **c.encode_grid(g, res, o))
    for p in poses:
        yield c.message("PoseSample", robot, frame, p["t"], x=p["x"], y=p["y"], yaw_deg=p["yaw"],
                        source="slam")
    yield c.message("SessionStatus", robot, frame, t1, source=label, phase="stopped",
                    text="end of the saved run", area_free_m2=free_area(g, res))
