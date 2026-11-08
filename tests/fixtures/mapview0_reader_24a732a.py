# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view contract, ``mapview/0`` (a draft; see CONTRACT.md): build, encode, decode and
check messages, and the state a viewer keeps.

Standard library only, so the adapter, the map-view service and any tool can use it without a
new dependency. Nothing here talks to a robot: the contract has no command message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
import time
import zlib

VERSION = "mapview/0"
KINDS = ("MapSnapshot", "PoseSample", "Overlays", "SessionStatus")
COMMON = ("contractVersion", "kind", "robot", "frame", "stamp")
REQUIRED = {
    "MapSnapshot": ("seq", "resolution", "origin", "width", "height", "cells", "sha256"),
    "PoseSample": ("x", "y", "yaw_deg", "source"),
    "Overlays": ("keepouts", "zones", "target"),
    "SessionStatus": ("source", "phase", "text", "area_free_m2"),
}
GRID_FIELDS = ("resolution", "origin", "width", "height", "cells", "sha256")
PHASES = ("started", "running", "stopped")

#: The largest grid a reader decodes (width x height): 4096 x 4096 cells at 5 cm is a
#: 200 m square. Refused from the header, before any cell is inflated.
MAX_CELLS = 4096 * 4096
#: The longest line a reader accepts, in characters.
MAX_LINE = 8 * 1024 * 1024
#: The longest SessionStatus text.
MAX_TEXT = 200

#: What a writer sends, and how often (CONTRACT.md, "Rates").
POSE_PERIOD_S = 0.5
MAP_CHECK_S = 5.0
MAP_REPEAT_S = 30.0
#: When a consumer calls what it holds stale, measured from when the latest message of that
#: kind ARRIVED, on the consumer's own clock (CONTRACT.md, "Staleness").
POSE_STALE_S = 2.0
POSE_LOST_S = 10.0
MAP_STALE_S = 35.0

# A robot is a local alias: a DNS label. No dot, colon or space can appear in one, so neither
# an IPv4 or IPv6 address, a host name nor a MAC can be sent as a robot.
_ROBOT = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_FRAME = re.compile(r"^[A-Za-z0-9_@./-]{1,128}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A message that does not follow the contract."""


class UnknownVersion(ContractError):
    """A message in a version this reader does not know. Nothing else in it is read."""


# -- small checks ----------------------------------------------------------------------------------

def _number(m: dict, k: str, where: str) -> float:
    v = m.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ContractError(f"{where}: {k} must be a finite number, got {v!r}")
    return float(v)


def _int(m: dict, k: str, where: str, lo: int = 0) -> int:
    v = m.get(k)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ContractError(f"{where}: {k} must be an integer >= {lo}, got {v!r}")
    return v


def _string(m: dict, k: str, where: str, longest: int = 128) -> str:
    v = m.get(k)
    if not isinstance(v, str) or not v or len(v) > longest:
        raise ContractError(f"{where}: {k} must be a string of 1 to {longest} characters")
    return v


def _point(p, where: str) -> None:
    if (not isinstance(p, (list, tuple)) or len(p) != 2
            or not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                       for v in p)):
        raise ContractError(f"{where} must be [x, y], two finite numbers")


def _grid_header(f: dict, where: str) -> tuple[int, int]:
    if not isinstance(f, dict):
        raise ContractError(f"{where}: a grid is an object")
    for k in GRID_FIELDS:
        if k not in f:
            raise ContractError(f"{where}: grid field missing: {k}")
    if _number(f, "resolution", where) <= 0:
        raise ContractError(f"{where}: resolution must be above 0 m per cell")
    _point(f["origin"], f"{where}: origin")
    w, h = _int(f, "width", where, 1), _int(f, "height", where, 1)
    if w * h > MAX_CELLS:
        raise ContractError(f"{where}: {w} x {h} cells is more than the {MAX_CELLS} allowed")
    if not isinstance(f["cells"], str):
        raise ContractError(f"{where}: cells must be a base64 string")
    if not isinstance(f["sha256"], str) or not _SHA.match(f["sha256"]):
        raise ContractError(f"{where}: sha256 must be 64 lowercase hex digits")
    return w, h


# -- grids as PNG ----------------------------------------------------------------------------------

def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def encode_flat(values, width: int, height: int, resolution: float,
                origin: tuple[float, float]) -> dict:
    """An occupancy grid given row-major from row 0 (the bottom, at ``origin``), values -1..100,
    as a ROS OccupancyGrid's ``data`` is, as the contract's grid fields."""
    if width < 1 or height < 1 or len(values) != width * height:
        raise ContractError(f"{len(values)} values for a {width} x {height} grid")
    if not (isinstance(resolution, (int, float)) and math.isfinite(resolution) and resolution > 0):
        raise ContractError("resolution must be above 0 m per cell")
    try:
        raw = bytes(int(v) + 1 for v in values)
    except ValueError:
        raw = b""
    if len(raw) != len(values) or max(raw) > 101:
        raise ContractError("grid values must be -1..100")
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(b"".join(
               b"\x00" + raw[r * width:(r + 1) * width] for r in range(height)), 9))
           + _chunk(b"IEND", b""))
    return {"resolution": float(resolution), "origin": [float(origin[0]), float(origin[1])],
            "width": width, "height": height, "cells": base64.b64encode(png).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest()}


def encode_grid(rows, resolution: float, origin: tuple[float, float]) -> dict:
    """An occupancy grid as rows (row 0 = the bottom, at ``origin``; values -1..100) as the
    contract's grid fields. Anything with ``tolist()`` (a numpy array) is taken as its rows."""
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    rows = [list(r) for r in rows]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ContractError("grid must be 2-D and rectangular")
    return encode_flat([v for r in rows for v in r], len(rows[0]), len(rows), resolution, origin)


def decode_raw(fields: dict, where: str = "grid") -> tuple[int, int, bytes]:
    """The grid fields back to (width, height, raw), raw being the cell values + 1, row-major
    from row 0. Checks the size against the header first, then the PNG, then sha256."""
    w, h = _grid_header(fields, where)
    try:
        png = base64.b64decode(fields["cells"], validate=True)
    except (ValueError, TypeError) as e:
        raise ContractError(f"{where}: cells is not base64: {e}") from None
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ContractError(f"{where}: cells is not a PNG")
    pos, idat, size = 8, b"", None
    while pos + 8 <= len(png):
        (n,) = struct.unpack(">I", png[pos:pos + 4])
        kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            if len(data) < 13:
                raise ContractError(f"{where}: PNG header is short")
            pw, ph, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data[:13])
            if depth != 8 or ctype != 0 or interlace != 0:
                raise ContractError(f"{where}: cells must be 8-bit grayscale, not interlaced")
            size = (pw, ph)
        elif kind == b"IDAT":
            idat += data
        elif kind == b"IEND":
            break
    if size != (w, h):
        got = "none" if size is None else f"{size[0]}x{size[1]}"
        raise ContractError(f"{where}: PNG size {got} does not match width/height {w}x{h}")
    try:
        d = zlib.decompressobj()
        rows = d.decompress(idat, h * (w + 1) + 1)
    except zlib.error as e:
        raise ContractError(f"{where}: PNG data does not inflate: {e}") from None
    if len(rows) != h * (w + 1) or d.unconsumed_tail:
        raise ContractError(f"{where}: PNG data length does not match its size")
    raw = bytearray()
    for r in range(h):
        line = rows[r * (w + 1):(r + 1) * (w + 1)]
        if line[0] != 0:
            raise ContractError(f"{where}: only PNG filter type 0 is used by this contract")
        raw += line[1:]
    if hashlib.sha256(bytes(raw)).hexdigest() != fields["sha256"]:
        raise ContractError(f"{where}: sha256 does not match the cells")
    if raw and max(raw) > 101:
        raise ContractError(f"{where}: cell value over 101")
    return w, h, bytes(raw)


def decode_grid(fields: dict, where: str = "grid") -> list[list[int]]:
    """The grid fields back to rows of occupancy (-1..100), row 0 = the bottom."""
    w, h, raw = decode_raw(fields, where)
    return [[v - 1 for v in raw[r * w:(r + 1) * w]] for r in range(h)]


# -- messages --------------------------------------------------------------------------------------

def _check_kind(m: dict) -> None:
    k = m["kind"]
    if k == "MapSnapshot":
        _int(m, "seq", k)
        _grid_header(m, k)
    elif k == "PoseSample":
        for f in ("x", "y", "yaw_deg"):
            _number(m, f, k)
        _string(m, "source", k, 32)
    elif k == "Overlays":
        if not isinstance(m["keepouts"], list) or not isinstance(m["zones"], list):
            raise ContractError(f"{k}: keepouts and zones must be lists")
        for i, ko in enumerate(m["keepouts"]):
            if not isinstance(ko, dict):
                raise ContractError(f"{k}: keepouts[{i}] must be an object")
            _string(ko, "name", f"{k}: keepouts[{i}]")
            _grid_header(ko.get("grid"), f"{k}: keepouts[{i}]")
        for i, z in enumerate(m["zones"]):
            if not isinstance(z, dict):
                raise ContractError(f"{k}: zones[{i}] must be an object")
            _string(z, "name", f"{k}: zones[{i}]")
            poly = z.get("polygon")
            if not isinstance(poly, list) or len(poly) < 3:
                raise ContractError(f"{k}: zones[{i}].polygon needs at least 3 points")
            for j, p in enumerate(poly):
                _point(p, f"{k}: zones[{i}].polygon[{j}]")
        t = m["target"]
        if t is not None:
            if not isinstance(t, dict):
                raise ContractError(f"{k}: target must be an object or null")
            _string(t, "label", f"{k}: target", 32)
            _number(t, "x", f"{k}: target")
            _number(t, "y", f"{k}: target")
    elif k == "SessionStatus":
        _string(m, "source", k)
        if m["phase"] not in PHASES:
            raise ContractError(f"{k}: phase must be one of {', '.join(PHASES)}, got "
                                f"{m['phase']!r}")
        if not isinstance(m["text"], str) or len(m["text"]) > MAX_TEXT:
            raise ContractError(f"{k}: text must be a string of at most {MAX_TEXT} characters")
        if m["area_free_m2"] is not None:
            _number(m, "area_free_m2", k)
        if "simulated" in m and not isinstance(m["simulated"], bool):
            raise ContractError(f"{k}: simulated must be true or false")
        if m.get("progress_pct") is not None:
            p = m["progress_pct"]
            if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= 100:
                raise ContractError(f"{k}: progress_pct must be an integer 0..100 or null")


def check(m: dict) -> dict:
    """``m`` if it follows the contract; raises ContractError (UnknownVersion for a version
    this reader does not know, before any other field is read) otherwise."""
    if not isinstance(m, dict):
        raise ContractError("a message is a JSON object")
    if m.get("contractVersion") != VERSION:
        raise UnknownVersion(f"contractVersion {m.get('contractVersion')!r}, want {VERSION!r}")
    for k in COMMON:
        if k not in m:
            raise ContractError(f"field missing: {k}")
    if m["kind"] not in REQUIRED:
        raise ContractError(f"unknown kind {m['kind']!r}")
    for k in REQUIRED[m["kind"]]:
        if k not in m:
            raise ContractError(f"{m['kind']}: field missing: {k}")
    if not isinstance(m["robot"], str) or not _ROBOT.match(m["robot"]):
        raise ContractError(f"robot {m['robot']!r} is not a local alias (a DNS label: lowercase "
                            f"letters, digits and '-'; never an address or a host name)")
    if not isinstance(m["frame"], str) or not _FRAME.match(m["frame"]):
        raise ContractError(f"frame {m['frame']!r} is not a frame name")
    _number(m, "stamp", "stamp")
    _check_kind(m)
    return m


def message(kind: str, robot: str, frame: str, stamp: float, **fields) -> dict:
    m = {"contractVersion": VERSION, "kind": kind, "robot": robot, "frame": frame,
         "stamp": float(stamp)}
    m.update(fields)
    return check(m)


def dumps(m: dict) -> str:
    return json.dumps(check(m), separators=(",", ":"), sort_keys=True, allow_nan=False)


def parse(line: str):
    """A line as JSON, not yet checked against the contract; None when it is not JSON or is
    over MAX_LINE (a ViewState counts None as refused)."""
    if len(line) > MAX_LINE:
        return None
    try:
        return json.loads(line, parse_constant=_no_constant)
    except (json.JSONDecodeError, ValueError):
        return None


def loads(line: str) -> dict:
    if len(line) > MAX_LINE:
        raise ContractError(f"a line of {len(line)} characters is over the {MAX_LINE} allowed")
    try:
        m = json.loads(line, parse_constant=_no_constant)
    except (json.JSONDecodeError, ValueError) as e:
        raise ContractError(f"not JSON: {e}") from None
    return check(m)


def _no_constant(name: str):
    raise ValueError(f"{name} is not a number this contract carries")


# -- who drives ------------------------------------------------------------------------------------

def driver_of(source: str) -> str:
    """Who moves the robot in a session, read from ``SessionStatus.source``: ``operator`` for a
    guided session (a person drives it), ``replay`` for a stored run played back, ``robot``
    for a FleetAction the robot runs itself (an explore-floor segment)."""
    if source.startswith("guided:"):
        return "operator"
    if source.startswith("replay:"):
        return "replay"
    return "robot"


def describe_source(source: str, simulated: bool) -> str:
    """One line for people saying what produced the session. A guided session is always said
    to be operator-driven."""
    d = driver_of(source)
    label = source.split(":", 1)[1] if ":" in source else source
    if d == "operator":
        text = f"guided session {label}: an operator drives the robot"
    elif d == "replay":
        text = f"replay of {label}"
    else:
        text = f"{source}: the robot explores inside its bound"
    return text + (" (simulated)" if simulated else "")


# -- the viewer's state: what it has accepted, and what it refused -------------------------------------

class ViewState:
    """What a viewer knows. Locks to the first accepted frame. A SessionStatus ``started`` in
    another frame replaces everything (a new session); any other message in another frame,
    and any message that fails the contract, is refused and counted. Never talks to a robot.

    ``now`` is when a message arrived, on the viewer's clock (default: ``time.monotonic``);
    staleness is judged from it, never from a message's ``stamp``."""

    def __init__(self, path_step: float = 0.02, clock=time.monotonic):
        self.path_step, self._clock = path_step, clock
        self.refused = {"frame": 0, "contract": 0, "version": 0}
        self.versions_seen: set[str] = set()
        self._reset(None)

    def _reset(self, frame) -> None:
        self.frame = frame
        self.map = None            # (rows, resolution, origin, seq, stamp)
        self.map_sha = None
        self.poses = []            # [(stamp, x, y, yaw_deg)], kept when the robot moved path_step
        self.last_pose = None
        self.overlays = {"keepouts": [], "zones": [], "target": None}
        self.keepout_grids = []    # [(name, rows, resolution, origin)]
        self.status = None
        self.arrived: dict[str, float] = {}
        self.map_changed_at = None

    def accept(self, m: dict, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        try:
            check(m)
            if self.frame is None:
                self.frame = m["frame"]
            elif m["frame"] != self.frame:
                if m["kind"] == "SessionStatus" and m["phase"] == "started":
                    self._reset(m["frame"])
                else:
                    self.refused["frame"] += 1
                    return False
            k = m["kind"]
            if k == "MapSnapshot":
                g = decode_grid(m)
                if self.map is None or m["seq"] >= self.map[3]:
                    if m["sha256"] != self.map_sha:
                        self.map_changed_at = now
                    self.map = (g, m["resolution"], tuple(m["origin"]), m["seq"], m["stamp"])
                    self.map_sha = m["sha256"]
            elif k == "PoseSample":
                p = (m["stamp"], float(m["x"]), float(m["y"]), float(m["yaw_deg"]))
                self.last_pose = p
                if not self.poses or math.hypot(p[1] - self.poses[-1][1],
                                                p[2] - self.poses[-1][2]) >= self.path_step:
                    self.poses.append(p)
            elif k == "Overlays":
                grids = [(ko["name"], decode_grid(ko["grid"], "keepout"), ko["grid"]["resolution"],
                          tuple(ko["grid"]["origin"])) for ko in m["keepouts"]]
                self.overlays = {"keepouts": m["keepouts"], "zones": m["zones"],
                                 "target": m["target"]}
                self.keepout_grids = grids
            elif k == "SessionStatus":
                self.status = m
            self.arrived[k] = now
            return True
        except UnknownVersion:
            self.refused["version"] += 1
            v = m.get("contractVersion") if isinstance(m, dict) else None
            self.versions_seen.add(str(v)[:32])
            return False
        except (ContractError, KeyError, TypeError, ValueError):
            self.refused["contract"] += 1
            return False

    def staleness(self, now: float | None = None) -> dict:
        """``pose``: fresh, stale (no PoseSample for POSE_STALE_S) or lost (POSE_LOST_S);
        ``map``: fresh or stale (no MapSnapshot for MAP_STALE_S); None when there is none, or
        once the session has stopped, when nothing is expected any more."""
        now = self._clock() if now is None else now
        ended = bool(self.status and self.status["phase"] == "stopped")
        if ended:
            return {"pose": None, "map": None, "ended": True}
        out = {"pose": None, "map": None, "ended": False}
        if "PoseSample" in self.arrived:
            age = now - self.arrived["PoseSample"]
            out["pose"] = "lost" if age > POSE_LOST_S else "stale" if age > POSE_STALE_S else "fresh"
        if "MapSnapshot" in self.arrived:
            out["map"] = "stale" if now - self.arrived["MapSnapshot"] > MAP_STALE_S else "fresh"
        return out
