# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`Custom` explore-floor and `Charge`: the catalog entries, their validation, and the jobs.

Pure: no ROS, no protobuf. The jobs drive an :class:`ExploreStack` -- the seam to SLAM, the
keep-out servers, the costmap, the explorer and the battery -- and report through a callback,
so everything here runs against a fake stack and a fake clock.

What reaches the control plane is ``ActionStatusUpdate.message`` and ``progress_pct`` only.
``AssignAction`` carries the action type and the payload, so ``FleetAction.spec.customType``
never reaches an adapter: the verb is named in the payload, as ``operation: explore-floor``,
and the catalog declares ``operation`` as an enum with that one allowed value.

One explore-floor segment
-------------------------

1. The bound (``bound_ref``) is resolved, and the keep-out set ``keepouts_ref`` names -- the one
   rendered from the applied zones -- is read only if its digest is the one the reference names.
2. The live speed cap is read and proven within ``max_speed_mps`` (which is itself refused above
   0.20 m/s). The adapter never sets the cap; it refuses to move under one above what was asked.
   A battery reading no older than ``BATTERY_STALE_S`` and above the return threshold is read,
   or nothing moves: without one the return and the hard stop could never fire.
3. SLAM starts on a fresh map, or -- ``resume: true`` -- from the pose graph the last segment of
   the same bound saved.
4. ALIGNED: the live map's walls lie on the saved map's (the bound's, or on a resume the saved
   segment's own map) once turned by the boot's delta. The mark must not lie in a keep-out.
   The keep-out mask, with the set's zones drawn in, is turned into this boot's frame (the delta between this boot's yaw at the
   mark and the mask's), loaded, the keep-out servers brought up (restarted once if their reply
   was lost), and the global costmap reset once.
5. The bound is proven in the live costmap. Only then does anything move: the explorer starts.
6. At the battery threshold, a battery reading gone stale, the time box or the explorer's
   completion: the explorer is stopped
   (unless it stopped itself), the return goal is sent once it has paused and settled, and the
   robot returns to the mark. N recovery episodes in one area mark it EXHAUSTED (never an area
   at the mark): the robot is stopped, the area is drawn into the running segment's mask (all
   but the way from where the robot stands to the mark), the mask is reloaded and the bound
   proven again with the robot still; the robot leaves by that way, back to the mark; the whole
   area is kept out, reloaded and proven again; and the explorer goes on to what is left. The area is
   saved with the segment, and every later resumed segment draws it in whole. A bound that
   cannot be proven again, or a path the planner draws into a keep-out, stops the robot where
   it stands: the map is saved there, and nothing more moves.
7. The map, the pose graph and the frame delta are saved at ``<namespace>/<name>/`` -- the path
   IS the action id -- with a ``MANIFEST.sha256``, and the final message carries
   ``map-ref=<store>/<namespace>/<name> map-sha256=<digest of the manifest>``.

Every read and every bring-up is bounded (``explore_steps``): a dropped reply is tried again, a
wrong answer never is, and a failure before step 5 ends the action FAILED with "no motion".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fleet_adapter_ros2 import explore_steps as steps

OPERATION_EXPLORE = "explore-floor"

#: The speed ceiling for exploration. A payload above it is refused, never clamped.
MAX_SPEED_MPS = 0.20

#: N recovery episodes in one area exhaust it, when the payload does not say.
DEFAULT_EXHAUSTED_AFTER = 3

#: How long a robot standing in an area it exhausted may take to leave it, back to the mark.
LEAVE_AREA_S = 120.0

#: A battery reading older than this is stale: before motion the segment is refused, while
#: exploring the robot returns to the mark. The robot's own topic reports once a second.
BATTERY_STALE_S = 10.0

#: The map_server thresholds a saved map declares. Unknown is written grey 205, which reads as
#: occupancy (255 - 205) / 255 = 0.19608; a free threshold at or above that reads every cell
#: nothing observed as free floor. 0.196 is the historical value that keeps it unknown.
FREE_THRESH = 0.196
OCCUPIED_THRESH = 0.65
UNKNOWN_GREY_OCC = (255 - 205) / 255


def map_yaml_refusal(yaml: bytes) -> str:
    """Why a saved map's YAML cannot be handed on, or "": a free threshold that reads unknown
    (grey 205) as free, or none at all."""
    for line in yaml.decode("utf-8", "replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "free_thresh":
            try:
                ft = float(value)
            except ValueError:
                return f"free_thresh {value.strip()!r} is not a number"
            if not ft < UNKNOWN_GREY_OCC:
                return (f"free_thresh {ft:g} reads unknown (grey 205, occupancy "
                        f"{UNKNOWN_GREY_OCC:.5f}) as free; a map must declare below that")
            return ""
    return "no free_thresh: a reader would apply its own default"


@dataclass(frozen=True)
class Param:
    """One declared parameter: what the catalog advertises (RFC-0001 ActionParam) and what
    `refusal` enforces, from the same record."""

    name: str
    kind: str                      # "number" | "string" | "bool" | "enum"
    unit: str = ""
    required: bool = False
    min: float | None = None
    max: float | None = None
    allowed: tuple[str, ...] = ()
    integer: bool = False


EXPLORE_PARAMS: tuple[Param, ...] = (
    Param("operation", "enum", required=True, allowed=(OPERATION_EXPLORE,)),
    Param("bound_ref", "string", required=True),
    Param("keepouts_ref", "string"),
    Param("max_speed_mps", "number", "m/s", required=True, min=0.01, max=MAX_SPEED_MPS),
    Param("battery_return_mv", "number", "mV", min=1.0),
    Param("exhausted_after_recoveries", "number", min=1, max=100, integer=True),
    Param("time_box_s", "number", "s", min=1.0),
    Param("resume", "bool"),
)

CHARGE_PARAMS: tuple[Param, ...] = (
    Param("full_mv", "number", "mV", min=1.0),
)

#: What a successful explore-floor segment reports, declared as SupportedAction.results.
EXPLORE_RESULTS: tuple[tuple[str, str, str], ...] = (
    ("map-ref", "string", ""),
    ("map-sha256", "string", ""),
    ("covered-m2", "number", "m2"),
)

EXPLORE_DESCRIPTION = (
    "Custom with operation explore-floor: explore one segment inside the bound named by "
    "bound_ref, at most max_speed_mps (refused above 0.2 m/s), after the bound is proven in "
    "the live costmap; return to the start mark at battery_return_mv, the time box or when no "
    "frontier is left; save the map, the pose graph and the boot frame delta at "
    "<namespace>/<action> in the artifact store. The final message carries map-ref and "
    "map-sha256. resume: true continues from the saved graph of the same bound. An area needing "
    "exhausted_after_recoveries recovery episodes (a whole number, default 3; one episode is one "
    "stop, however many recoveries Nav2 runs for it) is kept out of the rest of the segment once "
    "the bound is proven again with it, and later resumed segments keep it out; a bound that "
    "cannot be proven again stops the robot where it stands. No area at the start mark is ever "
    "exhausted.")

CHARGE_DESCRIPTION = (
    "Charge: waits, reporting the battery in mV, until it reads full (full_mv, or the adapter's "
    "configured full) and the cable is out (the IMU streaming again). Never moves the robot.")


@dataclass(frozen=True)
class BatteryLimits:
    """The robot's battery thresholds in mV: the hard stop, full, and the return threshold to
    use when a payload names none (None: a payload must name one)."""

    hard_mv: float
    full_mv: float
    return_mv: float | None = None


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_params(action_type: str, params: tuple[Param, ...], doc: dict) -> str:
    for p in params:
        if p.name not in doc:
            if p.required:
                return f"payload has no {p.name}; {action_type} requires it"
            continue
        v = doc[p.name]
        suffix = f" {p.unit}" if p.unit else ""
        if p.kind == "enum":
            if v not in p.allowed:
                return (f"{p.name} {json.dumps(v)} is not served; this adapter serves "
                        + ", ".join(repr(a) for a in p.allowed))
        elif p.kind == "string":
            if not isinstance(v, str) or not v:
                return f"{p.name} must be a non-empty string"
        elif p.kind == "bool":
            if not isinstance(v, bool):
                return f"{p.name} must be true or false"
        else:
            if not _is_finite_number(v):
                return f"{p.name} must be a finite number{suffix}"
            if p.integer and v != int(v):
                return f"{p.name} must be a whole number"
            if p.min is not None and v < p.min:
                return f"{p.name} {v:g}{suffix} is below the minimum of {p.min:g}{suffix}"
            if p.max is not None and v > p.max:
                return (f"{p.name} {v:g}{suffix} is above the maximum of {p.max:g}{suffix}; "
                        f"refused rather than run at less than was asked")
    return ""


def refusal(action_type: str, doc: dict, limits: BatteryLimits) -> str:
    """Why a Custom or Charge payload is refused, or "" when it is servable.

    Parameters the catalog does not declare are left alone, so a payload that also carries
    something another consumer reads is not refused for it."""
    if action_type == "Custom":
        if "operation" not in doc:
            return (f"payload has no operation; Custom serves operation "
                    f"{OPERATION_EXPLORE!r} only")
        why = _check_params(action_type, EXPLORE_PARAMS, doc)
        if why:
            return why
        if "keepouts_ref" in doc and parse_keepouts_ref(doc["keepouts_ref"]) is None:
            return (f"keepouts_ref {json.dumps(doc['keepouts_ref'])} is not "
                    f"<store>/<namespace>/<name>@sha256:<64 hex digits>: a keep-out set is "
                    f"named with the digest it must have")
        ret = doc.get("battery_return_mv", limits.return_mv)
        if ret is None:
            return ("no battery_return_mv in the payload and none configured on the adapter; "
                    "a segment must know when to return")
        if ret <= limits.hard_mv:
            return (f"battery_return_mv {ret:g} mV is at or below the hard stop "
                    f"{limits.hard_mv:g} mV")
        if ret >= limits.full_mv:
            return f"battery_return_mv {ret:g} mV is at or above full {limits.full_mv:g} mV"
        return ""
    if action_type == "Charge":
        why = _check_params(action_type, CHARGE_PARAMS, doc)
        if why:
            return why
        full = doc.get("full_mv", limits.full_mv)
        if full <= limits.hard_mv:
            return f"full_mv {full:g} mV is at or below the hard stop {limits.hard_mv:g} mV"
        return ""
    return "unsupported action type: " + (action_type or "(none stated)")


# ── the stack the jobs drive ────────────────────────────────────────────────


@dataclass
class Bound:
    """A room's bound: its keep-out mask, drawn in a saved map's frame, and theta0 -- the
    robot's yaw on the mark in that frame, from which this boot's delta is taken."""

    mask: steps.Mask
    theta0_deg: float
    ref_map: steps.Grid | None = None   # the saved map the mask was drawn on (ALIGNED)
    # Keep-outs measured by hand for what no sensor sees, in the mask's frame; drawn into every
    # segment's mask and proven in the costmap like the room (see read_hand_keepouts).
    hand_keepouts: list[steps.Keepout] = field(default_factory=list)


@dataclass
class SavedMap:
    """What the stack saved: files by name (the map and the pose graph), and the reference a
    later segment passes back to load the graph."""

    files: dict[str, bytes]
    graph_ref: str


class ExploreStack(Protocol):
    """The seam between the jobs and a robot's SLAM, keep-out, costmap, explorer and battery.

    Every read answers None when its reply is lost; every call answers None when its reply is
    lost and False when it was refused. Nothing here may block for long: a job bounds its own
    retries. ``halt``, ``hold`` and ``stop_explorer`` are called from any thread, a stop path's
    among them, and must never block and never raise: a step that fails is reported on stderr
    and the steps after it still run."""

    def battery_limits(self) -> BatteryLimits: ...
    def arm(self, halted: threading.Event) -> None:
        """The running job's halt event: ``halt`` sets it."""
    def halt(self) -> None:
        """Stop the explorer and the return goal, and end the job. Any thread; never blocks,
        never raises."""
    def hold(self) -> None:
        """Stop the explorer and the return goal, and keep the job (pause)."""
    def load_bound(self, bound_ref: str) -> Bound:
        """Raises LookupError, with the reason, when the bound cannot be had."""
    def read_param(self, node: str, name: str) -> str | None: ...
    def start_slam(self, graph_ref: str | None) -> bool | None: ...
    def robot_pose(self) -> tuple[float, float, float] | None:
        """x, y (m) and yaw (degrees) in the map frame."""
    def load_keepout_mask(self, mask: steps.Mask, delta_deg: float) -> bool | None: ...
    def reload_keepout_mask(self) -> bool | None:
        """The running keep-out servers serve the mask last loaded, in place of the one they
        were serving (an area kept out while a segment runs)."""
    def read_keepout_mask(self) -> steps.Grid | None:
        """The mask as the keep-out server publishes it, read off its topic (keep-out 100),
        and only one published since the last load: what the filter is given, not the
        file."""
    def activate_keepout(self) -> bool | None: ...
    def restart_keepout(self) -> bool | None: ...
    def reset_costmap(self) -> bool | None: ...
    def read_costmap(self) -> steps.Grid | None: ...
    def read_map(self) -> steps.Grid | None: ...
    def start_explorer(self) -> bool | None: ...
    def stop_explorer(self) -> None: ...
    def explorer_status(self) -> str: ...
    def recoveries(self) -> list[tuple[float, float]]:
        """Where each recovery episode since the last call began, in the map frame: one
        position per stop, however many recoveries the navigation stack ran for it."""
    def plans(self) -> list[list[tuple[float, float]]]:
        """Every global path the planner has published since the last call, as points in the
        map frame, the start first and the goal last."""
    def local_plans(self) -> list[list[tuple[float, float]] | None]:
        """Every local trajectory the controller has published since the last call -- what
        the robot is about to drive -- placed in the map frame; None for one that could not
        be placed (no transform from its frame)."""
    def skip_area(self, name: str) -> None:
        """Send the explorer no further goal into this area."""
    def go_to_mark(self, x: float, y: float, yaw_deg: float) -> None: ...
    def mark_result(self) -> str | None:
        """None while the return is under way; then SUCCEEDED, ABORTED, CANCELED or a reason."""
    def save_map(self, name: str) -> SavedMap | None: ...
    def battery_mv(self) -> float | None:
        """The latest reading in mV; None if none has ever arrived."""
    def battery_age_s(self) -> float | None:
        """Seconds since that reading arrived; None if none has."""
    def battery_source(self) -> str:
        """Where the reading comes from (the topic and its type), for a refusal to name."""
    def imu_streaming(self) -> bool | None: ...
    def charge_wait(self) -> None:
        """A notice that a Charge is waiting; never a command."""
    def end_segment(self) -> None:
        """Stop what a segment started on the robot (the explorer, the keep-out servers, SLAM),
        after the save or however else the segment ended."""


class Clock(Protocol):
    def now(self) -> float: ...
    def wait(self, event: threading.Event, seconds: float) -> bool:
        """Wait ``seconds``, or less if ``event`` is set; True if it is set."""


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def wait(self, event: threading.Event, seconds: float) -> bool:
        return event.wait(seconds)


# ── the artifact directory ──────────────────────────────────────────────────

MANIFEST = "MANIFEST.sha256"
_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def split_action_id(action_id: str, namespace: str) -> tuple[str, str] | None:
    """(namespace, name) for an AssignAction.action_id, or None if it is not one.

    The control plane sends the FleetAction's name, and the namespace is the adapter's own; an
    id of the form ``<namespace>/<name>`` is taken as it is. The artifact path is built from
    these, so anything that is not a plain DNS name is refused rather than joined into a path.
    """
    parts = action_id.split("/") if "/" in action_id else [namespace, action_id]
    if len(parts) != 2 or not all(_DNS_LABEL.match(p) for p in parts):
        return None
    return parts[0], parts[1]


def manifest_text(files: dict[str, bytes]) -> bytes:
    """``sha256sum``-format lines for every file, sorted by name."""
    return "".join(f"{hashlib.sha256(files[n]).hexdigest()}  {n}\n"
                   for n in sorted(files)).encode()


_KEEPOUTS_REF = re.compile(
    r"^([a-z0-9](?:[-a-z0-9.]*[a-z0-9])?)/([a-z0-9](?:[-a-z0-9.]*[a-z0-9])?)/"
    r"([a-z0-9](?:[-a-z0-9.]*[a-z0-9])?)@sha256:([0-9a-f]{64})$")


def parse_keepouts_ref(ref: object) -> tuple[str, str, str, str] | None:
    """(store, namespace, name, digest) for ``<store>/<namespace>/<name>@sha256:<hex>``."""
    m = _KEEPOUTS_REF.match(ref) if isinstance(ref, str) else None
    return m.groups() if m else None


@dataclass
class KeepoutSet:
    """The keep-outs a robot obeys, rendered from the applied zone set: named polygons in the
    frame of the map the zones came from, whose robot yaw on the mark is ``theta0_deg``."""

    ref: str
    theta0_deg: float
    keepouts: list[steps.Keepout]
    zone_set_sha256: str = ""
    source_map: str = ""


def _verified_files(root: Path, label: str, digest: str) -> dict[str, bytes]:
    """Every file of an artifact directory, read only if its MANIFEST.sha256 hashes to
    ``digest``, every file hashes to its manifest line, and no file is unlisted."""
    if not root.is_dir():
        raise LookupError(f"{label} is missing: no directory at {root.name}")
    try:
        manifest = (root / MANIFEST).read_bytes()
    except OSError as exc:
        raise LookupError(f"{label} has no {MANIFEST}") from exc
    got = hashlib.sha256(manifest).hexdigest()
    if got != digest:
        raise LookupError(f"{label}: its {MANIFEST} hashes to {got}, not the {digest} the "
                          f"reference names")
    listed: dict[str, str] = {}
    for line in manifest.decode("utf-8", "replace").splitlines():
        want, _, fname = line.partition("  ")
        if not re.fullmatch(r"[0-9a-f]{64}", want) or not _FILE_NAME.match(fname):
            raise LookupError(f"{label}: {MANIFEST} line {line!r} is not '<sha256>  <name>'")
        listed[fname] = want
    unlisted = sorted({p.name for p in root.iterdir()} - set(listed) - {MANIFEST})
    if unlisted:
        raise LookupError(f"{label}: files not listed in {MANIFEST}: {', '.join(unlisted)}")
    files = {}
    for fname, want in listed.items():
        try:
            data = (root / fname).read_bytes()
        except OSError as exc:
            raise LookupError(f"{label}: {MANIFEST} lists {fname}, which is missing") from exc
        if hashlib.sha256(data).hexdigest() != want:
            raise LookupError(f"{label}: {fname} hashes to {hashlib.sha256(data).hexdigest()}, "
                              f"not the {want} {MANIFEST} lists")
        files[fname] = data
    return files


def load_keepout_set(artifacts: ArtifactDir, ref: str) -> KeepoutSet:
    """The keep-out set ``ref`` names, proven by its digest; LookupError naming why not."""
    parts = parse_keepouts_ref(ref)
    if parts is None:
        raise LookupError(f"keep-out set reference {ref!r} is not "
                          f"<store>/<namespace>/<name>@sha256:<hex>")
    store, namespace, name, digest = parts
    label = f"keep-out set {store}/{namespace}/{name}"
    if store != artifacts.store:
        raise LookupError(f"{label} names store {store!r}; this adapter reads store "
                          f"{artifacts.store!r}")
    local = artifacts.root / namespace / name
    try:
        files = _verified_files(local, label, digest)
    except LookupError:
        if artifacts.remote is None:
            raise
        # The set is named by its digest, so a copy proven by it is the set; anything else is
        # fetched again, and proven again.
        artifacts.fetch(namespace, name, digest, f"keep-out set {ref}")
        files = _verified_files(local, label, digest)
    try:
        doc = json.loads(files["keepouts.json"])
        theta0 = doc["theta0_deg"]
        if not _is_finite_number(theta0):
            raise ValueError("theta0_deg is not a number")
        keepouts = []
        for k in doc["keepouts"]:
            poly = tuple((float(p["x"]), float(p["y"])) for p in k["polygon"])
            if not k.get("name") or not k.get("reason") or len(poly) < 3:
                raise ValueError(f"keep-out {k.get('name')!r} needs a name, a reason and at "
                                 f"least three points")
            keepouts.append(steps.Keepout(name=str(k["name"]), reason=str(k["reason"]),
                                          polygon=poly))
    except KeyError as exc:
        raise LookupError(f"{label}: keepouts.json has no {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise LookupError(f"{label}: keepouts.json is not a keep-out set: {exc}") from exc
    return KeepoutSet(ref=ref, theta0_deg=float(theta0), keepouts=keepouts,
                      zone_set_sha256=str(doc.get("zone_set_sha256", "")),
                      source_map=str(doc.get("source_map", "")))


def _turn(x: float, y: float, deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * x - s * y, s * x + c * y


def turned_keepouts(kset: KeepoutSet, deg: float) -> list[steps.Keepout]:
    """The set's polygons turned by ``deg`` about the mark."""
    return [steps.Keepout(k.name, k.reason, tuple(_turn(x, y, deg) for x, y in k.polygon))
            for k in kset.keepouts]


def goal_refusal(x: float, y: float, kset: KeepoutSet | None, frame_deg: float = 0.0) -> str:
    """Why a goal at (x, y) is refused, or "": inside a keep-out of the set, named.
    ``frame_deg`` is the goal frame's turn from the set's (0 when the robot localizes on the
    map the zones came from)."""
    if kset is None:
        return ""
    qx, qy = _turn(x, y, -frame_deg)
    for k in kset.keepouts:
        if steps.point_in_polygon(qx, qy, k.polygon):
            return (f"the goal ({x:.2f}, {y:.2f}) lies inside keep-out zone {k.name!r} "
                    f"({k.reason})")
    return ""


class ArtifactDir:
    """The directory an ArtifactStore declares; a segment is written at ``<namespace>/<name>/``.

    ``store`` is the ArtifactStore's name, which is what the map reference names. With no
    directory configured, a scratch directory of the adapter's own is used; the reference then
    names a store no cluster declares, and a consumer refuses it rather than reading a path
    nobody else can see.

    With ``remote`` (an ArtifactStore of kind ObjectStore), the directory is this adapter's own
    copy: a segment is saved here, then uploaded to ``<namespace>/<name>/`` in the store, the
    manifest last, so a reader that finds the manifest finds every file it lists. A keep-out
    set is fetched from the store by its reference and proven by its digest before it is read.
    """

    def __init__(self, root: str | None, store: str, remote=None) -> None:
        self.store = store
        self.remote = remote
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="explore-artifacts-"))

    def upload(self, namespace: str, name: str) -> None:
        """Upload the saved segment ``<namespace>/<name>``: each file, then the manifest. Each
        object's md5 is compared with the bytes sent (``StoreError`` otherwise)."""
        if self.remote is None:
            return
        root = self.root / namespace / name
        names = sorted(p.name for p in root.iterdir() if p.name != MANIFEST)
        for fname in [*names, MANIFEST]:
            self.remote.put(f"{namespace}/{name}/{fname}", (root / fname).read_bytes())

    def fetch(self, namespace: str, name: str, digest: str, label: str) -> None:
        """Fetch ``<namespace>/<name>`` from the store into this directory, reading only what
        a manifest hashing to ``digest`` lists. LookupError naming why not."""
        from fleet_adapter_ros2 import objectstore
        prefix = f"{namespace}/{name}"
        try:
            manifest = self.remote.get(f"{prefix}/{MANIFEST}")
            if manifest is None:
                raise LookupError(f"{label} is not in the store {self.remote.describe()}")
            got = hashlib.sha256(manifest).hexdigest()
            if got != digest:
                raise LookupError(f"{label}: its {MANIFEST} in the store hashes to {got}, not "
                                  f"the {digest} the reference names; not loaded")
            files = {}
            for line in manifest.decode("utf-8", "replace").splitlines():
                want, _, fname = line.partition("  ")
                if not re.fullmatch(r"[0-9a-f]{64}", want) or not _FILE_NAME.match(fname):
                    raise LookupError(f"{label}: {MANIFEST} line {line!r} is not "
                                      f"'<sha256>  <name>'")
                data = self.remote.get(f"{prefix}/{fname}")
                if data is None:
                    raise LookupError(f"{label}: {MANIFEST} lists {fname}, which the store "
                                      f"does not hold")
                files[fname] = data
        except objectstore.StoreError as exc:
            raise LookupError(f"{label}: {exc}") from exc
        final = self.root / namespace / name
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=final.parent))
        for fname, data in {**files, MANIFEST: manifest}.items():
            (staging / fname).write_bytes(data)
        if final.exists():
            for child in final.iterdir():
                child.unlink()
            final.rmdir()
        staging.rename(final)

    def save(self, namespace: str, name: str, files: dict[str, bytes]) -> tuple[str, str]:
        """Write ``files`` and the manifest; return (map-ref, map-sha256).

        Written beside its final place and renamed into it, so a reader never sees a half-written
        map. A directory already at the path is replaced: the path is the action's, and a re-run
        of the same action id is the same artifact."""
        for fname in files:
            if not _FILE_NAME.match(fname) or fname == MANIFEST:
                raise ValueError(f"artifact file name {fname!r} is not servable")
        manifest = manifest_text(files)
        final = self.root / namespace / name
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=final.parent))
        for fname, data in {**files, MANIFEST: manifest}.items():
            (staging / fname).write_bytes(data)
            os.chmod(staging / fname, 0o644)
        os.chmod(staging, 0o755)
        if final.exists():
            for child in final.iterdir():
                child.unlink()
            final.rmdir()
        staging.rename(final)
        return f"{self.store}/{namespace}/{name}", hashlib.sha256(manifest).hexdigest()

    def saved_map(self, action: str) -> steps.Grid:
        """The map a saved segment (``<namespace>/<name>``) wrote, as a Grid."""
        root = self.root / action
        try:
            meta = {}
            for line in (root / "map.yaml").read_text().splitlines():
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
            origin = [float(t) for t in meta["origin"].strip("[]").split(",")]
            return steps.grid_from_pgm((root / meta["image"]).read_bytes(),
                                       res=float(meta["resolution"]), ox=origin[0], oy=origin[1])
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise LookupError(f"the saved map of {action} cannot be read ({exc})") from exc

    def latest_segment(self, namespace: str, bound_ref: str) -> dict | None:
        """The most recently saved segment of ``bound_ref`` in ``namespace``, or None."""
        best = None
        for seg in sorted((self.root / namespace).glob("*/segment.json")):
            try:
                doc = json.loads(seg.read_text())
            except (OSError, ValueError):
                continue
            if doc.get("bound_ref") != bound_ref or not doc.get("graph_ref"):
                continue
            if best is None or doc.get("saved_at_ms", 0) >= best.get("saved_at_ms", 0):
                best = doc
        return best


def render_map(grid: steps.Grid) -> tuple[bytes, bytes]:
    """map_saver's trinary rendering of an occupancy grid: (map.yaml, map.pgm)."""
    w, h = grid.width, grid.height

    def pix(v: int) -> int:
        if v < 0:
            return 205
        return 0 if v >= 65 else (254 if v <= 25 else 205)
    rows = (bytes(pix(grid.data[iy * w + ix]) for ix in range(w)) for iy in range(h - 1, -1, -1))
    pgm = b"P5\n%d %d\n255\n" % (w, h) + b"".join(rows)
    yaml = (f"image: map.pgm\nmode: trinary\nresolution: {grid.res}\n"
            f"origin: [{grid.ox}, {grid.oy}, 0]\nnegate: 0\noccupied_thresh: {OCCUPIED_THRESH}\n"
            f"free_thresh: {FREE_THRESH}\n").encode()
    return yaml, pgm


# ── the jobs ────────────────────────────────────────────────────────────────

# How often a running segment looks at the battery, the explorer and the return.
LOOP_S = 0.5
# How often coverage is recomputed (a pass over the map), and the least change reported.
PROGRESS_S = 5.0
PROGRESS_STEP = 5
# How often a moving segment reads the costmap for the keep-out cells that came into it.
KEEPOUT_WATCH_S = 2.0

# (node, parameter) of each live speed cap proven before motion.
SPEED_READS: tuple[tuple[str, str], ...] = (
    ("/controller_server", "FollowPath.max_vel_x"),
    ("/velocity_smoother", "max_velocity"),
)

# (node, name) of every costmap a planner plans on: the global costmap carries the planner's
# path, the local costmap the controller's trajectory. Nav2 gives each its own filters, so each
# must carry the keep-out filter, or the controller can drive into a keep-out the path avoids.
PLANNING_COSTMAPS: tuple[tuple[str, str], ...] = (
    ("/global_costmap/global_costmap", "global costmap"),
    ("/local_costmap/local_costmap", "local costmap"),
)
KEEPOUT_FILTER_PLUGIN = "nav2_costmap_2d::KeepoutFilter"


def _names(answer: object) -> list[str]:
    """The entries of a string-array parameter as it reads, ``[a, b]``; anything else: none."""
    text = str(answer).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    return [n.strip().strip("'\"") for n in text[1:-1].split(",") if n.strip()]


Emit = Callable[..., None]


class Halted(Exception):
    """A stop path ended the job. The stop path reports; the job says nothing more."""


class _Failed(Exception):
    pass


class _StoppedHere(Exception):
    """The robot was stopped where it stands, and nothing more may move: the map is saved
    there and the segment ends failed with this reason."""


def _speed_within(answer: object, cap: float) -> bool:
    """A cap reading at or below ``cap``: a number, or a list whose first entry (x) is."""
    text = str(answer).strip().strip("[]()")
    try:
        value = float(text.split(",")[0])
    except ValueError:
        return False
    return math.isfinite(value) and value <= cap + 1e-9


def cable_out(imu: bool | None) -> bool:
    """Whether the charging cable is out. On this robot the IMU is silent while the cable is
    in, so only an IMU streaming says out; silent or not reporting is not out. Charge ends on
    this, and an explore-floor segment starts only on it."""
    return imu is True


class _Job:
    NAME = ""

    def __init__(self, *, action_id: str, doc: dict, stack: ExploreStack,
                 limits: BatteryLimits, clock: Clock, emit: Emit) -> None:
        self.action_id, self.doc, self.stack = action_id, doc, stack
        self.limits, self.clock, self._emit = limits, clock, emit
        self.halted = threading.Event()
        self._paused = threading.Event()

    def run(self) -> None:
        self.stack.arm(self.halted)
        try:
            self._run()
        except Halted:
            return
        except _Failed as exc:
            self._emit("ACTION_STATE_FAILED", f"{self.NAME}: {exc}", self._pct(),
                       "ACTION_END_CAUSE_EXECUTION_FAILURE")
        except Exception as exc:  # noqa: BLE001 - the job's boundary: nothing may end it silently
            # A job runs on its own thread: whatever else it raises would end that thread and
            # leave the action RUNNING, with nothing watching the battery. So it stops what can
            # move, and ends FAILED naming the exception.
            traceback.print_exc(file=sys.stderr)
            self._stop_on_error()
            where = "; no motion" if self._before_motion else ""
            self._emit("ACTION_STATE_FAILED",
                       f"{self.NAME}: stopped on an unexpected error: {type(exc).__name__}: "
                       f"{exc}{where}", self._pct(), "ACTION_END_CAUSE_EXECUTION_FAILURE")

    def _stop_on_error(self) -> None:
        """Stop whatever this job may have set moving. Never raises."""

    def _run(self) -> None:
        raise NotImplementedError

    def _pct(self) -> int:
        return 0

    def _say(self, text: str, pct: int | None = None) -> None:
        self._emit("ACTION_STATE_RUNNING", f"{self.NAME}: {text}",
                   self._pct() if pct is None else pct)

    def _wait(self, seconds: float) -> None:
        if self.clock.wait(self.halted, seconds):
            raise Halted

    def _gap(self, seconds: float) -> bool:
        return self.clock.wait(self.halted, seconds)

    def _read(self, read, accept, what: str) -> object:
        v = steps.bounded_read(read, accept, what=what, wait=self._gap)
        if v.verdict == "HALTED":
            raise Halted
        if not v.ok:
            raise _Failed(f"{v.summary()}; no motion" if self._before_motion else v.summary())
        return v.value

    def _call(self, call, what: str) -> steps.ReadVerdict:
        v = steps.bounded_call(call, what=what, wait=self._gap)
        if v.verdict == "HALTED":
            raise Halted
        return v

    _before_motion = True


class ExploreJob(_Job):
    """One explore-floor segment (see the module docstring)."""

    NAME = "explore-floor"

    def __init__(self, *, artifacts: ArtifactDir, namespace: str, **kw) -> None:
        super().__init__(**kw)
        self.artifacts, self.namespace = artifacts, namespace
        self.phase = "start"
        self._resume = threading.Event()
        self._mask: steps.Mask | None = None
        self._bound_m2 = 0.0
        self._covered_m2 = 0.0
        self._mark = (0.0, 0.0, 0.0)
        self._kept_out: list[steps.Keepout] = []
        # Areas exhausted in this segment, in the bound's frame, each drawn in whole once the
        # robot is back on the mark.
        self._live_areas: list[steps.Keepout] = []
        # Each hand keep-out's interior cells in the costmap when the bound was last proven,
        # and now: the watch reads the cells that come into the costmap as it grows.
        self._hand_proven: dict[str, int] = {}
        self._hand_seen: dict[str, int] = {}
        self._watch_at: float | None = None
        self._watch_said_at: float | None = None

    # ── pause / resume, from the adapter's control thread ───────────────────

    def pause(self) -> None:
        """Hold: the explorer and the return goal are stopped now; the job waits."""
        self._resume.clear()
        self._paused.set()
        self.stack.hold()

    def resume(self) -> None:
        self._resume.set()

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.stack.end_segment()

    def _stop_on_error(self) -> None:
        # The explorer, and the binding's goal (the return): each on its own, so one that
        # raises too does not keep the other from being stopped.
        for stop in (self.stack.stop_explorer, self.stack.hold):
            try:
                stop()
            except Exception:  # noqa: BLE001 - reported above; the next stop still runs
                traceback.print_exc(file=sys.stderr)

    def _hold_while_paused(self) -> bool:
        """Wait out a pause. True when one was waited out (the caller restarts its phase)."""
        if not self._paused.is_set():
            return False
        while not self._resume.is_set():
            self._wait(LOOP_S)
        self._paused.clear()
        self._resume.clear()
        return True

    # ── the segment ─────────────────────────────────────────────────────────

    def _pct(self) -> int:
        if not self._bound_m2:
            return 0
        return min(100, round(100 * self._covered_m2 / self._bound_m2))

    def _run(self) -> None:
        doc = self.doc
        ids = split_action_id(self.action_id, self.namespace)
        if ids is None:
            raise _Failed(f"action id {self.action_id!r} in namespace {self.namespace!r} is not "
                          f"a DNS name; no artifact path can be built; no motion")
        namespace, name = ids
        bound_ref = doc["bound_ref"]
        try:
            bound = self.stack.load_bound(bound_ref)
        except (LookupError, ValueError) as exc:
            raise _Failed(f"bound {bound_ref!r}: {exc}; no motion") from exc
        if bound.ref_map is None:
            raise _Failed(f"bound {bound_ref!r} has no saved map to align to; ALIGNED cannot be "
                          f"proven; no motion")
        kset = None
        if doc.get("keepouts_ref"):
            try:
                kset = load_keepout_set(self.artifacts, doc["keepouts_ref"])
            except LookupError as exc:
                raise _Failed(f"{exc}; no motion") from exc

        cap = float(doc["max_speed_mps"])
        self.phase = "verify"
        self._say(f"verifying the live speed cap is within {cap:g} m/s")
        for node, param in SPEED_READS:
            try:
                self._read(lambda n=node, p=param: self.stack.read_param(n, p),
                           lambda a: _speed_within(a, cap), f"speed cap {node} {param}")
            except _Failed as exc:
                raise _Failed(f"the live speed cap is not proven within {cap:g} m/s: "
                              f"{exc}") from exc
        self._keepout_filters_before_motion()
        self._cable_out_before_motion()
        self._battery_before_motion()

        prev = None
        if doc.get("resume"):
            prev = self.artifacts.latest_segment(namespace, bound_ref)
            if prev is None:
                raise _Failed(f"resume requested, but there is no saved segment of bound "
                              f"{bound_ref!r} in namespace {namespace!r}; no motion")
        self.phase = "slam"
        self._say("starting SLAM " + (f"from the saved graph of {prev['action']}" if prev
                                      else "on a fresh map"))
        v = self._call(lambda: self.stack.start_slam(prev["graph_ref"] if prev else None),
                       "SLAM start")
        if not v.ok:
            raise _Failed(f"SLAM did not start: {v.summary()}; no motion")
        pose = self._read(self.stack.robot_pose,
                          lambda p: all(_is_finite_number(c) for c in p), "robot pose on the mark")
        self._mark = tuple(float(c) for c in pose)
        delta = float(prev["delta_deg"]) if prev else self._mark[2] - bound.theta0_deg

        self.phase = "aligned"
        try:
            ref_map = self.artifacts.saved_map(prev["action"]) if prev else bound.ref_map
        except LookupError as exc:
            raise _Failed(f"{exc}; ALIGNED cannot be proven; no motion") from exc
        live = self._read(self.stack.read_map, lambda g: True, "map")
        aligned, line = steps.align(live, ref_map, delta_deg=0.0 if prev else delta,
                                    max_new_frac=0.05 if prev else None)
        if aligned is None:
            raise _Failed(f"alignment not proven: {line}; no motion")
        if not aligned:
            raise _Failed(f"not aligned to {'the saved segment' if prev else 'the bound'}'s map: "
                          f"{line}; no motion")
        self._say(f"aligned: {line}")
        keepouts = []
        self._kept_out = []
        # The saved areas are in the bound's frame; the mark is in this boot's. An area at the
        # mark is one no segment should have exhausted (an adapter that counted Nav2's raw
        # recoveries did): it is said, and neither kept out nor carried forward.
        mx, my = _turn(self._mark[0], self._mark[1], -delta)
        for k in exhausted_keepouts(prev):
            if steps.distance_to_polygon(mx, my, k.polygon) < steps.MARK_CLEAR_M:
                self._say(f"not kept out: {k.name} ({k.reason}) lies within "
                          f"{steps.MARK_CLEAR_M:g} m of the start mark, where no area is "
                          f"exhausted; dropped from this segment's record")
            else:
                self._kept_out.append(k)
        if self._kept_out:
            self._say(f"kept out: {', '.join(k.name for k in self._kept_out)} "
                      f"(exhausted by an earlier segment)")
        if kset is not None:
            why = goal_refusal(self._mark[0], self._mark[1], kset,
                               frame_deg=self._mark[2] - kset.theta0_deg)
            if why:
                raise _Failed(f"{why.replace('the goal', 'the start mark')}; no motion")
            keepouts = turned_keepouts(kset, bound.theta0_deg - kset.theta0_deg)
        for k in bound.hand_keepouts:
            if steps.point_in_polygon(mx, my, k.polygon):
                raise _Failed(f"the start mark lies inside keep-out {k.name!r} ({k.reason}), "
                              f"measured by hand; no motion")

        self.phase = "bound"
        self._bound, self._delta, self._set_keepouts = bound, delta, keepouts
        mask = self._compose_mask()
        self._mask, self._bound_m2 = mask, steps.allowed_area_m2(mask)
        named = (f", with keep-outs {', '.join(k.name for k in keepouts)} from "
                 f"{kset.ref}" if kset else "")
        if bound.hand_keepouts:
            named += (f", with keep-outs measured by hand: "
                      f"{', '.join(k.name for k in bound.hand_keepouts)} (from the bound)")
        self._say(f"loading the keep-out mask turned by {delta:+.3f} deg into this boot's "
                  f"frame{named}")
        v = self._call(lambda: self.stack.load_keepout_mask(mask, delta), "keep-out mask load")
        if not v.ok:
            raise _Failed(f"the keep-out mask did not load: {v.summary()}; no motion")
        v = self._call(self.stack.activate_keepout, "keep-out servers bring-up")
        if not v.ok:
            self._say(f"keep-out servers did not come up ({v.verdict}); restarting them once")
            again = self._call(self.stack.restart_keepout, "keep-out servers restart")
            if not again.ok:
                raise _Failed(f"the keep-out servers did not come up: {v.summary()}; "
                              f"{again.summary()}; no motion")
        v = self._call(self.stack.reset_costmap, "global costmap reset")
        if not v.ok:
            raise _Failed(f"the global costmap was not reset: {v.summary()}; no motion")
        in_force, lines, slam = self._grade((self._mark[0], self._mark[1]))
        if not in_force:
            raise _Failed("bound not in force: " + "; ".join(lines) + "; no motion")
        self._covered_m2 = steps.covered_m2(slam, mask)
        self._say("bound proven in the costmap (" + "; ".join(lines) + "); starting the explorer")

        self._hold_while_paused()
        self.stack.plans()            # paths from before the bound was proven are not this one's
        v = self._call(self.stack.start_explorer, "explorer start")
        if not v.ok:
            raise _Failed(f"the explorer did not start: {v.summary()}; no motion")
        self._before_motion = False
        self.phase = steps.EXPLORING
        self._say(f"exploring; covered {self._covered_m2:.1f} of {self._bound_m2:.1f} m2")
        try:
            sup, counter = self._explore()
        except _StoppedHere as stop:
            ref, digest = self._save(namespace, name, bound_ref, delta, prev, "remaining",
                                     self._counter)
            raise _Failed(f"{stop}; saved where it stands; covered {self._covered_m2:.1f} of "
                          f"{self._bound_m2:.1f} m2 ({self._pct()}%); exhausted: "
                          f"{self._exhausted_text(self._counter)}; map-ref={ref} "
                          f"map-sha256={digest}") from None

        exhausted = self._exhausted_text(counter)
        frontier = "none-left" if sup.reason.startswith("explorer:") else "remaining"
        if sup.state == steps.HARD_STOP:
            raise _Failed(f"hard stop: {sup.reason}; stopped where it stands, not returned; "
                          f"power off")
        if sup.state == steps.RETURN_FAILED:
            ref, digest = self._save(namespace, name, bound_ref, delta, prev, frontier, counter)
            raise _Failed(f"return to the mark failed ({self._nav_result}); saved where it "
                          f"stands; covered {self._covered_m2:.1f} of {self._bound_m2:.1f} m2 "
                          f"({self._pct()}%); exhausted: {exhausted}; map-ref={ref} "
                          f"map-sha256={digest}")
        self.phase = "saving"
        self._say("at the mark; saving the map, the pose graph and the frame delta")
        ref, digest = self._save(namespace, name, bound_ref, delta, prev, frontier, counter)
        self._emit("ACTION_STATE_SUCCEEDED",
                   f"{self.NAME}: segment done; covered {self._covered_m2:.1f} of "
                   f"{self._bound_m2:.1f} m2 ({self._pct()}%); exhausted: {exhausted}; "
                   f"frontier: {frontier}; map-ref={ref} map-sha256={digest}", self._pct())

    def _exhausted_text(self, counter: steps.AreaCounter) -> str:
        text = ", ".join(counter.exhausted) or "none"
        if self._kept_out:
            text += f"; kept out: {', '.join(k.name for k in self._kept_out)}"
        return text

    def _compose_mask(self) -> steps.Mask:
        """The segment's keep-out mask: the bound's, with the keep-out set's zones, the areas
        earlier segments exhausted, the areas this one exhausted and the keep-outs measured by
        hand drawn in, in the bound's frame, then turned into this boot's."""
        m = steps.apply_keepouts(self._bound.mask,
                                 self._set_keepouts + self._kept_out + self._live_areas)
        return steps.rotate_mask(steps.apply_keepouts(m, self._bound.hand_keepouts), self._delta)

    def _grade(self, robot: tuple[float, float]) -> tuple[bool, list[str], steps.Grid]:
        """The loaded mask graded in the live costmap, and each keep-out measured by hand read
        back on its own -- whole in the mask the server serves, lethal wherever the costmap
        reaches: (in force, the lines that say why, the map read)."""
        costmap = self._read(self.stack.read_costmap, lambda g: True, "global costmap")
        slam = self._read(self.stack.read_map, lambda g: True, "map")
        in_force, lines = steps.grade_bound(costmap, slam, self._mask, robot=robot)
        if self._bound.hand_keepouts:
            served = self._read(self.stack.read_keepout_mask, lambda g: True,
                                "keep-out mask as served")
            hand = self._hand_turned()
            hand_ok, hand_lines = steps.grade_keepouts(costmap, hand, served=served)
            in_force, lines = in_force and hand_ok, lines + hand_lines
            self._hand_proven = {k.name: steps.keepout_cells(costmap, k)[0] for k in hand}
            self._hand_seen = dict(self._hand_proven)
        return in_force, lines, slam

    def _hand_turned(self) -> list[steps.Keepout]:
        """The keep-outs measured by hand, in this boot's frame."""
        return [steps.Keepout(k.name, k.reason, tuple(_turn(x, y, self._delta)
                                                      for x, y in k.polygon))
                for k in self._bound.hand_keepouts]

    def _keepout_at(self, x: float, y: float) -> str:
        """What the loaded mask keeps out at (x, y) in this boot's frame, named."""
        bx, by = _turn(x, y, -self._delta)
        for kind, ks in (("measured by hand", self._bound.hand_keepouts),
                         ("of the keep-out set", self._set_keepouts),
                         ("exhausted by an earlier segment", self._kept_out),
                         ("exhausted in this segment", self._live_areas)):
            for k in ks:
                if steps.point_in_polygon(bx, by, k.polygon):
                    return f"keep-out {k.name!r} ({k.reason}), {kind}"
        return "the room's bound"

    def _stop_here(self) -> None:
        self._stop_on_error()

    def _check_plans(self) -> None:
        """Every path the planner published, and every local trajectory the controller
        published, held against the loaded mask; and the hand keep-outs read in the costmap
        as it grows. The bound was proven lethal before motion, and the planner never draws a
        path through lethal cells: a path a cell deep in a keep-out -- to a goal there, or
        across one -- means the bound is not in force now. The controller's trajectory can
        leave the planner's path (TEB's arcs swung ~0.29 m off a straight line on the robot),
        and it is what the robot drives. Either stops the robot where it stands; nothing more
        moves."""
        for plan in self.stack.plans():
            for i, (x, y) in enumerate(plan):
                if steps.deep_in_mask(self._mask, x, y):
                    self._stop_here()
                    gx, gy = plan[-1]
                    where = ("its goal" if i == len(plan) - 1
                             else f"on its way to a goal at ({gx:.2f}, {gy:.2f})")
                    raise _StoppedHere(f"the planner's path enters {self._keepout_at(x, y)} at "
                                       f"({x:.2f}, {y:.2f}), {where}: the bound is not in force; "
                                       f"stopped where it stands, not returned")
        for plan in self.stack.local_plans():
            if plan is None:
                self._stop_here()
                raise _StoppedHere("the controller's local plan could not be placed in the map "
                                   "(no transform from its frame), so what the robot is about "
                                   "to drive cannot be held against the bound; stopped where it "
                                   "stands, not returned")
            for x, y in plan:
                if steps.deep_in_mask(self._mask, x, y):
                    self._stop_here()
                    raise _StoppedHere(f"the controller's local plan enters "
                                       f"{self._keepout_at(x, y)} at ({x:.2f}, {y:.2f}): the "
                                       f"robot was about to drive there; stopped where it "
                                       f"stands, not returned")
        self._watch_keepouts()

    def _watch_keepouts(self) -> None:
        """The hand keep-outs read in the costmap every KEEPOUT_WATCH_S while the robot moves.
        At a fresh start the costmap may not reach the whole of one (the bound was proven on
        the served mask for the rest); as it grows, every cell inside one that comes into it
        must be lethal. One that is not means the bound is not in force there."""
        if not self._hand_seen:
            return
        now = self.clock.now()
        if self._watch_at is not None and now - self._watch_at < KEEPOUT_WATCH_S:
            return
        self._watch_at = now
        costmap = self.stack.read_costmap()
        if costmap is None:
            return
        grew = []
        for k in self._hand_turned():
            n, leak, _, _ = steps.keepout_cells(costmap, k)
            new = n - self._hand_proven.get(k.name, 0)
            if leak:
                self._stop_here()
                raise _StoppedHere(f"keep-out {k.name!r} ({k.reason}), measured by hand: "
                                   f"{max(new, 0)} cells newly in the costmap; of its {n} cells "
                                   f"there, not lethal: {leak}: the bound is not in force; "
                                   f"stopped where it stands, not returned")
            if n > self._hand_seen.get(k.name, 0):
                grew.append(f"keep-out {k.name!r}: {n} cells in the costmap, {new} more than "
                            f"when the bound was proven; not lethal: 0")
            self._hand_seen[k.name] = n
        if grew and (self._watch_said_at is None or now - self._watch_said_at >= PROGRESS_S):
            self._watch_said_at = now
            self._say(f"{self.phase}; " + "; ".join(grew))

    def _keep_out_in_segment(self, area: str, counter: steps.AreaCounter,
                             sup: steps.Supervisor) -> None:
        """An exhausted area kept out of the rest of this segment. The robot stands in it, so
        it first drives back to the mark with the area NOT drawn in: the bound, the keep-out set
        and the keep-outs measured by hand stay in force, and every plan is held against them
        as while exploring. A way out drawn through the area left the robot a strip too narrow
        for Nav2's recoveries: a robot day's retreat aborted on a Spin refused "Collision Ahead"
        after the navigator's own BackUp, and no strip wide enough to turn in after a back-up
        left most of the area kept out (maintainer's ruling). On the mark the whole area is
        drawn in, reloaded and proven with the robot still, and only then is the explorer
        started again. Anything short of that stops the robot where it stands."""
        self.stack.skip_area(area)
        self._say(f"exploring; exhausted: {area} after {counter.limit} recovery episodes; "
                  f"stopping to keep it out of this segment")
        self._stop_here()
        self.phase = "keeping-out"
        self._wait(steps.Supervisor.SETTLE_S)

        def refused(why: str) -> _StoppedHere:
            return _StoppedHere(f"bound not in force after keeping out {area}: {why}; stopped "
                                f"where it stands, not returned")
        square = dict(counter.exhausted_polygons())[area]
        kept = steps.Keepout(area, f"{counter.limit} recovery episodes in this segment",
                             tuple(_turn(x, y, -self._delta) for x, y in square))
        self._say(f"exploring; leaving {area} for the mark; it is kept out once the robot is "
                  f"there")
        self._leave_area(area, sup)
        self._live_areas.append(kept)
        lines = self._reload_and_prove((self._mark[0], self._mark[1]), refused)
        self.stack.plans()            # drawn before the reload: not against this mask
        self.stack.recoveries()       # the stop's own, and the way out's
        v = self._call(self.stack.start_explorer, "explorer restart")
        if not v.ok:
            raise refused(f"the explorer did not start again: {v.summary()}")
        self.phase = steps.EXPLORING
        self._say(f"exploring; {area} kept out of this segment (bound proven again: "
                  + "; ".join(lines) + ")")

    def _reload_and_prove(self, robot: tuple[float, float], refused) -> list[str]:
        """The mask composed again, loaded, reloaded by the running servers, the costmap reset
        and the bound proven with the robot still: the grading's lines, or ``refused``."""
        mask = self._compose_mask()
        for call, what in ((lambda: self.stack.load_keepout_mask(mask, self._delta),
                            "keep-out mask load"),
                           (self.stack.reload_keepout_mask, "keep-out mask reload")):
            v = self._call(call, what)
            if not v.ok:
                raise refused(v.summary())
        self._mask, self._bound_m2 = mask, steps.allowed_area_m2(mask)
        v = self._call(self.stack.reset_costmap, "global costmap reset")
        if not v.ok:
            raise refused(v.summary())
        try:
            in_force, lines, _ = self._grade(robot)
        except _Failed as exc:
            raise refused(str(exc)) from None
        if not in_force:
            raise refused("; ".join(lines))
        return lines

    def _leave_area(self, area: str, sup: steps.Supervisor) -> None:
        """Back to the mark, the battery and every planned path watched as while exploring.
        Not there: stopped where it stands."""
        self.stack.go_to_mark(*self._mark)
        t0 = self.clock.now()
        while True:
            self._check_plans()
            now = self.clock.now()
            mv, age = self.stack.battery_mv(), self.stack.battery_age_s()
            if mv is not None and age is not None and age <= BATTERY_STALE_S:
                sup.battery(now, mv)
            if sup.win.sustained_at_or_below(sup.hard_mv, now):
                self._stop_here()
                raise _StoppedHere(f"hard stop: battery sustained at or below {sup.hard_mv:g} mV "
                                   f"while leaving {area}; stopped where it stands")
            result = self.stack.mark_result()
            if result == "SUCCEEDED":
                return
            if result is not None or now - t0 >= LEAVE_AREA_S:
                self._stop_here()
                why = result or f"not back within {LEAVE_AREA_S:g} s"
                raise _StoppedHere(f"could not leave {area} for the mark ({why}); "
                                   f"stopped where it stands, not returned")
            self._wait(LOOP_S)

    def _return_mv(self) -> float:
        return float(self.doc.get("battery_return_mv", self.limits.return_mv))

    def _fresh_battery_mv(self) -> float | None:
        age = self.stack.battery_age_s()
        if age is None or age > BATTERY_STALE_S:
            return None
        return self.stack.battery_mv()

    def _cable_out_before_motion(self) -> None:
        """The charging cable out, read as Charge reads it, or no segment: a robot on its
        cable is the operator's, charging it, and must not be driven off it."""
        self.phase = "cable"
        try:
            imu = self._read(self.stack.imu_streaming, lambda v: True, "IMU")
        except _Failed as exc:
            raise _Failed(f"the charging cable is not proven out (the IMU is not reporting): "
                          f"{exc}") from exc
        if not cable_out(imu):
            raise _Failed("the charging cable is in; no motion")

    def _keepout_filters_before_motion(self) -> None:
        """A keep-out filter, enabled, on every costmap a planner plans on, all reading one
        filter-info topic, or no segment. The mask is loaded and proven later in the segment;
        a costmap without the filter would never see it, and the controller planning on it
        could drive into a keep-out the planner's path avoids."""
        self.phase = "verify"
        self._say("verifying the keep-out filter on every planning costmap")
        found: list[tuple[str, str, str]] = []
        for node, layer in PLANNING_COSTMAPS:
            def read(param: str, n: str = node, layer: str = layer) -> object:
                try:
                    return self._read(lambda: self.stack.read_param(n, param), lambda a: True,
                                      f"{layer} {n} {param}")
                except _Failed as exc:
                    raise _Failed(f"the keep-out filter on the {layer} ({n}) is not proven: "
                                  f"{exc}") from exc

            filters = read("filters")
            names = _names(filters)
            keepout = [f for f in names if str(read(f"{f}.plugin")) == KEEPOUT_FILTER_PLUGIN]
            if not keepout:
                raise _Failed(f"the {layer} ({node}) has no keep-out filter: filters reads "
                              f"{filters}; it could plan into a keep-out; no motion")
            name = keepout[0]
            if str(read(f"{name}.enabled")).strip().lower() == "false":
                raise _Failed(f"the {layer} ({node}) has its keep-out filter {name!r} "
                              f"disabled; it could plan into a keep-out; no motion")
            found.append((layer, name, str(read(f"{name}.filter_info_topic"))))
        topics = {topic for _, _, topic in found}
        if len(topics) != 1:
            said = "; ".join(f"the {layer} reads {topic}" for layer, _, topic in found)
            raise _Failed(f"the planning costmaps' keep-out filters read different filter-info "
                          f"topics ({said}): one mask is loaded and proven, so a costmap on "
                          f"another is not held by it; no motion")
        self._say("keep-out filter on every planning costmap: " + "; ".join(
            f"{layer} {name} on {topic}" for layer, name, topic in found))

    def _battery_before_motion(self) -> None:
        """A fresh battery reading above the return threshold, or no segment: without one
        neither the return nor the hard stop could ever fire."""
        self.phase = "battery"
        src, ret = self.stack.battery_source(), self._return_mv()
        try:
            mv = self._read(self._fresh_battery_mv, lambda v: v > ret,
                            f"battery on {src}")
        except _Failed as exc:
            if "no answer" in str(exc).lower():
                raise _Failed(f"no battery reading on {src} within {BATTERY_STALE_S:g} s; the "
                              f"return threshold and the hard stop could not fire: "
                              f"{exc}") from exc
            raise _Failed(f"the battery is at or below the return threshold {ret:g} mV: "
                          f"{exc}") from exc
        self._say(f"battery {mv:.0f} mV on {src} (return at {ret:g} mV, hard stop at "
                  f"{self.limits.hard_mv:g} mV)")

    def _explore(self) -> tuple[steps.Supervisor, steps.AreaCounter]:
        doc = self.doc
        sup = steps.Supervisor(doc.get("battery_return_mv", self.limits.return_mv),
                               self.limits.hard_mv, time_box_s=doc.get("time_box_s"))
        self._nav_result: str | None = None
        n = int(doc.get("exhausted_after_recoveries", DEFAULT_EXHAUSTED_AFTER))
        counter = steps.AreaCounter(n, mark=(self._mark[0], self._mark[1]))
        self._counter = counter
        reported, last_cover_t = self._pct(), None
        while True:
            if self._hold_while_paused():
                if sup.state == steps.EXPLORING:
                    v = self._call(self.stack.start_explorer, "explorer restart")
                    if not v.ok:
                        raise _Failed(f"the explorer did not restart after the pause: "
                                      f"{v.summary()}")
                elif sup.state == steps.RETURNING:
                    self.stack.go_to_mark(*self._mark)
                self._say(f"resumed; {sup.state.lower()}")
            self._check_plans()
            now = self.clock.now()
            mv, age = self.stack.battery_mv(), self.stack.battery_age_s()
            if mv is not None and age is not None and age <= BATTERY_STALE_S:
                sup.battery(now, mv)
            elif sup.state == steps.EXPLORING:
                # Blind to the battery, the threshold and the hard stop cannot fire: return.
                since = "never" if age is None else f"{age:.0f} s ago"
                sup.end_exploring(f"battery reading stale: the last on "
                                  f"{self.stack.battery_source()} was {since}")
            if sup.state == steps.EXPLORING:
                for x, y in self.stack.recoveries():
                    spared = len(counter.spared)
                    area = counter.add(x, y)
                    if area:
                        self._keep_out_in_segment(area, counter, sup)
                    elif len(counter.spared) > spared:
                        self._say(f"exploring; a recovery in {counter.spared[-1]}, within "
                                  f"{steps.MARK_CLEAR_M:g} m of the start mark: that area is "
                                  f"never exhausted")
            status = self.stack.explorer_status()
            if sup.state == steps.RETURNING:
                self._nav_result = self.stack.mark_result()
            for act in sup.step(now, status, self._nav_result):
                if act[0] == "stop_explore":
                    self.stack.stop_explorer()
                elif act[0] == "cancel_nav":
                    self.stack.hold()
                elif act[0] == "send_mark":
                    self.stack.go_to_mark(*self._mark)
                    self.phase = steps.RETURNING
                    self._say("returning to the mark")
                elif act[0] == "say" and sup.state == steps.STOPPING:
                    self.phase = steps.STOPPING
                    self._say(f"{act[1]}; explorer stopped, waiting for it to pause")
            if sup.state in (steps.AT_MARK, steps.RETURN_FAILED, steps.HARD_STOP):
                return sup, counter
            if sup.state == steps.EXPLORING and (
                    last_cover_t is None or now - last_cover_t >= PROGRESS_S):
                last_cover_t = now
                slam = self.stack.read_map()
                if slam is not None:
                    self._covered_m2 = steps.covered_m2(slam, self._mask)
                    if self._pct() >= reported + PROGRESS_STEP:
                        reported = self._pct()
                        self._say(f"exploring; covered {self._covered_m2:.1f} of "
                                  f"{self._bound_m2:.1f} m2")
            self._wait(LOOP_S)

    def _save(self, namespace: str, name: str, bound_ref: str, delta: float, prev,
              frontier: str, counter: steps.AreaCounter) -> tuple[str, str]:
        slam = self._read(self.stack.read_map, lambda g: True, "map")
        self._covered_m2 = steps.covered_m2(slam, self._mask)
        saved = self._read(lambda: self.stack.save_map(f"{namespace}/{name}"), lambda s: True,
                           "map and pose graph save")
        why = map_yaml_refusal(saved.files.get("map.yaml", b""))
        if why:
            raise _Failed(f"the saved map would read unknown as free, so no zone could be "
                          f"proposed from it: {why}")
        files = dict(saved.files)
        files["frame-delta.json"] = (json.dumps({
            "dx_m": 0.0, "dy_m": 0.0, "dyaw_rad": math.radians(delta), "dyaw_deg": delta,
            "source": ("the resumed graph's frame, as saved by the segment it was loaded from"
                       if prev else "this boot's yaw on the mark minus the bound's theta0"),
        }, sort_keys=True) + "\n").encode()
        files["segment.json"] = (json.dumps({
            "action": f"{namespace}/{name}", "bound_ref": bound_ref,
            "keepouts_ref": self.doc.get("keepouts_ref", ""), "resumed_from":
            prev["action"] if prev else "", "graph_ref": saved.graph_ref, "delta_deg": delta,
            "mark": list(self._mark), "covered_m2": round(self._covered_m2, 3),
            "bound_m2": round(self._bound_m2, 3), "frontier": frontier,
            "exhausted": list(counter.exhausted),
            "exhausted_areas": self._exhausted_areas(counter, delta),
            "saved_at_ms": int(time.time() * 1000),
        }, sort_keys=True) + "\n").encode()
        try:
            ref, digest = self.artifacts.save(namespace, name, files)
        except (OSError, ValueError) as exc:
            raise _Failed(f"the map could not be saved at {namespace}/{name}: {exc}") from exc
        if self.artifacts.remote is not None:
            self._upload(namespace, name)
        return ref, digest

    def _upload(self, namespace: str, name: str) -> None:
        from fleet_adapter_ros2 import objectstore
        last: list[str] = []

        def once() -> bool | None:
            try:
                self.artifacts.upload(namespace, name)
            except (objectstore.StoreError, OSError) as exc:
                last[:] = [str(exc)]
                return None
            return True
        v = self._call(once, f"upload to {self.artifacts.remote.describe()}")
        if not v.ok:
            raise _Failed(f"the map is saved on the robot at {namespace}/{name} but did not "
                          f"reach the store: {last[0] if last else v.summary()}")
        self._say(f"uploaded to the store at {namespace}/{name}")

    def _exhausted_areas(self, counter: steps.AreaCounter, delta: float) -> list[dict]:
        """The earlier segments' exhausted areas and this one's, in the bound's frame. A
        resumed segment carries every earlier area forward, so the set only grows."""
        out = [{"name": k.name, "reason": k.reason, "polygon": [list(p) for p in k.polygon]}
               for k in self._kept_out]
        known = {a["name"] for a in out}
        for name, square in counter.exhausted_polygons():
            if name not in known:
                out.append({"name": name,
                            "reason": f"{counter.limit} recovery episodes in segment "
                                      f"{self.action_id}",
                            "polygon": [list(_turn(x, y, -delta)) for x, y in square]})
        return out


def exhausted_keepouts(prev: dict | None) -> list[steps.Keepout]:
    """The exhausted areas a saved segment carries, as keep-outs in the bound's frame."""
    if not prev:
        return []
    return [steps.Keepout(name=str(a["name"]), reason=str(a["reason"]),
                          polygon=tuple((float(x), float(y)) for x, y in a["polygon"]))
            for a in prev.get("exhausted_areas") or []]


#: The key in a bound's bound.json that names its hand-measured keep-out file.
HAND_KEEPOUTS_KEY = "hand_keepouts"


def read_hand_keepouts(bound_dir: Path, bound_doc: dict) -> list[steps.Keepout]:
    """The keep-outs a bound declares were measured by hand, or [] when it declares none.

    ``bound.json`` names the file (``"hand_keepouts": "keepouts.json"``), in the bound's own
    directory. The file holds its polygons in the bound's map frame -- the frame the mask is
    drawn in -- and the theta0 it was turned with; a theta0 that is not the bound's means the
    file was made for another heading line, and it is refused. A declared file that is
    missing or unreadable is refused too, never skipped: what the operator measured is part
    of the bound. LookupError naming why."""
    name = bound_doc.get(HAND_KEEPOUTS_KEY)
    if name is None:
        return []
    if not isinstance(name, str) or not _FILE_NAME.match(name):
        raise LookupError(f"{HAND_KEEPOUTS_KEY} {json.dumps(name)} is not a file name in the "
                          f"bound's directory")
    try:
        doc = json.loads((bound_dir / name).read_text())
    except OSError as exc:
        raise LookupError(f"the hand-measured keep-outs {name} it declares cannot be read "
                          f"({exc.strerror or exc})") from exc
    except ValueError as exc:
        raise LookupError(f"the hand-measured keep-outs {name} are not JSON ({exc})") from exc
    try:
        theta0 = doc["theta0_deg"]
        if not _is_finite_number(theta0):
            raise ValueError("theta0_deg is not a number")
        if abs(theta0 - float(bound_doc["theta0_deg"])) > 1e-6:
            raise ValueError(f"they were turned with theta0 {theta0:g} deg, and the bound's is "
                             f"{float(bound_doc['theta0_deg']):g} deg: made for another heading "
                             f"line")
        out = []
        for k in doc["keepouts"]:
            poly = tuple((float(p["x"]), float(p["y"])) for p in k["polygon"])
            if not k.get("name") or not k.get("reason") or len(poly) < 3:
                raise ValueError(f"keep-out {k.get('name')!r} needs a name, a reason and at "
                                 f"least three points")
            if not all(math.isfinite(c) for p in poly for c in p):
                raise ValueError(f"keep-out {k['name']!r} has a point that is not a number")
            out.append(steps.Keepout(name=str(k["name"]), reason=str(k["reason"]),
                                     polygon=poly))
        if not out:
            raise ValueError("it lists no keep-out")
    except KeyError as exc:
        raise LookupError(f"the hand-measured keep-outs {name} have no {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise LookupError(f"the hand-measured keep-outs {name}: {exc}") from exc
    return out


class ChargeJob(_Job):
    """Wait for a charge: RUNNING with the battery in mV until it reads full AND the cable is
    out -- the IMU streaming again, since on the robot the IMU is silent on the charging cable,
    the silence a boot gate fails. Never moves the robot."""

    NAME = "Charge"
    LOOP_S = 1.0
    REPORT_MV = 25.0

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.full_mv = float(self.doc.get("full_mv", self.limits.full_mv))
        self._mv: float | None = None

    def _pct(self) -> int:
        if self._mv is None:
            return 0
        span = self.full_mv - self.limits.hard_mv
        return max(0, min(99, round(100 * (self._mv - self.limits.hard_mv) / span)))

    def _run(self) -> None:
        self.stack.charge_wait()
        last_phase, last_mv = None, None
        while True:
            mv, imu = self.stack.battery_mv(), self.stack.imu_streaming()
            self._mv = mv
            if mv is not None and mv >= self.full_mv and cable_out(imu):
                self._emit("ACTION_STATE_SUCCEEDED",
                           f"{self.NAME}: battery full ({mv:.0f} mV, full at "
                           f"{self.full_mv:.0f} mV) and cable out (IMU streaming)", 100)
                return
            if mv is None:
                phase, text = "unread", "waiting for charge: no battery reading"
            elif mv < self.full_mv:
                phase, text = "charging", (f"waiting for charge: {mv:.0f} mV (full at "
                                           f"{self.full_mv:.0f} mV)")
            else:
                imu_text = "IMU silent" if imu is False else "IMU not reporting"
                phase, text = "full", (f"battery full ({mv:.0f} mV); waiting for the cable to "
                                       f"come out ({imu_text})")
            if phase != last_phase or (mv is not None and last_mv is not None
                                       and abs(mv - last_mv) >= self.REPORT_MV):
                self._emit("ACTION_STATE_RUNNING", text, self._pct())
                last_phase, last_mv = phase, mv
            self._wait(self.LOOP_S)
