# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The steps one floor-exploration segment is built from (pure: no ROS, no protobuf).

Each is a port of a tool that ran the first supervised explorations of one room on a robot
(2026-09-27/28), and each is here because the run needed it:

* **A bounded read and a bounded call.** Under SLAM and Nav2 load a service reply is computed
  and not delivered ("failed to send response ... client will not receive response"). It broke
  a speed-cap read, a keep-out lifecycle bring-up and a costmap parameter read on the same
  evening. A read that gets no answer is read again, up to three more times, five seconds
  apart, and the verdict is exact: a wrong answer is never read again, and silence is never
  taken for a value.
* **The keep-out mask turned into this boot's frame.** A fresh SLAM map's frame is the odometry
  frame at SLAM start, and on the robot the odometry's yaw at boot is not the heading line
  (offsets from -76 to +148 degrees were recorded). Both frames start on the mark, so they
  differ by a rotation about it, and the mask is turned by that angle before it is loaded.
* **ALIGNED.** The mask was drawn in a saved map's frame; before anything moves, the live map's
  walls must lie on that map's walls once turned by the boot's delta (and, on a resume, must not
  lie on its free space: a graph laid down in the wrong place draws walls there).
* **The bound proven in the live costmap.** The explorer reads the global costmap and nothing
  else, so that is where the bound must be in force before anything moves: every keep-out cell
  lethal, and the unknown cells of the map still unknown -- a keep-out filter that writes FREE
  erases them, and every frontier with them.
* **The supervisor.** A battery threshold is a window of samples, never one sag under load;
  and the explorer's stop cancels every navigation goal, so the return goal is sent only after
  the explorer has said it paused and a settle has passed, or after a floor wait.
* **Exhausted areas.** N recovery episodes in one area mark it exhausted, once, by name: the
  segment ends and returns to the mark, and later resumed segments keep the area out. Nav2's
  own count rises by several for one stop, so it is read as episodes; and no area at the start
  mark is ever exhausted.
"""

from __future__ import annotations

import collections
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

# The ruled fallback for a dropped reply: read again up to three more times, five seconds
# apart; continue only on an exact answer.
RETRIES = 3
RETRY_GAP_S = 5.0


@dataclass
class ReadVerdict:
    """A bounded read's outcome. ``verdict`` is EXACT, MISMATCH, NO ANSWER or HALTED;
    ``attempts`` has one line per read, which is what a failure report quotes."""

    verdict: str
    value: object = None
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict in ("EXACT", "DONE")

    def summary(self) -> str:
        return f"{self.verdict} after {len(self.attempts)} attempt(s): " + "; ".join(self.attempts)


def bounded_read(read: Callable[[], object], accept: Callable[[object], bool], *, what: str,
                 retries: int = RETRIES, gap_s: float = RETRY_GAP_S,
                 wait: Callable[[float], bool]) -> ReadVerdict:
    """Read ``what`` once, and again up to ``retries`` times ``gap_s`` apart while it gets no
    answer (``read()`` returns None). An answer is judged by ``accept``: EXACT continues, and
    MISMATCH ends the read at once -- a wrong value is never read again, only silence is.
    ``wait(seconds)`` returns True when a stop arrived during the gap (HALTED)."""
    attempts: list[str] = []
    for i in range(1 + retries):
        if i and wait(gap_s):
            attempts.append(f"{what}: stopped while waiting to read again")
            return ReadVerdict("HALTED", None, attempts)
        answer = read()
        if answer is None:
            attempts.append(f"{what}: no answer (read {i + 1} of {1 + retries})")
            continue
        if accept(answer):
            attempts.append(f"{what}: {answer} exact (read {i + 1} of {1 + retries})")
            return ReadVerdict("EXACT", answer, attempts)
        attempts.append(f"{what}: {answer} does not match; not read again")
        return ReadVerdict("MISMATCH", answer, attempts)
    return ReadVerdict("NO ANSWER", None, attempts)


def bounded_call(call: Callable[[], bool | None], *, what: str, retries: int = RETRIES,
                 gap_s: float = RETRY_GAP_S, wait: Callable[[float], bool]) -> ReadVerdict:
    """Make a service call (a lifecycle transition, a costmap reset, a graph load), and make it
    again while its reply is lost (``call()`` returns None). True is DONE; False is REFUSED --
    the server answered no, and asking again would not change its answer."""
    attempts: list[str] = []
    for i in range(1 + retries):
        if i and wait(gap_s):
            attempts.append(f"{what}: stopped while waiting to call again")
            return ReadVerdict("HALTED", None, attempts)
        answer = call()
        if answer is None:
            attempts.append(f"{what}: no answer (call {i + 1} of {1 + retries})")
            continue
        if answer:
            attempts.append(f"{what}: done (call {i + 1} of {1 + retries})")
            return ReadVerdict("DONE", True, attempts)
        attempts.append(f"{what}: refused; not called again")
        return ReadVerdict("REFUSED", False, attempts)
    return ReadVerdict("NO ANSWER", None, attempts)


# ── grids and masks ─────────────────────────────────────────────────────────
#
# Both are stored as nav_msgs/OccupancyGrid stores its data: row-major, row 0 at the BOTTOM
# (the origin corner), so cell (ix, iy) is at world (ox + (ix + 0.5) res, oy + (iy + 0.5) res).


@dataclass
class Grid:
    """An occupancy grid: -1 unknown, 0..100 cost or occupancy."""

    width: int
    height: int
    res: float
    ox: float
    oy: float
    data: list[int] = field(default_factory=list)


@dataclass
class Mask:
    """A keep-out mask. ``keep[i]`` True is keep-out; False is allowed (the mask file's
    UNKNOWN, which the keep-out filter leaves as it is)."""

    width: int
    height: int
    res: float
    ox: float
    oy: float
    keep: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class Keepout:
    """An operator's keep-out: a polygon in the mask's frame, for what no sensor sees."""

    name: str
    reason: str
    polygon: tuple[tuple[float, float], ...]


# The mask file's two values: 0 is keep-out; 205 is UNKNOWN, which with free_thresh 0.10 the
# keep-out filter leaves untouched. A FREE value would erase unknown cells, so none is allowed.
_PGM_KEEP, _PGM_ALLOWED = 0, 205


def _cell(x: float, y: float, ox: float, oy: float, res: float) -> tuple[int, int]:
    return math.floor((x - ox) / res), math.floor((y - oy) / res)


def mask_from_pgm_pixels(data: bytes) -> tuple[int, int, bytes]:
    """An 8-bit binary PGM (P5, row 0 at the TOP) -> (width, height, pixels)."""
    fields, i = [], 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    if fields[0] != b"P5" or int(fields[3]) > 255:
        raise ValueError("not an 8-bit binary PGM")
    w, h = int(fields[1]), int(fields[2])
    pix = data[i + 1:i + 1 + w * h]
    if len(pix) != w * h:
        raise ValueError(f"the PGM holds {len(pix)} pixels, not {w}x{h}")
    return w, h, pix


def mask_from_pgm(data: bytes, *, res: float, ox: float, oy: float) -> Mask:
    """An 8-bit binary PGM holding only 0 and 205 -> Mask."""
    w, h, pix = mask_from_pgm_pixels(data)
    if not set(pix) <= {_PGM_KEEP, _PGM_ALLOWED}:
        raise ValueError("not a keep-out mask: it holds values other than 0 and 205")
    keep = [pix[(h - 1 - iy) * w + ix] == _PGM_KEEP for iy in range(h) for ix in range(w)]
    return Mask(width=w, height=h, res=res, ox=ox, oy=oy, keep=keep)


def mask_to_pgm(mask: Mask) -> bytes:
    w, h = mask.width, mask.height
    rows = (bytes(_PGM_KEEP if mask.keep[iy * w + ix] else _PGM_ALLOWED for ix in range(w))
            for iy in range(h - 1, -1, -1))
    return b"P5\n%d %d\n255\n" % (w, h) + b"".join(rows)


def mask_yaml(mask: Mask, image: str) -> bytes:
    """The map_server YAML beside a mask: trinary, free_thresh 0.10, so 205 reads unknown."""
    return (f"image: {image}\nmode: trinary\nresolution: {mask.res}\n"
            f"origin: [{mask.ox}, {mask.oy}, 0]\nnegate: 0\noccupied_thresh: 0.65\n"
            f"free_thresh: 0.10\n").encode()


def rotate_mask(mask: Mask, delta_deg: float) -> Mask:
    """The mask in a frame turned by ``delta_deg`` about the origin (the mark).

    A point q of the mask's frame is at R(delta) q in the new one, so each new cell is looked
    up at R(-delta) p in the old mask, nearest cell. A cell that falls outside the old mask is
    KEEP-OUT: outside the old mask is outside the room. The new grid covers the turned extent.
    """
    if not delta_deg:
        return replace(mask, keep=list(mask.keep))
    res, w, h = mask.res, mask.width, mask.height
    d = math.radians(delta_deg)
    c, s = math.cos(d), math.sin(d)
    corners = [(mask.ox, mask.oy), (mask.ox + w * res, mask.oy), (mask.ox, mask.oy + h * res),
               (mask.ox + w * res, mask.oy + h * res)]
    turned = [(c * x - s * y, s * x + c * y) for x, y in corners]
    nx0, nx1 = min(p[0] for p in turned), max(p[0] for p in turned)
    ny0, ny1 = min(p[1] for p in turned), max(p[1] for p in turned)
    nw, nh = math.ceil((nx1 - nx0) / res), math.ceil((ny1 - ny0) / res)
    keep = []
    for iy in range(nh):
        py = ny0 + (iy + 0.5) * res
        for ix in range(nw):
            px = nx0 + (ix + 0.5) * res
            qx, qy = c * px + s * py, -s * px + c * py
            cx, cy = _cell(qx, qy, mask.ox, mask.oy, res)
            keep.append(mask.keep[cy * w + cx] if 0 <= cx < w and 0 <= cy < h else True)
    return Mask(width=nw, height=nh, res=res, ox=round(nx0, 4), oy=round(ny0, 4), keep=keep)


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Even-odd point-in-polygon."""
    inside = False
    n = len(polygon)
    for k in range(n):
        (x1, y1), (x2, y2) = polygon[k], polygon[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def apply_keepouts(mask: Mask, keepouts) -> Mask:
    """The mask with every operator keep-out drawn in: a cell whose centre is inside a polygon
    becomes keep-out. Drawn in the mask's own frame, before it is turned, so a keep-out moves
    with the room."""
    keep = list(mask.keep)
    for k in keepouts:
        if len(k.polygon) < 3:
            raise ValueError(f"keep-out {k.name!r}: a polygon needs at least three points")
        for iy in range(mask.height):
            y = mask.oy + (iy + 0.5) * mask.res
            for ix in range(mask.width):
                if point_in_polygon(mask.ox + (ix + 0.5) * mask.res, y, k.polygon):
                    keep[iy * mask.width + ix] = True
    return replace(mask, keep=keep)


def keep_out_area(mask: Mask, square, exit_from: tuple[float, float],
                  exit_to: tuple[float, float], clear_m: float,
                  trail: Sequence[tuple[float, float]] = ()) -> Mask:
    """The mask with an area kept out while the robot stands in it: every cell whose centre is
    inside ``square`` becomes keep-out, except within ``clear_m`` of the line from where the
    robot stands (``exit_from``) to ``exit_to`` (the mark), and of ``trail``, the way the
    robot drove in (``trail_into``). A whole area made lethal around the robot would leave the
    planner no start. The straight line alone is not the way it came in: an obstacle on it
    inside the area leaves a strip too narrow to drive beside it, and the controller finds no
    feasible trajectory (a stand-in retreat aborted so, with the robot exactly where the line
    began). Cells already keep-out stay so."""
    legs = [(exit_from, exit_to), *itertools.pairwise(trail)]
    keep = list(mask.keep)
    for iy in range(mask.height):
        y = mask.oy + (iy + 0.5) * mask.res
        for ix in range(mask.width):
            x = mask.ox + (ix + 0.5) * mask.res
            if (point_in_polygon(x, y, square)
                    and all(distance_to_segment(x, y, a, b) >= clear_m for a, b in legs)):
                keep[iy * mask.width + ix] = True
    return replace(mask, keep=keep)


def trail_into(trail: Sequence[tuple[float, float]], square) -> list[tuple[float, float]]:
    """The way the robot last came into ``square``: the trail from its last point outside the
    square to its end. All of it when it never was outside; none when it ends outside."""
    if not trail or not point_in_polygon(*trail[-1], square):
        return []
    i = len(trail) - 1
    while i > 0 and point_in_polygon(*trail[i - 1], square):
        i -= 1
    return list(trail[max(i - 1, 0):])


def allowed_area_m2(mask: Mask) -> float:
    return sum(1 for k in mask.keep if not k) * mask.res * mask.res


def deep_in_mask(mask: Mask, x: float, y: float) -> bool:
    """A point in a keep-out cell whose eight neighbours are keep-out too. A path the planner
    drew beside the edge of a keep-out can fall in an edge cell of a mask turned and sampled
    onto the costmap's grid; one a cell further in cannot. Beyond the mask is outside the room,
    as ``rotate_mask`` reads it: keep-out."""
    ix, iy = _cell(x, y, mask.ox, mask.oy, mask.res)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            if (0 <= cx < mask.width and 0 <= cy < mask.height
                    and not mask.keep[cy * mask.width + cx]):
                return False
    return True


def grade_keepouts(costmap: Grid, keepouts, served: Grid | None = None
                   ) -> tuple[bool, list[str]]:
    """Is each operator keep-out IN force? (in_force, one line per keep-out)

    ``keepouts`` are in the costmap's frame. An interior cell has its centre at least two cells
    inside the edge: the polygon is drawn at cell centres in the bound's frame and the mask is
    then turned and sampled again, each step moving the edge by up to half a cell's diagonal, so
    a cell one cell in can be free in the very mask the filter applied (a stand-in segment was
    refused so). The band inside the edge is not judged, but its free cells are counted and
    said, so a shortfall larger than sampling explains is in the line.

    A fresh segment's costmap covers only what SLAM has mapped from the mark, and a keep-out
    can reach where a fresh map never does (the hearth's far corner lies behind a wall: on the
    robot the costmap reached x 2.12 m and the keep-out 2.67 m). So with ``served`` -- the mask
    as the keep-out server publishes it, keep-out 100 -- a keep-out is in force when:

    1. the served mask holds the whole polygon, every interior cell keep-out;
    2. every interior cell inside the costmap's current extent is LETHAL, and there is one;
    3. the interior cells beyond that extent are counted, with how far past it the polygon
       reaches, for the line: the job watches them as the costmap grows.

    Without ``served`` nothing proves the part beyond the costmap, so the whole polygon must
    lie inside it. The bound's own grading counts only the keep-out cells the costmap happens
    to cover, so a keep-out beyond it would pass unseen."""
    ok, lines, res = True, [], costmap.res
    x0, y0 = costmap.ox, costmap.oy
    x1, y1 = x0 + costmap.width * res, y0 + costmap.height * res
    for k in keepouts:
        xs, ys = [p[0] for p in k.polygon], [p[1] for p in k.polygon]
        whole_inside = not (min(xs) < x0 or max(xs) > x1 or min(ys) < y0 or max(ys) > y1)
        head = f"keep-out {k.name!r}: "
        if served is not None:
            m_ok, m_text, beyond = _served_holds(served, k, (x0, y0, x1, y1))
            if not m_ok:
                lines.append(head + m_text)
                ok = False
                continue
            head += m_text + "; "
        elif not whole_inside:
            lines.append(f"keep-out {k.name!r} is not inside the costmap's extent "
                         f"(x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f} m)")
            ok = False
            continue
        n, leak, edge, edge_free = keepout_cells(costmap, k)
        line = (f"{head}{n} cells in the costmap; not lethal: {leak}; edge cells free: "
                f"{edge_free} of {edge}")
        if served is not None:
            reach = max(math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
                        for x, y in k.polygon)
            line += (f"; beyond the costmap's extent: {beyond} cells, the polygon to "
                     f"{reach:.2f} m past it (watched as the costmap grows)" if beyond else
                     "; none beyond the costmap's extent")
        lines.append(line)
        if n == 0 or leak:
            ok = False
    return ok, lines


def keepout_cells(costmap: Grid, k: Keepout) -> tuple[int, int, int, int]:
    """(interior cells in the costmap, of them not lethal, edge cells, of them free)."""
    res, ys = costmap.res, [p[1] for p in k.polygon]
    n = leak = edge = edge_free = 0
    for iy in range(costmap.height):
        y = costmap.oy + (iy + 0.5) * res
        if not min(ys) <= y <= max(ys):
            continue
        for ix in range(costmap.width):
            x = costmap.ox + (ix + 0.5) * res
            depth = depth_in_polygon(x, y, k.polygon)
            lethal = costmap.data[iy * costmap.width + ix] == 100
            if depth >= 2 * res:
                n += 1
                leak += not lethal
            elif depth > 0:
                edge += 1
                edge_free += not lethal
    return n, leak, edge, edge_free


def _served_holds(served: Grid, k: Keepout, extent) -> tuple[bool, str, int]:
    """Does the served mask hold the whole of ``k``? (held, its words, interior cells beyond
    ``extent`` (x0, y0, x1, y1))."""
    res = served.res
    sx0, sy0 = served.ox, served.oy
    sx1, sy1 = sx0 + served.width * res, sy0 + served.height * res
    xs, ys = [p[0] for p in k.polygon], [p[1] for p in k.polygon]
    if min(xs) < sx0 or max(xs) > sx1 or min(ys) < sy0 or max(ys) > sy1:
        where = f"x {sx0:.2f}..{sx1:.2f}, y {sy0:.2f}..{sy1:.2f} m"
        return False, f"not inside the served mask's extent ({where})", 0
    x0, y0, x1, y1 = extent
    n = free = beyond = 0
    for iy in range(served.height):
        y = sy0 + (iy + 0.5) * res
        if not min(ys) <= y <= max(ys):
            continue
        for ix in range(served.width):
            x = sx0 + (ix + 0.5) * res
            if depth_in_polygon(x, y, k.polygon) < 2 * res:
                continue
            n += 1
            free += served.data[iy * served.width + ix] != 100
            beyond += not (x0 <= x <= x1 and y0 <= y <= y1)
    if n == 0 or free:
        return False, f"in the served mask {n} cells; not keep-out: {free}", beyond
    return True, f"whole in the served mask ({n} cells)", beyond


def _mask_keep_at(mask: Mask, x: float, y: float) -> bool | None:
    """True keep-out, False allowed, None outside the mask."""
    ix, iy = _cell(x, y, mask.ox, mask.oy, mask.res)
    if 0 <= ix < mask.width and 0 <= iy < mask.height:
        return mask.keep[iy * mask.width + ix]
    return None


def covered_m2(slam: Grid, mask: Mask) -> float:
    """Known cells of the map (free or occupied, not -1) inside the bound's allowed area."""
    n = 0
    for iy in range(slam.height):
        y = slam.oy + (iy + 0.5) * slam.res
        row = iy * slam.width
        for ix in range(slam.width):
            if slam.data[row + ix] != -1 and _mask_keep_at(
                    mask, slam.ox + (ix + 0.5) * slam.res, y) is False:
                n += 1
    return n * slam.res * slam.res


# The obstacle layer clears and marks from live scans out to its raytrace range (3.0 m in the
# robot's overlay), while a still robot's map holds only its first scan. Near the robot the two
# disagree for a reason that is the lidar's, not a filter's, so only beyond this is judged.
LIDAR_REACH_M = 3.2
# Some unknown cells beyond the reach are legitimately known in the costmap (a live scan ahead
# of the map's next update: 4.3 % on the stand-in). The failure this guards loses ~100 %.
UNKNOWN_LOST_LIMIT = 0.25


def grade_bound(costmap: Grid, slam: Grid, mask: Mask, robot: tuple[float, float] = (0.0, 0.0),
                reach_m: float = LIDAR_REACH_M) -> tuple[bool, list[str]]:
    """Is the bound IN the explorer's input? (in_force, the lines that say why)

    1. every keep-out cell of the mask inside the costmap's extent is LETHAL (100);
    2. beyond the lidar's reach, the map's unknown cells inside the allowed area are still
       unknown (-1) in the costmap;
    3. the costmap and the map have the mask's resolution.
    """
    res = mask.res
    lines: list[str] = []
    ok = True
    for name, g in (("costmap", costmap), ("map", slam)):
        if abs(g.res - res) > 1e-6:
            lines.append(f"{name} resolution {g.res} is not the mask's {res}")
            ok = False
    if not ok:
        return False, lines
    n_keep = leak = 0
    for iy in range(costmap.height):
        y = costmap.oy + (iy + 0.5) * res
        for ix in range(costmap.width):
            if _mask_keep_at(mask, costmap.ox + (ix + 0.5) * res, y):
                n_keep += 1
                if costmap.data[iy * costmap.width + ix] != 100:
                    leak += 1
    lines.append(f"keep-out cells in the costmap: {n_keep}; not lethal: {leak}")
    if n_keep == 0 or leak:
        ok = False
    n_unk = lost = near = 0
    for iy in range(slam.height):
        wy = slam.oy + (iy + 0.5) * res
        for ix in range(slam.width):
            if slam.data[iy * slam.width + ix] != -1:
                continue
            wx = slam.ox + (ix + 0.5) * res
            cx, cy = _cell(wx, wy, costmap.ox, costmap.oy, res)
            if not (0 <= cx < costmap.width and 0 <= cy < costmap.height):
                continue
            if _mask_keep_at(mask, wx, wy) is not False:
                continue
            if math.hypot(wx - robot[0], wy - robot[1]) <= reach_m:
                near += 1
                continue
            n_unk += 1
            if costmap.data[cy * costmap.width + cx] != -1:
                lost += 1
    lines.append(f"unknown cells of the map inside the allowed area, beyond {reach_m:g} m of "
                 f"the robot: {n_unk}; not unknown in the costmap: {lost} (within "
                 f"{reach_m:g} m, not judged: {near})")
    if n_unk and lost > UNKNOWN_LOST_LIMIT * n_unk:
        ok = False
    return ok, lines


# ── ALIGNED: is the live map in the saved map's frame? ──────────────────────

# A saved map's occupancy, read as map_saver writes it: p = (255 - pixel) / 255, occupied above
# 0.65, free below 0.196 -- so 205, the unknown pixel, is neither.
_OCC_P, _FREE_P = 0.65, 0.196


def grid_to_pgm(grid: Grid) -> bytes:
    """map_saver's trinary pixels for a grid: 0 occupied, 254 free, 205 unknown."""
    w, h = grid.width, grid.height

    def pix(v: int) -> int:
        if v < 0:
            return 205
        return 0 if v >= 65 else (254 if v <= 25 else 205)
    rows = (bytes(pix(grid.data[iy * w + ix]) for ix in range(w)) for iy in range(h - 1, -1, -1))
    return b"P5\n%d %d\n255\n" % (w, h) + b"".join(rows)


def grid_from_pgm(data: bytes, *, res: float, ox: float, oy: float) -> Grid:
    """A saved map's PGM -> Grid: 100 occupied, 0 free, -1 unknown."""
    m = mask_from_pgm_pixels(data)
    w, h, pix = m
    out = []
    for iy in range(h):
        row = (h - 1 - iy) * w
        for ix in range(w):
            p = (255 - pix[row + ix]) / 255.0
            out.append(100 if p > _OCC_P else (0 if p < _FREE_P else -1))
    return Grid(width=w, height=h, res=res, ox=ox, oy=oy, data=out)


def align(live: Grid, ref: Grid, *, delta_deg: float = 0.0, max_new_frac: float | None = None,
          tol_m: float = 0.15, min_cells: int = 80) -> tuple[bool | None, str]:
    """Is the live map in the saved map's frame? (aligned, the line that says why)

    The share of the live map's occupied cells within ``tol_m`` of an occupied cell of the saved
    map, after turning them by -``delta_deg`` (the boot's delta about the mark). Judged only
    where the saved map knew the floor (seen free, or near a saved wall) when it knew any: a
    wall in space it never saw is new territory, not a disagreement. None -- NOT PROVEN, never
    "misaligned" -- when the live map has fewer than ``min_cells`` wall cells to judge by.

    With ``max_new_frac`` (a resume): fail when more than that share of the live walls lie on
    the saved map's FREE space, away from its walls -- a resumed graph laid down in the wrong
    place draws its walls there."""
    k = math.ceil(tol_m / ref.res)
    occ = {(i % ref.width, i // ref.width) for i, v in enumerate(ref.data) if v >= 65}
    free = {(i % ref.width, i // ref.width) for i, v in enumerate(ref.data) if 0 <= v < 65}
    near = set()
    for cx, cy in occ:
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy <= k * k:
                    near.add((cx + dx, cy + dy))
    walls = [(i % live.width, i // live.width) for i, v in enumerate(live.data) if v == 100]
    if len(walls) < min_cells:
        return None, (f"alignment: only {len(walls)} occupied cells in the map (want >= "
                      f"{min_cells}) -- too little wall in view to judge")
    c, s = math.cos(math.radians(delta_deg)), math.sin(math.radians(delta_deg))
    judge_all = not free
    hit = judged = on_free = 0
    for ix, iy in walls:
        wx, wy = live.ox + (ix + 0.5) * live.res, live.oy + (iy + 0.5) * live.res
        if delta_deg:
            wx, wy = c * wx + s * wy, -s * wx + c * wy
        cell = _cell(wx, wy, ref.ox, ref.oy, ref.res)
        is_hit = cell in near
        if judge_all or is_hit or cell in free:
            judged += 1
            hit += is_hit
        if cell in free and not is_hit:
            on_free += 1
    frac = hit / judged if judged else 0.0
    ok = frac >= 0.80
    line = (f"alignment: {hit}/{judged} occupied cells of the map within {tol_m:g} m of the saved "
            f"map's walls = {frac:.0%} (want >= 80%)")
    if not judge_all:
        new_frac = on_free / len(walls)
        line += (f"; walls on the saved map's free space: {on_free} ({new_frac:.1%}); walls in "
                 f"space it never saw (new territory, not judged): {len(walls) - judged}")
        if max_new_frac is not None and new_frac > max_new_frac:
            ok = False
            line += f" > {max_new_frac:.0%} -- DUPLICATE WALLS"
    return ok, line


# ── the supervisor ──────────────────────────────────────────────────────────

EXPLORING, STOPPING, RETURNING, AT_MARK, RETURN_FAILED, HARD_STOP = (
    "EXPLORING", "STOPPING", "RETURNING", "AT_MARK", "RETURN_FAILED", "HARD_STOP")


class Window:
    """Battery samples over a time window. ``sustained_at_or_below(v)`` is true only when the
    window is full (its oldest sample is at least ``span`` old) and has >= ``min_n`` samples,
    all <= v. One sagged sample under motor load is not a threshold."""

    def __init__(self, span_s: float, min_n: int = 5) -> None:
        self.span, self.min_n = span_s, min_n
        self.samples: collections.deque = collections.deque()

    def add(self, t: float, mv: float) -> None:
        self.samples.append((t, mv))
        while self.samples and t - self.samples[0][0] > self.span:
            self.samples.popleft()

    def sustained_at_or_below(self, mv: float, now: float) -> bool:
        s = self.samples
        if len(s) < self.min_n or now - s[0][0] < self.span * 0.9:
            return False
        return all(v <= mv for _, v in s)

    def last(self) -> float | None:
        return self.samples[-1][1] if self.samples else None


class Supervisor:
    """The end of one exploration segment, as a pure state machine. ``step()`` returns the
    actions for the caller to perform: ('stop_explore',), ('send_mark',), ('cancel_nav',),
    ('say', text).

    EXPLORING ends on a sustained battery threshold, on the explorer's own completion, on the
    time box, or when the caller ends it (an exhausted area); the explorer is stopped (unless it
    already stopped itself), and the return goal is sent only once the explorer has said it paused and SETTLE_S has passed, or STOP_WAIT_S has.
    A sustained hard-stop reading ends everything, from any state."""

    STOP_WAIT_S = 5.0
    SETTLE_S = 2.0

    def __init__(self, return_mv: float, hard_mv: float, sustain_s: float = 30.0,
                 min_n: int = 5, time_box_s: float | None = None) -> None:
        if not hard_mv < return_mv:
            raise ValueError(f"return_mv {return_mv} must be above hard_mv {hard_mv}")
        self.return_mv, self.hard_mv = return_mv, hard_mv
        self.time_box_s = time_box_s
        self.win = Window(sustain_s, min_n)
        self.state = EXPLORING
        self.reason = ""
        self.started_t: float | None = None
        self.stop_t: float | None = None
        self.paused_t: float | None = None
        self._ended = ""

    def end_exploring(self, reason: str) -> None:
        """End EXPLORING at the next step, as the battery threshold would. Ignored once
        exploring has ended."""
        if self.state == EXPLORING and not self._ended:
            self._ended = reason

    def battery(self, t: float, mv: float) -> None:
        self.win.add(t, mv)

    def step(self, now: float, explore_status: str | None = None,
             nav_result: str | None = None) -> list[tuple]:
        if self.started_t is None:
            self.started_t = now
        acts: list[tuple] = []
        if self.state != HARD_STOP and self.win.sustained_at_or_below(self.hard_mv, now):
            self.state, self.reason = HARD_STOP, f"battery sustained at or below {self.hard_mv:g} mV"
            return [("stop_explore",), ("cancel_nav",), ("say", f"hard stop: {self.reason}")]
        if self.state == EXPLORING:
            if self.win.sustained_at_or_below(self.return_mv, now):
                self.reason = f"battery sustained at or below {self.return_mv:g} mV"
            elif explore_status == "exploration_complete":
                self.reason = "explorer: exploration_complete"
            elif self.time_box_s is not None and now - self.started_t >= self.time_box_s:
                self.reason = f"time box of {self.time_box_s:g} s reached"
            elif self._ended:
                self.reason = self._ended
            if self.reason:
                self.state, self.stop_t = STOPPING, now
                # An explorer that reported completion has already stopped, and cancelled its
                # goals; stopping it again makes it cancel every goal again -- ours included.
                if explore_status != "exploration_complete":
                    acts.append(("stop_explore",))
                else:
                    self.paused_t = now
                acts.append(("say", f"returning: {self.reason}"))
        elif self.state == STOPPING:
            # The explorer publishes "paused" BEFORE it cancels its goals, so the status alone is
            # not enough: SETTLE_S more after it is first seen.
            if (explore_status in ("exploration_paused", "exploration_complete")
                    and self.paused_t is None):
                self.paused_t = now
            settled = self.paused_t is not None and now - self.paused_t >= self.SETTLE_S
            if settled or now - self.stop_t >= self.STOP_WAIT_S:
                self.state = RETURNING
                said = f"returning to the mark (explorer: {explore_status or 'no status'})"
                acts += [("send_mark",), ("say", said)]
        elif self.state == RETURNING and nav_result is not None:
            if nav_result == "SUCCEEDED":
                self.state = AT_MARK
                acts.append(("say", "at the mark"))
            else:
                self.state = RETURN_FAILED
                acts.append(("say", f"return to the mark failed ({nav_result})"))
        return acts


# ── exhausted areas ─────────────────────────────────────────────────────────

# The side of the square an area is counted in, in the map frame.
AREA_SIZE_M = 1.0

# No area within this distance of the start mark is ever exhausted. The robot stands there at the
# start of every segment, and a resumed segment that keeps its own start out cannot begin.
MARK_CLEAR_M = 0.35

# One recovery EPISODE is what Nav2 does about one stop: a controller abort, then clear the local
# costmap, clear the global one, spin or wait -- `number_of_recoveries` rises by one for each, by
# 3 inside a second on a real robot, and the explorer may send its next goal (a new goal id, its
# count from 0) while the robot is still standing there. An episode opens at the first rise
# and every rise after it belongs to it until the robot has moved EPISODE_PROGRESS_M from where
# it opened, or EPISODE_QUIET_S pass with no rise. 0.25 m is the progress watchdog's own minimum
# progress; 15 s outlasts a spin or wait (5 s on the robot), so one episode is never cut in two.
EPISODE_PROGRESS_M = 0.25
EPISODE_QUIET_S = 15.0


class RecoveryEpisodes:
    """Nav2's recovery count, read from the NavigateToPose feedback stream, as episodes."""

    def __init__(self, progress_m: float = EPISODE_PROGRESS_M,
                 quiet_s: float = EPISODE_QUIET_S) -> None:
        self.progress_m, self.quiet_s = progress_m, quiet_s
        self._seen: dict[tuple, int] = {}
        self._open_at: tuple[float, float] | None = None
        self._last_rise = 0.0

    def feed(self, t: float, goal: tuple, n: int, x: float, y: float) -> tuple[float, float] | None:
        """One feedback message, read at ``t``: where a new episode began, or None."""
        if self._open_at is not None and (
                math.hypot(x - self._open_at[0], y - self._open_at[1]) >= self.progress_m
                or t - self._last_rise >= self.quiet_s):
            self._open_at = None
        rose = n > self._seen.get(goal, 0)
        self._seen[goal] = max(n, self._seen.get(goal, 0))
        if not rose:
            return None
        self._last_rise = t
        if self._open_at is not None:
            return None
        self._open_at = (x, y)
        return x, y


def distance_to_polygon(x: float, y: float, polygon) -> float:
    """0 inside the polygon; else the distance to its nearest edge."""
    if point_in_polygon(x, y, polygon):
        return 0.0
    n = len(polygon)
    return min(distance_to_segment(x, y, polygon[k], polygon[(k + 1) % n]) for k in range(n))


def distance_to_segment(x: float, y: float, a: tuple[float, float],
                        b: tuple[float, float]) -> float:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    s = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy)
                                               / (dx * dx + dy * dy)))
    return math.hypot(x - (x1 + s * dx), y - (y1 + s * dy))


def depth_in_polygon(x: float, y: float, polygon) -> float:
    """0 outside the polygon; inside, the distance to its nearest edge."""
    if not point_in_polygon(x, y, polygon):
        return 0.0
    n = len(polygon)
    return min(distance_to_segment(x, y, polygon[k], polygon[(k + 1) % n]) for k in range(n))


class AreaCounter:
    """Recovery episodes per area. The ``limit``-th in one area exhausts it: the name is
    returned once, and the area is listed in ``exhausted``. An area within MARK_CLEAR_M of
    ``mark`` is never exhausted; it is listed once in ``spared``, by name."""

    def __init__(self, limit: int, size_m: float = AREA_SIZE_M,
                 mark: tuple[float, float] | None = None) -> None:
        self.limit, self.size, self.mark = limit, size_m, mark
        self.counts: dict[tuple[int, int], int] = {}
        self.exhausted: list[str] = []
        self.spared: list[str] = []

    def name(self, key: tuple[int, int]) -> str:
        x0, y0 = key[0] * self.size, key[1] * self.size
        return (f"area x {x0:.1f}..{x0 + self.size:.1f} m, "
                f"y {y0:.1f}..{y0 + self.size:.1f} m")

    def key(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.size), math.floor(y / self.size)

    def exhausted_polygons(self) -> list[tuple[str, tuple[tuple[float, float], ...]]]:
        """(name, square) for every exhausted area, in the frame the recoveries were in."""
        out = [(self.name(k), self.square(k)) for k, n in self.counts.items() if n >= self.limit]
        return sorted(out, key=lambda a: self.exhausted.index(a[0]))

    def square(self, key: tuple[int, int]) -> tuple[tuple[float, float], ...]:
        x0, y0 = key[0] * self.size, key[1] * self.size
        return (x0, y0), (x0 + self.size, y0), (x0 + self.size, y0 + self.size), (x0, y0 + self.size)

    def at_mark(self, key: tuple[int, int]) -> bool:
        return (self.mark is not None
                and distance_to_polygon(*self.mark, self.square(key)) < MARK_CLEAR_M)

    def add(self, x: float, y: float) -> str | None:
        k = self.key(x, y)
        if self.at_mark(k):
            if self.name(k) not in self.spared:
                self.spared.append(self.name(k))
            return None
        self.counts[k] = self.counts.get(k, 0) + 1
        if self.counts[k] == self.limit:
            self.exhausted.append(self.name(k))
            return self.name(k)
        return None
