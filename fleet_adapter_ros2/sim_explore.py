# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A simulated exploration stack: the fake SLAM, keep-out servers, costmap, explorer, return
and battery that `SimulatedRobot` runs explore-floor and Charge against, and that the tests
configure to fail the ways a robot did.

It moves nothing and runs no SLAM. Its room is a synthetic 8 m x 6 m bound whose border is
wall; the map it "builds" is that room in this boot's frame (turned by the boot's yaw on the
mark less the bound's theta0), known within a radius of the mark that grows with explore time.
Its costmap marks lethal what the keep-out mask the job loaded says, so the job's ALIGNED and
bound checks grade something real. Every knob is a failure seen on the robot or needed to reach a branch:

* ``drop``: a call or read answers None that many times -- the dropped service reply;
* ``keepout_leak``: the keep-out filter is loaded but not lethal; fixed by the costmap reset
  (``keepout_leak_fixed_by="reset"``, the filter that writes only in the update window) or by
  nothing (None);
* ``params``: the live parameters -- the speed caps, and the keep-out filter on both planning
  costmaps (an unlisted parameter reads "not set"; None is no answer); ``boot_yaw_deg``: this boot's yaw on the mark;
  ``misaligned_m``: the robot booted that far off the mark (ALIGNED fails);
  ``bound_has_ref_map``: the bound carries the saved map it was drawn on;
* ``explore_seconds``, ``recoveries``, ``mark_result``: the explorer and the return; the robot
  stands where its latest recovery was until it is back at the mark;
* ``hand_keepouts``: keep-outs the bound declares were measured by hand (the bound's frame);
  ``costmap_misses``: polygons in this boot's frame the costmap never makes lethal, whatever
  the mask says (a filter serving a mask without them); ``plans``: (explore seconds, points)
  -- the global paths the planner publishes as exploring goes on; ``reload_fails``: a mask
  reloaded while the servers run is written, and the filter goes on serving the old one;
  ``costmap_extents``: (explore seconds, (x0, y0, x1, y1)) -- the global costmap covers only
  that box of this boot's frame from then on (a fresh map covers what the lidar has seen, and
  grows); ``mask_misses``: polygons in this boot's frame the mask server's topic does not keep
  out, whatever the file says; ``local_plans``: (explore seconds, points in ``odom``) -- the
  controller's local trajectories, placed in the map with ``map_to_odom`` (x, y, yaw degrees;
  None: no transform, so none can be placed);
* ``tour``: waypoints (x, y) in this boot's frame the robot is shown driving through while it
  explores, from the mark and back to it, round and round, at ``tour_speed_mps``; the map is
  then known within ``tour_scan_m`` of where it has been (and within the first scan of the
  mark), and the return draws it back to the mark. The default, no tour, keeps the robot on the
  mark and grows the known radius instead. Nothing in the job reads it: a map viewer does;
* the battery (``start_mv``, ``drain_mv_per_s``, ``battery_reading``, ``battery_stale_after_s``:
  the reading stops arriving then) and a person who plugs
  the cable in when a Charge waits and pulls it out ``unplug_after_full_s`` after full.

Its battery thresholds are synthetic, chosen for the simulation, and belong to no robot.
"""

from __future__ import annotations

import json
import math
import threading

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2 import explore_steps as steps

SIM_BOUND = "sim-room"
SIM_THETA0_DEG = 12.5
SIM_LIMITS = explore.BatteryLimits(hard_mv=7000.0, full_mv=8400.0, return_mv=7200.0)

_RES = 0.1
_W, _H = 80, 60                       # 8 m x 6 m
_OX, _OY = -2.0, -1.5                 # the mark is at (0, 0), inside
_FIRST_SCAN_M = 3.5                   # what SLAM knows before anything moves (the lidar's view)
# Both planning costmaps carry the keep-out filter, as the robot's overlay gives them.
_COSTMAPS = ("/global_costmap/global_costmap", "/local_costmap/local_costmap")
_KEEPOUT_FILTER = {"filters": "[keepout_filter]",
                   "keepout_filter.plugin": "nav2_costmap_2d::KeepoutFilter",
                   "keepout_filter.enabled": "True",
                   "keepout_filter.filter_info_topic": "/costmap_filter_info"}


def _sim_mask() -> steps.Mask:
    keep = [x in (0, _W - 1) or y in (0, _H - 1) for y in range(_H) for x in range(_W)]
    return steps.Mask(width=_W, height=_H, res=_RES, ox=_OX, oy=_OY, keep=keep)


def _sim_ref_map() -> steps.Grid:
    """The saved map the bound was drawn on: the border is wall, the room seen free."""
    m = _sim_mask()
    return steps.Grid(width=_W, height=_H, res=_RES, ox=_OX, oy=_OY,
                      data=[100 if k else 0 for k in m.keep])


class SimulatedExploreStack:
    """See the module docstring. ``log`` records every call by name, in order."""

    def __init__(self, clock=None, *, drop: dict[str, int] | None = None,
                 keepout_leak: bool = False, keepout_leak_fixed_by: str | None = "reset",
                 params: dict[tuple[str, str], str] | None = None, boot_yaw_deg: float = 0.0,
                 explore_seconds: float = 60.0, recoveries=(), mark_result: str = "SUCCEEDED",
                 return_seconds: float = 3.0, start_mv: float = 8200.0,
                 drain_mv_per_s: float = 0.5, battery_reading: bool = True,
                 battery_stale_after_s: float | None = None,
                 charge_mv_per_s: float = 50.0, unplug_after_full_s: float | None = 10.0,
                 limits: explore.BatteryLimits = SIM_LIMITS, misaligned_m: float = 0.0,
                 bound_has_ref_map: bool = True, hand_keepouts=(), costmap_misses=(),
                 plans=(), reload_fails: bool = False, costmap_extents=(),
                 mask_misses=(), local_plans=(), map_to_odom=(0.0, 0.0, 0.0), tour=(),
                 tour_speed_mps: float = 0.18, tour_scan_m: float = 2.5) -> None:
        self.clock = clock or explore.RealClock()
        self.drop = dict(drop or {})
        self.keepout_leak, self.keepout_leak_fixed_by = keepout_leak, keepout_leak_fixed_by
        self.params = {("/controller_server", "FollowPath.max_vel_x"): "0.2",
                       ("/velocity_smoother", "max_velocity"): "[0.2, 0.0, 1.0]",
                       **{(node, name): value for node in _COSTMAPS
                          for name, value in _KEEPOUT_FILTER.items()},
                       **(params or {})}
        self.param_reads: list[tuple[str, str]] = []
        self.explore_seconds = explore_seconds
        self._recoveries = sorted(recoveries)
        self._mark_result, self.return_seconds = mark_result, return_seconds
        self.drain, self.battery_reading = drain_mv_per_s, battery_reading
        self.battery_stale_after_s = battery_stale_after_s
        self.charge_rate, self.unplug_after_full_s = charge_mv_per_s, unplug_after_full_s
        self.limits = limits
        self.misaligned_m, self.bound_has_ref_map = misaligned_m, bound_has_ref_map
        self.hand_keepouts, self.costmap_misses = list(hand_keepouts), list(costmap_misses)
        self._plans = sorted(plans, key=lambda p: p[0])
        self.reload_fails = reload_fails
        self._extents = sorted(costmap_extents, key=lambda e: e[0])
        self.mask_misses = list(mask_misses)
        self._local_plans = sorted(local_plans, key=lambda p: p[0])
        self.map_to_odom = map_to_odom
        self._tour = [(0.0, 0.0), *((float(x), float(y)) for x, y in tour), (0.0, 0.0)] if tour \
            else []
        self.tour_speed, self.tour_scan_m = tour_speed_mps, tour_scan_m
        self._room = _sim_mask()
        self.bound_theta0_deg = SIM_THETA0_DEG
        self.log: list[str] = []
        self.skipped: list[str] = []
        self.on_explorer_started = None
        self.on_tick = None
        self._lock = threading.Lock()
        self._graphs: dict[str, dict] = {}
        self._mv, self._mv_t = float(start_mv), self.clock.now()
        self._plugged, self._full_t = False, None
        self._halted: threading.Event | None = None
        self.reboot(boot_yaw_deg)

    def reboot(self, boot_yaw_deg: float = 0.0) -> None:
        """A power cycle: SLAM, the keep-out and the explorer are gone; the robot is on the
        mark with a new boot yaw. The battery and the saved graphs remain."""
        self.boot_yaw_deg = boot_yaw_deg
        self._slam = False
        self._yaw = boot_yaw_deg
        self._radius = _FIRST_SCAN_M
        self._explore_t = 0.0
        self._exploring = self._returning = False
        self._explore_since: float | None = None
        self._status = ""
        self._mask: steps.Mask | None = None
        self._served: steps.Mask | None = None
        self._keepout_up = self._leaking = False
        self._pos = (0.0, 0.0)
        self._next_plan = self._next_local = 0
        self._return_since: float | None = None
        self._next_recovery = 0
        self.loaded_mask_delta_deg: float | None = None
        self.slam_started_with: str | None = None
        self.mark_sent_at = self.explorer_stopped_at = None
        self._home = False                  # back on the mark after a return (tour only)
        self._seen: tuple | None = None     # (frame key, seen cells, explore time seen up to)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _answer(self, name: str, value):
        self.log.append(name)
        if self.drop.get(name, 0) > 0:
            self.drop[name] -= 1
            return None
        return value

    def _advance(self) -> None:
        """Bring the explorer's clock and the battery up to now."""
        now = self.clock.now()
        with self._lock:
            if self._exploring and self._explore_since is not None:
                self._explore_t += now - self._explore_since
                self._explore_since = now
                if self._explore_t >= self.explore_seconds:
                    self._exploring, self._explore_since = False, None
                    self._status = "exploration_complete"
            dt = now - self._mv_t
            self._mv_t = now
            if self._plugged:
                self._mv = min(self.limits.full_mv + 100.0, self._mv + self.charge_rate * dt)
                if self._mv >= self.limits.full_mv and self._full_t is None:
                    self._full_t = now
                if (self._full_t is not None and self.unplug_after_full_s is not None
                        and now - self._full_t >= self.unplug_after_full_s):
                    self._plugged = False
            elif self._exploring or self._returning:
                self._mv -= self.drain * dt
            r_max = math.hypot(max(abs(_OX), _OX + _W * _RES), max(abs(_OY), _OY + _H * _RES))
            self._radius = max(self._radius, _FIRST_SCAN_M + (r_max - _FIRST_SCAN_M)
                               * min(1.0, self._explore_t / self.explore_seconds))

    def set_battery(self, mv: float) -> None:
        self._mv, self._mv_t = float(mv), self.clock.now()

    def plug(self) -> None:
        """The person plugs the cable in (the IMU goes silent)."""
        with self._lock:
            self._plugged, self._full_t = True, None

    def moving(self) -> bool:
        self._advance()
        return self._exploring or self._returning

    def _grid(self, lethal) -> steps.Grid | None:
        """The live map (``lethal`` None) or the costmap, in this boot's frame: the room turned
        by the boot's delta and shifted by ``misaligned_m``; cells the loaded mask keeps out are
        lethal in the costmap when ``lethal`` is true."""
        if not self._slam:
            return None
        frame = steps.rotate_mask(self._room, self._yaw - self.bound_theta0_deg)
        d = math.radians(self._yaw - self.bound_theta0_deg)
        c, s_ = math.cos(d), math.sin(d)
        seen = self._seen_cells(frame) if self._tour else None
        data = []
        for iy in range(frame.height):
            y = frame.oy + (iy + 0.5) * _RES
            for ix in range(frame.width):
                x = frame.ox + (ix + 0.5) * _RES
                qx, qy = c * x + s_ * y - self.misaligned_m, -s_ * x + c * y
                room = steps._mask_keep_at(self._room, qx, qy)
                known = ((ix, iy) in seen if seen is not None
                         else math.hypot(x, y) <= self._radius)
                if room is None or not known:
                    v = -1
                else:
                    v = 100 if room else 0
                if lethal and self._served is not None and steps._mask_keep_at(
                        self._served, x, y) and not any(
                        steps.point_in_polygon(x, y, k.polygon) for k in self.costmap_misses):
                    v = 100
                data.append(v)
        grid = steps.Grid(width=frame.width, height=frame.height, res=_RES, ox=frame.ox,
                          oy=frame.oy, data=data)
        return self._crop(grid) if lethal is not None else grid

    # ── the tour (a viewer's knob; see the module docstring) ─────────────────

    def _tour_at(self, explore_t: float) -> tuple[float, float, float]:
        """Where the tour has the robot after ``explore_t`` s of exploring, and its heading."""
        legs = list(zip(self._tour, self._tour[1:]))
        total = sum(math.dist(a, b) for a, b in legs)
        if explore_t <= 0 or total <= 0:
            return 0.0, 0.0, self._yaw
        d = (self.tour_speed * explore_t) % total
        for a, b in legs:
            seg = math.dist(a, b)
            if seg > 0 and d <= seg:
                f = d / seg
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]),
                        math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
            d -= seg
        return 0.0, 0.0, self._yaw

    def _tour_pose(self) -> tuple[float, float, float]:
        if self._home:
            return 0.0, 0.0, self._yaw
        x, y, yaw = self._tour_at(self._explore_t)
        if self._returning and self._return_since is not None:
            f = min(1.0, (self.clock.now() - self._return_since) / max(self.return_seconds, 1e-9))
            yaw = math.degrees(math.atan2(-y, -x)) if (x, y) != (0.0, 0.0) else yaw
            x, y = x * (1.0 - f), y * (1.0 - f)
        return x, y, yaw

    def _seen_cells(self, frame: steps.Mask) -> set:
        """The cells of ``frame`` known so far: the first scan round the mark, and every cell
        within ``tour_scan_m`` of a point the tour has passed (sampled every 0.1 m)."""
        key = (frame.ox, frame.oy, frame.width, frame.height)
        if self._seen is None or self._seen[0] != key:
            cells = {(ix, iy) for iy in range(frame.height) for ix in range(frame.width)
                     if math.hypot(frame.ox + (ix + 0.5) * _RES, frame.oy + (iy + 0.5) * _RES)
                     <= _FIRST_SCAN_M}
            self._seen = (key, cells, 0.0)
        _, cells, upto = self._seen
        step = 0.1 / max(self.tour_speed, 1e-9)
        r = self.tour_scan_m
        t = upto
        while t + step <= self._explore_t:
            t += step
            px, py, _ = self._tour_at(t)
            ix0 = max(0, int((px - r - frame.ox) / _RES))
            ix1 = min(frame.width, int((px + r - frame.ox) / _RES) + 1)
            iy0 = max(0, int((py - r - frame.oy) / _RES))
            iy1 = min(frame.height, int((py + r - frame.oy) / _RES) + 1)
            for iy in range(iy0, iy1):
                cy = frame.oy + (iy + 0.5) * _RES
                for ix in range(ix0, ix1):
                    if math.hypot(frame.ox + (ix + 0.5) * _RES - px, cy - py) <= r:
                        cells.add((ix, iy))
        self._seen = (key, cells, t)
        return cells

    def _crop(self, grid: steps.Grid) -> steps.Grid:
        """The costmap cut to the extent it has at this explore time, if one is set."""
        boxes = [box for t, box in self._extents if t <= self._explore_t]
        if not boxes:
            return grid
        x0, y0, x1, y1 = boxes[-1]
        ix0 = max(0, math.ceil((x0 - grid.ox) / grid.res - 1e-9))
        iy0 = max(0, math.ceil((y0 - grid.oy) / grid.res - 1e-9))
        ix1 = min(grid.width, math.floor((x1 - grid.ox) / grid.res + 1e-9))
        iy1 = min(grid.height, math.floor((y1 - grid.oy) / grid.res + 1e-9))
        data = [grid.data[iy * grid.width + ix] for iy in range(iy0, iy1) for ix in range(ix0, ix1)]
        return steps.Grid(width=ix1 - ix0, height=iy1 - iy0, res=grid.res,
                          ox=grid.ox + ix0 * grid.res, oy=grid.oy + iy0 * grid.res, data=data)

    def loaded_mask_keep_at(self, x: float, y: float) -> bool | None:
        return steps._mask_keep_at(self._mask, x, y) if self._mask is not None else None

    # ── ExploreStack ────────────────────────────────────────────────────────

    def battery_limits(self) -> explore.BatteryLimits:
        return self.limits

    def arm(self, halted: threading.Event) -> None:
        self._halted = halted

    def halt(self) -> None:
        self.log.append("halt")
        with self._lock:
            self._stop_motion()
        if self._halted is not None:
            self._halted.set()

    def hold(self) -> None:
        self.log.append("hold")
        with self._lock:
            self._stop_motion()

    def _stop_motion(self) -> None:
        if self._exploring and self._explore_since is not None:
            self._explore_t += self.clock.now() - self._explore_since
        self._exploring, self._explore_since, self._returning = False, None, False
        if self._status != "exploration_complete":
            self._status = "exploration_paused"

    def load_bound(self, bound_ref: str) -> explore.Bound:
        self.log.append("load_bound")
        if bound_ref != SIM_BOUND:
            raise LookupError(f"not a bound the simulated stack knows (it knows {SIM_BOUND!r})")
        return explore.Bound(mask=_sim_mask(), theta0_deg=self.bound_theta0_deg,
                             ref_map=_sim_ref_map() if self.bound_has_ref_map else None,
                             hand_keepouts=list(self.hand_keepouts))

    def read_param(self, node: str, name: str) -> str | None:
        self.param_reads.append((node, name))
        return self._answer("read_param", self.params.get((node, name), "not set"))

    def start_slam(self, graph_ref: str | None) -> bool | None:
        answer = self._answer("start_slam", True)
        if answer is None:
            return None
        if graph_ref is not None:
            saved = self._graphs.get(graph_ref)
            if saved is None:
                return False
            self._radius, self._yaw = saved["radius"], saved["yaw"]
        self._slam, self.slam_started_with = True, graph_ref
        return True

    def robot_pose(self) -> tuple[float, float, float] | None:
        if self._slam and self._tour:
            self._advance()
            return self._answer("robot_pose", self._tour_pose())
        return self._answer("robot_pose", (*self._pos, self._yaw) if self._slam else None)

    def load_keepout_mask(self, mask: steps.Mask, delta_deg: float) -> bool | None:
        answer = self._answer("load_keepout_mask", True)
        if answer:
            self._mask, self.loaded_mask_delta_deg = mask, delta_deg
        return answer

    def reload_keepout_mask(self) -> bool | None:
        answer = self._answer("reload_keepout_mask", True)
        if answer and not self.reload_fails:
            self._served = self._mask
        return answer

    def read_keepout_mask(self) -> steps.Grid | None:
        """The mask the server publishes on its topic: keep-out 100, the rest unknown."""
        m = self._served if self._keepout_up else None
        if m is None:
            return self._answer("read_keepout_mask", None)
        data = []
        for iy in range(m.height):
            y = m.oy + (iy + 0.5) * m.res
            for ix in range(m.width):
                x = m.ox + (ix + 0.5) * m.res
                keep = m.keep[iy * m.width + ix] and not any(
                    steps.point_in_polygon(x, y, p) for p in self.mask_misses)
                data.append(100 if keep else -1)
        return self._answer("read_keepout_mask", steps.Grid(width=m.width, height=m.height,
                                                            res=m.res, ox=m.ox, oy=m.oy,
                                                            data=data))

    def activate_keepout(self) -> bool | None:
        answer = self._answer("activate_keepout", True)
        if answer:
            self._keepout_up, self._leaking = True, self.keepout_leak
            self._served = self._mask
        return answer

    def restart_keepout(self) -> bool | None:
        answer = self._answer("restart_keepout", True)
        if answer:
            self.drop.pop("activate_keepout", None)
            self._keepout_up, self._leaking = True, self.keepout_leak
            self._served = self._mask
        return answer

    def reset_costmap(self) -> bool | None:
        answer = self._answer("reset_costmap", True)
        if answer and self.keepout_leak_fixed_by == "reset":
            self._leaking = False
        return answer

    def read_costmap(self) -> steps.Grid | None:
        self._advance()
        return self._answer("read_costmap",
                            self._grid(self._keepout_up and not self._leaking))

    def read_map(self) -> steps.Grid | None:
        self._advance()
        return self._answer("read_map", self._grid(None))

    def start_explorer(self) -> bool | None:
        answer = self._answer("start_explorer", True)
        if answer:
            self._advance()
            with self._lock:
                if self._explore_t < self.explore_seconds:
                    self._exploring, self._explore_since = True, self.clock.now()
                    self._home = False
                    self._status = "exploration_in_progress"
                else:
                    self._status = "exploration_complete"
            if self.on_explorer_started is not None:
                self.on_explorer_started()
        return answer

    def stop_explorer(self) -> None:
        self.log.append("stop_explorer")
        self.explorer_stopped_at = self.clock.now()
        with self._lock:
            self._stop_motion()

    def explorer_status(self) -> str:
        self._advance()
        return self._status

    def recoveries(self) -> list[tuple[float, float]]:
        self._advance()
        out = []
        while (self._next_recovery < len(self._recoveries)
               and self._recoveries[self._next_recovery][0] <= self._explore_t):
            _, x, y = self._recoveries[self._next_recovery]
            out.append((x, y))
            self._pos = (x, y)
            self._next_recovery += 1
        return out

    def plans(self) -> list[list[tuple[float, float]]]:
        self._advance()
        out = []
        while (self._next_plan < len(self._plans)
               and self._plans[self._next_plan][0] <= self._explore_t):
            out.append([tuple(p) for p in self._plans[self._next_plan][1]])
            self._next_plan += 1
        return out

    def local_plans(self) -> list[list[tuple[float, float]] | None]:
        self._advance()
        out = []
        while (self._next_local < len(self._local_plans)
               and self._local_plans[self._next_local][0] <= self._explore_t):
            points = self._local_plans[self._next_local][1]
            self._next_local += 1
            if self.map_to_odom is None:
                out.append(None)
                continue
            tx, ty, deg = self.map_to_odom
            c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            out.append([(tx + c * x - s_ * y, ty + s_ * x + c * y) for x, y in points])
        return out

    def skip_area(self, name: str) -> None:
        self.log.append("skip_area")
        self.skipped.append(name)

    def go_to_mark(self, x: float, y: float, yaw_deg: float) -> None:
        self.log.append("go_to_mark")
        self.mark_sent_at = self.clock.now()
        with self._lock:
            self._returning, self._return_since = True, self.clock.now()

    def mark_result(self) -> str | None:
        self._advance()
        if self._return_since is None:
            return None
        if self.clock.now() - self._return_since < self.return_seconds:
            return None
        with self._lock:
            self._returning = False
        if self._mark_result == "SUCCEEDED":
            self._pos = (0.0, 0.0)
            self._home = True
        return self._mark_result

    def save_map(self, name: str) -> explore.SavedMap | None:
        grid = self.read_map()
        answer = self._answer("save_map", True)
        if answer is None or grid is None:
            return None
        ref = f"sim-graph:{name}"
        self._graphs[ref] = {"radius": self._radius, "yaw": self._yaw}
        yaml, pgm = explore.render_map(grid)
        graph = (json.dumps({"sim": True, "radius_m": self._radius, "yaw_deg": self._yaw},
                            sort_keys=True) + "\n").encode()
        return explore.SavedMap(files={"map.yaml": yaml, "map.pgm": pgm,
                                       "map.posegraph": graph, "map.data": b"sim\n"},
                                graph_ref=ref)

    def battery_mv(self) -> float | None:
        self._advance()
        if self.on_tick is not None:
            self.on_tick()
        return self._mv if self.battery_reading else None

    def battery_age_s(self) -> float | None:
        if not self.battery_reading:
            return None
        last = self.battery_stale_after_s
        now = self.clock.now()
        return 0.0 if last is None or now <= last else now - last

    def battery_source(self) -> str:
        return "the simulated battery"

    def imu_streaming(self) -> bool | None:
        self._advance()
        return not self._plugged

    def end_segment(self) -> None:
        self.log.append("end_segment")
        with self._lock:
            self._stop_motion()
        self._slam = self._keepout_up = False
        self._served = None

    def charge_wait(self) -> None:
        self.log.append("charge_wait")
        self._advance()
        with self._lock:
            if self._mv < self.limits.full_mv:
                self._plugged, self._full_t = True, None
