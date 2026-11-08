# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A synthetic room and a lidar that ray-casts it from the plant's TRUE pose.

The plant keeps two velocities: the one it was commanded and the one it really moves at. Only the
true pose is ever cast; the command is kept so a test can say what the robot was told, and nothing
here hands it to the channel. A wheel that slips, a base that is pushed, a motor that stalls: the
true velocity differs from the command, and the scan follows the truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from fleet_adapter_ros2.scan_rest import Scan

# The MS200 on /scan_raw: about 10 Hz, 449 to 451 beams over a full turn, 0.05 to 12 m. The range
# noise is the vendor's stated +-10 mm, taken as one standard deviation (no datasheet is in the
# tree; see the report).
RATE_HZ = 10.0
BEAMS = 450
RANGE_MIN, RANGE_MAX = 0.05, 12.0
NOISE_M = 0.010


def box(x0, y0, x1, y1) -> list:
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


# A furnished room around the origin: walls, a cabinet, a table leg block, a chair.
ROOM = (box(-2.2, -1.6, 3.1, 2.3) + box(1.6, 1.5, 2.6, 2.1) + box(-1.9, -1.3, -1.2, -0.9)
        + box(0.9, -1.1, 1.3, -0.7))
# Two parallel walls 2 m apart, open at both ends beyond the cap (nothing there at all).
CORRIDOR = [((-60.0, -1.0), (60.0, -1.0)), ((-60.0, 1.0), (60.0, 1.0))]


def _ray_segment(ox, oy, dx, dy, seg) -> float:
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return math.inf
    t = ((ax - ox) * ey - (ay - oy) * ex) / den
    u = ((ax - ox) * dy - (ay - oy) * dx) / den
    return t if t > 1e-9 and 0.0 <= u <= 1.0 else math.inf


def _ray_circle(ox, oy, dx, dy, cx, cy, r) -> float:
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return math.inf
    t = -b - math.sqrt(disc)
    return t if t > 1e-9 else math.inf


@dataclass
class Plant:
    """The robot as it really is. ``true`` (vx, vy, wz in the robot's frame) moves it; ``cmd`` is
    only what it was told."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    true: tuple = (0.0, 0.0, 0.0)
    cmd: tuple = (0.0, 0.0, 0.0)

    def step(self, dt: float) -> None:
        vx, vy, wz = self.true
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * c - vy * s) * dt
        self.y += (vx * s + vy * c) * dt
        self.yaw += wz * dt


@dataclass
class Person:
    """A walking cylinder: centre at ``start`` + ``vel`` * t."""
    start: tuple
    vel: tuple
    radius: float = 0.2

    def at(self, t: float) -> tuple:
        return self.start[0] + self.vel[0] * t, self.start[1] + self.vel[1] * t


@dataclass
class Lidar:
    walls: list = field(default_factory=lambda: list(ROOM))
    beams: int = BEAMS
    noise_m: float = NOISE_M
    drop: float = 0.0           # share of beams returned as NaN
    seed: int = 1
    angle_min: float = 0.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def cast(self, plant: Plant, stamp: float, people=(), beams: int | None = None,
             angle_min: float | None = None) -> Scan:
        n = beams or self.beams
        a0 = self.angle_min if angle_min is None else angle_min
        inc = 2.0 * math.pi / n
        out = []
        for i in range(n):
            a = plant.yaw + a0 + i * inc
            dx, dy = math.cos(a), math.sin(a)
            r = min((_ray_segment(plant.x, plant.y, dx, dy, s) for s in self.walls),
                    default=math.inf)
            for p in people:
                cx, cy = p.at(stamp)
                r = min(r, _ray_circle(plant.x, plant.y, dx, dy, cx, cy, p.radius))
            if math.isfinite(r):
                r += self._rng.gauss(0.0, self.noise_m)
                if r > RANGE_MAX:
                    r = math.inf
            if self.drop and self._rng.random() < self.drop:
                r = math.nan
            out.append(r)
        return Scan(stamp, out, RANGE_MIN, RANGE_MAX, a0, inc)


def record(plant: Plant, lidar: Lidar, seconds: float, *, t0: float = 0.0, people=(),
           layout=None, rate_hz: float = RATE_HZ) -> list:
    """Scans at ``rate_hz`` for ``seconds``, the plant moving at its TRUE velocity in between.
    ``layout(i)`` may give a scan its own (beams, angle_min)."""
    scans = []
    dt = 1.0 / rate_hz
    for i in range(round(seconds * rate_hz)):
        beams, amin = layout(i) if layout else (None, None)
        scans.append(lidar.cast(plant, t0 + i * dt, people, beams=beams, angle_min=amin))
        plant.step(dt)
    return scans
