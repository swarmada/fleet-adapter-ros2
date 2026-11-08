#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Generate the occupancy map Nav2 needs at R2.

Nav2's map_server will not start without a map, and the static costmap layer is
what makes goal rejection and recovery behaviours real rather than theoretical.
This writes a small warehouse: an open floor bounded by walls, with two racking
blocks in the middle so a planner has something to plan around and a costmap has
something to inflate.

Output (PGM P5 + the YAML map_server expects):

    map/warehouse.pgm
    map/warehouse.yaml

Coordinate frame: origin at the bottom-left of the image, which is where the
YAML's `origin` places it in the map frame. The robot starts at (0, 0) in the
middle of the open floor, so `origin` is negative on both axes.
"""

from __future__ import annotations

import argparse
import os

RESOLUTION = 0.05          # m per pixel
WIDTH_M, HEIGHT_M = 20.0, 20.0
WALL_M = 0.30              # wall thickness

FREE, OCCUPIED = 254, 0    # PGM greyscale: 254 free, 0 occupied (map_server convention)


def _rect(grid, w, x0, y0, x1, y1, value):
    """Fill an axis-aligned rectangle, in pixel coordinates."""
    for y in range(max(0, y0), min(len(grid) // w, y1)):
        for x in range(max(0, x0), min(w, x1)):
            grid[y * w + x] = value


def build(width_px: int, height_px: int) -> bytearray:
    grid = bytearray([FREE]) * (width_px * height_px)
    wall = max(1, int(WALL_M / RESOLUTION))

    # Perimeter walls.
    _rect(grid, width_px, 0, 0, width_px, wall, OCCUPIED)                     # bottom
    _rect(grid, width_px, 0, height_px - wall, width_px, height_px, OCCUPIED)  # top
    _rect(grid, width_px, 0, 0, wall, height_px, OCCUPIED)                    # left
    _rect(grid, width_px, width_px - wall, 0, width_px, height_px, OCCUPIED)  # right

    # Two racking blocks, leaving a central aisle. In map metres these sit at
    # roughly x in [-6,-2] and [2,6], y in [-4,4] relative to the (0,0) start.
    rack_w = int(4.0 / RESOLUTION)
    rack_h = int(8.0 / RESOLUTION)
    cy = height_px // 2 - rack_h // 2
    _rect(grid, width_px, int(4.0 / RESOLUTION), cy,
          int(4.0 / RESOLUTION) + rack_w, cy + rack_h, OCCUPIED)
    _rect(grid, width_px, int(12.0 / RESOLUTION), cy,
          int(12.0 / RESOLUTION) + rack_w, cy + rack_h, OCCUPIED)
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="map", help="output directory")
    ap.add_argument("--name", default="warehouse")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    w = int(WIDTH_M / RESOLUTION)
    h = int(HEIGHT_M / RESOLUTION)
    grid = build(w, h)

    pgm = os.path.join(args.out, f"{args.name}.pgm")
    with open(pgm, "wb") as fh:
        # P5 is binary greyscale. Rows are written top-to-bottom, but map_server
        # reads the image with the origin at the BOTTOM-left, so flip on write.
        fh.write(f"P5\n{w} {h}\n255\n".encode())
        for y in range(h - 1, -1, -1):
            fh.write(bytes(grid[y * w:(y + 1) * w]))

    yaml_path = os.path.join(args.out, f"{args.name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"image: {args.name}.pgm\n"
            f"mode: trinary\n"
            f"resolution: {RESOLUTION}\n"
            # Place map (0,0) in the middle of the open floor, where the robot starts.
            f"origin: [{-WIDTH_M / 2:.3f}, {-HEIGHT_M / 2:.3f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.25\n"
        )

    print(f"wrote {pgm} ({w}x{h} px, {WIDTH_M}x{HEIGHT_M} m @ {RESOLUTION} m/px)")
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
