# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A keep-out measured by hand, from the start mark, into the bound's map frame.

Something below the lidar's plane -- a raised slab, a low ledge -- is in no map and no costmap,
so an operator measures it from the mark: for each corner, how far AHEAD along the heading line
and how far LEFT of it (right is negative). This turns those corners into the polygon a bound's
``keepouts.json`` holds, in the frame the bound's mask is drawn in, where the heading line points
at the bound's theta0:

    python3 -m fleet_adapter_ros2.hand_keepouts CORNERS --theta0 DEG --margin M \\
        --name NAME --reason TEXT > keepouts.json

CORNERS holds one corner a line, ``C<n> <ahead> <left>`` in metres; ``#`` starts a comment. The
keep-out is the corners' box along the line, grown by the margin on every side. Then name the
file in the bound's ``bound.json``: ``"hand_keepouts": "keepouts.json"``.

The margin is the operator's call and has no default. The costmap's keep-out filter writes its
cells after the inflation layer has run, so nothing inflates them: the planner may draw a path
along the polygon's edge, and only the margin keeps the robot's body off what was measured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def read_corners(text: str) -> list[tuple[float, float]]:
    """(ahead, left) for every ``C<n> <ahead> <left>`` line; ValueError naming a bad line."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            if len(parts) != 3 or not parts[0].startswith("C"):
                raise ValueError
            a, b = float(parts[1]), float(parts[2])
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError
        except ValueError:
            raise ValueError(f"line {n}: want 'C<n> <ahead> <left>' in metres, got "
                             f"{line!r}") from None
        out.append((a, b))
    if len(out) < 3:
        raise ValueError(f"want at least three corners, got {len(out)}")
    return out


def to_bound_frame(corners, theta0_deg: float, margin_m: float) -> list[tuple[float, float]]:
    """The corners' box along the heading line, grown by ``margin_m``, as four points in the
    frame where the heading line from the mark (the origin) points at ``theta0_deg``."""
    if not margin_m >= 0:
        raise ValueError("the margin must be zero or more")
    a0, a1 = min(c[0] for c in corners) - margin_m, max(c[0] for c in corners) + margin_m
    l0, l1 = min(c[1] for c in corners) - margin_m, max(c[1] for c in corners) + margin_m
    t = math.radians(theta0_deg)
    u, n = (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))
    return [(round(a * u[0] + l * n[0], 3), round(a * u[1] + l * n[1], 3))
            for a, l in ((a0, l0), (a1, l0), (a1, l1), (a0, l1))]


def keepouts_doc(corners, *, theta0_deg: float, margin_m: float, name: str,
                 reason: str) -> dict:
    a = [c[0] for c in corners]
    b = [c[1] for c in corners]
    return {
        "frame": "the bound's map frame (its mask's); the heading line at theta0_deg",
        "theta0_deg": theta0_deg,
        "source": (f"measured from the mark: ahead {min(a):.3f}..{max(a):.3f} m, left "
                   f"{min(b):.3f}..{max(b):.3f} m; margin {margin_m:g} m on every side"),
        "keepouts": [{"name": name, "reason": reason, "polygon": [
            {"x": x, "y": y} for x, y in to_bound_frame(corners, theta0_deg, margin_m)]}],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A keep-out measured from the mark, into a "
                                             "bound's keepouts.json (on stdout)")
    ap.add_argument("corners", help="the corners file: one 'C<n> <ahead> <left>' a line, metres")
    ap.add_argument("--theta0", type=float, required=True, metavar="DEG",
                    help="the bound's theta0_deg (its bound.json): the heading line's direction")
    ap.add_argument("--margin", type=float, required=True, metavar="M",
                    help="metres added on every side; no default")
    ap.add_argument("--name", required=True)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    try:
        with open(args.corners, encoding="utf-8") as f:
            corners = read_corners(f.read())
        doc = keepouts_doc(corners, theta0_deg=args.theta0, margin_m=args.margin,
                           name=args.name, reason=args.reason)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {args.corners}: {exc}", file=sys.stderr)
        return 2
    json.dump(doc, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
