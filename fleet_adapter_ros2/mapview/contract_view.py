# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A read-only desktop viewer that knows only the map-view contract (CONTRACT.md). The browser
viewer the map-view service serves is the main one; this is the developer's and replay tool.

    python -m fleet_adapter_ros2.mapview.contract_view --source FILE.jsonl --follow [--every 5]
    python -m fleet_adapter_ros2.mapview.contract_view --source FILE.jsonl --replay [--speed 20]
    python -m fleet_adapter_ros2.mapview.contract_view --source FILE.jsonl --png OUT.png

It reads a file; it never connects to a robot and sends nothing anywhere. Needs matplotlib
(``pip install matplotlib``). A real map shows a site's layout: keep the window out of any
public recording.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from fleet_adapter_ros2.mapview import contract as c

SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, AQUA, RED, ORANGE, GREY = "#2a78d6", "#1baf7a", "#e34948", "#eb6834", "#9a9893"


class Feed:
    """Lines from the contract file: all at once, as they are appended (follow), or by stamp
    (replay). In replay the viewer's clock is the stamps', so staleness reads as it did live."""

    def __init__(self, path: Path, mode: str, speed: float):
        self.f = open(path)  # noqa: SIM115 - followed for as long as the viewer runs
        self.mode, self.speed = mode, speed
        self.pending = None
        self.t0 = self.w0 = None

    def clock(self) -> float:
        if self.mode == "replay" and self.t0 is not None:
            return self.t0 + (time.time() - self.w0) * self.speed
        return time.monotonic()

    def take(self, st: c.ViewState):
        while True:
            if self.pending is None:
                line = self.f.readline()
                if not line:
                    return
                if not line.endswith("\n"):          # a line still being written: read it next time
                    self.f.seek(self.f.tell() - len(line))
                    return
                try:
                    self.pending = c.loads(line)
                except c.ContractError:
                    # Counted by the state, under version or contract, as the service counts it.
                    st.accept(c.parse(line))
                    continue
            if self.mode == "replay":
                if self.t0 is None:
                    self.t0, self.w0 = self.pending["stamp"], time.time()
                if (self.pending["stamp"] - self.t0) / self.speed > time.time() - self.w0:
                    return
                st.accept(self.pending, now=self.pending["stamp"])
            else:
                st.accept(self.pending)
            self.pending = None


def draw(ax, st: c.ViewState, every: float, now: float):
    import numpy as np

    ax.clear()
    ax.set_facecolor(SURF)
    fresh = st.staleness(now)
    if st.map is not None:
        g, res, (ox, oy), _, _ = st.map
        g = np.array(g, np.int16)
        h, w = g.shape
        img = np.full((h, w, 3), 0.86)
        img[g == 0] = (0.99, 0.99, 0.985)
        img[g >= 65] = (0.11, 0.11, 0.11)
        ax.imshow(img, extent=[ox, ox + w * res, oy, oy + h * res], origin="lower",
                  interpolation="nearest", alpha=0.5 if fresh["map"] == "stale" else 1.0)
        ax.set_xlim(ox, ox + w * res)
        ax.set_ylim(oy, oy + h * res)
    for _, k, kres, (kox, koy) in st.keepout_grids:
        k = np.array(k, np.int16)
        kh, kw = k.shape
        xs = kox + (np.arange(kw) + 0.5) * kres
        ys = koy + (np.arange(kh) + 0.5) * kres
        mask = (k == 100).astype(float)
        ax.contourf(xs, ys, mask, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(xs, ys, mask, levels=[0.5], colors=RED, linewidths=1.5)
    for z in st.overlays["zones"]:
        pts = np.array(z["polygon"] + [z["polygon"][0]])
        ax.plot(pts[:, 0], pts[:, 1], color=BLUE, lw=1.5, alpha=0.8)
    if st.poses:
        ax.plot([p[1] for p in st.poses], [p[2] for p in st.poses], color=AQUA, lw=2, zorder=5)
    if st.last_pose:
        _, x, y, yaw = st.last_pose
        live = fresh["pose"] in ("fresh", None)
        colour = BLUE if live else GREY
        if fresh["pose"] != "lost":
            ax.annotate("", (x + 0.6 * math.cos(math.radians(yaw)),
                             y + 0.6 * math.sin(math.radians(yaw))), (x, y),
                        arrowprops={"arrowstyle": "-|>,head_length=0.8,head_width=0.5",
                                    "color": colour, "lw": 3}, zorder=10)
        ax.plot(x, y, "o", ms=9, color=colour if live else SURF, mec=colour, mew=2, zorder=10)
    t = st.overlays["target"]
    if t:
        ax.plot(t["x"], t["y"], "o", ms=16, mfc="none", mec=ORANGE, mew=3, zorder=9)
        ax.annotate(t["label"], (t["x"], t["y"]), (10, 10), textcoords="offset points", color=INK,
                    fontsize=11, bbox={"boxstyle": "round,pad=0.2", "fc": SURF, "ec": "none"},
                    zorder=9)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title(st, every, now), color=RED if sum(st.refused.values()) else INK,
                 fontsize=10, loc="left")


def title(st: c.ViewState, every: float, now: float) -> str:
    fresh = st.staleness(now)
    src = (c.describe_source(st.status["source"], st.status.get("simulated", False))
           if st.status else "no session yet")
    if st.last_pose:
        pos = (f"robot ({st.last_pose[1]:+.2f}, {st.last_pose[2]:+.2f}) m, facing "
               f"{st.last_pose[3]:+.0f} deg")
        if fresh["pose"] in ("stale", "lost"):
            pos += f" ({fresh['pose']}: {now - st.arrived['PoseSample']:.0f} s old)"
    else:
        pos = "no pose yet"
    if st.map:
        age = f"map #{st.map[3]}, changed {now - st.map_changed_at:.0f} s ago"
        if fresh["map"] == "stale":
            age += f" (stale: {now - st.arrived['MapSnapshot']:.0f} s since the last)"
    else:
        age = "no map yet"
    end = "; session ended" if fresh["ended"] else ""
    ref = sum(st.refused.values())
    refused = ""
    if ref:
        refused = f"; {ref} refused"
        if st.refused["version"]:
            refused += f" ({st.refused['version']} in {', '.join(sorted(st.versions_seen))})"
    return f"{src}\n{pos}; {age}{end}; every {every:g} s{refused}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--every", type=float, default=5.0)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--follow", action="store_true", help="keep reading as the file grows")
    m.add_argument("--replay", action="store_true", help="play by the messages' stamps")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--frames", type=int, default=0)
    a = ap.parse_args(argv)

    import matplotlib
    if a.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = c.ViewState()
    feed = Feed(a.source, "replay" if a.replay else "follow", a.speed)
    fig, ax = plt.subplots(figsize=(7, 8.5))
    fig.patch.set_facecolor(SURF)
    n = 0
    try:
        while True:
            feed.take(st)
            draw(ax, st, a.every, feed.clock())
            fig.tight_layout()
            n += 1
            if a.png:
                fig.savefig(a.png, facecolor=SURF, dpi=120)
                if a.frames and n >= a.frames:
                    break
                if not (a.follow or a.replay):
                    break
                time.sleep(a.every)
            else:
                plt.pause(a.every)
                if not plt.fignum_exists(fig.number) or (a.frames and n >= a.frames):
                    break
    except KeyboardInterrupt:
        pass
    print(f"frame {st.frame}; refused {st.refused}", flush=True)


if __name__ == "__main__":
    main()
