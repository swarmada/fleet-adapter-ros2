# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view Hello World's data source: this adapter's explore-floor segment, run on its
simulated exploration stack, feeding the map-view service. No robot, no ROS, no cluster needed.

    python -m fleet_adapter_ros2.mapview.hello_world --url http://localhost:8080 [--segments 0]

Each segment is the adapter's own `ExploreJob` -- the code an AssignAction runs -- on a fresh
`SimulatedExploreStack`: a synthetic 8 m x 6 m room with one keep-out measured by hand, and a
robot shown driving a loop round it at 0.18 m/s while the map grows. The feed is the one the
adapter runs with ``--mapview-url``. What it does not run is the control plane: no FleetTask,
no admission, no lease. Every message says it is simulated.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2 import sim_explore as sim
from fleet_adapter_ros2.mapview import emitter as mv

NAMESPACE = "hello"
ROBOT = "sim-robot"
PAYLOAD = {"operation": "explore-floor", "bound_ref": sim.SIM_BOUND, "max_speed_mps": 0.2}
# In the bound's frame, which is this boot's: the boot's yaw on the mark is the bound's theta0.
KEEPOUT = steps.Keepout(name="keep-out-1", reason="a low obstacle the lidar does not see",
                        polygon=((2.0, 1.6), (3.0, 1.6), (3.0, 2.4), (2.0, 2.4)))
TOUR = [(4.8, 0.0), (4.8, 3.6), (1.0, 3.6), (-1.0, 3.6), (-1.0, 1.2), (1.2, 1.2)]
TOUR_SPEED_MPS = 0.18


def run_segment(n: int, sink, artifacts: explore.ArtifactDir, *, explore_seconds: float,
                clock=None, feed_kw=None, say=print) -> str:
    """One explore-floor segment, start to end; its final report."""
    clock = clock or explore.RealClock()
    stack = sim.SimulatedExploreStack(
        clock=clock, boot_yaw_deg=sim.SIM_THETA0_DEG, explore_seconds=explore_seconds,
        tour=TOUR, tour_speed_mps=TOUR_SPEED_MPS, hand_keepouts=[KEEPOUT], start_mv=8200.0)
    action_id = f"{NAMESPACE}/explore-{n}"
    feed = mv.MapviewEmitter(sink, ROBOT, simulated=True, **(feed_kw or {})).segment(stack,
                                                                                      action_id)
    done = threading.Event()
    final: list[str] = []

    def report(state, message, progress_pct=0, *rest):
        say(f"[segment {n}] {state.removeprefix('ACTION_STATE_').lower()}: {message}")
        if state in ("ACTION_STATE_SUCCEEDED", "ACTION_STATE_FAILED"):
            final.append(message)
            done.set()

    job = explore.ExploreJob(action_id=action_id, doc=dict(PAYLOAD), stack=stack,
                             limits=stack.battery_limits(), clock=clock,
                             emit=feed.observe(report), artifacts=artifacts, namespace=NAMESPACE)
    thread = threading.Thread(target=job.run, daemon=True, name=f"segment-{n}")
    thread.start()
    feed.start(alive=lambda: not done.is_set() and thread.is_alive())
    thread.join()
    done.set()
    feed.join(10)
    return final[0] if final else "the segment ended without a final report"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="the map-view service (POST <url>/v0/ingest)")
    where.add_argument("--out", help="a JSON Lines file instead")
    ap.add_argument("--segments", type=int, default=0, help="how many segments; 0 runs until "
                    "stopped (default)")
    ap.add_argument("--explore-seconds", type=float, default=120.0,
                    help="each segment's explore time (default 120 s: one loop of the room)")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds between segments")
    a = ap.parse_args(argv)

    sink = (mv.HttpSink(a.url, os.environ.get("SWARMADA_MAPVIEW_TOKEN")) if a.url
            else mv.FileSink(a.out))
    artifacts = explore.ArtifactDir(tempfile.mkdtemp(prefix="mapview-hello-"), "hello-local")
    print(f"simulated explore-floor segments, {a.explore_seconds:g} s each, feeding "
          f"{'the map-view service' if a.url else a.out}", flush=True)
    n = 0
    try:
        while a.segments == 0 or n < a.segments:
            n += 1
            final = run_segment(n, sink, artifacts, explore_seconds=a.explore_seconds,
                                say=lambda t: print(t, flush=True))
            print(f"[segment {n}] ended: {final}", flush=True)
            if a.segments == 0 or n < a.segments:
                time.sleep(a.pause)
    except KeyboardInterrupt:
        pass
    finally:
        close = getattr(sink, "close", None)
        if close is not None:
            close()
        if isinstance(sink, mv.HttpSink) and sink.failed_posts:
            print(f"posts that failed: {sink.failed_posts}; last: {sink.last_error}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
