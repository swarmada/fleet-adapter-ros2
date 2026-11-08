# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Write the map-view contract (JSON Lines) from an operator-driven session.

  record   live and read-only, through a runner that executes one shell line where ROS 2 can
           see the robot: ``ros2 topic echo /map --full-length`` and ``tf2_echo map
           base_footprint 5``, with keep-out and target overlays from the run directory.
           Ctrl-C ends it.
  convert  a saved guided run (guide-log.jsonl and a saved map) into the contract, for replay.

    python -m fleet_adapter_ros2.mapview.contract_tool record  --runner ./robot-shell \\
        --run-dir DIR --session s1 --out DIR/mapview.jsonl
    python -m fleet_adapter_ros2.mapview.contract_tool convert --run-dir DIR --map DIR/map.yaml \\
        --session s1 --out FILE [--zones zones.yaml]

The runner is a program taking one argument, the shell line (for example a script that runs it
over ssh). How it reaches the robot is the runner's: this tool never reads or prints an
address. Where rclpy is available, ``bridge`` does the same job as a ROS 2 node.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import sources as s

MAP_CMD = "PYTHONUNBUFFERED=1 ros2 topic echo /map --full-length"
POSE_CMD = "stdbuf -oL ros2 run tf2_ros tf2_echo map base_footprint 5"
# Ends only the two echoes this tool started; nothing else on the robot is touched.
END_CMD = ('pkill -f "[r]os2 topic echo /map --full-length"; '
           'pkill -f "[t]f2_echo map base_footprint 5"')


class Writer:
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)  # noqa: SIM115 - held for the recording
        self.lock = threading.Lock()
        self.counts = {}

    def put(self, m: dict):
        line = c.dumps(m)
        with self.lock:
            self.f.write(line + "\n")
            self.counts[m["kind"]] = self.counts.get(m["kind"], 0) + 1


def record(a):
    w = Writer(a.out)
    frame, robot, label = f"map@{a.session}", a.robot, f"guided:{a.session}"
    w.put(c.message("SessionStatus", robot, frame, time.time(), source=label, phase="started",
                    text="live recording of an operator-driven session started",
                    area_free_m2=None))
    keepouts = s.keepout_overlay(a.run_dir)
    last_target = object()
    procs = []
    runner = os.path.expanduser(a.runner)
    free = {"m2": None}

    def map_reader(p):
        buf, seq = [], 0
        for line in p.stdout:
            if line.strip() == "---":
                r = s.parse_map_echo("".join(buf))
                buf = []
                if r:
                    g, res, o, stamp = r
                    w.put(c.message("MapSnapshot", robot, frame, stamp or time.time(), seq=seq,
                                    **c.encode_grid(g, res, o)))
                    free["m2"] = s.free_area(g, res)
                    seq += 1
            else:
                buf.append(line)

    def pose_reader(p):
        ps = s.Tf2EchoParser()
        for line in p.stdout:
            r = ps.feed(line)
            if r:
                w.put(c.message("PoseSample", robot, frame, r[0], x=r[1], y=r[2], yaw_deg=r[3],
                                source="slam"))

    errlog = open(a.out.with_suffix(".stderr.log"), "a")  # noqa: SIM115 - the echoes' stderr
    for cmd, fn in ((MAP_CMD, map_reader), (POSE_CMD, pose_reader)):
        p = subprocess.Popen([runner, cmd], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=errlog, text=True, bufsize=1, start_new_session=True)
        procs.append(p)
        threading.Thread(target=fn, args=(p,), daemon=True).start()
    print(f"recording {frame} to {a.out}; Ctrl-C ends it", flush=True)
    try:
        while True:
            t = s.read_target(a.run_dir)
            if t != last_target:
                w.put(c.message("Overlays", robot, frame, time.time(), keepouts=keepouts, zones=[],
                                target=t))
                last_target = t
            time.sleep(1.0)
            if all(p.poll() is not None for p in procs):
                print("both streams ended", flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, 15)
            except ProcessLookupError:
                pass
        subprocess.run([runner, END_CMD], stdin=subprocess.DEVNULL, capture_output=True,
                       timeout=30, check=False)
        w.put(c.message("SessionStatus", robot, frame, time.time(), source=label, phase="stopped",
                        text="live recording stopped", area_free_m2=free["m2"]))
        print("written:", ", ".join(f"{k} {v}" for k, v in sorted(w.counts.items())), flush=True)


def convert(a):
    w = Writer(a.out)
    for m in s.convert_guided_run(a.run_dir, a.map, a.robot, f"map@{a.session}", a.zones):
        w.put(m)
    print("written:", ", ".join(f"{k} {v}" for k, v in sorted(w.counts.items())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "convert"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--session", required=True,
                       help="the session's label, e.g. s1 (the frame is map@<session>)")
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--robot", default="robot-a", help="a local alias, never a network name")
        if name == "record":
            p.add_argument("--runner", required=True,
                           help="a program that runs its one argument, a shell line, where ROS 2 "
                                "sees the robot")
        if name == "convert":
            p.add_argument("--map", required=True, type=Path)
            p.add_argument("--zones", type=Path)
    a = ap.parse_args(argv)
    if a.out.exists() and a.cmd == "convert":
        sys.exit(f"{a.out} exists: choose a new file")
    (record if a.cmd == "record" else convert)(a)


if __name__ == "__main__":
    main()
