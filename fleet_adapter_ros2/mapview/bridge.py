# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A read-only ROS 2 bridge for an operator-driven (guided) session: the map-view contract
from SLAM's ``/map`` and the robot's pose through TF, while an operator drives the robot.

    python -m fleet_adapter_ros2.mapview.bridge --session s1 --url http://localhost:8080
    python -m fleet_adapter_ros2.mapview.bridge --session s1 --out s1.jsonl [--run-dir DIR]

The bridge never drives. It subscribes to one topic, listens to TF and runs two timers; it
creates no publisher, no service or action client and no goal, and a test reads this file and
fails if it names one. No FleetTask exists for a guided session, so its SessionStatus names
the session (``guided:<session>``), and a viewer shows it as operator-driven.

With ``--run-dir``, the keep-out mask the session built (``mask/keepout.yaml``) and the target
the operator is sent to (the last ``<label> <x> <y>`` line of ``targets.txt``) are drawn on top,
and re-read when they change.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import sources as s


def yaw_deg(q) -> float:
    """The heading of quaternion ``q`` (x, y, z, w attributes), in degrees."""
    return math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))


class GuidedBridge:
    """The ROS-free half: what the node's callbacks hand it becomes contract messages."""

    def __init__(self, sink, robot: str, session: str, *, run_dir: Path | None = None,
                 simulated: bool = False, clock=time.time) -> None:
        self.sink, self.robot, self.run_dir, self.simulated = sink, robot, run_dir, simulated
        self.frame, self.source = f"map@{session}", f"guided:{session}"
        self._clock = clock
        self._seq = 0
        self._last_map = None          # (fields, sha256, sent at)
        self._map_checked_at = None
        self._overlay_key = None
        self._area = None
        self.refused_maps = 0
        c.check(self._msg("SessionStatus", source=self.source, phase="started", text="",
                          area_free_m2=None))

    def _msg(self, kind: str, **fields) -> dict:
        return {"contractVersion": c.VERSION, "kind": kind, "robot": self.robot,
                "frame": self.frame, "stamp": float(self._clock()), **fields}

    def _put(self, kind: str, **fields) -> None:
        try:
            self.sink.put(c.check(self._msg(kind, **fields)))
        except Exception as exc:  # noqa: BLE001 - a bad message is reported, never raised
            print(f"mapview bridge: {kind} not written: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    def _status(self, phase: str, text: str) -> None:
        self._put("SessionStatus", source=self.source, phase=phase, text=text[:c.MAX_TEXT],
                  area_free_m2=self._area, simulated=self.simulated)

    def start(self) -> None:
        self._status("started", "guided session: an operator drives the robot; this view "
                     "only reads")
        self.tick(force_overlays=True)

    def stop(self, text: str = "guided session view stopped") -> None:
        self._status("stopped", text)

    def on_map(self, msg) -> None:
        """A ``nav_msgs/OccupancyGrid``. Sent when it changed, checked at most every
        MAP_CHECK_S; a grid whose origin is turned in its frame is refused (the contract's grid
        is not rotated) and counted."""
        now = self._clock()
        if self._map_checked_at is not None and now - self._map_checked_at < c.MAP_CHECK_S:
            return
        self._map_checked_at = now
        info, q = msg.info, msg.info.origin.orientation
        if abs(yaw_deg(q)) > 1e-3:
            self.refused_maps += 1
            return
        try:
            fields = c.encode_flat(list(msg.data), int(info.width), int(info.height),
                                   float(info.resolution),
                                   (float(info.origin.position.x), float(info.origin.position.y)))
        except c.ContractError:
            self.refused_maps += 1
            return
        self._area = round(sum(v == 0 for v in msg.data) * info.resolution ** 2, 3)
        if self._last_map is None or fields["sha256"] != self._last_map[1]:
            self._send_map(fields, now)

    def _send_map(self, fields: dict, now: float) -> None:
        self._put("MapSnapshot", seq=self._seq, **fields)
        self._seq += 1
        self._last_map = (fields, fields["sha256"], now)

    def on_pose(self, x: float, y: float, yaw: float) -> None:
        if all(math.isfinite(v) for v in (x, y, yaw)):
            self._put("PoseSample", x=float(x), y=float(y), yaw_deg=float(yaw), source="slam")

    def tick(self, force_overlays: bool = False) -> None:
        """Once a second: the overlays when the run directory's mask or target changed, and the
        latest map again once MAP_REPEAT_S has passed without one."""
        now = self._clock()
        if self._last_map is not None and now - self._last_map[2] >= c.MAP_REPEAT_S:
            self._send_map(self._last_map[0], now)
        if self.run_dir is None and not force_overlays:
            return
        mask = self.run_dir / "mask" / "keepout.yaml" if self.run_dir else None
        target = s.read_target(self.run_dir) if self.run_dir else None
        key = (mask.stat().st_mtime if mask and mask.exists() else None, json_key(target))
        if key != self._overlay_key or force_overlays:
            self._overlay_key = key
            keepouts = s.keepout_overlay(self.run_dir) if self.run_dir else []
            self._put("Overlays", keepouts=keepouts, zones=[], target=target)


def json_key(target) -> tuple | None:
    return None if target is None else (target["label"], target["x"], target["y"])


def attach(node, bridge: GuidedBridge, lookup_pose, *, map_type, map_qos) -> None:
    """Wire ``bridge`` to ``node``: one subscription and two timers, nothing else.
    ``lookup_pose()`` answers (x, y, yaw degrees) from TF, or None."""
    node.create_subscription(map_type, "/map", bridge.on_map, map_qos)

    def pose() -> None:
        p = lookup_pose()
        if p is not None:
            bridge.on_pose(*p)

    node.create_timer(c.POSE_PERIOD_S, pose)
    node.create_timer(1.0, bridge.tick)


def _sink(a):
    from fleet_adapter_ros2.mapview import emitter
    if a.url:
        return emitter.HttpSink(a.url, os.environ.get("SWARMADA_MAPVIEW_TOKEN"))
    return emitter.FileSink(a.out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", required=True, help="the session's label (frame map@<session>)")
    ap.add_argument("--robot", default="robot-a", help="a local alias, never a network name")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="the map-view service (POST <url>/v0/ingest)")
    where.add_argument("--out", help="a JSON Lines file")
    ap.add_argument("--run-dir", type=Path, help="mask/keepout.yaml and targets.txt, if any")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_footprint")
    a = ap.parse_args(argv)

    import rclpy
    import tf2_ros
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time

    sink = _sink(a)
    bridge = GuidedBridge(sink, a.robot, a.session, run_dir=a.run_dir)
    rclpy.init()
    node = rclpy.create_node("mapview_guided_bridge")
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer, node)

    def lookup_pose():
        try:
            t = buffer.lookup_transform(a.map_frame, a.base_frame, Time()).transform
        except Exception:  # noqa: BLE001 - no transform yet is no pose
            return None
        return t.translation.x, t.translation.y, yaw_deg(t.rotation)

    attach(node, bridge, lookup_pose, map_type=OccupancyGrid,
           map_qos=QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL))
    bridge.start()
    print(f"reading /map and {a.map_frame} -> {a.base_frame} into {bridge.frame}; Ctrl-C ends it",
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        node.destroy_node()
        rclpy.try_shutdown()
        close = getattr(sink, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
