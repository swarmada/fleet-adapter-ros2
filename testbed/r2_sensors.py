#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R2 sensor node - the TF tree and laser scan Nav2 needs, without a simulator.

Nav2 will not plan without two things the R1 fake robot does not provide:

  1. a TF chain  map -> odom -> base_link -> laser
  2. a  sensor_msgs/LaserScan  the obstacle costmap can consume

This node supplies both by subscribing to the robot's /odom and raycasting
against the static map image. It publishes NOTHING the robot itself owns, so
fake_robot_node.py is untouched between R1 and R2 -- an R1 regression stays an
R1 regression, and R2 adds strictly on top.

Why not Gazebo? On Apple Silicon a headless Gazebo world is slow, fragile, and
adds a rendering dependency that proves nothing about the adapter. Nav2 cannot
tell the difference: it consumes /scan and TF, and both are real here. What this
does NOT give you is contact physics or wheel slip -- neither of which any Nav2
semantic under test depends on.

map -> odom is published as identity. That is a deliberate simplification: it
means "perfect localisation", so AMCL is not needed and R2 tests Nav2's PLANNING
and RECOVERY semantics rather than its particle filter. Swap in AMCL later if
localisation behaviour becomes interesting.

Run:
    python3 r2_sensors.py --map map/warehouse.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

SCAN_HZ = 10.0
# TF must run FASTER than every consumer, not at the same rate as them. Nav2's controller
# and collision_monitor each run at ~10 Hz on their own clock with no phase relationship
# to ours, and tf2 refuses to extrapolate -- so at a matched 10 Hz a consumer lands past
# the newest TF sample constantly. Near-misses are guaranteed, not unlucky: the observed
# failure was NINE MILLISECONDS of extrapolation, which was enough for collision_monitor
# to stop the robot and for goals to abort with an empty error_msg. 30-50 Hz on
# odom->base_link is the convention precisely so this cannot happen.
#
# The scan stays at SCAN_HZ: raycasting 360 beams in Python 50x a second is neither
# affordable nor needed.
TF_HZ = 50.0
N_BEAMS = 360
ANGLE_MIN, ANGLE_MAX = -math.pi, math.pi
RANGE_MIN, RANGE_MAX = 0.12, 12.0
RAY_STEP = 0.05          # m; matches the map resolution -- no point stepping finer
OCCUPIED_THRESH = 0.65


class OccupancyMap:
    """The PGM the map_server also reads, loaded once for raycasting."""

    def __init__(self, yaml_path: str) -> None:
        meta = {}
        with open(yaml_path, encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()

        self.resolution = float(meta["resolution"])
        ox, oy, _ = (float(v) for v in meta["origin"].strip("[]").split(","))
        self.origin_x, self.origin_y = ox, oy
        self.occupied_thresh = float(meta.get("occupied_thresh", OCCUPIED_THRESH))

        pgm = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), meta["image"])
        self.width, self.height, self.pixels = self._read_pgm(pgm)

    @staticmethod
    def _read_pgm(path: str):
        with open(path, "rb") as fh:
            data = fh.read()
        # P5 header: magic, width height, maxval -- each possibly comment-separated.
        fields, idx = [], 0
        while len(fields) < 4:
            while idx < len(data) and data[idx:idx + 1].isspace():
                idx += 1
            if data[idx:idx + 1] == b"#":
                while idx < len(data) and data[idx:idx + 1] != b"\n":
                    idx += 1
                continue
            start = idx
            while idx < len(data) and not data[idx:idx + 1].isspace():
                idx += 1
            fields.append(data[start:idx])
        idx += 1  # single whitespace after maxval
        w, h = int(fields[1]), int(fields[2])
        return w, h, data[idx:idx + w * h]

    def is_occupied(self, x: float, y: float) -> bool:
        """World metres -> pixel. Out of bounds counts as occupied (a wall)."""
        px = int((x - self.origin_x) / self.resolution)
        py = int((y - self.origin_y) / self.resolution)
        if not (0 <= px < self.width and 0 <= py < self.height):
            return True
        # The PGM was written bottom-row-first, matching map_server's origin.
        row = self.height - 1 - py
        value = self.pixels[row * self.width + px]
        occupancy = (255 - value) / 255.0
        return occupancy > self.occupied_thresh


class Sensors(Node):
    def __init__(self, occ: OccupancyMap) -> None:
        super().__init__("r2_sensors")
        self._map = occ
        self._x = self._y = self._yaw = 0.0

        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)

        # base_link -> laser is fixed geometry; publish once, latched.
        laser = TransformStamped()
        laser.header.stamp = self.get_clock().now().to_msg()
        laser.header.frame_id = "base_link"
        laser.child_frame_id = "laser"
        laser.transform.translation.z = 0.20
        laser.transform.rotation.w = 1.0
        self._static_tf.sendTransform([laser])

        # Sensor data QoS: BEST_EFFORT + VOLATILE is what Nav2's observation
        # sources expect. RELIABLE here silently produces an empty costmap.
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._scan_pub = self.create_publisher(LaserScan, "scan", scan_qos)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.create_timer(1.0 / TF_HZ, self._publish_tf)
        self.create_timer(1.0 / SCAN_HZ, self._publish_scan)

        self.get_logger().info(
            f"r2_sensors up: {occ.width}x{occ.height} px map, "
            f"{N_BEAMS} beams, {SCAN_HZ:.0f} Hz, map->odom = identity"
        )

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._x, self._y = p.x, p.y
        self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _send_tf(self, now) -> None:
        """Publish map->odom and odom->base_link, both stamped `now`.

        Frames and the map->odom identity are unchanged: that identity is what keeps AMCL
        out of the stack, deliberately. base_link->laser is static/latched and is not
        published here.
        """
        # map -> odom: identity (perfect localisation; see the module docstring).
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "odom"
        t.transform.rotation.w = 1.0

        # odom -> base_link: the robot's own reported pose.
        b = TransformStamped()
        b.header.stamp, b.header.frame_id, b.child_frame_id = now, "odom", "base_link"
        b.transform.translation.x = self._x
        b.transform.translation.y = self._y
        b.transform.rotation.z = math.sin(self._yaw / 2.0)
        b.transform.rotation.w = math.cos(self._yaw / 2.0)
        self._tf.sendTransform([t, b])

    def _publish_tf(self) -> None:
        self._send_tf(self.get_clock().now().to_msg())

    def _publish_scan(self) -> None:
        # Emit a TF sample at EXACTLY this scan's stamp, BEFORE the scan.
        #
        # A faster TF timer alone does not close the race. The timers have no phase
        # relationship, so a scan firing just before the TF timer is stamped NEWER than
        # the newest TF, and the consumer's lookup is once again an extrapolation into
        # the future -- the same 9 ms failure, just rarer. A sample at the scan's own
        # stamp makes that lookup land exactly ON a sample, which tf2 always allows.
        now = self.get_clock().now().to_msg()
        self._send_tf(now)

        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = now, "laser"
        scan.angle_min, scan.angle_max = ANGLE_MIN, ANGLE_MAX
        scan.angle_increment = (ANGLE_MAX - ANGLE_MIN) / N_BEAMS
        scan.range_min, scan.range_max = RANGE_MIN, RANGE_MAX
        scan.ranges = [self._cast(self._yaw + ANGLE_MIN + i * scan.angle_increment)
                       for i in range(N_BEAMS)]
        self._scan_pub.publish(scan)

    def _cast(self, world_angle: float) -> float:
        """March a ray until it hits an occupied cell or runs out of range."""
        dx, dy = math.cos(world_angle) * RAY_STEP, math.sin(world_angle) * RAY_STEP
        x, y, dist = self._x, self._y, 0.0
        while dist < RANGE_MAX:
            x += dx
            y += dy
            dist += RAY_STEP
            if dist >= RANGE_MIN and self._map.is_occupied(x, y):
                return dist
        return float("inf")   # no return within range -- Nav2 treats inf correctly


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", required=True, help="path to the map YAML (same file Nav2 loads)")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    occ = OccupancyMap(args.map)
    rclpy.init(args=ros_args)
    node = Sensors(occ)
    # `docker compose down` and --abort-on-container-exit send SIGTERM, which Python does
    # NOT turn into KeyboardInterrupt. Without this the process is killed inside
    # rclpy.spin() and raises
    #   RCLError: failed to initialize wait set: the given context is not valid
    # on the way out -- a non-zero exit, and a stack trace in the log, for an otherwise
    # clean shutdown. fake_robot_node.py already uses this exact handler.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
