#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Isaac-side VDA5050 shim - makes an Isaac robot look like a VDA5050 AMR.

WHY THIS EXISTS
---------------
`fleet-adapter-vda5050` is already written (3,488 lines, C1-C8 CONFORMANT, MQTT
client included) and has never spoken to anything. A real VDA5050 AMR runs
vendor navigation and publishes MQTT/JSON - it does NOT run ROS 2 or Nav2. So
the ROS 2 path cannot exercise it, and Isaac has no VDA5050 support of its own.

This shim is the missing piece, and it is the ONLY genuinely new code in the
multi-vendor demo.

THE PART THAT SAVES WORK
------------------------
The shim does NOT touch Isaac's Python API. It sits OUTSIDE Isaac and consumes
the same ROS 2 bridge output that `isaac_ros2_graph.py` already publishes for
every robot. So a VDA5050 robot in the scene is authored EXACTLY like a ROS 2
robot -- same Action Graph, same namespace -- and only the thing on the other
end of the topics differs:

    Isaac robot prim
      └── ROS2 Action Graph  (identical for every robot)
            ├── /robotN/odom, /robotN/cmd_vel
            │
            ├──► [robots 0,1]  fleet-adapter-ros2 + Nav2   (ROS 2 path)
            └──► [robot 2]     THIS SHIM  ──MQTT──►  fleet-adapter-vda5050

`isaac_ros2_graph.py --robot-count 3` therefore needs no VDA5050 awareness at all.

WHAT IT TRANSLATES
------------------
Northbound (Isaac -> MQTT), at `--state-hz`:
    /odom            -> state.agvPosition {x,y,theta}, state.velocity {vx,vy,omega}
    (twist non-zero) -> state.driving
    synthetic curve  -> state.batteryState.batteryCharge
    connection       -> ONLINE, with an MQTT last-will of CONNECTIONBROKEN
    factsheet        -> typeSpecification + protocolFeatures.agvActions

Southbound (MQTT -> Isaac):
    order            -> drive to the LAST released node, publish /cmd_vel
    instantActions   -> `cancelOrder` stops decisively (the confirmed-stop path)

Message shapes are written against `fleet_adapter_vda5050/profiles/vda5050.py`:
topic layout `uagv/v2/{manufacturer}/{serial}/{topic}`, version string "2.0.0",
`batteryCharge` already in percent, `errors[].errorLevel` FATAL -> Failed.

UNVALIDATED against a live broker. Compile-checked only.

RUN
---
    pip install paho-mqtt
    python3 vda5050_shim.py --robot-ns /robot2 --mqtt-host mosquitto \\
        --manufacturer swarmada --serial amr-1

    # then, pointing the real adapter at the same broker:
    python -m fleet_adapter_vda5050.adapter --comms vda5050 \\
        --mqtt-host mosquitto --manufacturer swarmada --serial amr-1 \\
        --endpoint <control-plane>:9090
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from datetime import datetime, timezone

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    print("FATAL: paho-mqtt is required.  pip install paho-mqtt")
    raise

VERSION = "2.0.0"          # must match profiles/vda5050.py _VERSION
INTERFACE = "uagv"
MAJOR = 2

CRUISE_SPEED = 0.5
TURN_SPEED = 1.0
ARRIVE_TOL = 0.25
HEADING_TOL = 0.15
CONTROL_HZ = 20.0


def _now_iso() -> str:
    """VDA5050 requires ISO 8601 UTC with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Vda5050Shim(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("vda5050_shim")
        self._args = args
        self._lock = threading.Lock()

        ns = args.robot_ns.rstrip("/")
        self._odom_topic = f"{ns}/odom" if ns else "odom"
        self._cmd_topic = f"{ns}/cmd_vel" if ns else "cmd_vel"

        # Robot state, mirrored from /odom.
        self._x = self._y = self._yaw = 0.0
        self._vx = self._vy = self._omega = 0.0
        self._have_odom = False
        self._battery = 100.0

        # Order state.
        self._goal: tuple[float, float] | None = None
        self._order_id = ""
        self._order_update_id = 0
        self._nodes_remaining = 0
        self._header_id = 0

        base = f"{INTERFACE}/v{MAJOR}/{args.manufacturer}/{args.serial}"
        self._t_connection = f"{base}/connection"
        self._t_state = f"{base}/state"
        self._t_factsheet = f"{base}/factsheet"
        self._t_order = f"{base}/order"
        self._t_instant = f"{base}/instantActions"

        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 10)
        self.create_timer(1.0 / CONTROL_HZ, self._control_tick)
        self.create_timer(1.0 / args.state_hz, self._publish_state)

        self._mqtt = self._connect_mqtt()
        self.get_logger().info(
            f"vda5050_shim: ROS 2 {self._odom_topic} <-> MQTT {base} "
            f"@ {args.mqtt_host}:{args.mqtt_port}"
        )

    # -- MQTT ---------------------------------------------------------------

    def _connect_mqtt(self) -> "mqtt.Client":
        client = mqtt.Client(client_id=f"shim-{self._args.serial}", clean_session=True)

        # LAST WILL. VDA5050 publishes CONNECTIONBROKEN via the broker's LWT when a
        # robot drops, and the adapter fails SAFE on it -- an unparseable or missing
        # connectionState is treated as broken, never as ONLINE. Registering the will
        # BEFORE connecting is what makes an ungraceful death visible.
        client.will_set(
            self._t_connection,
            json.dumps(self._envelope({"connectionState": "CONNECTIONBROKEN"})),
            qos=1, retain=True,
        )
        client.on_connect = self._on_mqtt_connect
        client.on_message = self._on_mqtt_message
        client.connect(self._args.mqtt_host, self._args.mqtt_port, keepalive=30)
        client.loop_start()
        return client

    def _on_mqtt_connect(self, client, _userdata, _flags, rc) -> None:
        if rc != 0:
            self.get_logger().error(f"MQTT connect failed rc={rc}")
            return
        client.subscribe([(self._t_order, 1), (self._t_instant, 1)])
        client.publish(self._t_connection,
                       json.dumps(self._envelope({"connectionState": "ONLINE"})),
                       qos=1, retain=True)
        client.publish(self._t_factsheet, json.dumps(self._factsheet()), qos=1, retain=True)
        self.get_logger().info("MQTT connected: ONLINE + factsheet published (retained)")

    def _on_mqtt_message(self, _client, _userdata, msg) -> None:
        try:
            obj = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            self.get_logger().warn(f"unparseable payload on {msg.topic}")
            return
        if msg.topic == self._t_order:
            self._handle_order(obj)
        elif msg.topic == self._t_instant:
            self._handle_instant_actions(obj)

    # -- southbound: MQTT -> motion ----------------------------------------

    def _handle_order(self, obj: dict) -> None:
        """A base order: nodes[] with sequenceIds. Drive to the last released node.

        headerId / orderUpdateId are the at-least-once dedup keys -- a re-delivered
        identical order MUST be a no-op, which is why the update id is compared
        rather than blindly accepted.
        """
        order_id = str(obj.get("orderId", ""))
        update_id = int(obj.get("orderUpdateId", 0) or 0)
        nodes = obj.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            return

        with self._lock:
            if order_id == self._order_id and update_id <= self._order_update_id:
                self.get_logger().info(f"duplicate order {order_id}/{update_id} - no-op")
                return

            released = [n for n in nodes
                        if isinstance(n, dict) and n.get("released", True)]
            if not released:
                return
            last = max(released, key=lambda n: int(n.get("sequenceId", 0) or 0))
            pos = last.get("nodePosition")
            if not isinstance(pos, dict):
                return
            self._goal = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))
            self._order_id, self._order_update_id = order_id, update_id
            self._nodes_remaining = len(released)
        self.get_logger().info(f"order {order_id}/{update_id} -> goal {self._goal}")

    def _handle_instant_actions(self, obj: dict) -> None:
        """`cancelOrder` is the adapter's decisive halt (blocking HARD).

        Clearing the goal stops the controller; the resulting deceleration shows up
        in /odom, travels back out as state.velocity, and is what the adapter's
        confirmed-stop check reads. Never report rest here -- report it when the
        odometry says so.
        """
        for action in obj.get("actions", []) or []:
            if not isinstance(action, dict):
                continue
            atype = str(action.get("actionType", ""))
            if atype in ("cancelOrder", "stopPause", "emergencyStop"):
                with self._lock:
                    self._goal = None
                    self._nodes_remaining = 0
                self._publish_cmd(0.0, 0.0)
                self.get_logger().info(f"instantAction {atype} - halting")

    # -- northbound: odom -> MQTT state -------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t = msg.twist.twist
        with self._lock:
            self._x, self._y = p.x, p.y
            self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self._vx, self._vy, self._omega = t.linear.x, t.linear.y, t.angular.z
            self._have_odom = True

    def _envelope(self, extra: dict) -> dict:
        self._header_id += 1
        return {
            "headerId": self._header_id,
            "timestamp": _now_iso(),
            "version": VERSION,
            "manufacturer": self._args.manufacturer,
            "serialNumber": self._args.serial,
            **extra,
        }

    def _factsheet(self) -> dict:
        return self._envelope({
            "typeSpecification": {
                "seriesName": self._args.series,
                "agvClass": "CARRIER",
                "maxLoadMass": 100.0,
                "localizationTypes": ["NATURAL"],
                "navigationTypes": ["AUTONOMOUS"],
            },
            "protocolFeatures": {
                "agvActions": [
                    {"actionType": "cancelOrder", "actionScopes": ["INSTANT"]},
                    {"actionType": "stopPause", "actionScopes": ["INSTANT"]},
                    {"actionType": "startPause", "actionScopes": ["INSTANT"]},
                ],
            },
        })

    def _publish_state(self) -> None:
        with self._lock:
            if not self._have_odom:
                return   # positionInitialized false is better than a bogus (0,0)
            x, y, yaw = self._x, self._y, self._yaw
            vx, vy, om = self._vx, self._vy, self._omega
            order_id, update_id = self._order_id, self._order_update_id
            remaining, battery = self._nodes_remaining, self._battery

        driving = math.hypot(vx, vy) > 1e-3 or abs(om) > 1e-3
        errors = []
        if self._args.inject_error:
            errors.append({
                "errorType": self._args.inject_error,
                "errorLevel": self._args.inject_error_level,
                "errorDescription": "injected by the testbed shim",
            })

        state = self._envelope({
            "orderId": order_id,
            "orderUpdateId": update_id,
            "lastNodeId": "",
            "lastNodeSequenceId": 0,
            "nodeStates": [{"nodeId": f"n{i}", "sequenceId": i, "released": True}
                           for i in range(remaining)],
            "edgeStates": [],
            "actionStates": [],
            "agvPosition": {
                "x": x, "y": y, "theta": yaw,
                "mapId": self._args.map_id,
                "positionInitialized": True,
            },
            "velocity": {"vx": vx, "vy": vy, "omega": om},
            "driving": driving,
            "operatingMode": self._args.operating_mode,
            "batteryState": {
                "batteryCharge": round(battery, 1),   # already PERCENT, not 0-1
                "charging": False,
                "batteryVoltage": 24.0 * (battery / 100.0),
            },
            "errors": errors,
            "information": [],
            "safetyState": {"eStop": "NONE", "fieldViolation": False},
        })
        self._mqtt.publish(self._t_state, json.dumps(state), qos=0)

    # -- the controller -----------------------------------------------------

    def _control_tick(self) -> None:
        """Turn-then-drive toward the order's goal node.

        A real AMR runs vendor navigation here. That is precisely the point: VDA5050
        is a task-level protocol, so the fleet layer never sees the motion controller
        and a crude one is representative enough for what the adapter is being tested
        against.
        """
        with self._lock:
            self._battery = max(0.0, self._battery - 0.02 / CONTROL_HZ)
            goal, x, y, yaw = self._goal, self._x, self._y, self._yaw
            if goal is None:
                return
            dx, dy = goal[0] - x, goal[1] - y
            dist = math.hypot(dx, dy)
            if dist <= ARRIVE_TOL:
                self._goal = None
                self._nodes_remaining = 0
                arrived = True
            else:
                arrived = False
                heading = math.atan2(dy, dx)
                err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))

        if arrived:
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info("goal reached - order complete")
            return
        if abs(err) > HEADING_TOL:
            self._publish_cmd(0.0, math.copysign(TURN_SPEED, err))
        else:
            self._publish_cmd(CRUISE_SPEED, 0.0)

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self._cmd_pub.publish(msg)

    def shutdown(self) -> None:
        try:
            self._mqtt.publish(
                self._t_connection,
                json.dumps(self._envelope({"connectionState": "OFFLINE"})),
                qos=1, retain=True)
            self._mqtt.loop_stop()
            self._mqtt.disconnect()
        except Exception:  # pragma: no cover - best effort on the way out
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot-ns", default="", help="ROS 2 namespace of the Isaac robot, e.g. /robot2")
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--manufacturer", default="swarmada")
    ap.add_argument("--serial", default="amr-1")
    ap.add_argument("--series", default="isaac-sim-amr")
    ap.add_argument("--map-id", default="warehouse")
    ap.add_argument("--operating-mode", default="AUTOMATIC",
                    help="AUTOMATIC | SEMIAUTOMATIC | MANUAL | SERVICE | TEACHIN. "
                         "MANUAL makes the adapter derive phase=Maintenance.")
    ap.add_argument("--state-hz", type=float, default=2.0,
                    help="VDA5050 state publish rate (real AMRs run 1-5 Hz)")
    ap.add_argument("--inject-error", default="",
                    help="fault injection: emit this errorType in state.errors[]")
    ap.add_argument("--inject-error-level", default="WARNING", choices=["WARNING", "FATAL"],
                    help="WARNING -> Degraded, FATAL -> Failed in the adapter")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init(args=ros_args)
    node = Vda5050Shim(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
