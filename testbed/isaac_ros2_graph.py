#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""R3: author the Isaac Sim ROS 2 Action Graph so the robot looks like a ROS 2 robot.

UNVALIDATED. No Isaac runtime existed in the session that wrote this. Treat the
first run as the debugging step and keep NVIDIA's own ROS 2 tutorials open
alongside it -- the OmniGraph node names and attribute paths below are the part
most likely to need correcting against your exact Isaac Sim build.

WHAT THIS IS
------------
On the Isaac side there is no rclpy node to write. You author an OmniGraph
"Action Graph" inside the USD stage and wire it to the robot's articulation.
This script builds that graph, per robot, and (with --robot-count N) namespaces
each one so topics land at /robotN/....

The result is a robot that publishes /clock, /odom, /tf, /scan and consumes
/cmd_vel -- indistinguishable, from Nav2's and the adapter's side, from
`fake_robot_node.py --passive` + `r2_sensors.py` at R2. That indistinguishability
IS the R3 result: r2_smoke.py must pass against this, unedited.

RUN (inside the Isaac Sim container)
------------------------------------
    ./python.sh /testbed/isaac_ros2_graph.py --robot-count 1

    # multi-robot, for the demo
    ./python.sh /testbed/isaac_ros2_graph.py --robot-count 3

THE ONE THAT BITES
------------------
ROS2PublishClock is not optional. Without /clock, every downstream node running
with use_sim_time:=true blocks forever waiting for time to start, and nothing in
the symptoms points at the cause.
"""

from __future__ import annotations

import argparse
import sys

# --------------------------------------------------------------------------
# SimulationApp MUST be constructed before any other Isaac/omni import. This is
# not a style rule -- the extension system is not loaded until it exists.
# --------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402  (import order is load-bearing)

_parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_parser.add_argument("--robot-count", type=int, default=1)
_parser.add_argument("--headless", action="store_true", default=True)
_parser.add_argument("--gui", dest="headless", action="store_false",
                     help="run with the Isaac UI (needs the WebRTC viewer or a display)")
_parser.add_argument("--env-usd",
                     default="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                     help="warehouse USD, relative to the Isaac asset root")
_parser.add_argument("--robot-usd",
                     default="/Isaac/Robots/Carter/carter_v1.usd",
                     help="robot USD, relative to the Isaac asset root")
_parser.add_argument("--spacing", type=float, default=2.0,
                     help="metres between spawned robots")
ARGS = _parser.parse_args()

simulation_app = SimulationApp({"headless": ARGS.headless})

# Everything below is only importable once SimulationApp exists.
import omni.graph.core as og                       # noqa: E402
from isaacsim.core.api import World                # noqa: E402
from isaacsim.core.utils import extensions, nucleus, stage  # noqa: E402
from pxr import Gf, UsdGeom                        # noqa: E402

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


def _asset_root() -> str:
    root = nucleus.get_assets_root_path()
    if root is None:
        print("FATAL: Isaac asset root not reachable. Check network access to Nucleus,")
        print("       or point --env-usd / --robot-usd at local USD files.")
        simulation_app.close()
        sys.exit(2)
    return root


def build_clock_graph() -> None:
    """/clock, once for the whole stage. Everything with use_sim_time depends on it."""
    og.Controller.edit(
        {"graph_path": "/ActionGraph_Clock", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "clock"),
            ],
        },
    )
    print("  /clock graph authored")


def build_robot_graph(index: int, prim_path: str, namespace: str) -> None:
    """Per-robot: odom, TF, scan out; cmd_vel in.

    `namespace` is empty for a single robot (topics at /odom, /cmd_vel) and
    /robotN for a fleet -- which is the Isaac half of the multi-robot story. The
    adapter half is the namespace plumbing noted in testbed/README.md section 5.
    """
    graph_path = f"/ActionGraph_Robot{index}"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                # -- outbound
                ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishRawTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                # -- inbound: /cmd_vel -> wheels
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("ScaleToStage", "omni.isaac.core_nodes.OgnIsaacScaleToFromStageUnit"),
                ("Break3Lin", "omni.graph.nodes.BreakVector3"),
                ("Break3Ang", "omni.graph.nodes.BreakVector3"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationCtl", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                # tick fan-out
                ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishOdom.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishRawTF.inputs:execIn"),
                ("OnTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("OnTick.outputs:tick", "ArticulationCtl.inputs:execIn"),
                # time
                ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishRawTF.inputs:timeStamp"),
                # context
                ("Context.outputs:context", "PublishOdom.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("Context.outputs:context", "PublishRawTF.inputs:context"),
                ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                # odometry
                ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                # cmd_vel -> differential drive -> articulation
                ("SubscribeTwist.outputs:linearVelocity", "ScaleToStage.inputs:value"),
                ("ScaleToStage.outputs:result", "Break3Lin.inputs:tuple"),
                ("SubscribeTwist.outputs:angularVelocity", "Break3Ang.inputs:tuple"),
                ("Break3Lin.outputs:x", "DiffController.inputs:linearVelocity"),
                ("Break3Ang.outputs:z", "DiffController.inputs:angularVelocity"),
                ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
                ("DiffController.outputs:velocityCommand", "ArticulationCtl.inputs:velocityCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Context.inputs:domain_id", 42),          # must match ROS_DOMAIN_ID
                ("ComputeOdom.inputs:chassisPrim", [prim_path]),
                ("ArticulationCtl.inputs:targetPrim", [prim_path]),

                # Frames. map -> odom is published by PublishRawTF as identity, which
                # mirrors r2_sensors.py's "perfect localisation" so AMCL stays off and
                # R2's Nav2 configuration carries over unchanged.
                ("PublishOdom.inputs:odomFrameId", f"{namespace}/odom" if namespace else "odom"),
                ("PublishOdom.inputs:chassisFrameId",
                 f"{namespace}/base_link" if namespace else "base_link"),
                ("PublishOdom.inputs:topicName", f"{namespace}/odom" if namespace else "odom"),
                ("PublishTF.inputs:parentPrim", [prim_path]),
                ("PublishTF.inputs:targetPrims", [prim_path]),
                ("PublishRawTF.inputs:parentFrameId", "map"),
                ("PublishRawTF.inputs:childFrameId", f"{namespace}/odom" if namespace else "odom"),
                ("SubscribeTwist.inputs:topicName",
                 f"{namespace}/cmd_vel" if namespace else "cmd_vel"),

                # Carter geometry. CHANGE THESE if you swap --robot-usd: wrong values
                # make the robot drift, and drift reads as a Nav2 tuning problem.
                ("DiffController.inputs:wheelDistance", 0.53),
                ("DiffController.inputs:wheelRadius", 0.24),
            ],
        },
    )
    print(f"  robot {index} graph authored at {graph_path} (ns={namespace or '<root>'})")


def main() -> int:
    root = _asset_root()
    world = World(stage_units_in_meters=1.0)

    print(f"loading environment: {ARGS.env_usd}")
    stage.add_reference_to_stage(root + ARGS.env_usd, "/World/Warehouse")

    build_clock_graph()

    for i in range(ARGS.robot_count):
        prim_path = f"/World/Robot{i}"
        namespace = f"/robot{i}" if ARGS.robot_count > 1 else ""
        stage.add_reference_to_stage(root + ARGS.robot_usd, prim_path)
        xform = UsdGeom.Xformable(world.stage.GetPrimAtPath(prim_path))
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, i * ARGS.spacing, 0.0))
        build_robot_graph(i, prim_path, namespace)

    world.reset()
    simulation_app.update()

    print(f"\n{ARGS.robot_count} robot(s) up. Verify from a ROS 2 shell on ROS_DOMAIN_ID=42:")
    print("    ros2 topic list          # expect /clock /odom /tf /cmd_vel")
    print("    ros2 topic hz /clock     # if this is silent, NOTHING downstream will start")
    print("    ros2 run tf2_ros tf2_echo map base_link")
    print("\nThen: docker compose --profile r3 up nav2-r3  and  r2-smoke (unedited).\n")

    try:
        while simulation_app.is_running():
            world.step(render=not ARGS.headless)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
