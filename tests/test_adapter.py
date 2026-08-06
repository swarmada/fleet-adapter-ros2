# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Safety-wiring tests for ROS 2 / Nav2 Fleet Adapter.

The safety MUSTs themselves (C3/C4/C5) are tested in `swarmada-sdk`; these confirm
this adapter WIRES them correctly and that the default robot binding honours the
confirmed-stop contract. Add your own tests as you bind the real fleet API.
"""

from __future__ import annotations

from swarmada_sdk.safety import ESTOP_FAILED, ESTOP_STOPPED, confirm_estop

from fleet_adapter_ros2.robot import SimulatedRobot


def test_default_robot_confirms_a_real_stop() -> None:
    # C5: STOPPED only when the robot has actually halted.
    robot = SimulatedRobot()
    robot.spawn("r1")
    robot.command_move("r1", 5.0, 0.0)
    assert confirm_estop(robot, "r1") == ESTOP_STOPPED
    assert robot.is_stopped("r1")


class _NeverStops:
    def command_stop(self, robot_id: str) -> None: ...
    def tick(self, dt: float) -> None: ...
    def is_stopped(self, robot_id: str) -> bool:
        return False


def test_unconfirmed_stop_is_failed_never_stopped() -> None:
    # A binding that cannot confirm a halt MUST NOT be reported STOPPED.
    assert confirm_estop(_NeverStops(), "r1", tick_dt=0.01, max_ticks=5) == ESTOP_FAILED
