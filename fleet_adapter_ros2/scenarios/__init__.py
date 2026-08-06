# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Simulator-scenario presets for the ROS 2 / Nav2 SimulatedRobot binding.

Vendored from the in-tree ``adapters/scenarios`` package (same YAML format, same
``--scenario`` surface) so the ROS 2 adapter tells the SAME story by name as the
in-tree simulator. Proto-free and dependency-light; the binding maps the engine's
output onto the wire protocol.
"""

from fleet_adapter_ros2.scenarios.loader import (
    HW_DEGRADED,
    HW_FAILED,
    HW_HEALTHY,
    HardwareFaultOverrides,
    HardwareState,
    Scenario,
    ScenarioEngine,
    available_presets,
    load_scenario,
)

__all__ = [
    "HW_DEGRADED",
    "HW_FAILED",
    "HW_HEALTHY",
    "HardwareFaultOverrides",
    "HardwareState",
    "Scenario",
    "ScenarioEngine",
    "available_presets",
    "load_scenario",
]
