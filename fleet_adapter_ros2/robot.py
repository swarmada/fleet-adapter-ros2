# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The robot binding: how ROS 2 / Nav2 Fleet Adapter talks to real robots.

`RobotBinding` is the seam between the Swarmada protocol and your fleet API. The
generated adapter ships with `SimulatedRobot` so it is conformant out of the box;
**replace it with a real binding to your fleet API**.

SAFETY CONTRACT (never fake it): `is_stopped(robot_id)` MUST return True only when
the robot is *actually* at rest — confirmed from the robot, never inferred from a
timer or from silence. This is what makes emergency stop safe (CONFORMANCE.md C5).
"""

from __future__ import annotations

from typing import Protocol


class RobotBinding(Protocol):
    """The methods the adapter drives. Implement these against your fleet API."""

    def spawn(self, robot_id: str) -> None: ...

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Begin driving the robot toward a target pose (task dispatch)."""

    def command_stop(self, robot_id: str) -> None:
        """Command a safe stop. NOT a confirmation — that is `is_stopped`."""

    def command_pause(self, robot_id: str) -> None:
        """Pause navigation (cancel the goal + hold), keeping the target so it can
        resume. A REQUEST; confirmation the base is at rest is `is_stopped`."""

    def command_resume(self, robot_id: str) -> None:
        """Resume a paused navigation (re-dispatch the retained target)."""

    def tick(self, dt: float) -> None:
        """Advance any internal model by dt seconds (no-op for a real robot)."""

    def is_stopped(self, robot_id: str) -> bool:
        """CONFIRMED at rest — the ground truth the adapter waits on before it
        reports STOPPED. MUST reflect the real robot, never a timeout."""

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        """(x, y, floor) in the zone frame."""

    def battery_percent(self, robot_id: str) -> int: ...


class SimulatedRobot:
    """A minimal simulated robot with a REAL confirmed stop, so the generated
    adapter passes conformance immediately.

    Optionally scenario-driven: pass a scenario ``engine`` (see
    ``fleet_adapter_ros2.scenarios``) to make the simulated battery, hardware, comms,
    and estop follow a named preset — the same presets and behaviour as the in-tree
    simulator. With no engine it behaves exactly as the $0 skeleton (constant
    battery, no faults), so conformance is unchanged.

    TODO(vendor): delete this and implement `RobotBinding` against your fleet API.
    Keep the confirmed-stop contract — return `is_stopped=True` only when the robot
    has actually halted.
    """

    def __init__(self, engine=None) -> None:
        self._moving: dict[str, bool] = {}
        self._pos: dict[str, tuple[float, float]] = {}
        # Scenario engine (proto-free) + its clock. None ⇒ no scenario behaviour.
        self._engine = engine
        self._elapsed = 0.0                 # seconds since spawn, advanced by tick()
        self._hw_prev: dict[str, str] = {}  # last-reported hardware status (delta base)

    # ── RobotBinding contract (safety-critical; scenario never weakens it) ───────

    def spawn(self, robot_id: str) -> None:
        self._moving[robot_id] = False
        self._pos[robot_id] = (0.0, 0.0)

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        self._moving[robot_id] = True
        self._pos[robot_id] = (x, y)  # yaw ignored by the sim; a real binding honors it

    def command_stop(self, robot_id: str) -> None:
        self._moving[robot_id] = False  # a real robot confirms this via is_stopped

    def command_pause(self, robot_id: str) -> None:
        self._moving[robot_id] = False  # halt; is_stopped confirms rest, like a real base

    def command_resume(self, robot_id: str) -> None:
        self._moving[robot_id] = True   # re-dispatch the retained navigation goal

    def tick(self, dt: float) -> None:
        self._elapsed += dt  # advance the scenario clock; still a no-op for real motion

    def is_stopped(self, robot_id: str) -> bool:
        return not self._moving.get(robot_id, False)

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        x, y = self._pos.get(robot_id, (0.0, 0.0))
        return (x, y, 0)

    def battery_percent(self, robot_id: str) -> int:
        if self._engine is not None:
            curve = self._engine.battery_at(self._elapsed)  # battery-edge
            if curve is not None:
                return curve
        return 100

    # ── Scenario state the adapter maps onto the wire protocol ───────────────────

    def elapsed(self) -> float:
        return self._elapsed

    def hardware_delta(self) -> list:
        """Hardware components whose status changed since the last call (all of them
        on the first call) — the delta-compressed hardware-fault stream. Empty with
        no engine. Each item is a proto-free scenarios.HardwareState."""
        if self._engine is None:
            return []
        changed = self._engine.hardware_delta(self._elapsed, self._hw_prev)
        for hs in changed:
            self._hw_prev[hs.name] = hs.status
        return changed

    def reset_hardware_baseline(self) -> None:
        """Forget which components were already reported, so the NEXT hardware_delta()
        returns every component again. Used to satisfy C6.1: the first TelemetryPayload
        after a (re)connect must be a full snapshot, never a resumed delta."""
        self._hw_prev.clear()

    def manifest_hardware(self) -> list:
        """The full hardware manifest (all HEALTHY) for the initial CapabilitiesSnapshot."""
        return self._engine.hardware_at(0.0) if self._engine is not None else []

    def telemetry_suppressed(self) -> bool:
        """comms-flaky: whether telemetry is withheld right now (gap or stream down)."""
        return self._engine is not None and self._engine.telemetry_gap_at(self._elapsed)

    def stream_drop_window(self):
        """comms-flaky: (drop_at, reconnect_at) or None if the preset declares no drop."""
        return self._engine.stream_drop_window() if self._engine is not None else None

    def stream_down(self) -> bool:
        """comms-flaky: whether the ControlStream is currently dropped."""
        return self._engine is not None and self._engine.stream_down_at(self._elapsed)

    def estop_due(self) -> bool:
        """estop-drill: whether a timer-driven estop is due now."""
        return self._engine is not None and self._engine.estop_due(self._elapsed)
