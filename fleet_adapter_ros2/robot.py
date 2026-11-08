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

import math
import sys
import threading
import time
from typing import Protocol

from fleet_adapter_ros2 import task as task_map


class RobotBinding(Protocol):
    """The methods the adapter drives. Implement these against your fleet API."""

    # ── lifecycle ────────────────────────────────────────────────────────────
    # DECIDED, after an adapter shipped that never brought its binding up: the lifecycle
    # belongs IN this contract, and the adapter
    # calls it unconditionally, rather than being discovered by the adapter with
    # `if hasattr(robot, "start")`.
    #
    # The evidence that settles it, measured across all three reference adapters:
    # ALL THREE BINDINGS ALREADY HAD A `start()`, and two of the three adapters never
    # called it -- `fleet-adapter-ros2` (this one) and `fleet-adapter-vda5050`
    # (still inert there). All three Protocols omitted it. So the defect was never
    # that bindings disagreed about NAMING; it was that the contract did not require the
    # call, and "Protocol silent" tracked "call forgotten" almost perfectly.
    #
    # Be precise about what that does and does not prove, because an earlier version of
    # this comment overstated it: a `hasattr("start")` check would in fact have worked on
    # all three bindings as they exist today. It is rejected for what it ENCODES, not for
    # a naming divergence nobody has observed -- `if hasattr(...)` literally says "a
    # binding with no lifecycle is fine", which is the assumption that was wrong, and it
    # cannot tell an absent lifecycle from one spelled differently by a binding not yet
    # written. Declaring it puts the requirement where an implementer reads it, which is
    # the one place its absence demonstrably mattered:
    #
    #   `Nav2Binding.start()` creates the rclpy node, three subscriptions, the
    #   NavigateToPose action client and the /cmd_vel publisher. Nothing called it.
    #   Every live path in that binding is guarded on `self._node is not None`, so
    #   `--binding ros2` constructed an object that talked to nothing and NOT ONE
    #   CALL RAISED: pose() answered (0,0,0), battery_percent() answered 0, and
    #   is_stopped() answered False forever -- which is what made three conformance
    #   estop checks (C5.2, C8.3, C14.1) fail, correctly, for a reason no one could
    #   see. The adapter went through this Protocol, and this Protocol had no
    #   lifecycle to go through.
    #
    # Three bindings had three lifecycle conventions and none of them was in the
    # contract. A binding with no lifecycle implements both as documented no-ops
    # (see `SimulatedRobot.start`), which is a visible, reviewable two lines; an
    # absent lifecycle was invisible.

    def start(self) -> None:
        """Bring the binding's transport up: open connections, subscribe, and start
        whatever background I/O the fleet API needs. Called ONCE by the adapter
        before `spawn`, and before any command or telemetry read.

        MUST raise if the transport cannot be established. Returning normally is the
        adapter's only evidence that the binding is live, so a binding that cannot
        connect must not return quietly -- that is an inert binding reporting plausible
        values, the defect this method was added to prevent.

        A binding that needs no bring-up implements this as an explicit no-op."""

    def stop(self) -> None:
        """Tear the transport down: close connections and JOIN any background thread
        before the interpreter exits. MUST be idempotent, and MUST be safe to call
        when `start` was never called or raised.

        Declared alongside `start` deliberately. A lifecycle with a bring-up and no
        teardown is half a lifecycle, and the missing half was also a real defect:
        `Nav2Binding` spun rclpy on a daemon thread the interpreter killed mid-C++-frame
        at exit, so the process died on SIGABRT -- exit 134 on runs whose work had
        already succeeded."""

    def transport_alive(self) -> bool:
        """True only while this binding can still OBSERVE the robot — its transport is up
        and delivering, so the values the accessors below return are measurements rather
        than defaults or frozen copies.

        Declared here for the same reason as the lifecycle, and it is the other half of the
        same defect -- a contract that did not require the call. The adapter must ask this
        BEFORE it publishes a
        `TelemetryPayload`, because a binding that cannot observe the robot still answers
        every accessor: `pose()` returns the origin, `battery_percent()` returns 0. Sending
        those is a **C6.2 violation** — *"never send a defaulted 0/false where the true
        reading is unknown"* — and the control plane cannot distinguish it from a robot
        genuinely parked at the origin on a flat battery.

        A predicate, deliberately, and not only an exception out of `tick()`. An exception
        is a ONE-SHOT signal, so whichever thread reaches it first consumes it on behalf of
        every other: an estop arriving before bring-up spends the binding's single
        `tick()` raise inside `confirm_estop`, and the telemetry loop that starts
        afterwards then sees a silent, apparently healthy binding and publishes defaults at
        full cadence forever. A predicate cannot be consumed, so it cannot be raced.

        A binding with no transport to lose returns True — `SimulatedRobot` does, because
        its readings are exactly as live as the simulation it models. Returning True is
        therefore the honest answer for it and NOT a stub.
        """

    # ── robot ────────────────────────────────────────────────────────────────

    def spawn(self, robot_id: str) -> None: ...

    # ── threads ──────────────────────────────────────────────────────────────
    # The adapter calls a binding from several threads (control, telemetry, safety, edge),
    # and EVERY STOP PATH -- estop, lease self-stop, cancel, reconcile -- calls
    # `command_stop`, `tick`, `is_stopped` and `stop_settled` WITHOUT the adapter's robot
    # lock, so that no stop ever waits behind another call into the binding. Those four MUST
    # be safe to call concurrently with any method here. None of the methods here may block,
    # `stop_settled` excepted, and only for its `timeout`: a binding that has to wait for
    # something (a server to appear, a reply) does that wait on a thread of its own, holding
    # nothing the stop methods need.

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Begin driving the robot toward a target pose (task dispatch). MUST NOT wait for the
        fleet API: a dispatch that waited, with the adapter's robot lock held, once kept an
        estop from being commanded or acknowledged at all."""

    def command_stop(self, robot_id: str) -> None:
        """Command a safe stop. NOT a confirmation — that is `is_stopped`. Called from any
        thread, without the adapter's lock, and MUST NOT block."""

    def stop_settled(self, robot_id: str, timeout: float) -> bool:
        """After a stop: wait up to ``timeout`` s until nothing the stop was meant to end can
        still start moving the robot, and say whether that holds. A binding that dispatches
        asynchronously has a gap between sending a goal and holding it; a stop in that gap
        must be carried out when the goal arrives, and until it has been, no path may report
        the robot stopped or an action stopped safely. A binding with no such gap returns
        True."""

    def command_pause(self, robot_id: str) -> None:
        """Pause navigation (cancel the goal + hold), keeping the target so it can
        resume. A REQUEST; confirmation the base is at rest is `is_stopped`."""

    def command_resume(self, robot_id: str) -> None:
        """Resume a paused navigation (re-dispatch the retained target)."""

    def tick(self, dt: float) -> None:
        """Advance any built-in model by dt seconds (no-op for a real robot)."""

    def is_stopped(self, robot_id: str) -> bool:
        """CONFIRMED at rest — the ground truth the adapter waits on before it
        reports STOPPED. MUST reflect the real robot, never a timeout."""

    def rest_evidence_gap(self, robot_id: str) -> str:
        """``""`` when this binding's rest evidence may confirm a stop; otherwise ONE
        SENTENCE saying why it may not, which the adapter carries verbatim into every
        stop report it therefore cannot confirm.

        ADR-0071 (CONFORMANCE.md C5.3) admits a STOPPED only from a motion channel
        INDEPENDENT OF THE COMMAND PATH: one whose reading would differ if the drive did
        not do what it was told. A binding whose "motion" reading is the velocity command
        echoed back — clamped, filtered, integrated, fused, it makes no difference — has no
        such channel, and :meth:`is_stopped` on it answers the command, not the robot.
        Declare that here rather than letting the adapter discover it on a floor: the two
        cases are indistinguishable on the wire and in the ROS graph alike, and the
        difference only shows when the drive disobeys, which is the one moment a stop
        report matters.

        This is a declaration, not a measurement, so it is SAFE ONLY IF HONEST. A binding
        that answers ``""`` without an independent channel makes every stop report a lie
        the suite cannot catch.
        """

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        """(x, y, floor) in the zone frame."""

    def battery_percent(self, robot_id: str) -> int: ...

    def explore_stack(self):
        """The exploration stack (`explore.ExploreStack`) this binding drives explore-floor and
        Charge through, or None when it has none -- and then neither action is declared.
        `command_stop` MUST halt it: every stop path reaches the explorer and the return goal
        through that one call, exactly as it reaches a Navigate goal."""


# How fast the simulated robot travels to a goal. The conformance harness sizes the goals it sends
# at this speed (`_FIXTURE_SPEED_MPS`), so each keeps the robot moving for as long as the check
# that sent it needs it moving.
SIM_SPEED_MPS = 0.5


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

    _NO_STACK = object()

    def __init__(self, engine=None, explore=_NO_STACK, clock=time.monotonic) -> None:
        # tick() is called from the telemetry loop and, without the adapter's lock, from the
        # estop and self-stop confirmation, so the clock it advances is guarded.
        self._clock_lock = threading.Lock()
        self._moving: dict[str, bool] = {}
        self._pos: dict[str, tuple[float, float]] = {}
        # A goal is reached SIM_SPEED_MPS after it is sent, measured from where the robot was:
        # `_arrive_at` is when, on `_clock`, while it moves; `_left_s` the travel a pause kept.
        # `_goal_state` is the TaskState of the last goal, "" when a stop ended it.
        self._clock = clock
        self._arrive_at: dict[str, float] = {}
        self._left_s: dict[str, float] = {}
        self._goal_state: dict[str, str] = {}
        # Scenario engine (proto-free) + its clock. None ⇒ no scenario behaviour.
        self._engine = engine
        self._elapsed = 0.0                 # seconds since spawn, advanced by tick()
        self._hw_prev: dict[str, str] = {}  # last-reported hardware status (delta base)
        # The simulated exploration stack, so explore-floor and Charge are declared and run
        # here as they would on a robot with one. None declares neither.
        if explore is SimulatedRobot._NO_STACK:
            from fleet_adapter_ros2.sim_explore import SimulatedExploreStack
            explore = SimulatedExploreStack()
        self._explore = explore

    # ── RobotBinding contract (safety-critical; scenario never weakens it) ───────

    def start(self) -> None:
        """No-op, DELIBERATELY: a simulated robot has no transport to bring up.

        This is not an oversight and must not be deleted as dead code. `RobotBinding`
        declares a lifecycle (see the Protocol above) so that every binding answers the
        question "what does bring-up mean here?" in writing. For this one the answer is
        "nothing", and writing it down is what lets the adapter call `start()`
        unconditionally on whatever binding it was handed -- which is the fix for the
        defect where an adapter that could not call a lifecycle it did not know about
        shipped a `--binding ros2` that talked to nothing."""

    def stop(self) -> None:
        """No-op, for the same reason as `start`: nothing to tear down. Idempotent
        trivially."""

    def transport_alive(self) -> bool:
        """Always True, and that is the honest answer rather than a stub: this robot's
        readings come from an in-process model, so they are exactly as live as the
        simulation. There is no transport that can die and leave the accessors returning
        frozen copies, which is the condition the adapter is asking about."""
        return True

    def spawn(self, robot_id: str) -> None:
        self._moving[robot_id] = False
        self._pos[robot_id] = (0.0, 0.0)

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        with self._clock_lock:
            fx, fy = self._pos.get(robot_id, (0.0, 0.0))
            self._arrive_at[robot_id] = (self._clock()
                                         + math.hypot(x - fx, y - fy) / SIM_SPEED_MPS)
            self._left_s.pop(robot_id, None)
            self._goal_state[robot_id] = task_map.TASK_RUNNING
            self._moving[robot_id] = True
            self._pos[robot_id] = (x, y)  # yaw ignored by the sim; a real binding honors it

    def command_stop(self, robot_id: str) -> None:
        try:
            if self._explore is not None:
                self._explore.halt()    # the explorer and the return goal stop with the base
        except Exception as exc:  # noqa: BLE001 - a stop path: the base's own stop still runs
            print(f"stop: the exploration stack's halt raised {type(exc).__name__}: {exc}; "
                  f"stopping the base regardless", file=sys.stderr)
        finally:
            self.command_hold(robot_id)

    def command_hold(self, robot_id: str) -> None:
        """The base's own stop, without the exploration stack (see `Nav2Binding.command_hold`)."""
        with self._clock_lock:
            self._moving[robot_id] = False  # a real robot confirms this via is_stopped
            self._arrive_at.pop(robot_id, None)
            self._left_s.pop(robot_id, None)
            if self._goal_state.get(robot_id) == task_map.TASK_RUNNING:
                self._goal_state[robot_id] = ""   # stopped short: not reached

    def stop_settled(self, robot_id: str, timeout: float) -> bool:
        """True: a move here takes effect in the call, so a stop has nothing in flight to
        wait for."""
        return True

    def command_pause(self, robot_id: str) -> None:
        with self._clock_lock:
            self._arrive(robot_id)
            if robot_id in self._arrive_at:  # the travel still to go waits for the resume
                self._left_s[robot_id] = self._arrive_at.pop(robot_id) - self._clock()
            self._moving[robot_id] = False  # halt; is_stopped confirms rest, like a real base

    def command_resume(self, robot_id: str) -> None:
        with self._clock_lock:
            if robot_id in self._left_s:
                self._arrive_at[robot_id] = self._clock() + self._left_s.pop(robot_id)
            self._moving[robot_id] = True   # re-dispatch the retained navigation goal

    def task_state(self, robot_id: str) -> str:
        """TaskState of the last goal: RUNNING until it is reached, then SUCCEEDED, as a Nav2
        goal reports it (C6.4). "" once a stop ended it short, which the stop reports."""
        with self._clock_lock:
            self._arrive(robot_id)
            return self._goal_state.get(robot_id, "")

    def _arrive(self, robot_id: str) -> None:
        """The robot comes to rest at its goal once the travel to it is over. Caller holds
        `_clock_lock`."""
        due = self._arrive_at.get(robot_id)
        if due is not None and self._clock() >= due:
            del self._arrive_at[robot_id]
            self._moving[robot_id] = False
            self._goal_state[robot_id] = task_map.TASK_SUCCEEDED

    def tick(self, dt: float) -> None:
        with self._clock_lock:
            self._elapsed += dt  # advance the scenario clock; still a no-op for real motion

    def is_stopped(self, robot_id: str) -> bool:
        exploring = self._explore is not None and self._explore.moving()
        with self._clock_lock:
            self._arrive(robot_id)
            moving = self._moving.get(robot_id, False)
        return not moving and not exploring

    def rest_evidence_gap(self, robot_id: str) -> str:
        """No gap: this binding's plant model IS the robot, and `_moving` is that model's
        own state, not a command echoed back. It goes on reporting motion for as long as
        the model moves, whatever was last commanded, which is the independence ADR-0071
        asks for on the only robot there is here. A run against it is still a SIMULATED
        run, and says nothing about the platform a real binding will drive."""
        return ""

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        x, y = self._pos.get(robot_id, (0.0, 0.0))
        return (x, y, 0)

    def battery_percent(self, robot_id: str) -> int:
        if self._engine is not None:
            curve = self._engine.battery_at(self._elapsed)  # battery-edge
            if curve is not None:
                return curve
        return 100

    def explore_stack(self):
        return self._explore

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
