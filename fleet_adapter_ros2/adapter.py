# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 / Nav2 Fleet Adapter — a safety-complete Swarmada Fleet Adapter.

Feature-basic but safety-complete (ADR-0005): the CONFORMANCE.md safety MUSTs are
wired from the audited `swarmada_sdk` primitives; optional commands are declined
with `unsupported = true` (C7). Bind `RobotBinding` (see robot.py) to your fleet
API — every `TODO(vendor)` marks a point where you do so.

Run:  python -m fleet_adapter_ros2.adapter --endpoint localhost:9090
"""

from __future__ import annotations

import argparse
import pathlib
import queue
import sys
import threading
import time
from collections.abc import Iterator

import grpc

# Make the package importable when run directly as a script (a real install
# `pip install -e .` makes this unnecessary).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarmada_sdk.safety import (  # noqa: E402
    ESTOP_STOPPED,
    FenceDecision,
    FenceGuard,
    LeaseMonitor,
    confirm_estop,
)

from fleet_adapter_ros2 import task as task_map  # noqa: E402
from fleet_adapter_ros2.robot import SimulatedRobot  # noqa: E402

_TELEMETRY_INTERVAL = 0.2
_TICK_DT = 0.2


# The fleet-adapter CONTRACT version this adapter implements (ADR-0032). Distinct from
# adapter_version (this build) and protocol_version (the wire package identity, not a semver, which
# is why it cannot express compatibility). A control plane refuses REGISTRATION for an adapter that
# reports nothing or reports out of range, so this is not optional -- telemetry, heartbeat and
# emergency stop keep working either way. Bump it only when this adapter is re-qualified against a
# new contract (`make conformance`), and update the adapters/REGISTRY.md row with it.
CONTRACT_VERSION = "1.0.0"


class fleet_adapter_ros2Adapter:  # noqa: N801  (generated class name)
    def __init__(self, pb, pb_grpc, robot, robot_id: str) -> None:
        self._pb = pb
        self._pb_grpc = pb_grpc
        self._robot = robot
        self._robot_lock = threading.Lock()
        self._robot_id = robot_id
        self._fence = FenceGuard()                              # C3
        self._lease = LeaseMonitor(on_expiry=self._self_stop)   # C4
        self._current_action = ""
        self._current_generation = 0
        self._current_fencing_token = None   # echoed on task-status streaming (C6.4)
        self._last_task_state = ""           # last streamed TaskState, for transition detect
        self._last_progress = -1             # last streamed progress_pct, to throttle updates
        self._edge_addr: queue.Queue = queue.Queue()  # advertised edge endpoints (C8)
        self._outbox: queue.Queue = queue.Queue()
        self._safety_outbox: queue.Queue = queue.Queue()  # → SafetyStream (acks + drills)
        self._shutdown = threading.Event()
        self._telemetry_started = False
        # C6.1: the first TelemetryPayload on any (re)connect must be a FULL
        # snapshot, not a resumed delta — otherwise a reconnected control plane
        # inherits a stale hardware picture it cannot detect.
        self._snapshot_pending = True
        # C2.3: per-robot cadence from RegisterAck; the adapter default until told.
        self._telemetry_interval = _TELEMETRY_INTERVAL

        # Scenario handle: a binding that exposes the scenario surface (SimulatedRobot
        # with an engine). None ⇒ no scenario (the ros2/Nav2 binding, or the $0
        # skeleton with no --scenario) → the streams run exactly as before.
        self._sim = robot if hasattr(robot, "hardware_delta") else None
        self._estop_drilled = False       # estop-drill fires once
        self._reconnect_pending = False   # comms-flaky: a stream drop is in progress
        self._active_call = None          # live ControlStream call, for cancel-on-drop

    # ── ControlStream ────────────────────────────────────────────────────────

    def _control_requests(self) -> Iterator:
        pb = self._pb
        yield pb.AdapterMessage(hello=pb.AdapterHello(  # C1.2
            vendor="ros2", adapter_version="0.1.0",
            protocol_version="fleet_adapter.v1", contract_version=CONTRACT_VERSION,
            namespace="default"))
        while True:
            msg = self._outbox.get()
            if msg is None:
                return
            yield msg

    def run_control_stream(self, stub) -> None:
        with self._robot_lock:
            self._robot.spawn(self._robot_id)
        # The edge loop blocks until an endpoint is advertised (C8), so starting it
        # now is safe; it stays idle for zones with no edge node.
        threading.Thread(target=self._edge_loop, daemon=True).start()

        drop_window = self._sim.stream_drop_window() if self._sim else None
        if drop_window is None:
            # No scenario-driven drop: one logical ControlStream session, but resilient.
            # An UNEXPECTED transport drop (a demo port-forward blip, a manager restart)
            # raises RpcError from the response iterator; log and RE-OPEN with a fresh
            # Hello/Register rather than dying with a traceback. The gRPC channel
            # auto-heals once connectivity returns.
            #
            # The conformance harness faults this stream ON PURPOSE — twice: the
            # accepting-reconnect probe (C6.1/C2.2) and the contract-version refusal
            # probe (C13.2/C14.1). Without the re-open below the process exited on the
            # first fault, so all four of those checks could only ever be skipped
            # ("adapter did not redial within 5.0s") and the run printed an alarming
            # but expected traceback.
            while not self._shutdown.is_set():
                try:
                    # Fresh outbox per session. The dying generator is not stopped the
                    # instant its stream faults — it can still be blocked in get() — and
                    # with a shared queue it swallows the RegisterRobot the NEW session
                    # enqueues on hello_ack, so the control plane never sees the
                    # re-registration §9.2.5(b) requires.
                    self._outbox = queue.Queue()
                    for cp in stub.ControlStream(self._control_requests()):
                        self._handle_control_plane(cp, stub)
                except grpc.RpcError as exc:
                    if self._shutdown.is_set():
                        break
                    print(f"control stream dropped ({exc.code()}); reconnecting", file=sys.stderr)
                    time.sleep(0.5)
                    continue
                break  # clean close (response iterator ended without error)
            self._shutdown.set()
            return

        # comms-flaky: reconnect across a scenario-driven stream drop. Each pass is a
        # full Hello/Register handshake, exercising the RegisterRobot reconnect and the
        # persisted fencing/lease state (C3/C4) across the outage.
        while not self._shutdown.is_set():
            self._reconnect_pending = False
            call = stub.ControlStream(self._control_requests())
            self._active_call = call
            try:
                for cp in call:
                    self._handle_control_plane(cp, stub)
            except grpc.RpcError as exc:  # the drop surfaces as a CANCELLED RPC
                if not self._reconnect_pending:
                    raise
                print(f"control stream dropped ({exc.code()}); reconnecting", file=sys.stderr)
            finally:
                self._active_call = None
            if self._shutdown.is_set() or not self._reconnect_pending:
                break
            self._await_reconnect()
        self._shutdown.set()

    def _handle_control_plane(self, cp, stub) -> None:
        pb = self._pb
        kind = cp.WhichOneof("payload")
        if kind == "hello_ack":
            if not cp.hello_ack.accepted:
                raise RuntimeError(f"handshake rejected: {cp.hello_ack.message}")
            self._outbox.put(pb.AdapterMessage(  # C2.1: register
                register=pb.RegisterRobot(robot_id=self._robot_id)))
            self._send_capabilities_snapshot()  # scenario hardware manifest, if any
            # A fresh ControlStream: the next payload must be a full snapshot (C6.1).
            self._snapshot_pending = True
            if not self._telemetry_started:  # C6: stream telemetry once registered
                self._telemetry_started = True
                threading.Thread(target=self._telemetry_loop, args=(stub,), daemon=True).start()
        elif kind == "register_ack":
            for e in cp.register_ack.edge_endpoints:  # C8.1: dial each advertised edge node
                self._edge_addr.put(e.address)
            # C2.3 — adopt the per-robot cadence the control plane returned. Only a
            # POSITIVE value is a cadence: 0 means "not specified" and a negative value
            # is malformed — clamping that to the 1s minimum would let a buggy control
            # plane force the fastest permitted rate. Legitimate values are clamped to
            # the protocol's 1-30s range.
            secs = cp.register_ack.telemetry_interval_seconds
            if secs > 0:
                self._telemetry_interval = max(1.0, min(30.0, float(secs)))
        elif kind == "heartbeat":
            self._outbox.put(pb.AdapterMessage(
                heartbeat=pb.HeartbeatResponse(robot_id=cp.heartbeat.robot_id)))
        elif kind == "command":
            self._handle_command(cp.command)

    def _handle_command(self, command) -> None:
        pb = self._pb
        result = pb.CommandResult(command_id=command.command_id, robot_id=command.robot_id)
        which = command.WhichOneof("command")
        if which == "assign_action":
            self._on_assign(command.assign_action, result)
        elif which == "renew_lease":
            self._on_renew(command.renew_lease, result)
        elif which == "cancel_action":
            cancelled = self._current_action
            with self._robot_lock:
                self._robot.command_stop(self._robot_id)
            self._lease.release(self._robot_id)
            self._current_action = ""
            self._last_task_state = ""
            self._last_progress = -1
            self._current_fencing_token = None
            result.cancel_action.CopyFrom(pb.CancelActionResult(
                acknowledged=True,
                disposition=pb.CANCEL_DISPOSITION_STOPPED_SAFELY))
            if cancelled:
                self._emit_action_status(cancelled, task_map.TASK_CANCELLED, None)
        elif which == "pause":
            self._on_pause(command.pause, result)
        elif which == "resume":
            self._on_resume(command.resume, result)
        else:
            # C7.1: decline every optional command not implemented.
            # TODO(vendor): implement any optional commands your robots support.
            result.unsupported = True
        self._outbox.put(pb.AdapterMessage(command_result=result))

    def _on_pause(self, pause, result) -> None:
        """Map Swarmada pause → Nav2 goal cancel-and-hold (the base halts, the target
        is retained for resume). require_stop_before_ack acks paused=true only once the
        base is CONFIRMED at rest from odom (is_stopped), never inferred."""
        pb = self._pb
        with self._robot_lock:
            self._robot.command_pause(self._robot_id)
            stopped = self._robot.is_stopped(self._robot_id)
        paused = stopped if pause.require_stop_before_ack else True
        result.pause.CopyFrom(pb.PauseResult(
            paused=paused,
            message="nav2: goal cancelled + hold" if paused else "nav2: pausing (awaiting confirmed rest)"))

    def _on_resume(self, resume, result) -> None:
        """Map Swarmada resume → re-dispatch the retained NavigateToPose goal."""
        pb = self._pb
        with self._robot_lock:
            self._robot.command_resume(self._robot_id)
        result.resume.CopyFrom(pb.ResumeResult(resumed=True, message="nav2: goal re-dispatched"))

    def _on_assign(self, task, result) -> None:
        pb = self._pb
        decision = self._fence.check(
            self._robot_id, task.HasField("fencing_token"), task.fencing_token, task.assignment_id)
        if decision is FenceDecision.MISSING:
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False, rejection=pb.ASSIGN_ACTION_REJECTION_MISSING_FENCING_TOKEN))
            return
        if decision is FenceDecision.STALE:
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False, rejection=pb.ASSIGN_ACTION_REJECTION_STALE_FENCING_TOKEN))
            return
        goal = task_map.goal_from_assign(task.destination, task.payload_json)  # FleetTask → Nav2 goal
        with self._robot_lock:
            self._robot.command_move(self._robot_id, goal.x, goal.y, goal.yaw)
        self._current_action = task.action_id
        if task.lease_duration_ms:
            self._current_generation = task.lease_generation
            self._lease.grant(self._robot_id, task.lease_duration_ms / 1000.0, task.lease_generation)
        result.assign_action.CopyFrom(pb.AssignActionResult(
            accepted=True, accepted_fencing_token=task.fencing_token))  # C3.5
        # Task now RUNNING; echo the executing fencing_token so the control plane can
        # detect a robot still acting on a superseded assignment (C6.4).
        token = task.fencing_token if task.HasField("fencing_token") else None
        self._current_fencing_token = token
        self._last_task_state = task_map.TASK_RUNNING
        self._last_progress = 0
        self._emit_action_status(task.action_id, task_map.TASK_RUNNING, token)

    def _on_renew(self, renew, result) -> None:
        pb = self._pb
        renewed = self._lease.renew(
            self._robot_id, renew.lease_duration_ms / 1000.0, renew.lease_generation)  # C4.3
        if renewed:
            self._current_generation = renew.lease_generation
        running = self._current_action == renew.action_id and self._current_action != ""  # C4.4
        result.renew_lease.CopyFrom(pb.RenewActionLeaseResult(
            renewed=renewed, running=running, current_generation=self._current_generation))

    def _emit_action_status(self, action_id: str, state_name: str,
                          fencing_token: int | None, progress_pct: int = 0) -> None:
        """Send a ActionStatusUpdate (AdapterMessage.action_status, field 6). state_name is
        a task.TASK_* name bound here to the proto TaskState enum. When set, the
        executing fencing_token is echoed for superseded-assignment detection (C6.4)."""
        pb = self._pb
        upd = pb.ActionStatusUpdate(
            action_id=action_id, state=getattr(pb, state_name), progress_pct=progress_pct)
        if fencing_token is not None:
            upd.fencing_token = fencing_token
        self._outbox.put(pb.AdapterMessage(action_status=upd))

    def _self_stop(self, robot_id: str) -> None:
        with self._robot_lock:  # C4.2
            self._robot.command_stop(robot_id)
        # Report the stop as a TERMINAL action_status before dropping the action.
        # Without this the control plane sees only silence after the lease lapses,
        # which is indistinguishable from a robot that merely stopped reporting —
        # so dual execution cannot be ruled out and C4.2 fails. STOPPED (not
        # FAILED) because a lease self-stop is a clean commanded stop.
        stopped = self._current_action
        if stopped:
            self._emit_action_status(stopped, task_map.TASK_STOPPED,
                                     self._current_fencing_token)
        self._current_action = ""

    def _telemetry_loop(self, stub) -> None:
        pb = self._pb
        while not self._shutdown.is_set():
            self._lease.tick()  # C4 self-stop check
            with self._robot_lock:
                self._robot.tick(_TICK_DT)  # advances the scenario clock too
                x, y, floor = self._robot.pose(self._robot_id)
                battery = self._robot.battery_percent(self._robot_id)  # battery-edge curve
            self._maybe_estop_drill()   # estop-drill: confirmed stop on a timer
            self._maybe_stream_drop()   # comms-flaky: tear the stream down on a timer
            # comms-flaky: withhold telemetry during a gap / while the stream is down.
            if self._sim is not None and self._sim.telemetry_suppressed():
                time.sleep(self._telemetry_interval)
                continue
            phase = (pb.RobotPhase.ROBOT_PHASE_IN_PROGRESS if self._current_action
                     else pb.RobotPhase.ROBOT_PHASE_IDLE)
            self._outbox.put(pb.AdapterMessage(telemetry=pb.TelemetryPayload(  # C6
                robot_id=self._robot_id, timestamp_ms=int(time.time() * 1000),
                position=pb.RobotPosition(x=x, y=y, floor=floor),  # C6.2 explicit presence
                battery=pb.BatteryStatus(percent=battery), phase=phase,
                current_action=self._current_action,
                hardware=self._hardware_updates())))  # hardware-fault, delta-compressed
            self._maybe_emit_running_progress()  # intermediate progressPct while running
            self._maybe_emit_task_progress()     # stream Nav2 goal completion / failure (C6.4)
            time.sleep(self._telemetry_interval)

    def _maybe_emit_running_progress(self) -> None:
        """Stream an intermediate RUNNING ActionStatusUpdate carrying progressPct derived
        from the NavigateToPose feedback distance, so the control-plane
        FleetTask.status.progressPct climbs during the task. Throttled: only emitted
        when progress advances by ≥5 points (RA-1: not per telemetry tick). Terminal
        transitions are handled by _maybe_emit_task_progress. A binding without
        task_progress (the $0 SimulatedRobot) is a no-op."""
        if not self._current_action or not hasattr(self._robot, "task_progress"):
            return
        with self._robot_lock:
            pct = self._robot.task_progress(self._robot_id)
        if pct - self._last_progress >= 5 and pct < 100:  # 100 is the terminal path's job
            self._emit_action_status(
                self._current_action, task_map.TASK_RUNNING, self._current_fencing_token,
                progress_pct=pct)
            self._last_progress = pct

    def _maybe_emit_task_progress(self) -> None:
        """Stream a TERMINAL task-status transition derived from the Nav2 goal status
        (C6.4). The binding reports SUCCEEDED (NavigateToPose SUCCEEDED) / FAILED
        (ABORTED) via task_state; on that transition the adapter reports it once —
        echoing the executing fencing token — and clears the task. RUNNING is already
        emitted on assign, and a CANCELED goal is reported via the cancel path, so only
        SUCCEEDED / FAILED are streamed here. A binding without goal state (the $0
        SimulatedRobot) is a no-op, so the conformance path is unchanged."""
        if not self._current_action or not hasattr(self._robot, "task_state"):
            return
        with self._robot_lock:
            state = self._robot.task_state(self._robot_id)
        if state in (task_map.TASK_SUCCEEDED, task_map.TASK_FAILED) and state != self._last_task_state:
            self._emit_action_status(
                self._current_action, state, self._current_fencing_token,
                progress_pct=100 if state == task_map.TASK_SUCCEEDED else 0)
            self._last_task_state = state
            self._current_action = ""            # terminal: the goal completed/failed
            self._current_fencing_token = None

    # ── Scenario mapping (proto only; schedule/state live on the binding) ────────

    def _hw_status_enum(self, status: str):
        pb = self._pb
        return {
            "HEALTHY": pb.HARDWARE_STATUS_HEALTHY,
            "DEGRADED": pb.HARDWARE_STATUS_DEGRADED,
            "FAILED": pb.HARDWARE_STATUS_FAILED,
        }.get(status, pb.HARDWARE_STATUS_UNSPECIFIED)

    def _hardware_updates(self) -> list:
        """TelemetryPayload.hardware for this tick — the components the scenario
        changed since the last payload (all on the first). Empty with no scenario."""
        if self._sim is None:
            return []
        pb = self._pb
        # C6.1 — clearing the delta baseline makes every component look changed, so
        # the whole inventory is sent on the first payload of a new stream.
        if self._snapshot_pending:
            self._sim.reset_hardware_baseline()
            self._snapshot_pending = False
        return [pb.HardwareStatusUpdate(
            component_name=hs.name, status=self._hw_status_enum(hs.status),
            degradation_reason=hs.reason) for hs in self._sim.hardware_delta()]

    def _send_capabilities_snapshot(self) -> None:
        """Emit the scenario's full hardware manifest once after registration."""
        if self._sim is None:
            return
        pb = self._pb
        components = [pb.HardwareComponent(
            name=hs.name, type=hs.type, model=hs.model,
            status=self._hw_status_enum(hs.status), degradation_reason=hs.reason)
            for hs in self._sim.manifest_hardware()]
        if not components:
            return
        self._outbox.put(pb.AdapterMessage(capabilities=pb.CapabilitiesSnapshot(
            robot_id=self._robot_id, hardware=components,
            snapshot_ms=int(time.time() * 1000))))

    def _maybe_estop_drill(self) -> None:
        """estop-drill: once due, bring the robot to a REAL confirmed stop via
        confirm_estop (never inferred) and report the EstopAck. Fires once."""
        if self._sim is None or self._estop_drilled or not self._sim.estop_due():
            return
        self._estop_drilled = True
        pb = self._pb
        # stop_initiated_at is stamped around the call that actually commands the
        # hardware, so it attests WHEN the stop was issued. C5.3 requires it:
        # reporting STOPPED with nothing attesting a hardware stop is exactly the
        # "inferred, not confirmed" case the estop protocol forbids.
        initiated_ms = int(time.time() * 1000)
        with self._robot_lock:
            state = confirm_estop(self._robot, self._robot_id)  # C5 discipline: never faked
        self._current_action = ""  # safe-hold: drop the task, like a real estop
        self._safety_outbox.put(pb.AdapterSafetyMessage(
            robot_id=self._robot_id, estop_ack=pb.EstopAck(
                estop_id=f"drill-{self._robot_id}",
                stop_initiated_at=initiated_ms,
                state=(pb.ESTOP_STATE_STOPPED if state == ESTOP_STOPPED
                       else pb.ESTOP_STATE_FAILED),
                message="estop drill (scenario)")))

    def _maybe_stream_drop(self) -> None:
        """comms-flaky: at the drop time, cancel the ControlStream RPC so it tears
        down; run_control_stream then waits out the outage and reconnects."""
        if self._sim is None or self._reconnect_pending or not self._sim.stream_down():
            return
        self._reconnect_pending = True
        call = self._active_call
        if call is not None:
            call.cancel()           # raises CANCELLED in the control loop
        self._outbox.put(None)      # unblock the request generator so it returns

    def _await_reconnect(self) -> None:
        """Block until the scenario's reconnect time (from the binding's clock), then
        return so the control loop re-establishes the stream. The LeaseMonitor keeps
        ticking meanwhile — an un-renewed lease self-stops (C4) during the outage."""
        window = self._sim.stream_drop_window() if self._sim else None
        if window is None:
            return
        _, reconnect_at = window
        while not self._shutdown.is_set():
            if self._sim.elapsed() >= reconnect_at:
                return
            time.sleep(_TELEMETRY_INTERVAL)

    # ── SafetyStream (C5) ────────────────────────────────────────────────────

    def run_safety_stream(self, stub) -> None:
        pb = self._pb
        outbox = self._safety_outbox  # shared so the estop-drill can push an ack too

        def requests() -> Iterator:
            while True:
                msg = outbox.get()
                if msg is None:
                    return
                yield msg

        for cp in stub.SafetyStream(requests()):
            if cp.WhichOneof("payload") == "estop":
                # See the drill path: stamped around the hardware stop so the ACK
                # carries evidence the stop was commanded, not merely asserted (C5.3).
                initiated_ms = int(time.time() * 1000)
                with self._robot_lock:
                    state = confirm_estop(self._robot, self._robot_id)  # C5: confirmed, never inferred
                outbox.put(pb.AdapterSafetyMessage(
                    robot_id=cp.robot_id, estop_ack=pb.EstopAck(
                        estop_id=cp.estop.estop_id,
                        stop_initiated_at=initiated_ms,
                        state=(pb.ESTOP_STATE_STOPPED if state == ESTOP_STOPPED
                               else pb.ESTOP_STATE_FAILED))))

    # ── EdgeStream (C8) ──────────────────────────────────────────────────────

    def _edge_loop(self) -> None:
        pb = self._pb
        addr = self._edge_addr.get()  # blocks until an edge endpoint is advertised (C8.1)
        # TODO(vendor): use grpc.secure_channel with mTLS, same discipline as ControlStream.
        estub = self._pb_grpc.EdgeServiceStub(grpc.insecure_channel(addr))
        outbox: queue.Queue = queue.Queue()
        # C8.2: tee one PositionFrame so the edge node has a pose to evaluate, fire-and-forget.
        with self._robot_lock:
            x, y, floor = self._robot.pose(self._robot_id)
        outbox.put(pb.AdapterEdgeMessage(position=pb.PositionFrame(
            robot_id=self._robot_id,
            position=pb.RobotPosition(x=x, y=y, floor=floor))))

        def requests() -> Iterator:
            while True:
                msg = outbox.get()
                if msg is None:
                    return
                yield msg

        for ec in estub.EdgeStream(requests()):
            if ec.WhichOneof("msg") == "estop":  # C8.3: same confirmed discipline as C5
                initiated_ms = int(time.time() * 1000)
                with self._robot_lock:
                    state = confirm_estop(self._robot, self._robot_id)
                outbox.put(pb.AdapterEdgeMessage(estop_ack=pb.EstopAck(
                    estop_id=ec.estop.estop_id,
                    stop_initiated_at=initiated_ms,
                    state=(pb.ESTOP_STATE_STOPPED if state == ESTOP_STOPPED
                           else pb.ESTOP_STATE_FAILED))))


def main() -> None:
    ap = argparse.ArgumentParser(description="ROS 2 / Nav2 Fleet Adapter")
    ap.add_argument("--endpoint", default="localhost:9090")
    ap.add_argument("--robot-id", default="robot-conformance-1")
    ap.add_argument("--binding", default="simulated", choices=["simulated", "ros2"],
                    help="robot binding: 'simulated' ($0, no ROS 2) or 'ros2' (rclpy + Nav2)")
    ap.add_argument("--scenario", default="healthy-fleet",
                    help="simulated-binding scenario preset name or path (default: healthy-fleet)")
    ap.add_argument("--fault-component", default="camera_front",
                    help="hardware-fault: component to degrade (default: camera_front)")
    ap.add_argument("--fault-at", type=float, default=30.0,
                    help="hardware-fault: seconds until the component degrades (default: 30)")
    ap.add_argument("--recover-at", type=float, default=90.0,
                    help="hardware-fault: seconds until the component recovers (default: 90)")
    args = ap.parse_args()

    from fleet_adapter.v1 import fleet_adapter_pb2 as pb
    from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc

    if args.binding == "ros2":
        from fleet_adapter_ros2.nav2_binding import Nav2Binding
        robot = Nav2Binding()  # ROS 2 / Nav2 (sub-items 2-4)
    else:
        # Scenarios are additive tooling, not a safety requirement (ADR-0005): if
        # scenario support is unavailable (e.g. PyYAML missing in a minimal conformance
        # env) run the $0 skeleton. A bad --scenario *name* is still a hard error.
        engine = None
        try:
            from fleet_adapter_ros2.scenarios import (
                HardwareFaultOverrides, ScenarioEngine, load_scenario)
        except ImportError as exc:
            print(f"scenario support unavailable ({exc}); running without a scenario. "
                  f"Install pyyaml to enable --scenario.", file=sys.stderr)
        else:
            scenario = load_scenario(args.scenario, HardwareFaultOverrides(
                component=args.fault_component, fault_at=args.fault_at, recover_at=args.recover_at))
            engine = ScenarioEngine(scenario)
        robot = SimulatedRobot(engine)  # $0 skeleton; scenario-driven when an engine is set
    adapter = fleet_adapter_ros2Adapter(pb, pb_grpc, robot, args.robot_id)

    # TODO(vendor): use grpc.secure_channel with mTLS credentials (C1.3).
    channel = grpc.insecure_channel(args.endpoint)
    stub = pb_grpc.FleetAdapterServiceStub(channel)
    threading.Thread(target=adapter.run_safety_stream, args=(stub,), daemon=True).start()  # C1.1
    adapter.run_control_stream(stub)


if __name__ == "__main__":
    main()
