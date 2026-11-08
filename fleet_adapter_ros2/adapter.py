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
import contextlib
import json
import math
import os
import pathlib
import queue
import signal
import sys
import threading
import time
from collections.abc import Iterator

import grpc

# Make the package importable when run directly as a script (a real install
# `pip install -e .` makes this unnecessary).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from swarmada_sdk.safety import (  # noqa: E402
    ESTOP_FAILED,
    ESTOP_STOPPED,
    FenceDecision,
    FenceGuard,
    LeaseMonitor,
    await_estop_confirmation,
    command_estop,
)

# An SDK from before `FenceGuard.adopt` cannot honour the RegisterAck's fencing token (C3.8).
# Refuse to start on one, rather than fail on the first RegisterAck with a robot already bound.
if not hasattr(FenceGuard, "adopt"):
    raise ImportError("swarmada_sdk.safety.FenceGuard has no adopt(): this adapter needs a "
                      "swarmada-sdk that can adopt the control plane's fencing token")

from fleet_adapter_ros2 import explore, handover, manual_drive
from fleet_adapter_ros2 import task as task_map  # noqa: E402
from fleet_adapter_ros2.robot import SimulatedRobot  # noqa: E402
from fleet_adapter_ros2.watchdog import (  # noqa: E402
    ODOMETRY_SOURCES,
    resolve_odometry_source,
    resolve_response_bound,
    resolve_stall_limits,
    rest_evidence_gap,
)

_TELEMETRY_INTERVAL = 0.2
_TICK_DT = 0.2

# A confirmed stop's wall-clock budget: 40 looks at the robot, 0.05 s apart (2.0 s), the
# SDK's confirm_estop default. Rest not confirmed within it is ESTOP_FAILED (C5.2), and a lease
# self-stop's FAILED (C4.2).
_ESTOP_TICK_S = 0.05
_ESTOP_TICKS = 40
# How often the lease is looked at (C4.2): a lapse is noticed at most this long after it
# happens. Not a plant property and not the telemetry cadence, which the control plane sets
# anywhere in 1-30 s and which used to be how late a lapse could be noticed.
_LEASE_TICK_S = 0.05
# How long a stop path waits for the navigation server to answer a goal sent before the stop
# (RobotBinding.stop_settled). The same 2.0 s budget. Not a plant property: it decides only
# whether a report can say "stopped" yet, and waiting less can only make it more cautious.
_STOP_SETTLE_S = 2.0
# The wait before a stream that failed is dialled again: the control stream and the SafetyStream
# alike, so neither comes back long before the other.
_REDIAL_S = 0.5
# A SafetyStream counts as up once the control plane's response headers have come on it: it sends
# them when it serves the stream (registered for estops, under a verified identity), and a call
# returns before the server has accepted anything, so nothing earlier proves an estop can reach the
# robot. One that ends sooner than this after it opened was refused or never reached: its loss is
# said once, not at every redial.
_SAFETY_BRIEF_S = 0.5
# How long a dialled SafetyStream may go without those headers. Until they come the estop path
# stays closed and motion is refused; not come within this, the call is cancelled, that is said
# once, and the stream is dialled again.
_SAFETY_HEADERS_S = 5.0

# The end cause an action ended on a CancelAction carries, by the cancel's cause. Only the
# causes that name an end cause outright: SUPERSEDED needs the superseding action's id, which
# a CancelAction does not carry, and REVOCATION (a lease revoked after a connectivity loss)
# is not a lapse this adapter observed -- neither can be stated without guessing, which the
# proto forbids, so they stay UNSPECIFIED and the message says why.
_END_CAUSE_BY_CANCEL_CAUSE = {
    "CANCEL_CAUSE_OPERATOR": "ACTION_END_CAUSE_CANCELLED_BY_OPERATOR",
    "CANCEL_CAUSE_CAPABILITY_LOSS": "ACTION_END_CAUSE_CAPABILITY_LOSS",
}


# The fleet-adapter CONTRACT version this adapter implements (ADR-0032). Distinct from
# adapter_version (this build) and protocol_version (the wire package identity, not a semver, which
# is why it cannot express compatibility). A control plane refuses REGISTRATION for an adapter that
# reports nothing or reports out of range, so this is not optional -- telemetry, heartbeat and
# emergency stop keep working either way. Bump it only when this adapter is re-qualified against a
# new contract (`make conformance`), and update the adapters/REGISTRY.md row with it.
#
# 1.3.0 IS A CLAIM, AND THIS IS WHAT MAKES IT TRUE. The control plane implements 1.3.0 and
# accepts ">=1.2.0 <1.4.0" (N and N-1 minor, ADR-0032); the conformance harness derives the same
# window. This adapter declared 1.0.0 until it met the obligations below, and was refused
# registration VERSION_MISMATCH for as long as the window excluded it. Raising the string alone
# would have claimed a contract it did not implement.
#
# Every 1.1.0 and 1.2.0 change was additive: new fields on existing messages and new optional
# Command arms. Most need nothing from an adapter that ignores what it does not know. 1.2.0's do,
# and this adapter meets each:
#
#   * FA-R5 / FA-R6 -- an assignment carrying an operating rule it cannot honour, or whose subject
#     it does not recognise, is DECLINED OPERATING_CONSTRAINT_UNMET naming every subject
#     (`_unhonourable`, C17.1 / C17.2). It honours no subject and declares none, which FA-R1 says
#     is not a breach (`_HONOURED_SUBJECTS`, C17.3). To read the rule at all, the vendored
#     fleet_adapter.v1 stubs carry AssignAction.operating_constraints: an adapter on stubs that
#     predate it cannot see a rule, so cannot decline one.
#   * ScanCapabilities is answered with a full CapabilitiesSnapshot carrying the action catalog
#     (`_capabilities_snapshot`, C12.1).
#   * The RegisterAck's authoritative action state is reconciled and its fencing token adopted
#     (`_reconcile_action_state`, C2.2 / C3.8). C3.8 is graded advisory at N-1; it is met anyway.
#
#   * 1.3.0 adds one thing, the zone_constraints push (ADR-0059, FA-R7). It is answered rather
#     than declined (`zone_constraints_ack`): an empty set is acknowledged as permission, a set
#     carrying rules is acknowledged with every subject named unhonourable, and a set with
#     unresolved zones is not acknowledged, since nothing could have applied rules nobody could
#     state. This adapter declined the push, and declared 1.2.0, until it did this; declaring
#     1.3.0 while declining it would have asserted an adoption it had not made.
#
# 1.3.0 is the version a 1.3.0 control plane implements; it stays inside the window until the
# control plane implements 1.5.0, so re-qualify before then.
#
# What no version here covers: C3.7, the fencing high-water mark surviving a process restart. The
# mark is held in memory, as in every reference adapter, and the harness records C3.7 SKIP.
CONTRACT_VERSION = "1.3.0"

# The operating subjects this adapter can honour while a robot is in a zone that restricts one
# (FA-R1). EMPTY, and that is a conformant answer, not a gap: an adapter declaring none is not
# in breach -- its robots are simply withheld from zones restricting a subject. Nothing here can
# switch a camera or microphone off, stop water, chemical or UV dispensing, or hold a speed or
# noise cap for as long as the robot is inside a zone: RobotBinding drives a base to a pose and
# nothing else. Claiming a subject it does not enforce would be a FALSE ATTESTATION, which
# ADR-0058 makes both a conformance failure and a security event.
_HONOURED_SUBJECTS: frozenset[str] = frozenset()


def _unhonourable(constraints, honoured) -> list[str]:
    """The subjects in `constraints` this adapter cannot honour (FA-R5 and FA-R6).

    One predicate covers both obligations, as in the reference simulation adapter. A subject
    the adapter does not RECOGNISE is not in `honoured` either, so it lands here exactly as a
    recognised subject it cannot honour does. An unrecognised constraint is not an absent
    constraint: treating it as absent would make the namespaced extension form
    (`example.com/...`) a universal bypass.
    """
    return [c.subject for c in constraints if c.subject not in honoured]

# The Command arms that act on the robot (its motion, its lease, a hold, an install or a
# stream) or vouch that it can take work (validate_action). Each is refused when
# Command.robot_id is not the robot this adapter serves (`_refuse_unserved`), as the reference
# simulation adapter refuses them. `scan`, `zone_constraints` and the three `verify_*` arms are
# not listed: they inspect or acknowledge and move nothing, and a scan names its own robot in
# CapabilitiesSnapshot.robot_id. Listed arms this adapter does not implement stay declined.
_ROBOT_ACTING_ARMS = frozenset({
    "assign_action", "renew_lease", "cancel_action", "pause", "resume", "zone_admission",
    "validate_action", "activate_stream", "deactivate_stream", "push_firmware", "model_update",
})


def zone_constraints_ack(pb, push) -> object:
    """Answer the zone_constraints push (FA-R7): the operating rules in force where the
    robot now is. This adapter honours no subject, so the answer is a receipt that says so.

    * An empty set is permission (a robot that has left a restricted region), acknowledged
      with nothing named -- never read as a lockout.
    * A set carrying rules is acknowledged with every subject named unhonourable. The
      robot is already in the zone; naming what it cannot comply with is all it can do, and
      the control plane can report it but not prevent it.
    * A set with unresolved zones is incomplete, and the claim fails closed: nothing here
      acknowledges having applied rules nobody could state. The zones are named, and the
      subjects that were known are still listed. The robot is not stopped: the push
      arrives after the crossing and does not make a zone safe (ADR-0059).
    """
    subjects = _unhonourable(push.operating_constraints, _HONOURED_SUBJECTS)
    unresolved = list(push.unresolved_zones)
    if unresolved:
        return pb.ZoneConstraintsAck(
            acknowledged=False, unhonourable_subjects=subjects,
            message="unresolved zone(s) " + ", ".join(unresolved) + ": their rules are "
                    "unknown, so this adapter cannot claim to have applied them")
    return pb.ZoneConstraintsAck(
        acknowledged=True, unhonourable_subjects=subjects,
        message=("this adapter honours none of these subjects; the robot is in a zone it "
                 "cannot comply with") if subjects else "")


# The action catalog this adapter serves (RFC-0001 SupportedAction):
# (type, required_capabilities, [(param, kind, unit, required)]).
#
# THE ONE SOURCE. It is advertised on every CapabilitiesSnapshot, and `_refusal` below judges
# every assignment and every validate_action against it, so what the adapter says it serves and
# what it accepts cannot drift apart.
#
# ONE entry, because the adapter serves one thing: a Nav2 NavigateToPose goal from the
# payload's `x`, `y` and `yaw`, in the `map` frame. A named destination is not offered, and
# neither is any frame but `map`: RobotBinding.command_move takes only x, y and yaw, so the
# goal is always sent in `map` and nothing else could reach it.
_SUPPORTED_ACTIONS = [
    ("Navigate", ["navigation.2d"],
     [("x", "number", "m", True), ("y", "number", "m", True), ("yaw", "number", "rad", False)]),
]

# The actions a binding with an exploration stack serves beside Navigate (explore.py): Custom
# with operation explore-floor, and Charge. Declared only when the binding has a stack, so a
# binding that cannot run them does not advertise them. Their parameters are one record each
# (explore.Param), read by the catalog below and by explore.refusal alike.
_STACK_ACTIONS = [
    ("Custom", ["navigation.2d"], explore.EXPLORE_PARAMS, explore.EXPLORE_DESCRIPTION,
     explore.EXPLORE_RESULTS),
    ("Charge", [], explore.CHARGE_PARAMS, explore.CHARGE_DESCRIPTION, ()),
]
_JOB_TYPES = frozenset(t for t, *_ in _STACK_ACTIONS)

# The one frame a goal is sent in. A payload may say so (`"frame": "map"`) or say nothing.
_GOAL_FRAME = "map"


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _refusal(action_type: str, payload_json: bytes, destination: str = "",
             limits: explore.BatteryLimits | None = None,
             teleop_topic: str | None = None, hands_over: bool = False) -> str:
    """Why this adapter cannot serve an action, or "" when it can. Judged against
    `_SUPPORTED_ACTIONS` alone, for AssignAction (INVALID_ACTION) and validate_action alike.

    * A type the catalog does not list is refused, never run as the nearest verb it has
      (RFC-0001: an adapter "rejects the assignment"). The message is the reference simulation
      adapter's.
    * An EMPTY type (proto3's unset string: no type was stated) is refused for the same reason,
      whatever the catalog holds. A control plane never sends one (FleetAction.spec.type is a
      required enum), and guessing which verb was meant is the nearest-neighbour reading the
      catalog exists to prevent.
    * A required parameter that is absent, or a parameter that is not a finite number, is
      refused. NO COORDINATE IS DEFAULTED: a Navigate with no target used to become (0, 0) in
      `map`, an unrequested drive to the map origin.
    * A named destination is refused: this adapter has nothing that can resolve one, and the
      deprecated `destination` field is not in the catalog. It is refused beside coordinates
      too, since a message naming two targets names one this adapter cannot read.
    * A payload `frame` other than `map` is refused. The goal is always sent in `map`, so
      `{"x": 1, "y": 2, "frame": "odom"}` would be a drive to (1, 2) in `map`: on hardware
      `odom` drifts from `map`, and that is a drive to the wrong place. An absent frame is
      `map`; any other value, `null` or a case variant included, is not.

    Custom and Charge are judged by `explore.refusal` when the binding has an exploration stack
    (``limits`` is its battery limits), and are not in the catalog when it has none. A Custom
    naming ``operation: manual-drive`` is judged by `manual_drive.refusal` against the binding's
    teleop topic (``teleop_topic``; None when it forwards none), whatever the stack, and one naming
    ``operation: navigate-handover`` by `handover.refusal`, which also needs a binding that can
    turn its gate to the operator (``hands_over``).
    """
    if handover.is_navigate_handover(action_type, payload_json):
        if destination:
            return (f"destination {destination!r} is not served: {action_type} takes its "
                    f"parameters in the payload")
        return handover.refusal(payload_json, teleop_topic, hands_over)
    if manual_drive.is_manual_drive(action_type, payload_json):
        if destination:
            return (f"destination {destination!r} is not served: {action_type} takes its "
                    f"parameters in the payload")
        return manual_drive.refusal(payload_json, teleop_topic)
    if limits is not None and action_type in _JOB_TYPES:
        if destination:
            return (f"destination {destination!r} is not served: {action_type} takes its "
                    f"parameters in the payload")
        try:
            doc = json.loads(payload_json or b"{}")
        except (ValueError, UnicodeDecodeError):
            doc = None
        if not isinstance(doc, dict):
            return "payload is not a JSON object"
        return explore.refusal(action_type, doc, limits)
    params = {t: p for t, _, p in _SUPPORTED_ACTIONS}.get(action_type)
    if params is None:
        return "unsupported action type: " + (action_type or "(none stated)")
    if destination:
        return (f"destination {destination!r} is not served: a named destination cannot be "
                f"resolved here, and the target travels in the payload as x/y")
    try:
        payload = json.loads(payload_json or b"{}")
    except (ValueError, UnicodeDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return "payload is not a JSON object"
    if "frame" in payload and payload["frame"] != _GOAL_FRAME:
        return (f"payload frame {json.dumps(payload['frame'])} is not served: goals are sent "
                f"in the {_GOAL_FRAME!r} frame only")
    for name, _kind, unit, required in params:  # every kind in the catalog is "number"
        if name not in payload:
            if required:
                return f"payload has no {name} ({unit}); {action_type} requires it"
        elif not _is_finite_number(payload[name]):
            return f"payload {name} is not a finite number ({unit})"
    return ""


class fleet_adapter_ros2Adapter:  # noqa: N801  (generated class name)
    def __init__(self, pb, pb_grpc, robot, robot_id: str, *, namespace: str = "default",
                 artifacts: explore.ArtifactDir | None = None, job_clock=None,
                 keepouts_ref: str | None = None,
                 provisioning_interval_s: float | None = None, mapview=None,
                 handover_rest_s: float = _ESTOP_TICK_S * _ESTOP_TICKS,
                 handover_ready_s: float = handover.READY_S) -> None:
        self._pb = pb
        self._pb_grpc = pb_grpc
        self._robot = robot
        # Serialises the adapter's ordinary calls into the binding (dispatch, pause, resume,
        # the telemetry reads). NO STOP PATH TAKES IT -- estop, the lease self-stop, cancel and
        # reconcile call the binding directly, which RobotBinding requires to be safe -- and it
        # is never held across anything that waits: no dispatch waits for the navigation
        # server (the binding parks the goal instead), and no stop's confirmation sleeps
        # under it. So nothing a stop needs can be held by something that blocks.
        self._robot_lock = threading.Lock()
        self._robot_id = robot_id
        self._fence = FenceGuard()                              # C3
        # C4. The lease is looked at by its own thread (`_lease_loop`), never at the telemetry
        # cadence. `_lease_lock` guards the lease table, and pairs it with the action it
        # belongs to: an assignment's dispatch, its becoming the current action and its lease
        # grant happen in one step under it, and a lapse is recorded -- with the action that
        # held the lease, and its lowest-level stop commanded -- in one step under it. Nothing
        # under it waits; the self-stop's confirmation and report run after it is released,
        # so a renewal or an assignment never waits behind a stop in progress. Order: this
        # lock, then `_robot_lock`, never the other way.
        self._lease_lock = threading.Lock()
        self._lapsed: list[str] = []
        self._lease = LeaseMonitor(on_expiry=self._lapsed.append)
        self._current_action = ""
        # The action a cancel has stopped. It stays the current action, with its lease, until
        # rest is confirmed (`_on_cancel`), but it is no longer executing, so a renewal says
        # running=false for it (C4.4).
        self._halted_action = ""
        # The action a cancel has already reported a TERMINAL state for. A cancelled action
        # ends once, so a retried cancel does not report a second, contradicting end.
        self._ended_on_cancel = ""
        self._current_generation = 0
        self._current_fencing_token = None   # echoed on task-status streaming (C6.4)
        self._last_task_state = ""           # last streamed TaskState, for transition detect
        self._last_progress = -1             # last streamed progress_pct, to throttle updates
        self._edge_addr: queue.Queue = queue.Queue()  # advertised edge endpoints (C8)
        self._outbox: queue.Queue = queue.Queue()
        self._safety_outbox: queue.Queue = queue.Queue()  # → SafetyStream (acks + drills)
        # The SafetyStream, the estop path (C1.1). `main()` runs it (`start_safety_stream`), and
        # from then on the robot moves only while it is up: registration waits for it, and while
        # it is down the robot is stopped and every assignment and resume is declined.
        # `_safety_gen` counts the streams dialled; `_safety_flapping` keeps a stream refused at
        # every redial from being announced, lost and up, at every redial.
        self._safety_required = False
        self._safety_up = threading.Event()
        self._safety_lock = threading.Lock()
        self._safety_gen = 0
        self._safety_flapping = False
        # A run of dials the control plane has not served (no headers): said once, not each time.
        self._safety_unserved = False
        self._shutdown = threading.Event()
        self._telemetry_started = False
        # Whether the binding could observe the robot on the previous telemetry cycle, so
        # the degrade/recover transitions are logged once each instead of 5x a second.
        self._transport_was_live = True
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
        # Reports that must wait for a stop to settle (cancel, reconcile) are made from here,
        # in the order their stops were commanded, never on the control thread: a message read
        # behind a cancel -- a renewal, a second cancel, an assignment -- is never held up by it.
        self._stop_reports: queue.Queue = queue.Queue()
        self._stop_reporter: threading.Thread | None = None
        # explore-floor and Charge. A binding with no exploration stack declares neither (a
        # test double that predates the method has none). The running job, if any, is the
        # current action's; every stop path halts it through the binding's command_stop.
        self._namespace = namespace
        self._stack = getattr(robot, "explore_stack", lambda: None)()
        self._limits = self._stack.battery_limits() if self._stack is not None else None
        # manual-drive: the topic the binding forwards teleop from, through its gate. None (a
        # binding that forwards none) declares no manual-drive and refuses every one.
        self._teleop_topic = getattr(robot, "teleop_topic", lambda: None)()
        # navigate-handover: also needs a binding that turns its armed gate to the operator, under
        # its own stop lock, when Nav2 declines (`hand_over_teleop`).
        self._hands_over = (bool(self._teleop_topic) and hasattr(robot, "hand_over_teleop")
                            and hasattr(robot, "hold_for_handover"))
        # navigate-handover: how long, on the job's clock, a declined goal waits for the robot to
        # be confirmed at rest before the gate turns; past it the action ends Failed and the gate
        # never turns. By default the estop's confirmed-stop budget (`handover.REST_S`).
        self._handover_rest_s = handover_rest_s
        # navigate-handover on a binding that takes the rest from the operator's ready: how long,
        # on the job's clock, a declined goal waits for it; past it the action ends Failed and
        # the gate never turns (`handover.READY_S`).
        self._handover_ready_s = handover_ready_s
        self._artifacts = artifacts or explore.ArtifactDir(None, "adapter-local")
        self._job_clock = job_clock or explore.RealClock()
        self._job = None
        # The threads of the explore-floor segments started, while they run: a segment ends
        # what it launched on the robot when its thread ends, and the shutdown ends it for one
        # still running, since the process will not wait for that thread.
        self._segments: list[threading.Thread] = []
        # The keep-out set every Navigate goal is checked against: the one rendered from the
        # applied zones, named with its digest. Read afresh, and proven, on every check.
        self._keepouts_ref = keepouts_ref
        # The map-view feed (mapview/emitter.py), when one was asked for: it watches each
        # explore-floor segment from its own thread through the stack's read accessors. No stop
        # path calls it or waits for it.
        self._mapview = mapview
        # C2.1's provisioning loop: a robot the control plane has no Robot for is announced
        # (DiscoverRobot) and registered again after an interval, until an operator admits it.
        self._provisioning_override = provisioning_interval_s
        self._provisioning_s = _provisioning_interval(0, provisioning_interval_s)
        self._admitted = True
        self._admission_said = False     # whether this admission's ADMITTED line was printed
        self._catalog_unseen = False

    # ── ControlStream ────────────────────────────────────────────────────────

    def _control_requests(self) -> Iterator:
        pb = self._pb
        yield pb.AdapterMessage(hello=pb.AdapterHello(  # C1.2
            vendor="ros2", adapter_version="0.1.0",
            protocol_version="fleet_adapter.v1", contract_version=CONTRACT_VERSION,
            namespace=self._namespace))
        while True:
            msg = self._outbox.get()
            if msg is None:
                return
            yield msg

    def bring_up_robot(self) -> None:
        """Bring the robot binding up, then spawn the robot on it.

        THE ONE PLACE the adapter drives `RobotBinding.start()`. It is called
        unconditionally -- no `hasattr`, no `--binding` branch -- because the Protocol
        declares the lifecycle (robot.py) and a binding with nothing to bring up says so
        with an explicit no-op.

        `start()` runs BEFORE `spawn()`, and that order is load-bearing rather than
        stylistic: `spawn` is the first thing the adapter asks OF the robot, and on
        `Nav2Binding` nothing can be asked until the rclpy node, the three subscriptions
        and the action client exist. Spawning first would be addressing a transport that
        is not up yet.

        Separate from :meth:`run_control_stream` so the bring-up the shipped entry point
        performs can be driven WITHOUT a control plane or a gRPC stub -- which is what
        `testbed/r1_smoke.py` needs, and the reason it needs it matters: that harness used
        to call `binding.start()` itself. It was the only thing in the tree that did, so it
        exercised a started binding while the adapter did not, and R1 passed every
        assertion for months against a transport the shipped adapter never opened.
        The harness now calls THIS, so there is exactly one bring-up
        implementation and the harness cannot be green while the adapter is inert.

        A failure here is fatal on purpose: `start()` raising stops the run, rather than
        letting the adapter register and stream plausible telemetry (pose `(0,0,0)`,
        battery `0`, `is_stopped` permanently False) read off a binding connected to
        nothing."""
        with self._robot_lock:
            self._robot.start()
            self._robot.spawn(self._robot_id)

    def run_control_stream(self, stub) -> None:
        self.bring_up_robot()
        # The lease is watched from the start, by its own thread, whatever the streams do.
        threading.Thread(target=self._lease_loop, daemon=True).start()
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
                    time.sleep(_REDIAL_S)
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
            self._register_when_safe(self._outbox)  # C2.1, once the SafetyStream is up (C1.1)
            # A fresh ControlStream: the next payload must be a full snapshot (C6.1).
            self._snapshot_pending = True
            if not self._telemetry_started:  # C6: stream telemetry once registered
                self._telemetry_started = True
                threading.Thread(target=self._telemetry_loop, args=(stub,), daemon=True).start()
        elif kind == "register_ack":
            # C2.1: register FIRST, and announce for admission only when the control plane
            # says there is no Robot for this robot yet. Any other refusal is a real one and is
            # not answered by announcing: that would be reaching for a robot this adapter was
            # told it may not touch.
            if not cp.register_ack.accepted:
                if cp.register_ack.rejection == pb.REGISTRATION_REJECTION_NOT_ADMITTED:
                    self._admitted = False
                    self._admission_said = False
                    self._catalog_unseen = True   # a catalog sent before admission is dropped
                    self._outbox.put(pb.AdapterMessage(discover=self._discover_message()))
                    outbox = self._outbox
                    t = threading.Timer(self._provisioning_s,
                                        lambda: self._provisioning_iteration(outbox))
                    t.daemon = True
                    t.start()
                    print(f"NOT_ADMITTED robot={self._robot_id}; announced for an operator to "
                          f"admit, next register in {self._provisioning_s:.0f}s (C2.1)",
                          file=sys.stderr)
                else:
                    print(f"register refused robot={self._robot_id}: "
                          f"{cp.register_ack.message}", file=sys.stderr)
                return
            if not self._admission_said:
                # Said once per admission: the first accepted registration, or the first after a
                # NOT_ADMITTED. The robot's name only, never an address.
                self._admission_said = True
                print(f"ADMITTED robot={self._robot_id}: "
                      + ("an operator admitted it; " if not self._admitted else "")
                      + "registration accepted (C2.1)", file=sys.stderr)
            self._admitted = True
            if self._catalog_unseen:
                self._catalog_unseen = False
                self._send_capabilities_snapshot()
            self._reconcile_action_state(cp.register_ack)  # C2.2, C3.8
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
        elif kind == "discover_ack":
            # Staged, not admitted: an operator admits it, and the loop picks that up. The
            # ack's TTL sets the loop's interval.
            self._provisioning_s = _provisioning_interval(cp.discover_ack.ttl_seconds,
                                                          self._provisioning_override)
            print(f"DISCOVERED robot={self._robot_id}: {cp.discover_ack.message}",
                  file=sys.stderr)
        elif kind == "heartbeat":
            self._outbox.put(pb.AdapterMessage(
                heartbeat=pb.HeartbeatResponse(robot_id=cp.heartbeat.robot_id)))
        elif kind == "command":
            self._handle_command(cp.command)

    def _register_when_safe(self, outbox) -> None:
        """Register on this control stream, and push the catalog, once the SafetyStream is up: a
        reconnection counts only when both streams are back. Waits on its own thread, and gives
        up when the session it was for has been replaced."""
        pb = self._pb

        def register() -> None:
            outbox.put(pb.AdapterMessage(register=pb.RegisterRobot(robot_id=self._robot_id)))
            self._send_capabilities_snapshot()  # the catalog, and scenario hardware if any

        if not self._safety_required or self._safety_up.is_set():
            register()
            return
        print("control stream up; registering once the SafetyStream is up too (C1.1)",
              file=sys.stderr)

        def wait() -> None:
            while not self._shutdown.is_set() and outbox is self._outbox:
                if self._safety_up.wait(0.05):
                    if outbox is self._outbox:
                        register()
                    return
        threading.Thread(target=wait, daemon=True, name="register-when-safe").start()

    def _no_estop_path(self) -> str:
        """Why the robot may not move now, or "": it may not while its SafetyStream is down."""
        if self._safety_required and not self._safety_up.is_set():
            return ("the SafetyStream is down: no estop can reach this robot, so it does not "
                    "move until the stream is back")
        return ""

    def _provisioning_iteration(self, outbox) -> None:
        """Register again, unless admitted since, shut down, or on a stream since replaced
        (whose own hello registers)."""
        if self._admitted or self._shutdown.is_set() or outbox is not self._outbox:
            return
        outbox.put(self._pb.AdapterMessage(register=self._pb.RegisterRobot(
            robot_id=self._robot_id)))

    def _discover_message(self):
        """What an operator needs to decide the admission: the identity and the capabilities
        this robot claims, from the same catalog the snapshot carries. Nothing is said that
        the adapter does not know: no firmware, no MAC."""
        caps = sorted({c for _, cs, _p in _SUPPORTED_ACTIONS for c in cs})
        return self._pb.DiscoverRobot(robot_id=self._robot_id, manufacturer="ROS 2",
                                      model="Nav2", reported_capabilities=caps)

    def _reconcile_action_state(self, ack) -> None:
        """C2.2 — adopt the control plane's authoritative action state from a RegisterAck.

        The control plane is the source of truth across a connectivity gap. A task may have
        been cancelled while this adapter could not hear about it, and resuming it on
        reconnect would drive the robot to a goal nobody wants any more. The rules are the
        reference simulation adapter's:

        * IN_PROGRESS for the action being executed: keep executing, under the same lease.
        * IN_PROGRESS for any other action, or naming none: a divergence, treated as CANCELLED.
        * CANCELLED, UNKNOWN or IDLE: stop the robot through the same `command_stop` the
          cancel and lease self-stop paths use (on Nav2: cancel the goal and publish a zero
          Twist), release the lease, drop the action, and report CANCELLED, UNKNOWN or
          STOPPED for the plane's action id, or for the local one when the plane names none.
          With no action id on either side there is nothing to report.
        * REVOKING or UNSPECIFIED: nothing here. A REVOKING lease is left to lapse (C4).

        Only an accepted RegisterAck is read (a refused registration hands this adapter no
        robot), and nothing is read when the state is absent.

        C3.8 comes first, whatever the phase. `fencing_token` is the token this robot should
        be executing under, and "the adapter MUST reject any command with a lower token"
        (fleet_adapter.proto). `FenceGuard.check` only raises the mark from an AssignAction
        this adapter accepted, so without this, after a control-plane failover, the adapter
        keeps honouring commands the dead plane authorised: two executors on one robot.
        `adopt` is monotonic, so a zero, equal or lower token (a replayed or reordered
        RegisterAck) leaves the mark where it is; a higher one also drops the assignment held
        at the old mark. The mark is held in memory only, so a restart forgets it (C3.7)."""
        pb = self._pb
        if not ack.accepted or not ack.HasField("authoritative_action_state"):
            return
        st = ack.authoritative_action_state
        if st.fencing_token:
            self._fence.adopt(self._robot_id, st.fencing_token)
        phase = st.phase
        if phase == pb.ROBOT_ACTION_PHASE_IN_PROGRESS:
            if st.action_id and st.action_id == self._current_action:
                return  # matches local state; keep executing under the same lease
            phase = pb.ROBOT_ACTION_PHASE_CANCELLED  # divergence: treat as cancelled
        state = {pb.ROBOT_ACTION_PHASE_CANCELLED: task_map.TASK_CANCELLED,
                 pb.ROBOT_ACTION_PHASE_UNKNOWN: task_map.TASK_UNKNOWN,
                 pb.ROBOT_ACTION_PHASE_IDLE: task_map.TASK_STOPPED}.get(phase)
        if state is None:
            return
        self._robot.command_stop(self._robot_id)
        with self._lease_lock:
            self._lease.release(self._robot_id)
            reported = st.action_id or self._current_action
            self._current_action = ""
        self._last_task_state = ""
        self._last_progress = -1
        self._current_fencing_token = None
        token = self._fence.highest(self._robot_id) or None

        def report(settled: bool) -> None:
            if not reported:
                return
            # The plane's phase says the action is over, not why it ended: no end cause can be
            # stated without guessing, so it stays unset and the message says what is known.
            message = (f"reconciled on registration to the control plane's authoritative "
                       f"phase {pb.RobotActionPhase.Name(st.phase)}; the cause is not known here")
            status = state
            if not settled:
                status = task_map.TASK_UNKNOWN
                message = self._UNSETTLED + " " + message
            self._emit_action_status(reported, status, token, message=message)
        self._when_settled(report)

    # Said by every stop path whose stop has not settled: the report must not claim a stop.
    _UNSETTLED = ("the stop was commanded, but a goal sent to the navigation server before it "
                  "has not been answered: if accepted it is cancelled on arrival, and until "
                  "then the robot is not known to be stopped.")

    def _when_settled(self, report) -> None:
        """Call ``report(settled)`` once the stop just commanded has settled -- the binding
        says nothing the stop was meant to end can still start moving the robot -- or once
        _STOP_SETTLE_S has passed without that. Cancel and reconcile report from this, so
        neither can report a stop while a goal sent before it may still be accepted.

        At once, on the caller's thread, when the stop has already settled, which is the
        common case. Otherwise the wait and the report are handed to the stop reporter thread
        (`_report_stops`), so the control thread goes on reading at once."""
        if self._robot.stop_settled(self._robot_id, 0):
            report(True)
            return
        self._defer_stop_report(
            lambda: report(self._robot.stop_settled(self._robot_id, _STOP_SETTLE_S)))

    def _when_confirmed(self, report) -> None:
        """Call ``report(confirmed, settled, why_not)`` once rest after the stop just commanded
        is confirmed by the estop's rule (`_confirm_stop`), or once its budget has passed
        without that. The cancel reports from this, as the estop and the lease self-stop do, so
        no stop path says the robot stopped before it is observed at rest.

        The first look is taken here, on the caller's thread, and never sleeps: rest already
        observed, or a transport that cannot be read, is reported at once. Otherwise the looks
        and the report are handed to the stop reporter thread, so the control thread goes on
        reading while the robot brakes."""
        confirmed, settled, why = self._confirm_stop(looks=1)
        if confirmed or why:
            report(confirmed, settled, why)
            return
        self._defer_stop_report(lambda: report(*self._confirm_stop()))

    def _defer_stop_report(self, job) -> None:
        self._stop_reports.put(job)
        if self._stop_reporter is None:
            self._stop_reporter = threading.Thread(target=self._report_stops, daemon=True)
            self._stop_reporter.start()

    def _report_stops(self) -> None:
        """The stop reporter: one deferred report at a time, in the order the stops were
        commanded. Each waits at most one stop budget (2.0 s); no lock is held while it waits."""
        while True:
            job = self._stop_reports.get()
            try:
                job()
            except Exception as exc:  # noqa: BLE001 - a boundary: later reports must still go
                print(f"a stop's report raised: {exc!r}", file=sys.stderr)

    def _handle_command(self, command) -> None:
        pb = self._pb
        result = pb.CommandResult(command_id=command.command_id, robot_id=command.robot_id)
        which = command.WhichOneof("command")
        if command.robot_id != self._robot_id and which in _ROBOT_ACTING_ARMS:
            self._refuse_unserved(command, which, result)
        elif which == "assign_action":
            self._on_assign(command.assign_action, result)
        elif which == "renew_lease":
            self._on_renew(command.renew_lease, result)
        elif which == "cancel_action":
            self._on_cancel(command.cancel_action, result)
            return  # answered by _on_cancel, once its stop has settled
        elif which == "pause":
            self._on_pause(command.pause, result)
        elif which == "resume":
            self._on_resume(command.resume, result)
        elif which == "scan":
            # C12.1: a required command. Declining it leaves the control plane without this
            # adapter's action catalog, the pre-dispatch gate when validate_action is declined.
            result.scan.CopyFrom(self._capabilities_snapshot())
        elif which == "zone_constraints":  # FA-R7, contract 1.3.0
            result.zone_constraints.CopyFrom(zone_constraints_ack(pb, command.zone_constraints))
        elif which == "validate_action":
            # Answered from the same check assignment uses, as the reference simulation adapter
            # answers it from the same list it advertises. Pure inspection: nothing moves.
            reason = (self._no_estop_path()
                      or _refusal(command.validate_action.action_type,
                               command.validate_action.payload_json, limits=self._limits,
                               teleop_topic=self._teleop_topic, hands_over=self._hands_over)
                      or self._geometry_refusal(command.validate_action.action_type,
                                                command.validate_action.payload_json))
            result.validate_action.CopyFrom(pb.ValidateActionResult(
                servable=not reason, message=reason))
        else:
            # C7.1: decline every optional command not implemented.
            # TODO(vendor): implement any optional commands your robots support.
            result.unsupported = True
        self._outbox.put(pb.AdapterMessage(command_result=result))

    def _refuse_unserved(self, command, which: str, result) -> None:
        """Refuse a command addressed to a robot this adapter does not serve.

        One ControlStream carries the commands for every Robot bound to this adapter, and
        Command.robot_id selects which one
        (RFC-0001 #fleet-adapter-protocol-the-controlstream-service). This adapter serves
        exactly one. Running a command addressed to any other id on it moves the wrong robot: an
        assignment for a second Robot object bound to the same adapter sent a Nav2 goal on this
        one, and a cancel for it stopped this one's action. So nothing here reaches the robot,
        the fence or the lease, and each arm is answered with the refusal the contract already
        defines for it, as the reference simulation adapter answers it. An assignment is
        declined ROBOT_UNAVAILABLE: the robot it names is not available through this adapter,
        and the control plane releases the assignment to reschedule it. INVALID_ACTION would say
        the action could not be served, which is not true of the action.
        """
        pb = self._pb
        why = (f"robot_id {command.robot_id!r} is not served by this adapter, which serves "
               f"{self._robot_id!r}; nothing was done")
        if which == "assign_action":
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False, rejection=pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE,
                message=why))
        elif which == "renew_lease":
            result.renew_lease.CopyFrom(pb.RenewActionLeaseResult(
                renewed=False, running=False, message=why))
        elif which == "cancel_action":
            result.cancel_action.CopyFrom(pb.CancelActionResult(acknowledged=False, message=why))
        elif which == "pause":
            result.pause.CopyFrom(pb.PauseResult(paused=False, message=why))
        elif which == "resume":
            result.resume.CopyFrom(pb.ResumeResult(resumed=False, message=why))
        elif which == "validate_action":
            result.validate_action.CopyFrom(pb.ValidateActionResult(servable=False, message=why))
        else:
            result.unsupported = True  # C7.1: declined for every robot, this one included
        print(f"UNSERVED_REFUSED robot={self._robot_id} addressed={command.robot_id!r} "
              f"arm={which} command_id={command.command_id}", file=sys.stderr, flush=True)

    def _on_cancel(self, cancel, result) -> None:
        """Stop the action, and say it stopped safely only once the robot is OBSERVED AT REST,
        by the estop's rule and within the estop's budget (`_confirm_stop`). Every acknowledged
        disposition, UNSPECIFIED included, tells the control plane the action halted, and
        STOPPED_SAFELY lets it requeue the action to another robot: acknowledged while this one
        still brakes, or while a goal sent before the cancel may still be accepted, that is two
        executors. So until rest is confirmed the cancel is NOT acknowledged, and says why: the
        lease and the action are kept, the plane can retry, and the lease still self-stops if
        it lapses.

        What rest does NOT decide is whether the ACTION ended. It did, on the cancel, and
        every path here reports that with a terminal ActionStatusUpdate (`_end_on_cancel`,
        C16.1) -- CANCELLED when rest was confirmed, FAILED with the reason when it could not
        be. The refused acknowledgement alone leaves the plane nothing but lease expiry to
        resolve the action with, and on a platform that can never confirm rest that is every
        cancel it sends.

        The stop is commanded here, at once. The CommandResult is sent by `_when_confirmed`:
        at once when the robot is already at rest, otherwise from the stop reporter, so the
        control thread never waits for the answer.

        A cancel naming an action other than the one executing touches nothing. The control
        plane sends one after every in-place preemption: the robot has switched to the
        preemptor, and the preempted action's reconcile asks for that one to stop, by name, at a
        token that is not stale (tokens are counted per action). It is not moving the robot, so
        it is stopped already, and stopping the robot would abort the action that replaced it,
        which the fencing rule forbids (RFC-0001 #fleet-adapter-protocol-the-controlstream-service,
        C3.9). It is answered as the reference simulation adapter answers it, with a terminal
        acknowledgement for the named action, one of the two answers that rule accepts."""
        pb = self._pb
        named = cancel.action_id
        if named and self._current_action and named != self._current_action:
            result.cancel_action.CopyFrom(pb.CancelActionResult(
                acknowledged=True, cause=cancel.cause,
                disposition=pb.CANCEL_DISPOSITION_STOPPED_SAFELY,
                message=f"{named} is not executing on this robot ({self._current_action} is); "
                        f"nothing was stopped"))
            print(f"CANCEL_NOT_RUNNING robot={self._robot_id} action={named} "
                  f"running={self._current_action}", file=sys.stderr, flush=True)
            self._outbox.put(pb.AdapterMessage(command_result=result))
            return
        cancelled = self._current_action
        token = self._current_fencing_token
        self._robot.command_stop(self._robot_id)
        self._halted_action = cancelled

        def report(confirmed: bool, settled: bool, why: str) -> None:
            if not confirmed:
                message = self._unconfirmed(settled, why)
                # The task ENDED here, whatever can be confirmed about the robot, and saying so
                # is not the same claim as acknowledging the cancel. FAILED, never STOPPED: the
                # stop went out but nothing witnessed rest, and the message says which.
                self._end_on_cancel(cancelled, task_map.TASK_FAILED, token, cancel.cause,
                                    message=message)
                result.cancel_action.CopyFrom(pb.CancelActionResult(
                    acknowledged=False, cause=cancel.cause, message=message))
                self._outbox.put(pb.AdapterMessage(command_result=result))
                return
            # Only if it is still the current action: one assigned while this confirmed is not
            # the one cancelled, and neither it nor its lease may be dropped.
            with self._lease_lock:
                current = self._current_action == cancelled
                if current:
                    self._lease.release(self._robot_id)
                    self._current_action = ""
            if current:
                self._last_task_state = ""
                self._last_progress = -1
                self._current_fencing_token = None
            result.cancel_action.CopyFrom(pb.CancelActionResult(
                acknowledged=True, cause=cancel.cause,
                disposition=pb.CANCEL_DISPOSITION_STOPPED_SAFELY))
            self._end_on_cancel(cancelled, task_map.TASK_CANCELLED, token, cancel.cause)
            self._outbox.put(pb.AdapterMessage(command_result=result))
        self._when_confirmed(report)

    def _end_on_cancel(self, action: str, state_name: str, fencing_token: int | None,
                       cause_value, message: str = "") -> None:
        """The TERMINAL ActionStatusUpdate for an action a cancel ended (C16.1).

        Sent on EVERY cancel path, because what ends the action is the cancel, not what could
        afterwards be confirmed about the robot: CANCELLED when rest was confirmed, FAILED with
        the reason when it could not be. Without one the control plane sees only the refused
        acknowledgement and has nothing to resolve the action with but lease expiry -- and on a
        platform that can never confirm rest, that is every cancel it ever sends.

        Once per action. A cancel the plane retries is answered again, but an action that has
        already been reported ended does not end a second time in a different state."""
        if not action or action == self._ended_on_cancel:
            return
        self._ended_on_cancel = action
        cause = self._pb.CancelCause.Name(cause_value)
        end_cause = _END_CAUSE_BY_CANCEL_CAUSE.get(cause, "")
        note = "" if end_cause else (
            f"cancelled by the control plane ({cause}); the end cause cannot be stated from "
            f"it without guessing")
        self._emit_action_status(
            action, state_name, fencing_token, end_cause=end_cause,
            message="; ".join(part for part in (message, note) if part))

    def _on_pause(self, pause, result) -> None:
        """Map Swarmada pause → Nav2 goal cancel-and-hold (the base halts, the target
        is retained for resume). require_stop_before_ack acks paused=true only once the
        base is CONFIRMED at rest from odom (is_stopped), never inferred."""
        pb = self._pb
        job = self._running_job()
        with self._robot_lock:
            if isinstance(job, explore.ExploreJob):
                job.pause()   # the explorer and the return goal stop; the segment waits
            else:
                self._robot.command_pause(self._robot_id)
            stopped = self._robot.is_stopped(self._robot_id)
        paused = stopped if pause.require_stop_before_ack else True
        if paused:
            message = "nav2: goal cancelled + hold"
        else:
            gap = self._robot.rest_evidence_gap(self._robot_id)
            # "awaiting" promises an answer that is coming. On a platform with no channel
            # that can confirm rest it never is, and saying so is the difference between a
            # pause that is slow and one that can never be acknowledged.
            message = (f"nav2: goal cancelled + hold, but the pause cannot be acknowledged "
                       f"stopped: {gap}") if gap else "nav2: pausing (awaiting confirmed rest)"
        result.pause.CopyFrom(pb.PauseResult(paused=paused, message=message))

    def _on_resume(self, resume, result) -> None:
        """Map Swarmada resume → re-dispatch the retained NavigateToPose goal."""
        pb = self._pb
        refused = self._no_estop_path()
        if refused:
            result.resume.CopyFrom(pb.ResumeResult(resumed=False, message=refused))
            return
        job = self._running_job()
        if isinstance(job, manual_drive.ManualDriveJob):
            # A pause closed the forwarding and ended the drive; teleop is never reopened by a
            # resume. The operator takes over again with a new manual-drive.
            result.resume.CopyFrom(pb.ResumeResult(
                resumed=False, message="a manual-drive is not resumed: the pause ended it; "
                                       "assign a new one"))
            return
        with self._robot_lock:
            if isinstance(job, explore.ExploreJob):
                job.resume()  # the segment restarts the phase it was paused in
            else:
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
        # No estop path, no motion: declined, as for a robot that cannot take work now.
        refused = self._no_estop_path()
        if refused:
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False, rejection=pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE,
                message=refused))
            return
        # C17.1 / C17.2 (FA-R5, FA-R6) -- DECLINE an operating rule this adapter cannot honour,
        # naming every subject, never accept and ignore it. After the two fencing branches on
        # purpose: a superseded command is refused STALE so the control plane learns its
        # assignment lost, not refused for a rule that was never the reason it lost.
        unhonourable = _unhonourable(task.operating_constraints, _HONOURED_SUBJECTS)
        if unhonourable:
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False,
                rejection=pb.ASSIGN_ACTION_REJECTION_OPERATING_CONSTRAINT_UNMET,
                unhonourable_subjects=unhonourable))
            return
        # An action this adapter cannot serve is DECLINED INVALID_ACTION ("unparseable /
        # unsupported"), never run as navigation to a defaulted target. After the operating-rule
        # decline: a rule binds whatever the action is, and C17 grades that reason.
        reason = (_refusal(task.action_type, task.payload_json, task.destination, self._limits,
                           teleop_topic=self._teleop_topic, hands_over=self._hands_over)
                  or self._geometry_refusal(task.action_type, task.payload_json))
        operator_verb = (manual_drive.OPERATION
                         if manual_drive.is_manual_drive(task.action_type, task.payload_json)
                         else handover.OPERATION
                         if handover.is_navigate_handover(task.action_type, task.payload_json)
                         else "")
        if not reason and not task.lease_duration_ms and operator_verb:
            reason = (f"{operator_verb} needs a lease: teleop is forwarded only while "
                      f"the action holds one, and this assignment carries none")
        if reason:
            result.assign_action.CopyFrom(pb.AssignActionResult(
                accepted=False, rejection=pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION,
                message=reason))
            return
        # Task now RUNNING; echo the executing fencing_token so the control plane can
        # detect a robot still acting on a superseded assignment (C6.4).
        token = task.fencing_token if task.HasField("fencing_token") else None
        if task.action_type in _JOB_TYPES:
            self._start_job(task, token, result)
            return
        goal = task_map.goal_from_assign(task.destination, task.payload_json)  # FleetTask → Nav2 goal
        # One step against the lease: either this assignment's lease is granted before a lapse
        # is looked for (so the old lease cannot lapse on it), or the lapse and its stop come
        # first and this goal is dispatched after them. Never a stop meant for the lapsed action
        # landing on this one. Nothing here waits (command_move never does).
        with self._lease_lock:
            self._halt_job()  # a segment the new assignment supersedes stops first
            with self._robot_lock:
                self._robot.command_move(self._robot_id, goal.x, goal.y, goal.yaw)
            self._current_action = task.action_id
            self._halted_action = ""
            self._ended_on_cancel = ""
            self._current_fencing_token = token
            if task.lease_duration_ms:
                self._current_generation = task.lease_generation
                self._lease.grant(self._robot_id, task.lease_duration_ms / 1000.0,
                                  task.lease_generation)
        result.assign_action.CopyFrom(pb.AssignActionResult(
            accepted=True, accepted_fencing_token=task.fencing_token))  # C3.5
        self._last_task_state = task_map.TASK_RUNNING
        self._last_progress = 0
        self._emit_action_status(task.action_id, task_map.TASK_RUNNING, token)

    def _on_renew(self, renew, result) -> None:
        pb = self._pb
        with self._lease_lock:
            renewed = self._lease.renew(
                self._robot_id, renew.lease_duration_ms / 1000.0, renew.lease_generation)  # C4.3
            job = self._job
            if (renewed and isinstance(job, manual_drive.ManualDriveJob)
                    and job.action_id == renew.action_id == self._current_action):
                # The teleop gate's own deadline moves with the lease, in the same step.
                self._robot.extend_teleop(self._robot_id, renew.action_id,
                                          renew.lease_duration_ms / 1000.0)
        if renewed:
            self._current_generation = renew.lease_generation
        # C4.4: executing means assigned and not stopped by a cancel. A cancelled action is kept
        # until rest is confirmed, but the robot is braking, not executing it.
        running = (self._current_action == renew.action_id and self._current_action != ""
                   and self._current_action != self._halted_action)
        result.renew_lease.CopyFrom(pb.RenewActionLeaseResult(
            renewed=renewed, running=running, current_generation=self._current_generation))

    def _geometry_refusal(self, action_type: str, payload_json: bytes) -> str:
        """Why the keep-outs refuse an action this catalog can serve, or "".

        The zones an operator confirmed are the zones the robot obeys: the control plane refuses
        work addressed to an exclusion zone by name, and reads no polygon; this refuses a
        Navigate goal that lies inside one, and any Navigate while the keep-out set it is told to
        obey cannot be proven by its digest. A segment naming a keep-out set it cannot prove is
        refused here too, before it starts.

        A Navigate goal is in the robot's `map` frame, and the set's polygons in the frame of
        the map the zones came from: the check holds while the robot localizes on that map."""
        try:
            doc = json.loads(payload_json or b"{}")
        except (ValueError, UnicodeDecodeError):
            return ""
        if action_type == "Custom" and self._stack is not None and doc.get("keepouts_ref"):
            try:
                explore.load_keepout_set(self._artifacts, doc["keepouts_ref"])
            except LookupError as exc:
                return f"{exc}; nothing moves"
        # A navigate-handover's goal goes to Nav2 as a Navigate's does, and is refused as one.
        if ((action_type == "Navigate" or handover.is_navigate_handover(action_type, doc))
                and self._keepouts_ref):
            try:
                kset = explore.load_keepout_set(self._artifacts, self._keepouts_ref)
            except LookupError as exc:
                return f"{exc}; no Navigate is accepted until it is proven"
            goal = task_map.goal_from_assign("", payload_json)
            why = explore.goal_refusal(goal.x, goal.y, kset)
            if why:
                return f"{why}; refused before any motion"
        return ""

    def _start_job(self, task, token: int | None, result) -> None:
        """Accept an explore-floor, a manual-drive or a Charge and run it on its own thread. The
        lease and the current action are taken in one step, as for a Navigate; nothing here
        waits. A manual-drive's teleop gate is opened in that same step, with the lease's
        duration, and by nothing else."""
        pb = self._pb
        doc = json.loads(task.payload_json or b"{}")
        common = {"action_id": task.action_id, "doc": doc, "stack": self._stack,
                  "limits": self._limits, "clock": self._job_clock}
        feed = None
        manual = manual_drive.is_manual_drive(task.action_type, doc)
        handing = handover.is_navigate_handover(task.action_type, doc)
        if handing:
            job = handover.NavigateHandoverJob(
                action_id=task.action_id, params=handover.parse_payload(doc),
                robot=self._robot, robot_id=self._robot_id, clock=self._job_clock,
                emit=lambda *a, **k: self._job_emit(job, *a, **k), rest_s=self._handover_rest_s,
                ready_s=self._handover_ready_s)
            feed = self._circle_feed(job, task.action_id, handover.OPERATION)
        elif manual:
            job = manual_drive.ManualDriveJob(
                action_id=task.action_id, params=manual_drive.parse_payload(doc),
                robot=self._robot, robot_id=self._robot_id, clock=self._job_clock,
                emit=lambda *a, **k: self._job_emit(job, *a, **k))
            feed = self._circle_feed(job, task.action_id, manual_drive.OPERATION)
        elif task.action_type == "Custom":
            report = lambda *a, **k: self._job_emit(job, *a, **k)
            if self._mapview is not None:
                feed = self._mapview.segment(self._stack, task.action_id)
                report = feed.observe(report)      # each report is passed on first, then noted
            job = explore.ExploreJob(artifacts=self._artifacts, namespace=self._namespace,
                                     emit=report, **common)
        else:
            job = explore.ChargeJob(emit=lambda *a, **k: self._job_emit(job, *a, **k), **common)
        job.fencing_token = token
        with self._lease_lock:
            self._halt_job()
            self._job = job
            self._current_action = task.action_id
            self._halted_action = ""
            self._ended_on_cancel = ""
            self._current_fencing_token = token
            if task.lease_duration_ms:
                self._current_generation = task.lease_generation
                self._lease.grant(self._robot_id, task.lease_duration_ms / 1000.0,
                                  task.lease_generation)
            if manual or handing:
                # Armed for a navigate-handover: it forwards nothing until Nav2 declines and the
                # binding hands it over. Opened first, because opening holds the base and drops
                # any goal; the handover's own goal is sent after, in the same step.
                self._robot.open_teleop(self._robot_id, task.action_id, job.drive,
                                        task.lease_duration_ms / 1000.0)
            if handing:
                goal = task_map.goal_from_assign("", task.payload_json)
                with self._robot_lock:
                    self._robot.command_move(self._robot_id, goal.x, goal.y, goal.yaw)
        result.assign_action.CopyFrom(pb.AssignActionResult(
            accepted=True, accepted_fencing_token=task.fencing_token))  # C3.5
        self._last_task_state = task_map.TASK_RUNNING
        self._last_progress = 0
        self._emit_action_status(task.action_id, task_map.TASK_RUNNING, token,
                                 message=job.drive.message if manual or handing else "")
        thread = threading.Thread(target=job.run, daemon=True, name=f"job-{task.action_id}")
        if isinstance(job, explore.ExploreJob):
            self._segments = [t for t in self._segments if t.is_alive()] + [thread]
        thread.start()
        if feed is not None:
            # It ends itself once this job is no longer the current action, whatever ended it.
            feed.start(alive=lambda: self._running_job() is job)

    def _circle_feed(self, job, aid: str, operation: str):
        """The map-view feed of a navigate-handover's or a manual-drive's circle, or None without
        a map view: the circle's state from this job alone, and the baton from the binding's gate
        record (mapview/handover_feed.py). It reads; it never acts."""
        if self._mapview is None:
            return None
        rid = self._robot_id
        split = explore.split_action_id(aid, self._namespace)
        log = getattr(self._robot, "teleop_log", None)
        return self._mapview.handover(
            job=job, gate_open=lambda: self._robot.teleop_open(rid, aid), action_id=aid,
            label=split[1] if split else aid, last=job.params.last, operation=operation,
            gate_log=(lambda after: log(rid, after)) if log is not None else None)

    def _job_emit(self, job, state_name: str, message: str, progress_pct: int = 0,
                  end_cause: str = "") -> None:
        """A job's report. Said only while the job is the current action and no stop path has
        halted it, in one step with the release a terminal report makes, so a job never
        reports after the cancel, estop or lease lapse that ended it has."""
        terminal = state_name in (task_map.TASK_SUCCEEDED, task_map.TASK_FAILED)
        with self._lease_lock:
            if (job is not self._job or job.halted.is_set()
                    or self._current_action != job.action_id):
                return
            if (isinstance(job, manual_drive.ManualDriveJob)
                    and self._halted_action == job.action_id):
                return  # the cancel that closed its forwarding reports its end
            if terminal:
                if isinstance(job, handover.NavigateHandoverJob):
                    # Its Nav2 target must not outlive it. A pause keeps a target for its resume,
                    # and this action ended on the pause: a resume would drive the robot to it
                    # with no action and no lease. The hold drops it, in the step that releases.
                    self._robot.command_hold(self._robot_id)
                self._lease.release(self._robot_id)
                self._current_action = ""
                self._job = None
        self._emit_action_status(job.action_id, state_name, job.fencing_token,
                                 progress_pct=progress_pct, message=message, end_cause=end_cause)
        if terminal:
            self._last_task_state = state_name
            self._last_progress = -1
            self._current_fencing_token = None

    def _running_job(self):
        job = self._job
        if job is None or job.halted.is_set() or job.action_id != self._current_action:
            return None
        return job

    def _halt_job(self) -> None:
        """End the running job, if any: its explorer and return goal stop, and it reports
        nothing more. Never blocks."""
        job, self._job = self._job, None
        if job is not None and not job.halted.is_set():
            job.halted.set()
            if isinstance(job, manual_drive.ManualDriveJob):
                self._robot.close_teleop(self._robot_id, job.action_id)  # one zero Twist
            else:
                self._stack.halt()

    def shutdown(self) -> None:
        """Stop what this adapter commanded, before the process ends. Every exit from `main()`
        comes here: the control stream closing, an exception out of its loop, SIGTERM, SIGINT.

        In this order, each step on its own, so one that raises is reported and the steps after
        it still run: the base held and the navigation goal cancelled (the binding's hold, with
        its zero Twist); the explorer told to stop, if a segment runs; the running job ended, so
        it commands and reports nothing more; rest looked for, within the estop's budget; and
        the processes a running segment launched ended (SLAM, the keep-out servers, the
        explorer). The caller tears rclpy down after this: every step here needs it. Nothing
        here waits on a lock another thread may hold."""
        self._shutdown.set()
        rid = self._robot_id
        hold = getattr(self._robot, "command_hold", None) or self._robot.command_stop
        job = self._job
        segment = self._stack is not None and any(t.is_alive() for t in self._segments)
        steps = [("holding the base and cancelling the navigation goal", lambda: hold(rid))]
        if segment:
            steps.append(("stopping the explorer", self._stack.stop_explorer))
        if job is not None:
            steps.append(("ending the running job", job.halted.set))
        steps.append(("looking for rest", self._report_rest_at_shutdown))
        if segment:
            steps.append(("ending the processes the segment launched", self._stack.end_segment))
        print("shutdown: stopping the robot before the adapter exits", file=sys.stderr)
        for what, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - a stop path: the next step still runs
                print(f"shutdown: {what} raised {type(exc).__name__}: {exc}; going on with the "
                      f"steps after it", file=sys.stderr)

    def _report_rest_at_shutdown(self) -> None:
        confirmed, settled, why = self._confirm_stop()
        print("shutdown: " + ("the robot is at rest" if confirmed else
                              "rest was not confirmed: " + self._unconfirmed(settled, why)),
              file=sys.stderr)

    def _emit_action_status(self, action_id: str, state_name: str,
                          fencing_token: int | None, progress_pct: int = 0,
                          message: str = "", end_cause: str = "") -> None:
        """Send a ActionStatusUpdate (AdapterMessage.action_status, field 6). state_name is
        a task.TASK_* name bound here to the proto TaskState enum. When set, the
        executing fencing_token is echoed for superseded-assignment detection (C6.4).
        `message` and `end_cause` (an ActionEndCause name) are set only when known."""
        pb = self._pb
        extra = {}
        if message:
            extra["message"] = message
        if end_cause:
            extra["end_cause"] = getattr(pb, end_cause)
        upd = pb.ActionStatusUpdate(
            action_id=action_id, state=getattr(pb, state_name), progress_pct=progress_pct,
            **extra)
        if fencing_token is not None:
            upd.fencing_token = fencing_token
        self._outbox.put(pb.AdapterMessage(action_status=upd))

    def _lease_loop(self) -> None:
        """The lease's own thread: one look every _LEASE_TICK_S, for as long as the adapter
        runs, whatever telemetry, the streams or a stop in progress are doing. A self-stop that
        raises is logged and the thread goes on: a dead lease thread would turn C4.2 off
        without a word."""
        while not self._shutdown.wait(_LEASE_TICK_S):
            try:
                self._tick_lease()
            except Exception as exc:  # noqa: BLE001 - a boundary: the watch must outlive a fault
                print(f"lease self-stop raised; the lease is still being watched: {exc!r}",
                      file=sys.stderr)

    def _tick_lease(self) -> None:
        """One look at the lease (C4.2). Under `_lease_lock`, in one step: the lapse is found,
        the action that held the lease is recorded, and the stop is commanded at the lowest
        level (the binding's command_stop, which never waits). Then, with the lock released,
        each stop is confirmed and reported (`_self_stop`)."""
        with self._lease_lock:
            self._lease.tick()
            lapsed = []
            for robot_id in self._lapsed:
                self._robot.command_stop(robot_id)   # closes a manual-drive's teleop gate too
                if isinstance(self._job, manual_drive.ManualDriveJob):
                    self._job.halted.set()   # the self-stop below reports its end, not the job
                lapsed.append((robot_id, self._current_action, self._current_fencing_token))
            self._lapsed.clear()
        for robot_id, action, token in lapsed:
            self._self_stop(robot_id, action, token)

    def _self_stop(self, robot_id: str, action: str, token: int | None) -> None:
        """C4.2: bring the task to a CONFIRMED stop and report it -- STOPPED only once the
        robot is observed at rest, FAILED when rest cannot be confirmed within the stopping budget.

        The stop was commanded at the lapse (`_tick_lease`), at the lowest level (the binding's
        command_stop: the goal cancelled and a zero Twist), whatever state the goal was in, and
        without the robot lock. Here rest is confirmed by the estop's rule and within the
        estop's budget (`is_stopped`), so the two stops never answer differently for the same
        robot: a goal sent before the stop and not yet answered may still be accepted and
        drive, so it refutes rest whatever odom says -- it is cancelled on arrival, and until
        then the report is FAILED and says why. It is never UNKNOWN, which says nothing and
        which C4.2 does not count.

        Reported for `action`, the one whose lease lapsed, as a TERMINAL action_status. Without
        one the control plane sees only silence after the lease lapses, which is
        indistinguishable from a robot that merely stopped reporting, so dual execution cannot
        be ruled out. An action assigned while this confirms is neither reported nor dropped."""
        confirmed, settled, why = self._confirm_stop()
        if action:
            if confirmed:
                state, message = task_map.TASK_STOPPED, ""
            else:
                state, message = task_map.TASK_FAILED, self._unconfirmed(settled, why)
            self._emit_action_status(action, state, token,
                                     end_cause="ACTION_END_CAUSE_LEASE_EXPIRED", message=message)
        with self._lease_lock:
            if self._current_action == action:
                self._current_action = ""

    def _unconfirmed(self, settled: bool, why: str) -> str:
        """Why a stop's report cannot say the robot stopped, from `_confirm_stop`'s answer."""
        return why or (self._UNSETTLED if not settled else (
            f"the stop was commanded, but rest was not observed within "
            f"{_ESTOP_TICK_S * _ESTOP_TICKS:g} s."))

    def _confirm_stop(self, looks: int | None = None) -> tuple[bool, bool, str]:
        """Confirm rest after a commanded stop: ``(confirmed, settled, why_not)``.

        Up to `looks` looks, _ESTOP_TICK_S apart -- by default the estop's confirmed-stop budget --
        each asking `is_stopped`, the estop's own test. Returns at once when it holds. Otherwise
        `settled` says whether a goal sent before the stop is what kept it from holding. A
        transport that cannot be read is not at rest, and `why_not` says so (C5.3: never rest
        from silence). No adapter lock is held, so nothing waits behind this."""
        rid = self._robot_id
        looks = _ESTOP_TICKS if looks is None else looks
        # The binding's evidence is judged BEFORE the robot is looked at. On a platform whose
        # odometry reproduces the velocity command there is nothing to wait for: that channel
        # reads zero the moment the stop is commanded and would answer "at rest" while the
        # base is still braking (C5.3, ADR-0071). Looking anyway would spend the stopping
        # budget to arrive at the same answer, and a caller could not tell the two apart.
        gap = self._robot.rest_evidence_gap(rid)
        if gap:
            return False, self._robot.stop_settled(rid, 0), gap
        try:
            for look in range(looks):
                self._robot.tick(_ESTOP_TICK_S)
                if self._robot.is_stopped(rid):
                    return True, True, ""
                if look < looks - 1:
                    time.sleep(_ESTOP_TICK_S)
            return False, self._robot.stop_settled(rid, 0), ""
        except RuntimeError as exc:
            print(f"a stop could not be confirmed - the robot binding's transport is not "
                  f"live: {exc}", file=sys.stderr)
            return False, False, (f"the stop was commanded, but rest cannot be read from the "
                                  f"robot: {exc}")

    def _report_transport_not_live(self, detail: str) -> None:
        """Announce a transport that cannot observe the robot, on the TRANSITION only.

        At a 0.2 s cadence an unconditional log is 5 lines a second, which buries the
        diagnosis it is trying to deliver. Logged again if the transport recovers, so the
        operator sees an interval rather than an unbounded complaint."""
        if self._transport_was_live:
            self._transport_was_live = False
            print(f"telemetry DEGRADED to ROBOT_PHASE_OFFLINE: the robot binding cannot "
                  f"observe the robot, so position and battery are being OMITTED rather "
                  f"than sent as defaults (C6.2). {detail}", file=sys.stderr)

    def _telemetry_loop(self, stub) -> None:
        """Stream TelemetryPayload at the registered cadence (C6).

        Every cycle asks the binding whether it can still OBSERVE the robot, and stops
        claiming a position or a battery the moment it cannot. Two things make that
        necessary rather than defensive, and both were seen on an un-started binding:

        * **It was a C6.2 violation.** An un-started or dead binding answers every
          accessor — `pose()` with the origin, `battery_percent()` with 0 — and this loop
          published them at full cadence with a fresh `timestamp_ms`. C6.2 says *"never
          send a defaulted 0/false where the true reading is unknown"*, and a robot
          "parked at (0,0) on a 0% battery" is indistinguishable from a real one.
        * **`tick()` raising is not enough on its own.** It raises ONCE, so the first
          thread to reach it consumes the signal for all the others. `main()` starts the
          safety stream BEFORE `bring_up_robot()`, so an estop arriving in that window
          spends the binding's single raise inside `_confirm_estop` — and this loop, which
          starts later on `hello_ack`, then sees a binding that ticks silently and looks
          healthy. Measured: 0 raises in 200 ticks, publishing `(0,0,0)` and 0% forever.
          A predicate cannot be consumed; an exception can.

        So the payload degrades instead of lying: `phase = ROBOT_PHASE_OFFLINE` with
        `position` and `battery` OMITTED. That is better than the silence the first fix
        settled for, and it does not reverse its principle — nothing false is published
        either way. It is the protocol saying exactly what is wrong, in the field built for
        it, at the cadence the control plane already expects: a gap is ambiguous between a
        crashed adapter, a network fault and a blind one, whereas OFFLINE is the adapter
        reporting "I am up and I cannot see the robot". Proto3 message fields have
        presence, so omitting them is expressible and is what C6.2 asks for.

        The lease is NOT looked at here. It used to be, once per cycle, so a lapse was noticed
        up to one telemetry interval late -- up to 30 s -- and a self-stop held telemetry up
        while it confirmed rest. It has its own thread (`_lease_loop`), which keeps watching
        through a transport outage and the scenario stream drop exactly as this loop did."""
        pb = self._pb
        while not self._shutdown.is_set():
            x = y = floor = battery = None
            with self._robot_lock:
                try:
                    self._robot.tick(_TICK_DT)  # advances the scenario clock too
                    live = self._robot.transport_alive()
                    detail = ""
                except RuntimeError as exc:
                    # tick() diagnosing a transport that never started or has died. Handled
                    # identically to the predicate answering False, so the outcome does not
                    # depend on which thread happened to consume the one-shot raise.
                    live, detail = False, str(exc)
                if live:
                    x, y, floor = self._robot.pose(self._robot_id)
                    battery = self._robot.battery_percent(self._robot_id)  # battery-edge curve
            self._maybe_estop_drill()   # estop-drill: confirmed stop on a timer
            self._maybe_stream_drop()   # comms-flaky: tear the stream down on a timer
            # comms-flaky: withhold telemetry during a gap / while the stream is down.
            if self._sim is not None and self._sim.telemetry_suppressed():
                time.sleep(self._telemetry_interval)
                continue
            if not live:
                self._report_transport_not_live(detail)
                # OFFLINE, with position and battery ABSENT — never defaulted (C6.2).
                self._outbox.put(pb.AdapterMessage(telemetry=pb.TelemetryPayload(
                    robot_id=self._robot_id, timestamp_ms=int(time.time() * 1000),
                    phase=pb.RobotPhase.ROBOT_PHASE_OFFLINE,
                    current_action=self._current_action,
                    hardware=self._hardware_updates())))
                # NOT emitted while blind: both derive from the binding's goal state, which
                # is frozen at whatever it held when the transport stopped. Reporting
                # SUCCEEDED off a frozen GoalStatus tells the control plane an action
                # finished when nothing is known about it — the same class of false terminal
                # report as a superseded goal's result (see on_goal_result).
                time.sleep(self._telemetry_interval)
                continue
            if not self._transport_was_live:
                self._transport_was_live = True
                print("telemetry RECOVERED: the binding can observe the robot again; "
                      "position and battery are being reported.", file=sys.stderr)
            phase = (pb.RobotPhase.ROBOT_PHASE_IN_PROGRESS if self._current_action
                     else pb.RobotPhase.ROBOT_PHASE_IDLE)
            if isinstance(self._running_job(), explore.ChargeJob):
                phase = pb.RobotPhase.ROBOT_PHASE_CHARGING
            self._outbox.put(pb.AdapterMessage(telemetry=pb.TelemetryPayload(  # C6
                robot_id=self._robot_id, timestamp_ms=int(time.time() * 1000),
                position=pb.RobotPosition(x=x, y=y, floor=floor),  # C6.2 explicit presence
                # A battery nothing has reported is absent, never 0 (C6.2).
                battery=(pb.BatteryStatus(percent=battery) if battery is not None else None),
                phase=phase,
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
        if (not self._current_action or not hasattr(self._robot, "task_progress")
                or self._running_job() is not None):  # a job reports its own progress
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
        (ABORTED, or a stall the binding's progress watchdog declared) via task_state; on
        that transition the adapter reports it once — echoing the executing fencing token —
        releases the lease and clears the task, as the cancel path does. RUNNING is already
        emitted on assign, and a CANCELED goal is reported via the cancel path, so only
        SUCCEEDED / FAILED are streamed here. A binding without goal state (the $0
        SimulatedRobot) is a no-op, so the conformance path is unchanged.

        A stall is not a second failure path: it arrives here as FAILED like any other, and
        carries the binding's reason as the message, with the cause EXECUTION_FAILURE."""
        if (not self._current_action or not hasattr(self._robot, "task_state")
                or self._running_job() is not None):
            # A segment's return goal is a Nav2 goal too; its SUCCEEDED is not the segment's.
            return
        with self._robot_lock:
            state = self._robot.task_state(self._robot_id)
            failure = (self._robot.task_failure(self._robot_id)
                       if state == task_map.TASK_FAILED and hasattr(self._robot, "task_failure")
                       else "")
        if state in (task_map.TASK_SUCCEEDED, task_map.TASK_FAILED) and state != self._last_task_state:
            self._emit_action_status(
                self._current_action, state, self._current_fencing_token,
                progress_pct=100 if state == task_map.TASK_SUCCEEDED else 0,
                message=failure,
                # FAILED here is always the goal ended in failure -- aborted or refused by the
                # server, or a stall -- so the cause is known. SUCCEEDED needs none.
                end_cause=("ACTION_END_CAUSE_EXECUTION_FAILURE"
                           if state == task_map.TASK_FAILED else ""))
            # The action is over: its lease must not outlive it and self-stop the robot
            # later, which is the release every other ending (cancel, reconcile) makes.
            with self._lease_lock:
                self._lease.release(self._robot_id)
                self._current_action = ""        # terminal: the goal completed/failed
            self._last_task_state = state
            self._last_progress = -1
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

    def _capabilities_snapshot(self):
        """The full CapabilitiesSnapshot: the hardware manifest (empty with no scenario), the
        action catalog, and nothing this adapter does not have.

        One builder for the scan reply (C12.1) and the push after registration, as in the
        reference simulation adapter, so the robot does not describe itself one way at
        registration and another way when scanned. No installed models, sensor streams or
        firmware state are reported, because none are implemented, and no honoured operating
        subject is declared (see _HONOURED_SUBJECTS)."""
        pb = self._pb
        hardware = [] if self._sim is None else [pb.HardwareComponent(
            name=hs.name, type=hs.type, model=hs.model,
            status=self._hw_status_enum(hs.status), degradation_reason=hs.reason)
            for hs in self._sim.manifest_hardware()]
        actions = [pb.SupportedAction(
            action_type=kind, required_capabilities=caps,
            params=[pb.ActionParam(name=n, kind=k, unit=u, required=r) for n, k, u, r in params])
            for kind, caps, params in _SUPPORTED_ACTIONS]
        jobs = list(_STACK_ACTIONS) if self._stack is not None else []
        if self._teleop_topic:
            jobs = manual_drive.catalog(   # one Custom entry, every operation
                jobs, extra=((handover.OPERATION, handover.DESCRIPTION),) if self._hands_over
                else ())
        if jobs:
            for kind, caps, params, description, results in jobs:
                actions.append(pb.SupportedAction(
                    action_type=kind, required_capabilities=caps, description=description,
                    params=[pb.ActionParam(
                        name=p.name, kind=p.kind, unit=p.unit, required=p.required,
                        allowed=list(p.allowed),
                        **{b: getattr(p, b) for b in ("min", "max") if getattr(p, b) is not None})
                        for p in params],
                    results=[pb.ActionParam(name=n, kind=k, unit=u) for n, k, u in results]))
        return pb.CapabilitiesSnapshot(
            robot_id=self._robot_id, hardware=hardware, supported_actions=actions,
            snapshot_ms=int(time.time() * 1000))

    def _send_capabilities_snapshot(self) -> None:
        """Push the full snapshot right after RegisterRobot, on every binding.

        A v0.4 control plane issues no ScanCapabilities, so this push is the only way the
        action catalog reaches it; admission fails open for as long as no catalog has arrived.
        It was gated on a scenario with hardware, so under the ros2 binding the catalog was
        never sent. The snapshot always carries the catalog, so it is always sent, at the same
        point the reference simulation adapter sends its own."""
        self._outbox.put(self._pb.AdapterMessage(capabilities=self._capabilities_snapshot()))

    def _confirm_estop(self, on_receipt=None) -> str:
        """`confirm_estop`, but a BROKEN TRANSPORT still produces an answer, and a stop that
        is not confirmed at the first look calls ``on_receipt`` before waiting for rest, so the
        caller can send the STOPPING receipt within C5.5's 500 ms. The terminal answer is
        confirm_estop's: STOPPED only from the robot's own ground truth, FAILED otherwise.

        `confirm_estop` polls the binding's `tick()` and `is_stopped()`. Since the adapter
        took over bring-up, `Nav2Binding.tick()` raises when its transport was never started
        or has died —
        which is correct, and on this path alone it would make things worse if it escaped:

        * `run_safety_stream` and `_edge_loop` call this from inside their response loops,
          so an escaping exception kills that thread and the control plane gets **no
          EstopAck at all**. Silence on an estop is strictly less actionable than
          `FAILED`, which at least says "rest was not confirmed — escalate".
        * the safety stream is started BEFORE `bring_up_robot()` runs (see `main`), so
          there is a real window in which an estop can arrive at an un-started binding.
          Losing the safety thread permanently to a startup race would be a worse defect
          than the one being fixed.

        So the exception is caught, reported to stderr with its diagnosis intact, and
        mapped to ESTOP_FAILED. That mapping is not a fudge and not a weakening of C5:
        FAILED is the TRUE answer, because a binding whose transport is not delivering
        odom cannot confirm rest, and C5.3 forbids inferring rest from silence. The only
        thing that changes is that the reason now reaches the operator instead of being
        lost — previously this returned FAILED with no explanation anywhere, which is how
        C5.2 / C8.3 / C14.1 were misattributed for weeks.

        Deliberately NOT a blanket `except Exception`: only the transport's own
        RuntimeError is absorbed. A bug in the safety primitive itself must still crash.
        """
        try:
            command_estop(self._robot, self._robot_id)
            if self._robot.rest_evidence_gap(self._robot_id):
                # The stop IS commanded (above, first, always). What cannot follow is a
                # confirmation: this platform's rest evidence is the command echoed back, so
                # STOPPED would be the inference C5.3 forbids. C5.2's honest terminal answer
                # on such a platform is FAILED, and it is reached without spending the
                # budget looking at a channel that cannot answer.
                if on_receipt is not None:
                    on_receipt()
                return ESTOP_FAILED
            # confirm_estop's first look, unchanged. Rest confirmed now is a single STOPPED,
            # which C5.2 allows only when rest is confirmed within the 500 ms of C5.5.
            if await_estop_confirmation(self._robot, self._robot_id, _ESTOP_TICK_S, 1,
                                        sleep=lambda _s: None) == ESTOP_STOPPED:
                return ESTOP_STOPPED
            # Otherwise the first ack is the receipt, sent now that the stop is commanded:
            # STOPPING, "received and braking; rest not yet confirmed". A goal sent before the
            # stop and not yet answered keeps is_stopped() False until it is cancelled on
            # arrival, so this is the path such a stop takes.
            if on_receipt is not None:
                on_receipt()
            return await_estop_confirmation(self._robot, self._robot_id, _ESTOP_TICK_S,
                                            _ESTOP_TICKS - 1)
        except RuntimeError as exc:
            print(f"estop could not be confirmed - the robot binding's transport is not "
                  f"live, so rest cannot be read from the robot and C5.3 forbids inferring "
                  f"it: {exc}", file=sys.stderr)
            return ESTOP_FAILED

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
        self._estop(f"drill-{self._robot_id}", lambda ack: self._safety_outbox.put(
            pb.AdapterSafetyMessage(robot_id=self._robot_id, estop_ack=ack)),
            message="estop drill (scenario)")  # C5 discipline: never faked
        self._current_action = ""  # safe-hold: drop the task, like a real estop

    def _estop(self, estop_id: str, send, message: str = "") -> None:
        """One estop, acknowledged per C5, for every channel it arrives on (SafetyStream,
        EdgeStream, the drill). A single STOPPED when rest is confirmed at once; otherwise a
        STOPPING receipt as soon as the stop is commanded, then STOPPED or FAILED.

        stop_initiated_at is stamped around the call that actually commands the hardware, so
        it attests WHEN the stop was issued; C5.3 requires it on a STOPPED."""
        pb = self._pb
        initiated_ms = int(time.time() * 1000)

        def ack(state) -> None:
            # A FAILED says why, and on a platform that cannot confirm rest at all the reason
            # is the platform's, not a fault: without it the ack is indistinguishable from a
            # dead transport or a robot that would not brake.
            text = message
            if state == pb.ESTOP_STATE_FAILED:
                gap = self._robot.rest_evidence_gap(self._robot_id)
                if gap:
                    text = f"{message}; {gap}" if message else (
                        f"the stop was commanded, but it cannot be reported confirmed: {gap}")
            send(pb.EstopAck(estop_id=estop_id, stop_initiated_at=initiated_ms, state=state,
                             **({"message": text} if text else {})))
        # No adapter lock: the stop, and every look at the robot after it, go straight to the
        # binding, so an estop never waits behind a dispatch or any other call (C5.1, C5.5).
        state = self._confirm_estop(on_receipt=lambda: ack(pb.ESTOP_STATE_STOPPING))
        ack(pb.ESTOP_STATE_STOPPED if state == ESTOP_STOPPED else pb.ESTOP_STATE_FAILED)

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
        return so the control loop re-establishes the stream. The lease thread keeps
        watching meanwhile — an un-renewed lease self-stops (C4) during the outage."""
        window = self._sim.stream_drop_window() if self._sim else None
        if window is None:
            return
        _, reconnect_at = window
        while not self._shutdown.is_set():
            if self._sim.elapsed() >= reconnect_at:
                return
            time.sleep(_TELEMETRY_INTERVAL)

    # ── SafetyStream (C5) ────────────────────────────────────────────────────

    def start_safety_stream(self, stub) -> None:
        """Run the SafetyStream (C1.1) on its own thread, and from now on move the robot only
        while it is up."""
        self._safety_required = True
        threading.Thread(target=self.run_safety_stream, args=(stub,), daemon=True,
                         name="safety-stream").start()

    def run_safety_stream(self, stub) -> None:
        """The SafetyStream, dialled again after every drop with the control stream's wait, for
        as long as the adapter runs. Each stream has its own outbox, so the request generator of
        a stream that has ended cannot swallow an ack meant for the next. When it ends while the
        adapter runs, the robot is stopped (`_safety_down`); when the adapter is the one closing
        it, to exit, it ends quietly."""
        pb = self._pb
        while not self._shutdown.is_set():
            outbox: queue.Queue = queue.Queue()
            self._safety_outbox = outbox   # the estop-drill's acks go down the stream now open
            with self._safety_lock:
                self._safety_gen += 1
                gen = self._safety_gen

            def requests(outbox=outbox) -> Iterator:
                while True:
                    msg = outbox.get()
                    if msg is None:
                        return
                    yield msg

            why = "the control plane ended it"
            opened = time.monotonic()
            try:
                call = stub.SafetyStream(requests())
                served = self._safety_headers(call)
                if served == "timeout":
                    call.cancel()
                    why = (f"no response headers within {_SAFETY_HEADERS_S:g} s: the control "
                           f"plane has not confirmed it serves this stream")
                    if not self._safety_unserved:
                        print(f"SafetyStream not served: {why}; the estop path stays closed and "
                              f"motion is refused; redialling", file=sys.stderr)
                    self._safety_unserved = True
                    call = ()
                elif served == "served":
                    self._safety_unserved = False
                    self._safety_opened(gen)
                for cp in call:
                    if cp.WhichOneof("payload") == "estop" and cp.robot_id != self._robot_id:
                        self._estop_unserved(cp, outbox)
                    elif cp.WhichOneof("payload") == "estop":
                        robot_id = cp.robot_id  # C5: confirmed, never inferred (see _estop)
                        self._estop(cp.estop.estop_id, lambda ack, r=robot_id, o=outbox: o.put(
                            pb.AdapterSafetyMessage(robot_id=r, estop_ack=ack)))
            except grpc.RpcError as exc:
                code = exc.code()
                why = getattr(code, "name", str(code))
            outbox.put(None)
            if self._shutdown.is_set():
                return  # the adapter closed the channel itself, to exit
            self._safety_down(why, time.monotonic() - opened)
            time.sleep(_REDIAL_S)

    def _estop_unserved(self, cp, outbox) -> None:
        """An estop for a robot this adapter does not serve. It cannot confirm that robot is at
        rest, so the only true answer is FAILED, which escalates; a STOPPED would confirm a stop
        nobody made. This robot is not stopped by it: the estop was addressed to another one.
        Answered under the addressed id, with no stop_initiated_at, since no stop was commanded."""
        pb = self._pb
        print(f"ESTOP_UNSERVED robot={self._robot_id} addressed={cp.robot_id!r} "
              f"id={cp.estop.estop_id}", file=sys.stderr, flush=True)
        outbox.put(pb.AdapterSafetyMessage(
            robot_id=cp.robot_id,
            estop_ack=pb.EstopAck(
                estop_id=cp.estop.estop_id, state=pb.ESTOP_STATE_FAILED,
                message=(f"robot_id {cp.robot_id!r} is not served by this adapter, which "
                         f"serves {self._robot_id!r}; no stop was commanded and none can be "
                         f"confirmed"))))

    @staticmethod
    def _safety_headers(call) -> str:
        """Wait for the SafetyStream's response headers: "served" once they have come on a call
        still open, "ended" when the call ended first (refused, or never reached: its error comes
        out of the call), "timeout" when neither happened within _SAFETY_HEADERS_S."""
        came = threading.Event()

        def wait() -> None:
            with contextlib.suppress(Exception):  # the call ended: done() says so
                call.initial_metadata()
            came.set()
        threading.Thread(target=wait, daemon=True, name="safety-headers").start()
        if not came.wait(_SAFETY_HEADERS_S):
            return "timeout"
        return "ended" if call.done() else "served"

    def _safety_opened(self, gen: int) -> None:
        with self._safety_lock:
            if gen != self._safety_gen or self._shutdown.is_set() or self._safety_up.is_set():
                return
            self._safety_up.set()
        if not self._safety_flapping:
            print("SafetyStream up: the estop path is open", file=sys.stderr)
            return
        # Coming out of a run of refusals: said once it has stayed open. The robot's gating does
        # not wait for this; only the line does.
        steady = threading.Timer(_SAFETY_BRIEF_S, self._safety_steady, args=(gen,))
        steady.daemon = True
        steady.start()

    def _safety_steady(self, gen: int) -> None:
        with self._safety_lock:
            if gen != self._safety_gen or not self._safety_up.is_set():
                return
            self._safety_flapping = False
        print("SafetyStream up: the estop path is open", file=sys.stderr)

    def _safety_down(self, why: str, open_for: float) -> None:
        """The SafetyStream ended while the adapter runs, `open_for` seconds after it opened. No
        estop can reach the robot any more, so it stops now: the base held and its goal
        cancelled, a running segment ended, and the action it was running ended FAILED with the
        reason. Every assignment and resume is declined until the stream is back
        (`_no_estop_path`). Said once per loss, not once per redial."""
        with self._safety_lock:
            was_up = self._safety_up.is_set()
            self._safety_up.clear()
        if not was_up:
            return
        brief = open_for < _SAFETY_BRIEF_S
        if not (brief and self._safety_flapping):
            print(f"SafetyStream lost ({why}): no estop can reach the robot; stopping it and "
                  f"refusing motion until the stream is back; redialling", file=sys.stderr)
        self._safety_flapping = brief
        rid = self._robot_id
        with self._lease_lock:
            self._robot.command_stop(rid)
            self._halt_job()
            action, token = self._current_action, self._current_fencing_token
            if action and action != self._halted_action:
                self._lease.release(rid)
                self._current_action = ""
            else:
                action = ""   # nothing running, or a cancel in progress that reports its own end
        if not action:
            return
        self._last_task_state = task_map.TASK_FAILED
        self._last_progress = -1
        self._current_fencing_token = None

        def report() -> None:
            confirmed, settled, why_not = self._confirm_stop()
            message = (f"stopped: the SafetyStream was lost ({why}) and no estop could reach the "
                       f"robot" + ("" if confirmed else
                                   f"; {self._unconfirmed(settled, why_not)}"))
            self._emit_action_status(action, task_map.TASK_FAILED, token, message=message,
                                     end_cause="ACTION_END_CAUSE_EXECUTION_FAILURE")
        threading.Thread(target=report, daemon=True, name="safety-lost-report").start()

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
                self._estop(ec.estop.estop_id,
                            lambda ack: outbox.put(pb.AdapterEdgeMessage(estop_ack=ack)))


# The provisioning loop's interval when the control plane states no DiscoveredRobot TTL.
_PROVISIONING_INTERVAL_S = 60.0


def _provisioning_interval(ttl_seconds: int, override_s: float | None) -> float:
    """C2.1's interval I: min(ttl/2, 300), 60 s with no TTL; an override only shortens it."""
    rule = min(ttl_seconds / 2.0, 300.0) if ttl_seconds > 0 else _PROVISIONING_INTERVAL_S
    if override_s is None:
        return rule
    return min(override_s, rule) if ttl_seconds > 0 else override_s


def _tls_material(spec: str) -> bytes:
    """A PEM named by a TLS flag: a file, or ``env:NAME`` -- the PEM in this process's
    environment, so a client key handed to an adapter inside a robot's container is never a
    file there. A value that is neither is refused, naming the flag's form, never the value."""
    if spec.startswith("env:"):
        name = spec[4:]
        value = os.environ.get(name, "")
        if not name or not value.strip():
            raise ValueError(f"environment variable {name or '(no name)'} is empty or unset")
        return value.encode()
    with open(spec, "rb") as fh:
        return fh.read()


def _resolve_tls(args) -> str:
    """Why the TLS flags cannot be used as given, or "". The sim adapter's semantics: the CA,
    the certificate and the key together, or none of them."""
    flags = (args.tls_ca, args.tls_cert, args.tls_key)
    if any(flags) and not all(flags):
        return ("--tls-ca, --tls-cert and --tls-key must be supplied together (mTLS needs a CA "
                "to trust, plus a client keypair to present)")
    if args.tls_server_name and not all(flags):
        return "--tls-server-name is used only with --tls-ca, --tls-cert and --tls-key"
    for flag, spec in (("--tls-ca", args.tls_ca), ("--tls-cert", args.tls_cert),
                       ("--tls-key", args.tls_key)):
        if spec:
            try:
                _tls_material(spec)
            except (OSError, ValueError) as exc:
                why = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
                return f"{flag} cannot be read: {why}"
    return ""


def open_control_channel(args, grpc_mod=None):
    """The channel to the control plane's ControlStream and SafetyStream (C1.3).

    With the three TLS flags: mTLS, presenting the client certificate. The control plane reads
    this adapter's name from the certificate's SAN (<adapter>.<namespace>.svc.cluster.local)
    and registers the stream under it; without a verified identity it refuses every message.
    ``--tls-server-name`` verifies the server against the name its certificate carries when
    the endpoint is an address (a forward) no SAN covers. Without TLS flags: plaintext, which
    the conformance harness serves."""
    g = grpc_mod or grpc
    if not args.tls_cert:
        return g.insecure_channel(args.endpoint)
    creds = g.ssl_channel_credentials(root_certificates=_tls_material(args.tls_ca),
                                      private_key=_tls_material(args.tls_key),
                                      certificate_chain=_tls_material(args.tls_cert))
    options = ()
    if args.tls_server_name:
        options = (("grpc.ssl_target_name_override", args.tls_server_name),)
    return g.secure_channel(args.endpoint, creds, options=options)


def _resolve_store(args):
    """(the object store, "") or (None, why not). Names the credential's variables, never
    their values."""
    from fleet_adapter_ros2 import objectstore
    try:
        remote = objectstore.ObjectStore.from_env(
            args.store_endpoint, args.store_bucket or args.artifact_store, os.environ)
    except ValueError as exc:
        return None, f"--store-endpoint: {exc}"
    print(f"artifact store {args.artifact_store!r}: object store, {remote.describe()}; maps are "
          f"uploaded after each save, keep-out sets fetched by digest (credential from "
          f"{objectstore.ACCESS_KEY_ENV} and {objectstore.SECRET_KEY_ENV})", file=sys.stderr)
    return remote, ""


HANDOVER_REST_ODOMETRY, HANDOVER_REST_SCAN, HANDOVER_REST_OPERATOR = (
    "odometry", "scan", "operator")
HANDOVER_REST_SOURCES = (HANDOVER_REST_ODOMETRY, HANDOVER_REST_SCAN, HANDOVER_REST_OPERATOR)


def _resolve_handover_ready(args) -> tuple[tuple[float, str] | None, str]:
    """``--handover-ready-s`` and ``--handover-ready-topic``: ``((bound, topic) or None, "")``
    to run, or ``(None, why)`` to refuse. Only with ``--handover-rest-source operator``."""
    from fleet_adapter_ros2 import operator_ready
    if args.handover_rest_source != HANDOVER_REST_OPERATOR:
        for flag, v in (("--handover-ready-s", args.handover_ready_s),
                        ("--handover-ready-topic", args.handover_ready_topic)):
            if v is not None:
                return None, f"{flag} is used only with --handover-rest-source operator"
        return None, ""
    bound = handover.READY_S if args.handover_ready_s is None else args.handover_ready_s
    if not handover.READY_S_MIN <= bound <= handover.READY_S_MAX:
        # NaN fails this too. A timeout never opens the gate; it only ends the action.
        return None, (f"--handover-ready-s {bound:g} is not served: {handover.READY_S_MIN:g} to "
                      f"{handover.READY_S_MAX:g} s, the time a declined goal waits for the "
                      f"operator's ready")
    topic = operator_ready.READY_TOPIC if args.handover_ready_topic is None \
        else args.handover_ready_topic
    if topic.strip() in ("", "/cmd_vel", "cmd_vel", args.teleop_topic or "",
                         args.rest_scan_topic or ""):
        return None, (f"--handover-ready-topic {topic!r} must be the ready's own topic: not "
                      f"cmd_vel, the teleop topic or the scan's")
    return (bound, topic), ""


def _resolve_handover_rest(args) -> tuple[tuple[float, str, str] | None, str]:
    """``--handover-rest-source`` and its record: ``((T, summary, md5) or None, "")`` to run, or
    ``(None, why)`` to refuse. The scan channel is declared measured only by a valid record; with
    the operator source it is a veto only, and takes a veto-only record and nothing else."""
    from fleet_adapter_ros2 import scan_rest
    if args.handover_rest_source == HANDOVER_REST_OPERATOR:
        if args.binding != "ros2":
            return None, ("--handover-rest-source operator needs --binding ros2: the simulated "
                          "binding has no lidar, and the veto needs the scan")
        if not args.teleop_topic:
            return None, ("--handover-rest-source operator needs --teleop-topic: without it "
                          "there is no navigate-handover to confirm")
        if not (args.rest_scan_topic or "").strip():
            return None, ("--handover-rest-source operator needs --rest-scan-topic, the lidar's "
                          "raw LaserScan topic: the scan vetoes the operator's ready")
        if not args.rest_scan_calibration:
            return None, ("--handover-rest-source operator needs --rest-scan-calibration: the "
                          "veto's T comes from a veto-only record (stationary runs, 'scan_rest "
                          "calibrate --veto-only'); there is no default threshold")
        try:
            return scan_rest.load_veto_record(args.rest_scan_calibration), ""
        except scan_rest.CalibrationRefused as exc:
            return None, f"--rest-scan-calibration is not a veto's evidence: {exc}"
    if args.handover_rest_source != HANDOVER_REST_SCAN:
        for flag, v in (("--rest-scan-topic", args.rest_scan_topic),
                        ("--rest-scan-calibration", args.rest_scan_calibration)):
            if v is not None:
                return None, f"{flag} is used only with --handover-rest-source scan"
        return None, ""
    if args.binding != "ros2":
        return None, ("--handover-rest-source scan needs --binding ros2: the simulated "
                      "binding has no lidar")
    if not args.teleop_topic:
        return None, ("--handover-rest-source scan needs --teleop-topic: without it there is "
                      "no navigate-handover to wait for rest")
    if not (args.rest_scan_topic or "").strip():
        return None, ("--handover-rest-source scan needs --rest-scan-topic, the lidar's raw "
                      "LaserScan topic")
    if not args.rest_scan_calibration:
        return None, ("--handover-rest-source scan needs --rest-scan-calibration: the scan "
                      "channel is declared measured only by the calibration record of a session "
                      "that measured it (s0 and T from stationary runs, no miss in any motion "
                      "run); there is no default threshold")
    try:
        return scan_rest.load_record(args.rest_scan_calibration), ""
    except scan_rest.CalibrationRefused as exc:
        return None, (f"--rest-scan-calibration is not evidence the scan channel is measured: "
                      f"{exc}")


def _resolve_battery_source(args) -> str:
    """Why the battery flags cannot be used together, or "". A millivolt topic carries no
    percent, so both ends of its scale are the robot's to state; neither has a default."""
    if args.battery_mv_topic is None:
        if args.battery_empty_mv is not None:
            return "--battery-empty-mv is used only with --battery-mv-topic"
        return ""
    if args.binding != "ros2":
        return "--battery-mv-topic is a ros2 binding flag"
    if args.battery_empty_mv is None or args.battery_full_mv is None:
        return ("--battery-mv-topic needs --battery-empty-mv and --battery-full-mv: the "
                "percent is read between them, and no robot property has a default")
    if not args.battery_empty_mv < args.battery_full_mv:
        return "--battery-empty-mv must be below --battery-full-mv"
    if args.battery_hard_mv is not None and args.battery_hard_mv < args.battery_empty_mv:
        return "--battery-hard-mv must not be below --battery-empty-mv"
    return ""


def _resolve_mapview(args):
    """(emitter or None, refusal) for --mapview-url / --mapview-file. The service's ingest
    token, if it has one, is read from SWARMADA_MAPVIEW_TOKEN, never from a flag."""
    if not (args.mapview_url or args.mapview_file):
        if args.mapview_robot or args.mapview_session:
            return None, ("--mapview-robot and --mapview-session are used only with --mapview-url "
                          "or --mapview-file")
        return None, ""
    if args.mapview_url and args.mapview_file:
        return None, "--mapview-url and --mapview-file: give one"
    if args.mapview_url and not args.mapview_url.startswith(("http://", "https://")):
        return None, "--mapview-url must be an http:// or https:// URL"
    from fleet_adapter_ros2.mapview import contract
    from fleet_adapter_ros2.mapview import emitter as mapview
    robot = args.mapview_robot or args.robot_id
    try:
        sink = (mapview.HttpSink(args.mapview_url, os.environ.get("SWARMADA_MAPVIEW_TOKEN"))
                if args.mapview_url else mapview.FileSink(args.mapview_file))
        return mapview.MapviewEmitter(sink, robot, simulated=args.binding == "simulated",
                                      session=args.mapview_session), ""
    except contract.ContractError as exc:
        return None, f"--mapview-robot {robot!r} or --mapview-session: {exc}"
    except OSError as exc:
        return None, f"--mapview-file: {exc}"


def _resolve_explore(args) -> tuple[tuple | None, str]:
    """((site, limits) or None, refusal) for the ros2 binding's exploration stack.

    No battery thresholds: no stack, said so, and the catalog is Navigate alone. Thresholds
    that disagree, or a site file that cannot say how to reach the robot's stack: refused, so
    the binding never declares actions it cannot run. The site file is read here, at run time,
    and only its missing KEYS are ever printed."""
    hard, full, ret = args.battery_hard_mv, args.battery_full_mv, args.battery_return_mv
    if hard is None and full is None and ret is None:
        print("explore-floor and Charge: not declared -- the ros2 binding needs "
              "--battery-hard-mv and --battery-full-mv (and a site file) to run them",
              file=sys.stderr)
        return None, ""
    if hard is None or full is None:
        return None, ("explore-floor and Charge need both --battery-hard-mv and "
                      "--battery-full-mv; a robot property, so neither has a default")
    if not hard < full:
        return None, "--battery-hard-mv must be below --battery-full-mv"
    if ret is not None and not hard < ret < full:
        return None, "--battery-return-mv must lie between the hard stop and full"
    from fleet_adapter_ros2 import ros2_explore
    try:
        site = ros2_explore.SiteConfig.load(args.site_env, args.shell)
    except ros2_explore.SiteError as exc:
        return None, f"explore-floor and Charge: {exc}"
    where = ("from inside its container (--shell local)" if args.shell == "local" else
             "over ssh as the site file says (--shell ssh)")
    print(f"explore-floor and Charge: declared -- battery hard {hard:g} mV, return "
          f"{'per payload' if ret is None else f'{ret:g} mV'}, full {full:g} mV; the robot's "
          f"stack is reached {where}", file=sys.stderr)
    return (site, explore.BatteryLimits(hard_mv=hard, full_mv=full, return_mv=ret)), ""


class _ExitOnSignal:
    """SIGTERM and SIGINT end the adapter the way the control stream closing does: the
    shutdown flag is set and the channel closed (from a thread of its own, never inside the
    handler), the control-stream loop ends, and `main()`'s `finally` stops the robot. Nothing
    is raised into whatever the main thread was doing, so no step of that stop is cut short.

    A second signal while the robot is being stopped is reported and ignored. A second signal
    before that -- the main thread is not in the stream, so closing the channel did not end
    it -- raises SystemExit, which reaches the same `finally`."""

    def __init__(self, adapter, close) -> None:
        self._adapter, self._close = adapter, close
        self.signum: int | None = None
        self._stopping = False
        self._saved = {s: signal.signal(s, self._on_signal)
                       for s in (signal.SIGTERM, signal.SIGINT)}

    def _on_signal(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        if self._stopping:
            print(f"{name} while the robot is being stopped: ignored; the stop goes on",
                  file=sys.stderr)
            return
        if self.signum is not None:
            self._stopping = True
            raise SystemExit(128 + self.signum)
        self.signum = signum
        print(f"{name}: stopping the robot, then exiting", file=sys.stderr)
        self._adapter._shutdown.set()
        threading.Thread(target=self._close, daemon=True, name="close-channel").start()

    def stopping(self) -> None:
        self._stopping = True

    def restore(self) -> None:
        for s, h in self._saved.items():
            signal.signal(s, h)


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
    ap.add_argument("--stall-window", type=float, default=None, metavar="SECONDS",
                    help="progress watchdog: a goal must get closer by --stall-min-progress "
                         "within this many seconds, or it is cancelled and FAILED. Required "
                         "with --binding ros2; a property of the robot, so no default")
    ap.add_argument("--stall-min-progress", type=float, default=None, metavar="METRES",
                    help="progress watchdog: the least reduction in distance to go that counts "
                         "as progress. Required with --binding ros2; no default")
    ap.add_argument("--goal-response-bound", type=float, default=None, metavar="SECONDS",
                    help="how long a goal sent to the navigation server may go unanswered before "
                         "it is abandoned: the robot stopped, the goal cancelled by its id and "
                         "FAILED. Required with --binding ros2; a property of the server and its "
                         "transport, so no default")
    ap.add_argument("--odometry-source", default=None,
                    choices=list(ODOMETRY_SOURCES),
                    help="what the odometry this binding confirms rest from actually is: "
                         "'measured' (a sensor that observes the robot) or 'command-derived' "
                         "(the velocity command published back). Declared command-derived, no "
                         "stop is ever reported confirmed from it. Required with --binding "
                         "ros2; a property of the platform, so no default")
    ap.add_argument("--namespace", default="default",
                    help="the namespace this adapter reports in its hello, and the first part of "
                         "the path an explore-floor map is saved at (default: default)")
    # ControlStream mTLS (RFC-0001 section 9.2.7, C1.3), with the sim adapter's semantics: all
    # three or none. The control plane derives this adapter's NAME from the certificate's SAN.
    ap.add_argument("--tls-ca", default=None, metavar="PEM",
                    help="the CA bundle that signed the ControlStream server certificate: a file, "
                         "or env:NAME to read it from this process's environment")
    ap.add_argument("--tls-cert", default=None, metavar="PEM",
                    help="the client certificate (a file or env:NAME). Its SAN must be "
                         "<adapter>.<namespace>.svc.cluster.local: the control plane reads the "
                         "FleetAdapter's name from it")
    ap.add_argument("--tls-key", default=None, metavar="PEM",
                    help="the key for --tls-cert (a file or env:NAME; env:NAME keeps it "
                         "off the disk of the host the adapter runs on)")
    ap.add_argument("--tls-server-name", default=None, metavar="NAME",
                    help="verify the server against this name: needed when --endpoint is an "
                         "address (a port-forward) the server certificate's SAN does not cover")
    ap.add_argument("--provisioning-interval", type=float, default=None, metavar="SECONDS",
                    help="how long after announcing an unadmitted robot to register again "
                         "(C2.1); default min(TTL/2, 300) s, 60 s when no TTL is stated")
    ap.add_argument("--artifact-root", default=None, metavar="DIR",
                    help="the directory an ArtifactStore declares: an explore-floor segment is "
                         "saved at DIR/<namespace>/<action>/. Default: a scratch directory of "
                         "this process, which no cluster reads")
    ap.add_argument("--artifact-store", default="adapter-local", metavar="NAME",
                    help="the ArtifactStore's name, which the map-ref in the final message names "
                         "(default: adapter-local, a store no cluster declares)")
    ap.add_argument("--store-endpoint", default=None, metavar="URL",
                    help="the object store of an ArtifactStore of kind ObjectStore "
                         "(http[s]://host[:port]): each saved segment is uploaded to "
                         "<bucket>/<namespace>/<action>, and a keep-out set is fetched from it "
                         "by its reference and proven by its digest. The credential is read "
                         "from SWARMADA_STORE_ACCESS_KEY and SWARMADA_STORE_SECRET_KEY in this "
                         "process's environment, never from a flag. --artifact-root is then "
                         "this adapter's own copy")
    ap.add_argument("--store-bucket", default=None, metavar="NAME",
                    help="the bucket in --store-endpoint (default: the --artifact-store name)")
    ap.add_argument("--battery-hard-mv", type=float, default=None, metavar="MV",
                    help="ros2 binding: the battery's hard stop. With --battery-full-mv and a "
                         "site file, the binding declares explore-floor and Charge; a robot "
                         "property, so no default")
    ap.add_argument("--battery-full-mv", type=float, default=None, metavar="MV",
                    help="ros2 binding: the reading that counts as full, for Charge")
    ap.add_argument("--battery-return-mv", type=float, default=None, metavar="MV",
                    help="ros2 binding: the return threshold an explore-floor uses when its "
                         "payload names none; without it every payload must name one")
    ap.add_argument("--site-env", default="~/.config/swarmada/robot.env", metavar="FILE",
                    help="the file the ros2 binding reads, at run time, for how to reach the "
                         "robot's stack (host, users, container, work directory, bounds); "
                         "'env' reads the same keys from this process's environment")
    ap.add_argument("--shell", default="ssh", choices=["ssh", "local"],
                    help="ros2 binding: where the robot-side scripts run. 'ssh' reaches the "
                         "robot's container over ssh and docker exec; 'local' is this adapter "
                         "inside that container, and needs only the work directory, the "
                         "environment script and the overlay (default: ssh)")
    ap.add_argument("--teleop-topic", default=None, metavar="TOPIC",
                    help="ros2 binding: the geometry_msgs/Twist topic the operator's teleop "
                         "publishes on (never cmd_vel). Declares Custom manual-drive and "
                         "navigate-handover; a message reaches cmd_vel only while one of them "
                         "holds the lease and forwards. Unset: neither. With --binding "
                         "simulated: the stand-in that forwards teleop and hands a declined "
                         "goal to the operator (see the --sim-* flags)")
    ap.add_argument("--handover-rest-s", type=float, default=_ESTOP_TICK_S * _ESTOP_TICKS,
                    metavar="S",
                    help="navigate-handover: after Nav2 declines, how long to wait for the robot "
                         "to be confirmed at rest before teleop opens; not confirmed by then, "
                         "the action ends Failed and teleop never opens (default: the estop's "
                         "confirmed-stop budget, %(default)g s). A binding that cannot confirm "
                         "rest (--odometry-source command-derived) never opens it")
    ap.add_argument("--handover-rest-source", choices=HANDOVER_REST_SOURCES,
                    default=HANDOVER_REST_ODOMETRY,
                    help="navigate-handover: what confirms the robot at rest before teleop "
                         "opens. 'odometry' (default): the estop's own test (is_stopped, after "
                         "--odometry-source). 'scan': the lidar's scan-to-scan change on "
                         "--rest-scan-topic, declared measured only by --rest-scan-calibration. "
                         "'operator': the operator's ready (--handover-ready-topic), pressed "
                         "once they see the robot stopped, with the scan on --rest-scan-topic "
                         "as a veto only (a veto-only --rest-scan-calibration). The estop's "
                         "answer never reads the scan or the ready")
    ap.add_argument("--handover-ready-s", type=float, default=None, metavar="S",
                    help="with --handover-rest-source operator: after Nav2 declines, how long "
                         "the base is held waiting for the operator's ready; none by then, the "
                         "action ends Failed and teleop never opens (default "
                         f"{handover.READY_S:g} s; {handover.READY_S_MIN:g} to "
                         f"{handover.READY_S_MAX:g} s)")
    ap.add_argument("--handover-ready-topic", default=None, metavar="TOPIC",
                    help="with --handover-rest-source operator: the topic of the operator's "
                         "ready, std_msgs/msg/Empty, one message per press (default "
                         "/handover_ready); not cmd_vel, the teleop topic or the scan's")
    ap.add_argument("--rest-scan-topic", default=None, metavar="TOPIC",
                    help="ros2 binding with --handover-rest-source scan: the raw "
                         "sensor_msgs/LaserScan topic (not a filtered or fused scan)")
    ap.add_argument("--rest-scan-calibration", default=None, metavar="FILE",
                    help="with --handover-rest-source scan: the calibration record (JSON, from "
                         "'python -m fleet_adapter_ros2.scan_rest calibrate') whose threshold "
                         "the channel uses. Refused unless it shows the definition, T = 1.5 x "
                         "s0 and no miss in any motion run; there is no default threshold")
    ap.add_argument("--sim-coast-s", type=float, default=0.0, metavar="S",
                    help="simulated binding with --teleop-topic: after a decline's hold, the "
                         "stand-in's base reports motion for S more seconds, as a braking base "
                         "would (default 0)")
    ap.add_argument("--sim-decline-beyond-m", type=float, default=None, metavar="M",
                    help="simulated binding with --teleop-topic: the stand-in's nav2 declines "
                         "every goal farther than M from the robot (unset: none)")
    ap.add_argument("--sim-teleop-pulse-s", type=float, default=0.5, metavar="S",
                    help="simulated binding with --teleop-topic: a teleop command the gate "
                         "forwards moves the stand-in by its speed times S (default 0.5)")
    ap.add_argument("--sim-operator-file", default=None, metavar="PATH",
                    help="simulated binding with --teleop-topic: the operator's teleop, one "
                         "'vx vy wz' line appended per key press; only the gate decides what "
                         "reaches the stand-in's base")
    ap.add_argument("--battery-mv-topic", default=None, metavar="TOPIC",
                    help="ros2 binding: a std_msgs/UInt16 topic carrying the battery in mV, "
                         "read instead of sensor_msgs/BatteryState on battery_state. Needs "
                         "--battery-empty-mv and --battery-full-mv, which the percent is read "
                         "between")
    ap.add_argument("--battery-empty-mv", type=float, default=None, metavar="MV",
                    help="ros2 binding: the reading that counts as 0 %% on --battery-mv-topic; "
                         "a robot property, so no default")
    ap.add_argument("--imu-topic", default="imu",
                    help="ros2 binding: the IMU topic Charge watches for the cable coming out")
    ap.add_argument("--keepouts-ref", default=None, metavar="STORE/NS/NAME@sha256:HEX",
                    help="the keep-out set rendered from the applied zones, in the artifact "
                         "store: every Navigate goal inside one of its zones is refused, and no "
                         "Navigate is accepted while the set cannot be proven by its digest. The "
                         "polygons are in the frame of the map the zones came from, so the check "
                         "holds while the robot localizes on that map")
    ap.add_argument("--mapview-url", default=None, metavar="URL",
                    help="emit the map-view contract (mapview/0) while an explore-floor segment "
                         "runs, to the map-view service at URL. Read-only: the feed reads the "
                         "map, the pose and the served keep-out mask, and commands nothing")
    ap.add_argument("--mapview-file", default=None, metavar="FILE",
                    help="emit the map-view contract to FILE (JSON Lines) instead")
    ap.add_argument("--mapview-robot", default=None, metavar="ALIAS",
                    help="the robot's alias in the map view, a DNS label (default: --robot-id)")
    ap.add_argument("--mapview-session", default=None, metavar="SESSION",
                    help="the SLAM session a navigate-handover's circle is drawn in: its target "
                         "and target.state go out in frame map@SESSION, the frame the map in the "
                         "viewer comes from (default: a frame of its own per action)")
    args = ap.parse_args()

    # Before anything connects: a ros2 adapter without a progress watchdog would report a
    # goal the stack never terminates as RUNNING forever, and its thresholds are the plant's.
    stall_limits, refusal = resolve_stall_limits(
        args.binding, args.stall_window, args.stall_min_progress)
    if not refusal:
        response_bound, refusal = resolve_response_bound(args.binding, args.goal_response_bound)
    if not refusal:
        # And a ros2 adapter that has not been told what its odometry IS would confirm rest
        # from a channel that may be the stop command echoed back (C5.3, ADR-0071).
        odometry_source, refusal = resolve_odometry_source(args.binding, args.odometry_source)
    if not refusal:
        refusal = _resolve_battery_source(args)
    if not refusal and not 0.1 <= args.handover_rest_s <= 30.0:
        # NaN fails this too. Zero would turn the gate on no evidence of rest at all.
        refusal = (f"--handover-rest-s {args.handover_rest_s:g} is not served: 0.1 to 30 s, "
                   f"the time a declined goal waits for confirmed rest")
    scan_rest_record = None
    if not refusal:
        scan_rest_record, refusal = _resolve_handover_rest(args)
    ready = None
    if not refusal:
        ready, refusal = _resolve_handover_ready(args)
    if not refusal:
        refusal = _resolve_tls(args)
    remote = None
    if not refusal and args.store_endpoint:
        remote, refusal = _resolve_store(args)
    elif not refusal and args.store_bucket:
        refusal = "--store-bucket is used only with --store-endpoint"
    explore_stack_args = None
    if not refusal and args.binding == "ros2":
        explore_stack_args, refusal = _resolve_explore(args)
    mapview = None
    if not refusal:
        mapview, refusal = _resolve_mapview(args)
    if refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        raise SystemExit(2)
    if args.binding == "ros2":
        print(f"goal response bound: a goal sent to the navigation server and unanswered after "
              f"{response_bound:g} s is abandoned -- the robot stopped, the goal cancelled by its "
              f"id and FAILED (--goal-response-bound {response_bound:g})", file=sys.stderr)
        print(f"progress watchdog: armed -- {stall_limits.describe()} "
              f"(--stall-window {stall_limits.window_s:g}, "
              f"--stall-min-progress {stall_limits.min_progress_m:g})", file=sys.stderr)
        gap = rest_evidence_gap(odometry_source)
        print(f"rest evidence: odometry declared {odometry_source} "
              f"(--odometry-source {odometry_source}) -- "
              + ("a stop is reported confirmed only once that odometry shows the robot at rest"
                 if not gap else
                 f"NO STOP IS EVER REPORTED CONFIRMED on this robot: {gap}"), file=sys.stderr)
    else:
        print("progress watchdog: not armed -- the simulated binding reports no goal progress "
              "to measure" + (f"; --stall-window {stall_limits.window_s:g} and "
                             f"--stall-min-progress {stall_limits.min_progress_m:g} are unused"
                             if stall_limits else ""), file=sys.stderr)

    from fleet_adapter.v1 import fleet_adapter_pb2 as pb
    from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc

    if args.binding == "ros2":
        from fleet_adapter_ros2.nav2_binding import Nav2Binding
        # Constructed only. Bring-up is NOT done here: the adapter drives the
        # RobotBinding lifecycle in bring_up_robot(), for every binding alike. Starting
        # it in this branch would put the lifecycle back on the `--binding` switch, and
        # the next binding added to that switch would be inert again.
        robot = Nav2Binding(stall_limits=stall_limits,  # ROS 2 / Nav2 (sub-items 2-4)
                            response_bound_s=response_bound,
                            odometry_source=odometry_source,
                            battery_mv_topic=args.battery_mv_topic,
                            battery_empty_mv=args.battery_empty_mv,
                            battery_full_mv=args.battery_full_mv,
                            teleop_topic=args.teleop_topic,
                            rest_scan_topic=(args.rest_scan_topic if scan_rest_record else None),
                            rest_scan_threshold_m=(scan_rest_record[0] if scan_rest_record
                                                   else None),
                            ready_topic=ready[1] if ready else None)
        print("navigate-handover rest: "
              + (f"{handover.CONFIRMED} -- the operator's ready on {ready[1]} "
                 f"(std_msgs/msg/Empty), pressed once they see the robot stopped, within "
                 f"{ready[0]:g} s of the hold; the lidar scan on {args.rest_scan_topic} is a "
                 f"veto only (veto-only record md5 {scan_rest_record[2]}: {scan_rest_record[1]}): "
                 f"any moving or unknown keeps the gate shut, and it never confirms. The "
                 f"estop's answer reads neither"
                 if ready else
                 f"the lidar scan channel on {args.rest_scan_topic}, declared measured by the "
                 f"calibration record md5 {scan_rest_record[2]}: {scan_rest_record[1]}. The "
                 f"estop's answer does not read it"
                 if scan_rest_record else
                 "the estop's own test (--odometry-source)"), file=sys.stderr)
        print("manual-drive: " + (f"declared -- teleop on {args.teleop_topic} reaches cmd_vel "
                                  f"only while a manual-drive holds the lease"
                                  if args.teleop_topic else
                                  "not declared -- no --teleop-topic"), file=sys.stderr)
        print(f"battery: read from {robot.battery_source()}"
              + (f", 0-100 % between {args.battery_empty_mv:g} and {args.battery_full_mv:g} mV"
                 if args.battery_mv_topic else "")
              + "; a battery that never reports is absent from telemetry and a Failed "
                "'battery' entry, never 0", file=sys.stderr)
        if explore_stack_args is not None:
            from fleet_adapter_ros2 import ros2_explore
            site, limits = explore_stack_args
            robot._explore = ros2_explore.Ros2ExploreStack(
                binding=robot, robot_id=args.robot_id, site=site,
                shell=ros2_explore.RobotShell(site, mode=args.shell),
                ros=ros2_explore.RclpyFacade(robot, imu_topic=args.imu_topic), limits=limits,
                imu_topic=args.imu_topic)
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
        if args.teleop_topic:
            # The stand-in for a desk or kind run of manual-drive and navigate-handover.
            robot = handover.SimulatedHandoverRobot(
                teleop_topic=args.teleop_topic, engine=engine,
                decline_beyond_m=args.sim_decline_beyond_m, pulse_s=args.sim_teleop_pulse_s,
                coast_s=args.sim_coast_s)
            print(f"simulated binding: the handover stand-in, teleop on {args.teleop_topic}"
                  + (f", goals beyond {args.sim_decline_beyond_m:g} m declined"
                     if args.sim_decline_beyond_m is not None else ""), file=sys.stderr)
            if args.sim_operator_file:
                threading.Thread(
                    target=handover.follow_operator_file,
                    args=(robot, args.sim_operator_file, args.robot_id, threading.Event()),
                    daemon=True, name="operator-file").start()
        else:
            robot = SimulatedRobot(engine)  # $0 skeleton; scenario-driven when an engine is set
    adapter = fleet_adapter_ros2Adapter(
        pb, pb_grpc, robot, args.robot_id, namespace=args.namespace,
        artifacts=explore.ArtifactDir(args.artifact_root, args.artifact_store, remote=remote),
        keepouts_ref=args.keepouts_ref, provisioning_interval_s=args.provisioning_interval,
        mapview=mapview, handover_rest_s=args.handover_rest_s,
        handover_ready_s=ready[0] if ready else handover.READY_S)

    channel = open_control_channel(args)
    # The endpoint itself is not printed: on a robot run it is an address on a site's own
    # network, and these lines are kept as evidence.
    print("control stream: " + ("mTLS to --endpoint, presenting the client certificate"
                                + (f", server verified as {args.tls_server_name}"
                                   if args.tls_server_name else "")
                                if args.tls_cert else
                                "PLAINTEXT to --endpoint (no --tls-* flags)"), file=sys.stderr)
    stub = pb_grpc.FleetAdapterServiceStub(channel)
    adapter.start_safety_stream(stub)  # C1.1
    exiting = _ExitOnSignal(adapter, lambda: channel.close())
    try:
        adapter.run_control_stream(stub)
    finally:
        exiting.stopping()
        try:
            # The robot is stopped first, while rclpy is still up: the hold, the goal's cancel,
            # the explorer's stop and the end of the segment all go out through it.
            adapter.shutdown()
        finally:
            # The teardown half of the lifecycle the Protocol now declares. `Nav2Binding.start()`
            # also registers this via atexit -- that was the workaround for having no Protocol
            # lifecycle to hang it on, and it stays because `stop()` is idempotent. Calling it
            # here makes the shutdown explicit at the level that owns the robot, instead of a
            # side effect a reader has to find inside the binding.
            robot.stop()
            exiting.restore()
            if mapview is not None:
                mapview.close()          # after the robot is stopped: a last flush, bounded
    if exiting.signum is not None:
        raise SystemExit(128 + exiting.signum)


if __name__ == "__main__":
    main()
