# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""RegisterRobot goes out before any telemetry, on every connect and reconnect (C2.1).

The adapter registers on the ControlStream only once the SafetyStream is up too (C1.1). When the
SafetyStream comes up first, the registration is queued at the hello's answer, before the
telemetry thread starts. When the ControlStream comes up first, the registration waits on a thread
of its own, and the telemetry thread, already running, used to queue its first payload ahead of
it. The conformance harness reads the first message for C2.1: it saw a TelemetryPayload, recorded
"adapter initiated no registration", and with no registration skipped the edge checks (C8.1-C8.3)
and the catalog push (C12.2). It happened once in CI (run 43820854866), on an order the streams
come up in by chance.

Now a session's telemetry waits until that session's RegisterRobot is queued.
"""

from __future__ import annotations

import queue
import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "robot-conformance-1"
# Long enough for the telemetry thread to have queued several payloads had it not waited: its
# first pass is immediate and it repeats every _TELEMETRY_INTERVAL (0.2 s).
HELD_S = 0.6


def _adapter() -> Adapter:
    robot = SimulatedRobot()
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._safety_required = True          # as start_safety_stream sets it, with the stream not up
    return a


def _hello(a: Adapter) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)), None)


def _safety_up(a: Adapter) -> None:
    a._safety_opened(a._safety_gen)


def _kinds(q: queue.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait().WhichOneof("payload"))
    return out


def _until(pred, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_with_the_control_stream_up_first_the_first_message_is_the_registration():
    a = _adapter()
    try:
        _hello(a)                          # ControlStream up; the SafetyStream not yet
        time.sleep(HELD_S)
        assert _kinds(a._outbox) == [], "nothing may be queued before the registration"
        _safety_up(a)
        assert _until(lambda: not a._outbox.empty())
        assert _until(lambda: a._outbox.qsize() >= 3)    # register, the catalog, telemetry
        kinds = _kinds(a._outbox)
        assert kinds[0] == "register"
        assert "telemetry" in kinds, "telemetry resumes once the registration is queued"
        assert kinds.index("telemetry") > kinds.index("register")
    finally:
        a._shutdown.set()


def test_with_the_safety_stream_up_first_the_registration_is_queued_at_the_hello():
    a = _adapter()
    try:
        _safety_up(a)
        _hello(a)
        kinds = [a._outbox.get(timeout=1.0).WhichOneof("payload")]
        assert kinds == ["register"]
    finally:
        a._shutdown.set()


def test_after_a_reconnect_the_new_session_registers_before_its_telemetry():
    """A redial replaces the outbox (run_control_stream). The telemetry thread keeps running
    across it, and must not put anything on the new session ahead of its RegisterRobot."""
    a = _adapter()
    try:
        _safety_up(a)
        _hello(a)
        assert _until(lambda: "telemetry" in _kinds(a._outbox))   # the first session streams
        a._outbox = queue.Queue()          # the redial's fresh outbox, before its hello
        a._safety_up.clear()               # and its SafetyStream not back yet
        time.sleep(HELD_S)
        assert _kinds(a._outbox) == [], "no telemetry on a session that has not registered"
        _hello(a)
        time.sleep(HELD_S)
        assert _kinds(a._outbox) == [], "still none while its registration waits"
        a._safety_gen += 1
        _safety_up(a)
        assert _until(lambda: a._outbox.qsize() >= 3)
        kinds = _kinds(a._outbox)
        assert kinds[0] == "register" and kinds.index("telemetry") > 0
    finally:
        a._shutdown.set()


def test_the_wait_is_short_once_registered():
    """The first payload follows the registration within one short look (0.2 s), not one
    telemetry interval (1-30 s): C2.3 measures the cadence from there."""
    a = _adapter()
    a._telemetry_interval = 5.0
    try:
        _hello(a)
        time.sleep(0.3)
        _safety_up(a)
        t0 = time.monotonic()
        assert _until(lambda: "telemetry" in [m.WhichOneof("payload") for m in list(a._outbox.queue)],
                      timeout=2.0)
        assert time.monotonic() - t0 < 2 * adapter_mod._TELEMETRY_INTERVAL + 0.2
    finally:
        a._shutdown.set()
