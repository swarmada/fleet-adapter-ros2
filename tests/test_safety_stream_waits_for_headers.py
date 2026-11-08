# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The estop path opens only once the control plane says it serves the SafetyStream.

A gRPC call to SafetyStream returns before the server has accepted anything. The adapter used to
count the stream up at that moment, so a robot could be registered on a new control stream, and
accept motion, before the control plane served its estop path: a reconnection test caught the
register arriving first about one run in twelve. The control plane now sends the stream's response
headers once it serves the stream (registered for estops, under a verified identity), and the
adapter opens the estop path only when they have come (RFC-0001, safety emergency stop protocol).

It fails closed. No headers within a named timeout: the call is cancelled, the estop path stays
closed, every assignment is declined, one message says what it waited for, and the stream is
dialled again. A call refused before any headers is a loss as before, not a timeout.

The first half drives the adapter against a fake control plane; the second against a real gRPC
server on the loopback interface, so the adapter's own gRPC client and its initial_metadata()
are what wait.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent import futures

import grpc
import pytest
from safety_call import SafetyCall
from test_shutdown_stops_the_robot import _RpcError, _until

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


def _navigate(action_id: str, token: int) -> pb.AssignAction:
    return pb.AssignAction(action_id=action_id, assignment_id=f"a-{action_id}",
                           action_type="Navigate", payload_json=b'{"x": 1, "y": 0}',
                           fencing_token=token, lease_generation=1)


def _assign(a, command_id: int) -> pb.AssignActionResult:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=command_id, robot_id=ROBOT,
        assign_action=_navigate(f"nav-{command_id}", command_id))), None)
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"]
    assert res, out
    return res[-1]


def _adapter(monkeypatch, headers_s: float = 0.3):
    monkeypatch.setattr(adapter_mod, "_SAFETY_HEADERS_S", headers_s)
    return adapter_mod.fleet_adapter_ros2Adapter(pb, None, SimulatedRobot(), ROBOT)


def test_the_timeout_is_a_named_constant() -> None:
    assert adapter_mod._SAFETY_HEADERS_S == 5.0


# ── against a fake control plane ─────────────────────────────────────────────

def test_no_headers_keeps_the_estop_path_closed_and_says_so_once(monkeypatch, capsys) -> None:
    dials = []

    class _Silent:
        def SafetyStream(self, requests):
            dials.append(time.monotonic())

            def stream():
                threading.Event().wait()
                yield
            return SafetyCall(stream(), "silent")
    a = _adapter(monkeypatch)
    a.start_safety_stream(_Silent())
    _until(lambda: len(dials) >= 3, "the SafetyStream to be dialled three times", 5.0)
    assert not a._safety_up.is_set()
    declined = _assign(a, 1)
    assert not declined.accepted
    assert declined.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert "SafetyStream" in declined.message
    assert dials[1] - dials[0] >= 0.3 + 0.45, "redialled before the headers' timeout"
    a._shutdown.set()
    err = capsys.readouterr().err
    assert err.count("SafetyStream not served: no response headers within 0.3 s") == 1, err
    assert "the estop path stays closed and motion is refused" in err
    assert "SafetyStream up" not in err


def test_headers_that_come_late_open_the_path_only_then(monkeypatch) -> None:
    calls = []

    class _Late:
        def SafetyStream(self, requests):
            def stream():
                threading.Event().wait()
                yield
            call = SafetyCall(stream(), "silent")
            calls.append(call)
            return call
    a = _adapter(monkeypatch, headers_s=5.0)
    a.start_safety_stream(_Late())
    _until(lambda: calls, "the SafetyStream to be dialled", 2.0)
    time.sleep(0.3)
    assert not a._safety_up.is_set()
    assert not _assign(a, 1).accepted                  # dialled, not yet served: no motion
    calls[0].headers.set()                             # the control plane serves it now
    _until(a._safety_up.is_set, "the estop path to open on the headers", 2.0)
    assert _assign(a, 2).accepted
    a._shutdown.set()


def test_a_stream_refused_before_any_headers_is_a_loss_not_a_timeout(monkeypatch,
                                                                     capsys) -> None:
    dials = []

    class _Refusing:
        def SafetyStream(self, requests):
            dials.append(1)
            if len(dials) >= 3:
                a._shutdown.set()

            def stream():
                raise _RpcError(grpc.StatusCode.UNAVAILABLE)
                yield
            return SafetyCall(stream(), "refused")
    a = _adapter(monkeypatch, headers_s=5.0)
    t0 = time.monotonic()
    a.run_safety_stream(_Refusing())
    assert len(dials) == 3 and time.monotonic() - t0 < 4.0   # no 5 s waits for headers
    assert not a._safety_up.is_set()
    assert "SafetyStream not served" not in capsys.readouterr().err


# ── against a real gRPC server on the loopback interface ────────────────────

class _Servicer(pb_grpc.FleetAdapterServiceServicer):
    """The control plane's SafetyStream, in one of four behaviours: send the headers on accept
    (as it does for a verified identity), never send them, let a first estop carry them, or
    refuse the stream."""

    def __init__(self, mode: str) -> None:
        self.mode, self.acks, self.cancelled = mode, [], threading.Event()

    def SafetyStream(self, request_iterator, context):
        if self.mode == "refuse":
            context.abort(grpc.StatusCode.UNAVAILABLE, "SafetyStream not configured")

        def read() -> None:
            with contextlib.suppress(Exception):  # the stream ended
                for m in request_iterator:
                    if m.WhichOneof("payload") == "estop_ack":
                        self.acks.append(m.estop_ack)
        threading.Thread(target=read, daemon=True).start()
        context.add_callback(self.cancelled.set)
        if self.mode == "headers":
            context.send_initial_metadata(())
        elif self.mode == "first-estop":
            yield pb.ControlPlaneSafetyMessage(
                robot_id=ROBOT, estop=pb.Estop(estop_id="e-1", reason="test"))
        while context.is_active():
            time.sleep(0.01)


@pytest.fixture
def control_plane():
    servers = []

    def serve(mode: str):
        servicer = _Servicer(mode)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_FleetAdapterServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        servers.append((server, channel))
        return servicer, pb_grpc.FleetAdapterServiceStub(channel)
    yield serve
    for server, channel in servers:
        channel.close()
        server.stop(0)


def test_grpc_headers_on_accept_open_the_estop_path(monkeypatch, control_plane) -> None:
    _, stub = control_plane("headers")
    a = _adapter(monkeypatch, headers_s=5.0)
    a.start_safety_stream(stub)
    _until(a._safety_up.is_set, "the estop path to open on the server's headers", 5.0)
    assert _assign(a, 1).accepted
    a._shutdown.set()


def test_grpc_no_headers_fails_closed_and_cancels_the_call(monkeypatch, control_plane,
                                                           capsys) -> None:
    servicer, stub = control_plane("never")
    a = _adapter(monkeypatch, headers_s=0.5)
    a.start_safety_stream(stub)
    _until(servicer.cancelled.is_set, "the adapter to cancel a call never served", 5.0)
    assert not a._safety_up.is_set()
    assert not _assign(a, 1).accepted
    a._shutdown.set()
    assert "SafetyStream not served: no response headers within 0.5 s" in capsys.readouterr().err


def test_grpc_a_first_estop_carries_the_headers(monkeypatch, control_plane) -> None:
    servicer, stub = control_plane("first-estop")
    a = _adapter(monkeypatch, headers_s=5.0)
    a.start_safety_stream(stub)
    _until(a._safety_up.is_set, "the estop path to open on the estop's headers", 5.0)
    _until(lambda: any(k.estop_id == "e-1" for k in servicer.acks),
           "an EstopAck for the estop that carried the headers", 5.0)
    a._shutdown.set()


def test_grpc_a_refused_stream_is_a_loss_not_a_timeout(monkeypatch, control_plane,
                                                       capsys) -> None:
    _, stub = control_plane("refuse")
    a = _adapter(monkeypatch, headers_s=5.0)
    a.start_safety_stream(stub)
    time.sleep(1.5)
    assert not a._safety_up.is_set()
    assert not _assign(a, 1).accepted
    a._shutdown.set()
    assert "SafetyStream not served" not in capsys.readouterr().err
