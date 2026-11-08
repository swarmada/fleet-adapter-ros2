# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The SafetyStream comes back after a drop, and nothing moves while it is down.

The SafetyStream is the estop path (RFC-0001, fleet-adapter protocol: both streams are opened at
startup, and an adapter that cannot open the SafetyStream must not declare itself healthy). The
adapter used to open it once: a drop ended its thread, the control stream came back and took work,
and from then on no estop could reach the robot until the process was restarted.

Now the SafetyStream is redialled like the control stream, with the same backoff, and a
reconnection counts only once both are back: the robot is registered on a new control stream only
after the SafetyStream is up -- that is, once the control plane's response headers have come on it,
which it sends when it serves the stream. While it is down the adapter stops the robot, ends the action it was
running, and declines every assignment and resume, since a robot that cannot receive an estop must
not move.

Every test drives the shipped `main()` against a control plane whose two streams share one
connection, as they do over gRPC.
"""

from __future__ import annotations

import contextlib
import queue
import random
import threading
import time

import grpc
import pytest
from safety_call import SafetyCall
from test_shutdown_stops_the_robot import _main, _RpcError, _until, signals  # noqa: F401

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


class _Net:
    """Both streams over one connection. `drop()` ends both, as a lost connection does;
    `drop_safety()` ends the SafetyStream alone. While `refuse_safety` is set a new SafetyStream
    is refused at once (the control plane serving none)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.conn = 0            # bumped by drop(): every stream of the old connection ends
        self.safety_conn = 0     # bumped by drop_safety()
        self.refuse_safety = False
        self.session_q: queue.Queue | None = None
        self.safe_q: queue.Queue = queue.Queue()
        self.sent: list = []     # (monotonic time, AdapterMessage) on any control stream
        self.acks: list = []     # EstopAcks on any SafetyStream
        self.safety_opened: list = []   # monotonic times a SafetyStream was served
        self.registered = threading.Event()
        self.closed = threading.Event()

    def close(self) -> None:
        """The control plane ends the control stream cleanly, as at the end of a run."""
        self.closed.set()
        if self.session_q is not None:
            self.session_q.put(None)

    def drop(self) -> None:
        with self.lock:
            self.conn += 1
            self.registered.clear()

    def drop_safety(self) -> None:
        with self.lock:
            self.safety_conn += 1

    def _gone(self, conn: int, safety_conn: int | None = None) -> bool:
        return self.conn != conn or (safety_conn is not None and (
            self.safety_conn != safety_conn or self.closed.is_set()))

    def ControlStream(self, requests):
        conn, q = self.conn, queue.Queue()
        self.session_q = q

        def read():
            with contextlib.suppress(Exception):  # the stream ended
                for m in requests:
                    self.sent.append((time.monotonic(), m))
                    kind = m.WhichOneof("payload")
                    if kind == "hello":
                        q.put(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)))
                    elif kind == "register":
                        q.put(pb.ControlPlaneMessage(register_ack=pb.RegisterAck(accepted=True)))
                        self.registered.set()
        threading.Thread(target=read, daemon=True).start()

        def stream():
            while True:
                if self._gone(conn):
                    raise _RpcError(grpc.StatusCode.UNAVAILABLE)
                try:
                    m = q.get(timeout=0.02)
                except queue.Empty:
                    continue
                if m is None:
                    return
                yield m
        return stream()

    def SafetyStream(self, requests):
        conn, safety_conn = self.conn, self.safety_conn
        if self.refuse_safety:
            raise _RpcError(grpc.StatusCode.UNAVAILABLE)   # refused as it is dialled

        def read():
            with contextlib.suppress(Exception):  # the stream ended
                for m in requests:
                    if m.WhichOneof("payload") == "estop_ack":
                        self.acks.append(m.estop_ack)

        def stream():
            threading.Thread(target=read, daemon=True).start()
            while True:
                if self._gone(conn, safety_conn):
                    raise _RpcError(grpc.StatusCode.UNAVAILABLE)
                try:
                    m = self.safe_q.get(timeout=0.02)
                except queue.Empty:
                    continue
                yield m
        # Served a moment after the dial, as a real control plane accepts the stream after the
        # call has returned: then it records the time and sends the headers.
        call = SafetyCall(stream(), "silent")

        def serve() -> None:
            time.sleep(random.uniform(0.0, 0.005))
            self.safety_opened.append(time.monotonic())
            call.headers.set()
        threading.Thread(target=serve, daemon=True).start()
        return call

    def send(self, command_id: int, **command) -> None:
        self.session_q.put(pb.ControlPlaneMessage(command=pb.Command(
            command_id=command_id, robot_id=ROBOT, **command)))

    def results(self, kind: str) -> list:
        return [m.command_result for _, m in self.sent
                if m.WhichOneof("payload") == "command_result"
                and m.command_result.WhichOneof("result") == kind]

    def statuses(self, action: str) -> list:
        return [m.action_status for _, m in self.sent
                if m.WhichOneof("payload") == "action_status" and m.action_status.action_id == action]


def _navigate(action: str, token: int) -> pb.AssignAction:
    return pb.AssignAction(action_id=action, assignment_id=f"a-{action}", action_type="Navigate",
                           payload_json=b'{"x": 3, "y": 0}', fencing_token=token,
                           lease_generation=1, lease_duration_ms=60_000)


def _run(monkeypatch, drive):
    net = _Net()
    robot = SimulatedRobot()
    monkeypatch.setattr(adapter_mod, "SimulatedRobot", lambda engine=None: robot)
    monkeypatch.setattr(adapter_mod, "open_control_channel", lambda args: net)
    monkeypatch.setattr(pb_grpc, "FleetAdapterServiceStub", lambda ch: net)
    failures: list = []

    def driver():
        try:
            drive(net, robot)
        except BaseException as exc:  # noqa: BLE001 - reported on the test's thread
            failures.append(exc)
        finally:
            net.close()
    t = threading.Thread(target=driver, daemon=True)
    t.start()
    code, raised = _main(monkeypatch, "--binding", "simulated")
    t.join(10.0)
    assert not failures, failures
    assert code is None and raised is None, (code, raised)


def test_after_a_dropped_connection_both_streams_return_and_an_estop_reaches_the_robot(
        monkeypatch, signals) -> None:  # noqa: F811
    def drive(net, robot):
        _until(net.registered.is_set, "the first registration", 5.0)
        net.drop()                                          # the connection is lost
        _until(lambda: len(net.safety_opened) >= 2, "the SafetyStream to be redialled", 5.0)
        _until(net.registered.is_set, "the robot to be registered again", 5.0)
        second_register = max(t for t, m in net.sent if m.WhichOneof("payload") == "register")
        assert second_register >= net.safety_opened[1], (
            "the robot was registered on the new control stream before the SafetyStream was back")
        net.safe_q.put(pb.ControlPlaneSafetyMessage(
            robot_id=ROBOT, estop=pb.Estop(estop_id="e-after-drop", reason="test")))
        _until(lambda: any(a.estop_id == "e-after-drop" for a in net.acks),
               "an EstopAck for the estop sent after the reconnection", 5.0)
        assert not robot._moving.get(ROBOT)
    _run(monkeypatch, drive)


def test_while_the_safety_stream_is_down_the_robot_is_held_and_motion_is_refused(
        monkeypatch, signals) -> None:  # noqa: F811
    def drive(net, robot):
        _until(net.registered.is_set, "the registration", 5.0)
        net.send(1, assign_action=_navigate("nav-1", 1))
        _until(lambda: robot._moving.get(ROBOT), "nav-1 to be under way")
        net.refuse_safety = True
        net.drop_safety()                                   # the estop path alone is lost
        _until(lambda: not robot._moving[ROBOT],
               "the robot to be held when its SafetyStream was lost", 2.0)
        _until(lambda: any(s.state == pb.ACTION_STATE_FAILED for s in net.statuses("nav-1")),
               "nav-1 to end FAILED", 3.0)
        net.send(2, assign_action=_navigate("nav-2", 2))
        _until(lambda: len(net.results("assign_action")) >= 2, "an answer to nav-2")
        declined = net.results("assign_action")[-1].assign_action
        assert not declined.accepted
        assert declined.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
        assert "SafetyStream" in declined.message
        time.sleep(0.3)
        assert not robot._moving[ROBOT], "the robot moved with no estop path"
        net.refuse_safety = False                           # the control plane serves it again
        _until(lambda: len(net.safety_opened) >= 2, "the SafetyStream to come back", 5.0)
        time.sleep(0.8)
        net.send(3, assign_action=_navigate("nav-3", 3))
        _until(lambda: robot._moving.get(ROBOT), "nav-3 to run once the SafetyStream is back")
        assert net.results("assign_action")[-1].assign_action.accepted
    _run(monkeypatch, drive)


@pytest.mark.parametrize("_", [0])
def test_a_safety_stream_that_fails_while_running_is_redialled(_, capsys) -> None:
    opened = []

    class _Once:
        def SafetyStream(self, requests):
            opened.append(time.monotonic())

            def stream():
                if len(opened) >= 3:
                    a._shutdown.set()
                raise _RpcError(grpc.StatusCode.UNAVAILABLE)
                yield
            return SafetyCall(stream(), "refused")
    a = adapter_mod.fleet_adapter_ros2Adapter(pb, None, SimulatedRobot(), ROBOT)
    a.run_safety_stream(_Once())
    assert len(opened) == 3
    assert opened[2] - opened[1] >= 0.45, "redialled with no backoff"


def test_an_assignment_right_after_the_safety_stream_opens_is_served() -> None:
    """A SafetyStream counts as up once the control plane's headers have come on it, and not later:
    it sends them when it serves the stream, and nothing more until an estop. Waiting for anything
    beyond the headers declined the conformance harness's first assignments, sent the moment the
    robot was registered."""
    class _Open:
        def SafetyStream(self, requests):  # the control plane's end: served, then silent
            def stream():
                threading.Event().wait()
                yield
            return SafetyCall(stream())
    robot = SimulatedRobot()
    a = adapter_mod.fleet_adapter_ros2Adapter(pb, None, robot, ROBOT)
    a.start_safety_stream(_Open())
    _until(lambda: a._safety_gen >= 1, "the SafetyStream to be dialled", 2.0)
    time.sleep(0.05)
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=_navigate("nav-1", 1))), None)
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    res = [m.command_result.assign_action for m in out if m.WhichOneof("payload") == "command_result"]
    assert res and res[0].accepted, res
