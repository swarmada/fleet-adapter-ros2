"""The EdgeStream is dialled again after it ends or fails, and each advertised edge node once.

Before, the adapter dialled the first advertised edge node once: when that EdgeStream ended or
failed, no edge estop reached the robot again for as long as the adapter ran, and an edge node
advertised after the first was never dialled.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace as NS

import grpc

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


class _Unavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _Robot(SimulatedRobot):
    def __init__(self) -> None:
        super().__init__()
        self.stops = 0

    def command_stop(self, robot_id: str) -> None:
        self.stops += 1
        super().command_stop(robot_id)


class _Node:
    """An edge node whose EdgeStream fails at the first dial, ends at the second, and sends an
    estop at the third; later dials end at once."""

    def __init__(self) -> None:
        self.dials, self.sent = 0, []

    def EdgeStream(self, requests):  # the gRPC method name
        self.dials += 1
        if self.dials == 1:
            raise _Unavailable()
        if self.dials != 3:
            return iter(())

        def read() -> None:
            for m in requests:
                self.sent.append(m)
        threading.Thread(target=read, daemon=True).start()
        return iter([pb.EdgeControlMessage(estop=pb.Estop(estop_id="edge-3", reason="test"))])

    def answered(self) -> list:
        return [m.estop_ack for m in list(self.sent) if m.WhichOneof("msg") == "estop_ack"
                and m.estop_ack.state != pb.ESTOP_STATE_STOPPING]


def _until(cond, timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def _adapter(robot: SimulatedRobot) -> Adapter:
    robot.spawn(ROBOT)
    return Adapter(pb, None, robot, ROBOT)


def test_an_edge_stream_that_fails_or_ends_is_dialled_again_and_its_estop_answered(
        monkeypatch, capsys):
    monkeypatch.setattr(adapter_mod, "_REDIAL_S", 0.01)
    robot = _Robot()
    a = _adapter(robot)
    node = _Node()
    a._pb_grpc = NS(EdgeServiceStub=lambda channel: node)
    t = threading.Thread(target=a._edge_redial, args=("localhost:1",), daemon=True)
    t.start()
    try:
        assert _until(lambda: node.answered()), f"no estop answered after {node.dials} dials"
        assert node.answered()[0].estop_id == "edge-3" and robot.stops >= 1
        err = capsys.readouterr().err
        assert "EdgeStream to localhost:1 ended (UNAVAILABLE); redialling" in err, err
        assert err.count("ended (the edge node ended it)") == 1, err   # once, not per redial
    finally:
        a._shutdown.set()
    t.join(1.0)
    assert not t.is_alive(), "the redial outlived shutdown"


def test_each_advertised_edge_node_is_dialled_once(monkeypatch):
    dialled: list = []
    monkeypatch.setattr(Adapter, "_edge_redial", lambda self, addr: dialled.append(addr))
    a = _adapter(SimulatedRobot())
    for addr in ("edge:1", "edge:2", "edge:1", "edge:2"):   # a RegisterAck again on a reconnect
        a._edge_addr.put(addr)
    t = threading.Thread(target=a._edge_loop, daemon=True)
    t.start()
    assert _until(lambda: len(dialled) >= 2)
    time.sleep(0.1)
    assert sorted(dialled) == ["edge:1", "edge:2"]
    a._shutdown.set()
    a._edge_addr.put("edge:3")                              # wakes the loop, which exits
    t.join(1.0)
    assert not t.is_alive() and "edge:3" not in dialled
