"""An estop from the EdgeStream or the scenario drill whose stop raises.

The estop is answered FAILED, as before. Whether its stop went out cannot be known, so the adapter
now does what it does for a SafetyStream loss whose stop raises: it commands the stop again, every
_ESTOP_TICK_S within the stop budget and then every _SAFETY_STOP_RETRY_S, until one goes out, a
later loss or estop takes the retry over, or the adapter shuts down, and it refuses every
assignment and resume meanwhile (`_no_estop_path`). Before, nothing was tried again and nothing
was refused, so a running goal could keep driving and new work was accepted.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from types import SimpleNamespace as NS

import pytest

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"
FAILING = "the robot's stop is failing"
LATCHED = "set this robot's local latch"


class _Robot(SimulatedRobot):
    """A binding whose stop raises `fail` times, then goes out (None: always raises). Counts the
    stops, the threads that send them, and the motion it is sent."""

    def __init__(self, fail: int | None = None) -> None:
        super().__init__()
        self.fail, self.stops, self.moves, self.resumes = fail, 0, 0, 0
        self.senders: list = []

    def command_stop(self, robot_id: str) -> None:
        self.stops += 1
        self.senders.append(threading.get_ident())
        if self.fail is None or self.stops <= self.fail:
            raise ValueError("binding bug: command_stop")
        super().command_stop(robot_id)

    def command_move(self, robot_id, x, y, yaw=0.0) -> None:
        self.moves += 1
        super().command_move(robot_id, x, y, yaw)

    def command_resume(self, robot_id) -> None:
        self.resumes += 1
        super().command_resume(robot_id)


class _ConfirmRaises(_Robot):
    """A binding whose stop goes out, but whose confirmation raises (a bug in `state_seq`)."""

    def state_seq(self, robot_id: str) -> int:
        raise ValueError("binding bug: state_seq")


def _adapter(robot: _Robot) -> Adapter:
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._safety_required = True
    a._safety_up.set()          # the SafetyStream is up: only the failing stop refuses motion
    return a


def _fast_retries(monkeypatch) -> None:
    """The stop budget 40 x 5 ms (0.2 s), and the later retries every 20 ms."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.005)
    monkeypatch.setattr(adapter_mod, "_SAFETY_STOP_RETRY_S", 0.02)


class _Edge:
    """An edge node that sends the estops `ids`, and reads what the adapter sends back on its own
    thread."""

    def __init__(self, *ids: str) -> None:
        self.ids, self.sent = ids, []

    def EdgeStream(self, requests):  # the gRPC method name
        def read() -> None:
            for m in requests:
                self.sent.append(m)
        threading.Thread(target=read, daemon=True).start()
        return iter([pb.EdgeControlMessage(estop=pb.Estop(estop_id=i, reason="test"))
                     for i in self.ids])

    def acks(self, want: int, timeout: float = 2.0) -> dict:
        end = time.monotonic() + timeout
        while True:
            acks = {m.estop_ack.estop_id: m.estop_ack for m in list(self.sent)
                    if m.WhichOneof("msg") == "estop_ack"
                    and m.estop_ack.state != pb.ESTOP_STATE_STOPPING}
            if len(acks) >= want or time.monotonic() > end:
                return acks
            time.sleep(0.01)


def _edge_estop(a: Adapter, *ids: str) -> _Edge:
    edge = _Edge(*ids)
    a._pb_grpc = NS(EdgeServiceStub=lambda channel: edge)
    a._edge_stream("localhost:1")                    # returns when the edge stream ends
    return edge


def _drill(a: Adapter) -> list:
    a._sim = NS(estop_due=lambda: True)
    a._safety_outbox = queue.Queue()
    a._maybe_estop_drill()
    return [a._safety_outbox.get_nowait().estop_ack for _ in range(a._safety_outbox.qsize())]


def _navigate(action_id: str = "t-2", token: int = 2):
    return pb.AssignAction(action_id=action_id, action_type="Navigate",
                           payload_json=b'{"x": 3, "y": 4}', fencing_token=token,
                           lease_generation=1, lease_duration_ms=60_000)


def _until(cond, timeout: float = 1.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def _retry_threads() -> set:
    return {t for t in threading.enumerate() if t.name == "estop-stop-retry"}


def test_an_edge_estop_whose_stop_raises_refuses_new_work_and_keeps_commanding_the_stop(
        monkeypatch):
    _fast_retries(monkeypatch)
    robot = _Robot()
    a = _adapter(robot)
    before = _retry_threads()
    try:
        edge = _edge_estop(a, "edge-1")
        assert edge.acks(1)["edge-1"].state == pb.ESTOP_STATE_FAILED
        assert FAILING in a._no_estop_path(), "new work is accepted after a stop that raised"
        n = robot.stops
        assert _until(lambda: robot.stops > n + 2), "the stop is not commanded again"
        res = pb.CommandResult(command_id=1, robot_id=ROBOT)
        a._on_assign(_navigate(), res)
        assert not res.assign_action.accepted
        assert res.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
        assert FAILING in res.assign_action.message, res.assign_action
        assert robot.moves == 0, "a goal was sent while the stop is failing"
    finally:
        a._shutdown.set()
    assert _until(lambda: not (_retry_threads() - before)), "the retry outlived shutdown"


def test_the_action_an_edge_estop_stopped_is_not_new_work_that_ends_the_refusal(monkeypatch):
    """The action running at the estop is still current while its stop is retried: the retry
    must not take it for new work and give way (no stop went out)."""
    _fast_retries(monkeypatch)
    robot = _Robot()
    a = _adapter(robot)
    a._current_action, a._current_fencing_token = "t-1", 1
    try:
        _edge_estop(a, "edge-1")
        time.sleep(0.35)                             # past the 0.2 s budget
        assert FAILING in a._no_estop_path(), "the refusal ended though no stop went out"
        n = robot.stops
        assert _until(lambda: robot.stops > n), "the retry gave way to the estopped action"
    finally:
        a._shutdown.set()


def test_an_edge_estop_whose_stop_goes_out_on_a_retry_lifts_the_retrys_refusal(monkeypatch,
                                                                               capsys):
    """The retry's refusal ends; the edge estop's local latch still refuses."""
    _fast_retries(monkeypatch)
    robot = _Robot(fail=3)                           # the estop's own stop and two retries raise
    a = _adapter(robot)
    try:
        edge = _edge_estop(a, "edge-1")
        assert edge.acks(1)["edge-1"].state == pb.ESTOP_STATE_FAILED
        assert _until(lambda: FAILING not in a._no_estop_path()), a._no_estop_path()
        assert LATCHED in a._no_estop_path()
        # Three raised; then one went out. The stop after the latch is one of the four, or a
        # fifth if it went out while the retry was still to send its own.
        assert robot.stops in (4, 5), robot.stops
        assert _until(lambda: "once estop edge-1 raised" in capsys.readouterr().err)
    finally:
        a._shutdown.set()


def test_an_edge_estop_whose_confirmation_raises_commands_the_stop_again_once():
    """Whether the stop went out cannot be told from a fault: it is commanded again, and the
    retry's refusal ends as soon as one goes out (the local latch's does not)."""
    robot = _ConfirmRaises(fail=0)
    a = _adapter(robot)
    try:
        edge = _edge_estop(a, "edge-1")
        assert edge.acks(1)["edge-1"].state == pb.ESTOP_STATE_FAILED
        assert _until(lambda: FAILING not in a._no_estop_path()), a._no_estop_path()
        assert LATCHED in a._no_estop_path()
        # The estop's, the retry's, and the stop after the latch.
        assert _until(lambda: robot.stops == 3), robot.stops
        time.sleep(0.05)
        assert robot.stops == 3
    finally:
        a._shutdown.set()


def test_a_drill_estop_whose_stop_raises_refuses_new_work_and_keeps_commanding_the_stop(
        monkeypatch):
    _fast_retries(monkeypatch)
    robot = _Robot()
    a = _adapter(robot)
    a._current_action, a._current_fencing_token = "t-1", 1
    try:
        acks = _drill(a)
        assert acks and acks[-1].state == pb.ESTOP_STATE_FAILED
        assert FAILING in a._no_estop_path(), "new work is accepted after a stop that raised"
        n = robot.stops
        assert _until(lambda: robot.stops > n + 2), "the stop is not commanded again"
        res = pb.CommandResult(command_id=2, robot_id=ROBOT)
        a._on_resume(type(pb.Command().resume)(), res)
        assert not res.resume.resumed
        assert FAILING in res.resume.message, res.resume
        assert robot.resumes == 0
    finally:
        a._shutdown.set()


def test_a_drill_estop_whose_stop_goes_out_on_a_retry_lifts_the_refusal(monkeypatch):
    _fast_retries(monkeypatch)
    robot = _Robot(fail=2)
    a = _adapter(robot)
    try:
        _drill(a)
        assert _until(lambda: a._no_estop_path() == ""), a._no_estop_path()
        assert robot.stops == 3
    finally:
        a._shutdown.set()


def test_an_edge_estop_whose_stop_raises_takes_a_lost_streams_retry_over(monkeypatch):
    """One retry at a time: the estop's retry takes over the loss's, and the lost action says so.
    The budget is 2 s here, so the estop lands inside it on a slow machine too."""
    _fast_retries(monkeypatch)
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICKS", 400)
    robot = _Robot()
    a = _adapter(robot)
    a._current_action, a._current_fencing_token = "t-1", 1
    try:
        a._safety_down("test: the stream ended", 10.0)
        time.sleep(0.03)
        a._safety_up.set()                           # back, so only the failing stop refuses
        _edge_estop(a, "edge-1")
        time.sleep(0.05)
        sent = len(robot.senders)
        time.sleep(0.15)
        assert len(set(robot.senders[sent:])) == 1, "more than one thread is sending the stop"
        assert FAILING in a._no_estop_path()
        ends = []
        end = time.monotonic() + 1.0
        while time.monotonic() < end and not ends:
            with contextlib.suppress(queue.Empty):
                m = a._outbox.get(timeout=0.02)
                if m.WhichOneof("payload") == "action_status":
                    ends.append(m.action_status)
        assert len(ends) == 1 and "or an estop raised" in ends[0].message, ends
    finally:
        a._shutdown.set()


@pytest.mark.parametrize("exc", [BrokenPipeError(32, "Broken pipe"), OSError(5, "I/O error")])
def test_a_stderr_that_raises_never_keeps_an_edge_estop_from_refusing_new_work(monkeypatch,
                                                                               exc):
    """The refusal and the retry come before the fault is logged, and a log that raises never
    ends the EdgeStream: the next estop on it is still answered."""
    _fast_retries(monkeypatch)
    robot = _Robot()
    a = _adapter(robot)

    class _RaisingStderr:
        def write(self, text: str) -> int:
            raise exc

        def flush(self) -> None:
            raise exc

    try:
        with monkeypatch.context() as m:
            m.setattr(adapter_mod.sys, "stderr", _RaisingStderr())
            edge = _edge_estop(a, "edge-1", "edge-2")
        assert set(edge.acks(2)) == {"edge-1", "edge-2"}, "a log that raised ended the EdgeStream"
        assert FAILING in a._no_estop_path(), "the log line came first, and raised"
        n = robot.stops
        assert _until(lambda: robot.stops > n), "the stop is not commanded again"
    finally:
        a._shutdown.set()
