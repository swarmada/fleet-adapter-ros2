# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""No stop waits behind a goal dispatch.

An assignment that arrived while the navigation server was not up blocked the control thread in
`wait_for_server()`, which has no bound, while it held the adapter's robot lock. Every stop path
took that lock before commanding anything. So an estop got no ack at all, not even the STOPPING
receipt, a lapsed lease could not self-stop, and a cancel was never read: it was queued on the
same control thread, behind the assignment.

Now nothing that waits for the server runs on a caller's thread. A goal the server is not up for
is parked, and the binding's own sender thread waits for the server while holding nothing. A stop
withdraws a parked goal, and no stop takes the adapter's robot lock.

Each test holds a real `_send_nav2_goal` in a `wait_for_server()` that does not return. It
requires the stop commanded (a zero Twist) and the path's first answer within C5.5's 500 ms.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from safety_call import SafetyCall
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
BUDGET_S = 0.5   # C5.5: the first answer to an estop


class _Future:
    """What send_goal_async returns: the goal response arrives through its done callback."""

    def __init__(self) -> None:
        self.cb = None

    def add_done_callback(self, cb) -> None:
        self.cb = cb


class _Server:
    """The NavigateToPose action client. Until `up` is set its server has not come up:
    `wait_for_server()` blocks as rclpy's does when given no timeout, and `server_is_ready()`
    looks without waiting. Records every goal that reaches the server."""

    def __init__(self, up: bool = False) -> None:
        self.up = threading.Event()
        if up:
            self.up.set()
        self.waiting = threading.Event()   # a caller is inside wait_for_server()
        self.sent: list = []
        self.futures: list[_Future] = []

    def server_is_ready(self) -> bool:
        return self.up.is_set()

    def wait_for_server(self, timeout_sec=None) -> bool:
        self.waiting.set()
        return self.up.wait(timeout_sec)

    def send_goal_async(self, goal, feedback_callback=None, goal_uuid=None) -> _Future:
        self.sent.append(goal)
        self.futures.append(_Future())
        return self.futures[-1]


class _Handle:
    def __init__(self) -> None:
        self.accepted = True
        self.cancels = 0

    def cancel_goal_async(self) -> None:
        self.cancels += 1

    def get_result_async(self):
        return NS(add_done_callback=lambda cb: None)


class _Twist:
    def __init__(self) -> None:
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class _Publisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


def _odom(vx: float = 0.0):
    return NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))))


@pytest.fixture
def ros_msgs(monkeypatch):
    """geometry_msgs, which `_send_nav2_goal` imports to build the goal's pose, and
    unique_identifier_msgs, for the id it sends the goal with."""
    ids = types.ModuleType("unique_identifier_msgs.msg")
    ids.UUID = lambda uuid: NS(uuid=uuid)
    ids_pkg = types.ModuleType("unique_identifier_msgs")
    ids_pkg.msg = ids
    monkeypatch.setitem(sys.modules, "unique_identifier_msgs", ids_pkg)
    monkeypatch.setitem(sys.modules, "unique_identifier_msgs.msg", ids)
    msg = types.ModuleType("geometry_msgs.msg")
    msg.PoseStamped = lambda: NS(header=NS(frame_id="", stamp=None),
                                 pose=NS(position=NS(x=0.0, y=0.0),
                                         orientation=NS(z=0.0, w=1.0)))
    pkg = types.ModuleType("geometry_msgs")
    pkg.msg = msg
    monkeypatch.setitem(sys.modules, "geometry_msgs", pkg)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", msg)


class _ControlPlane:
    """The control plane's end of ControlStream. The adapter's control thread reads these in
    order, one at a time, as `run_control_stream` does."""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.opened = threading.Event()

    def ControlStream(self, requests):  # noqa: N802 (gRPC method name)
        self.opened.set()   # the adapter has its session outbox by now
        def stream():
            while (m := self.q.get()) is not None:
                yield m
        return stream()

    def send(self, command_id: int, **command) -> None:
        self.q.put(pb.ControlPlaneMessage(command=pb.Command(
            command_id=command_id, robot_id=ROBOT, **command)))


class _SafetyStub:
    """A control plane that sends one Estop down the SafetyStream and keeps the stream open, as
    the control plane does: the adapter redials a stream that ends. Given the adapter, it ends
    the adapter's SafetyStream loop instead, once the estop has been handled, and keeps the
    acks in `acks`, so a test can call `run_safety_stream` on its own thread and read them."""

    def __init__(self, adapter=None) -> None:
        self.adapter = adapter
        self.acks: list = []

    def SafetyStream(self, requests):  # noqa: N802 (gRPC method name)
        def stream():
            yield pb.ControlPlaneSafetyMessage(
                robot_id=ROBOT, estop=pb.Estop(estop_id="e-1", reason="test"))
            if self.adapter is None:
                threading.Event().wait()
            outbox = self.adapter._safety_outbox
            while not outbox.empty():
                self.acks.append(outbox.get_nowait())
            self.adapter._shutdown.set()
        return SafetyCall(stream())


def _assign(action_id: str, token: int, lease_ms: int = 60_000):
    return pb.AssignAction(
        action_id=action_id, assignment_id=f"a-{action_id}", action_type="Navigate",
        payload_json=b'{"x": 3, "y": 0}', fencing_token=token, lease_generation=1,
        lease_duration_ms=lease_ms)


@pytest.fixture
def rig(ros_msgs):
    """The adapter on a ROS-free Nav2Binding whose real `_send_nav2_goal` talks to `_Server`,
    with its control thread running. Torn down by bringing the server up and ending the stream,
    so a thread still stuck at the end of a failing test is released."""
    server = _Server()
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b.start = lambda: None               # bring-up: the stand-ins above are the transport
    b.on_odom(_odom(0.0))                # odom says rest
    a = Adapter(pb, None, b, ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    assert cp.opened.wait(2.0)
    yield a, b, server, cp
    server.up.set()
    cp.q.put(None)
    control.join(timeout=5.0)


def _first(q: queue.Queue, pick, within: float):
    """The first message on `q` that `pick` accepts, within `within` s, else None."""
    deadline = time.monotonic() + within
    while (left := deadline - time.monotonic()) > 0:
        try:
            m = q.get(timeout=left)
        except queue.Empty:
            return None
        if pick(m):
            return m
    return None


def _halted(b: Nav2Binding) -> bool:
    """The lowest-level stop went out: a zero Twist, and nothing but zero Twists."""
    tw = b._cmd_vel.published
    return bool(tw) and all(t.linear.x == 0.0 and t.angular.z == 0.0 for t in tw)


def test_an_estop_is_acked_within_500_ms_while_a_dispatch_waits_for_the_server(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert server.waiting.wait(2.0)                        # the dispatch is in wait_for_server()
    t0 = time.monotonic()
    threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),), daemon=True).start()
    first = _first(a._safety_outbox, lambda m: True, BUDGET_S)
    assert first is not None, "no EstopAck within 500 ms"
    assert time.monotonic() - t0 <= BUDGET_S
    assert _halted(b)
    # The goal was never sent, so nothing can still start the robot: rest is confirmed at the
    # first look, and that is a single STOPPED.
    assert first.estop_ack.state == pb.ESTOP_STATE_STOPPED
    assert first.estop_ack.stop_initiated_at > 0
    server.up.set()                                        # the server comes up afterwards
    time.sleep(0.2)
    assert server.sent == []                               # the stopped goal is never sent


def test_a_lease_self_stop_reports_within_500_ms_while_a_dispatch_waits_for_the_server(rig):
    a, b, server, cp = rig
    server.up.set()
    cp.send(1, assign_action=_assign("t-1", 1, lease_ms=300))
    assert _first(a._outbox, lambda m: m.WhichOneof("payload") == "command_result", 2.0)
    h = _Handle()
    server.futures[0].cb(NS(result=lambda: h))             # t-1 accepted and adopted
    server.up.clear()                                      # then the server goes away
    server.waiting.clear()
    cp.send(2, assign_action=_assign("t-2", 2, lease_ms=300))
    assert server.waiting.wait(2.0)                        # t-2's dispatch waits for it
    time.sleep(0.4)                                        # both leases have lapsed
    t0 = time.monotonic()
    threading.Thread(target=a._tick_lease, daemon=True).start()   # the lease thread's call
    terminal = (pb.ACTION_STATE_STOPPED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED)
    upd = _first(a._outbox, lambda m: m.WhichOneof("payload") == "action_status"
                 and m.action_status.state in terminal, BUDGET_S)
    assert upd is not None, "no terminal action_status within 500 ms of the lapse"
    assert time.monotonic() - t0 <= BUDGET_S
    assert _halted(b) and h.cancels == 1
    assert upd.action_status.state == pb.ACTION_STATE_STOPPED
    assert upd.action_status.end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    server.up.set()
    time.sleep(0.2)
    assert len(server.sent) == 1                           # t-2's goal is never sent


def test_a_cancel_is_answered_within_500_ms_while_a_dispatch_waits_for_the_server(rig):
    a, b, server, cp = rig
    cp.send(1, assign_action=_assign("t-1", 1))
    assert server.waiting.wait(2.0)
    t0 = time.monotonic()
    cp.send(2, cancel_action=pb.CancelAction(
        action_id="t-1", fencing_token=1, cause=pb.CANCEL_CAUSE_OPERATOR))
    res = _first(a._outbox, lambda m: m.WhichOneof("payload") == "command_result"
                 and m.command_result.WhichOneof("result") == "cancel_action", BUDGET_S)
    assert res is not None, "no CancelActionResult within 500 ms"
    assert time.monotonic() - t0 <= BUDGET_S
    assert _halted(b)
    # Nothing was sent, so nothing is left that could start the robot: the cancel settles.
    assert res.command_result.cancel_action.acknowledged is True
    assert res.command_result.cancel_action.disposition == pb.CANCEL_DISPOSITION_STOPPED_SAFELY
    server.up.set()
    time.sleep(0.2)
    assert server.sent == []


# ── the binding: a dispatch never waits, and a parked goal is still delivered ────────────

def _binding(server: _Server) -> Nav2Binding:
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    return b


def _move(b: Nav2Binding, server: _Server, x: float) -> None:
    """command_move, on a thread of its own so that one which waits for the server fails the
    test instead of hanging it; the server is brought up to release it."""
    t = threading.Thread(target=b.command_move, args=(ROBOT, x, 0.0), daemon=True)
    t.start()
    t.join(BUDGET_S)
    if t.is_alive():
        server.up.set()
        t.join(5.0)
        pytest.fail("command_move waited for the navigation server")


def test_command_move_returns_at_once_while_the_server_is_down(ros_msgs):
    server = _Server()
    _move(_binding(server), server, 3.0)


def test_a_goal_parked_while_the_server_is_down_is_sent_when_it_comes_up(ros_msgs):
    server = _Server()
    b = _binding(server)
    _move(b, server, 3.0)
    assert server.waiting.wait(2.0) and server.sent == []
    assert b.stop_settled(ROBOT, 0) is True                # nothing stopped, nothing to settle
    server.up.set()
    deadline = time.monotonic() + 2.0
    while not server.sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(server.sent) == 1
    h = _Handle()
    server.futures[0].cb(NS(result=lambda: h))
    assert b._goal_handle is h                             # adopted, as a goal sent at once is


def test_only_the_newest_parked_goal_is_sent(ros_msgs):
    server = _Server()
    b = _binding(server)
    _move(b, server, 1.0)
    _move(b, server, 2.0)                                  # replaces the first before it is sent
    assert server.waiting.wait(2.0)
    server.up.set()
    deadline = time.monotonic() + 2.0
    while not server.sent and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert [g.pose.pose.position.x for g in server.sent] == [2.0]


def test_a_stop_withdraws_a_parked_goal_and_settles_at_once(ros_msgs):
    server = _Server()
    b = _binding(server)
    b.on_odom(_odom(0.0))
    _move(b, server, 3.0)
    assert server.waiting.wait(2.0)
    b.command_stop(ROBOT)
    assert b.stop_settled(ROBOT, 0) is True                # never sent: nothing to wait for
    assert b.is_stopped(ROBOT) is True
    server.up.set()
    time.sleep(0.2)
    assert server.sent == []
