# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A lapsed lease is noticed at once, whatever the telemetry cadence (C4.2).

What used to happen:

* **the lease was ticked by the telemetry loop,** once per cycle. A cycle ends in a sleep of the
  control plane's `telemetry_interval_seconds`, 1-30 s. So a lease that lapsed just after a tick
  was noticed up to 30 s later, and the robot went on executing all that time;
* **the self-stop ran on that thread,** and blocked telemetry while it confirmed rest.

Now the lease has a thread of its own that looks every `_LEASE_TICK_S`. Its table is read and
changed under a lock held only for the lookup, never across the self-stop, so a renewal or an
assignment never waits behind a stop in progress. A failure in one self-stop is logged, and the
thread goes on: a dead lease thread would silently turn C4.2 off.

The self-stop reports and clears only the action whose lease lapsed. An action assigned while
the stop was being confirmed is not reported as stopped, and it is not dropped.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace as NS

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"
TERMINAL = (pb.ACTION_STATE_STOPPED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED)


def _lapse(a: Adapter):
    """One look at the lease, the way the adapter takes it. (The red run predates
    `_tick_lease`; there the telemetry loop's `_lease.tick()` was the look.)"""
    return (getattr(a, "_tick_lease", None) or a._lease.tick)()


def _assign_msg(action_id: str, token: int, lease_ms: int):
    return pb.ControlPlaneMessage(command=pb.Command(
        command_id=token, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id=action_id, assignment_id=f"a-{action_id}", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=token, lease_generation=1,
            lease_duration_ms=lease_ms)))


class _ControlPlane:
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.opened = threading.Event()

    def ControlStream(self, requests):  # noqa: N802 (gRPC method name)
        self.opened.set()

        def stream():
            while (m := self.q.get()) is not None:
                yield m
        return stream()


def _first(q: queue.Queue, pick, within: float):
    deadline = time.monotonic() + within
    while (left := deadline - time.monotonic()) > 0:
        try:
            m = q.get(timeout=left)
        except queue.Empty:
            return None
        if pick(m):
            return m
    return None


def test_a_lapse_is_self_stopped_at_once_even_at_the_slowest_telemetry_cadence():
    # Through the shipped entry point: handshake, a RegisterAck at the slowest cadence C2.3
    # allows (30 s), then a 200 ms lease that is never renewed.
    a = Adapter(pb, None, SimulatedRobot(), ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    try:
        assert cp.opened.wait(2.0)
        cp.q.put(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)))
        cp.q.put(pb.ControlPlaneMessage(register_ack=pb.RegisterAck(
            accepted=True, telemetry_interval_seconds=30)))
        time.sleep(0.5)                  # telemetry is now in a 30 s sleep
        cp.q.put(_assign_msg("t-1", 1, lease_ms=200))
        t0 = time.monotonic()
        upd = _first(a._outbox, lambda m: m.WhichOneof("payload") == "action_status"
                     and m.action_status.action_id == "t-1"
                     and m.action_status.state in TERMINAL, 1.0)
        assert upd is not None, "the lapsed lease was not self-stopped within 1 s"
        assert time.monotonic() - t0 < 0.2 + 0.3   # the lease, plus a tick and the stop
        assert upd.action_status.end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    finally:
        a._shutdown.set()
        cp.q.put(None)
        control.join(timeout=3.0)


# ── the self-stop and the control thread, side by side ──────────────────────────────────

class _Twist:
    def __init__(self) -> None:
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class _Publisher:
    def publish(self, msg) -> None:
        pass


class _Handle:
    accepted = True

    def cancel_goal_async(self) -> None:
        pass

    def get_result_async(self):
        return NS(add_done_callback=lambda cb: None)


def _odom(vx: float):
    return NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))))


@pytest.fixture
def moving(monkeypatch):
    """A robot that stays in motion, so a self-stop spends its whole budget (40 x 0.01 s)
    confirming, with its first goal accepted and adopted."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.01)
    b = mark_transport_live(Nav2Binding(ROBOT))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(epoch)
    b._server_ready = lambda: True
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b.on_odom(_odom(0.4))
    a = Adapter(pb, None, b, ROBOT)
    a._handle_control_plane(_assign_msg("t-1", 1, lease_ms=1), None)
    b._on_goal_response(NS(result=lambda: _Handle()), sent[0])
    time.sleep(0.01)                     # t-1's lease lapses
    return a, b


def _statuses(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        m = a._outbox.get_nowait()
        if m.WhichOneof("payload") == "action_status":
            out.append(m.action_status)
    return out


def test_an_action_assigned_while_a_self_stop_confirms_is_neither_reported_nor_dropped(moving):
    a, b = moving
    stop = threading.Thread(target=_lapse, args=(a,), daemon=True)
    stop.start()
    time.sleep(0.1)                      # the self-stop is confirming rest
    a._handle_control_plane(_assign_msg("t-2", 2, lease_ms=60_000), None)
    stop.join(3.0)
    st = _statuses(a)
    t1 = [s for s in st if s.action_id == "t-1" and s.state in TERMINAL]
    t2 = [s for s in st if s.action_id == "t-2" and s.state in TERMINAL]
    assert len(t1) == 1 and t1[0].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    assert t2 == []                      # t-2's lease did not lapse; it was not stopped
    assert a._current_action == "t-2"    # and it is still the action being executed


def test_a_renewal_does_not_wait_behind_a_self_stop_in_progress(moving):
    a, b = moving
    stop = threading.Thread(target=_lapse, args=(a,), daemon=True)
    stop.start()
    time.sleep(0.1)                      # the self-stop is confirming rest
    t0 = time.monotonic()
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=9, robot_id=ROBOT, renew_lease=pb.RenewActionLease(
            action_id="t-1", lease_generation=1, lease_duration_ms=1000))), None)
    took = time.monotonic() - t0
    stop.join(3.0)
    assert took < 0.05, f"the renewal waited {took:.3f} s behind the self-stop"


def test_the_lease_thread_survives_concurrent_renewals_and_a_failing_self_stop(monkeypatch):
    a = Adapter(pb, None, SimulatedRobot(), ROBOT)
    real = a._self_stop
    calls = []

    def flaky(robot_id, *rest):
        calls.append(robot_id)
        if len(calls) == 1:
            raise RuntimeError("a self-stop that fails")
        real(robot_id, *rest)
    monkeypatch.setattr(a, "_self_stop", flaky)
    loop = threading.Thread(target=a._lease_loop, daemon=True)
    loop.start()
    try:
        # 1. A lapse whose self-stop raises.
        a._handle_control_plane(_assign_msg("t-1", 1, lease_ms=50), None)
        deadline = time.monotonic() + 1.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls, "the first lapse was never self-stopped"
        # 2. Renewals from another thread, hammering the lease table while it is watched;
        #    their 10 ms leases lapse now and then, and those self-stops run too.
        stop_renewing = threading.Event()

        def renew():
            while not stop_renewing.is_set():
                a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
                    command_id=9, robot_id=ROBOT, renew_lease=pb.RenewActionLease(
                        action_id="t-1", lease_generation=5, lease_duration_ms=10))), None)
        r = threading.Thread(target=renew, daemon=True)
        r.start()
        time.sleep(0.3)
        stop_renewing.set()
        r.join(2.0)
        # 3. After both, a lapse is still noticed and reported.
        a._handle_control_plane(_assign_msg("t-3", 3, lease_ms=50), None)
        deadline = time.monotonic() + 1.0
        seen = False
        while time.monotonic() < deadline and not seen:
            seen = any(s.action_id == "t-3" and s.state in TERMINAL for s in _statuses(a))
            time.sleep(0.02)
        assert loop.is_alive()
        assert seen, "after a failing self-stop and concurrent renewals, a lapse went unreported"
        assert len(calls) >= 2
    finally:
        a._shutdown.set()
        loop.join(timeout=2.0)
