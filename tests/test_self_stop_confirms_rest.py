# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The lease self-stop commands the stop first, then reports what it observed (C4.2).

C4.2 says: bring the task to a CONFIRMED stop and report it. STOPPED only once the robot is
observed at rest; FAILED when rest cannot be confirmed within the stopping budget.

What the self-stop used to do:

* it commanded the stop, then waited up to 2 s for the navigation server to answer every goal
  sent before it;
* if one was still unanswered, it reported UNKNOWN. C4.2 does not count UNKNOWN, so the stop was
  reported as no stop at all;
* once nothing was unanswered, it reported STOPPED without looking at the robot at all.

Now the stop is commanded first, whatever state the goal is in. Then the robot is looked at for
the confirmed-stop budget, the estop's 40 x 0.05 s, by the estop's own rule (`is_stopped`):

* **confirmed at rest, and nothing sent before the stop still unanswered:** STOPPED;
* **anything else:** FAILED, and the report says why. A goal the server has not answered may
  still be accepted and drive, so it refutes rest whatever odom says; it is cancelled on arrival.

It never reports UNKNOWN, and it never answers differently from an estop that saw the same.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace as NS

import pytest
from conftest import mark_transport_live
from test_stop_never_waits import _SafetyStub

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"
# What the harness's C4.2 counts as a stop, and how long it waits (harness.py _check_leases).
C42_TERMINAL = (pb.ACTION_STATE_STOPPED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED)
C42_WINDOW_S = 4.0
C42_LEASE_MS = 500
HARNESS_TELEMETRY_S = 1.0   # the telemetry_interval_seconds the harness's RegisterAck sets


class _Twist:
    def __init__(self) -> None:
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class _Publisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _Handle:
    def __init__(self) -> None:
        self.accepted = True
        self.cancels = 0
        self.on_result = None

    def cancel_goal_async(self) -> None:
        self.cancels += 1

    def get_result_async(self):
        handle = self
        return NS(add_done_callback=lambda cb: setattr(handle, "on_result", cb))


class _Done:
    def __init__(self, v) -> None:
        self._v = v

    def result(self):
        return self._v


class _TimedQueue(queue.Queue):
    """The adapter's outbox, stamping each message with when it was put."""

    def put(self, item, block=True, timeout=None) -> None:
        super().put((time.monotonic(), item), block, timeout)


def _odom(vx: float = 0.0):
    return NS(pose=NS(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))))


def _rig(vx: float = 0.0):
    b = mark_transport_live(Nav2Binding(ROBOT, odometry_source="measured"))
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(epoch)
    b._server_ready = lambda: True                         # the server is up: goals go at once
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b.on_odom(_odom(vx))
    a = Adapter(pb, None, b, ROBOT)
    a._outbox = _TimedQueue()
    return a, b, sent


def _assign(a: Adapter, lease_ms: int) -> float:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=pb.AssignAction(
            action_id="t-lease-expiry", assignment_id="a-lease", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=11, lease_generation=1,
            lease_duration_ms=lease_ms))), None)
    return time.monotonic()                                # the harness starts its wait here


def _statuses(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        at, m = a._outbox.get_nowait()
        if m.WhichOneof("payload") == "action_status":
            out.append((at, m.action_status))
    return out


def _zero_twists_only(b: Nav2Binding) -> bool:
    tw = b._cmd_vel.published
    return bool(tw) and all(t.linear.x == 0.0 and t.angular.z == 0.0 for t in tw)


# ── reproduced from hardware: C4.2 with the goal unanswered past 2 s at lease expiry ─────

def test_c4_2_a_goal_unanswered_past_2_s_at_lease_expiry_still_gets_a_terminal_report():
    """What the harness's C4.2 does, on the adapter's own clock: a 500 ms lease, no renewal,
    and a wait of 4 s for a STOPPED, FAILED or CANCELLED action_status. The adapter's own lease
    thread watches the lease, and its telemetry loop runs at the harness's 1 s cadence. The
    navigation server does not answer the goal until 4.5 s after the assignment, well past the
    2 s the stop used to wait for it. Real budgets throughout."""
    a, b, sent = _rig(vx=0.0)                              # odom: at rest
    a._telemetry_interval = HARNESS_TELEMETRY_S
    t0 = _assign(a, C42_LEASE_MS)
    loop = threading.Thread(target=a._telemetry_loop, args=(None,), daemon=True)
    loop.start()
    lease = threading.Thread(target=a._lease_loop, daemon=True)
    lease.start()
    h = _Handle()
    late = threading.Timer(4.5, lambda: b._on_goal_response(_Done(h), sent[0]))
    late.start()
    try:
        terminal = None
        while time.monotonic() - t0 < C42_WINDOW_S and terminal is None:
            time.sleep(0.02)
            terminal = next(((at, s) for at, s in _statuses(a)
                             if s.action_id == "t-lease-expiry" and s.state in C42_TERMINAL),
                            None)
        assert terminal is not None, "C4.2: no terminal action_status within 4 s of lease expiry"
        at, st = terminal
        assert at - t0 <= C42_WINDOW_S
        # Odom says rest, but a goal the server may still accept refutes it: FAILED, which is
        # terminal, and the report says why.
        assert st.state == pb.ACTION_STATE_FAILED
        assert st.end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
        assert "not been answered" in st.message and "not known to be stopped" in st.message
        assert _zero_twists_only(b)                        # the stop went out first
        late.join()
        assert h.cancels == 1 and b._goal_handle is None   # the late goal: cancelled on arrival
    finally:
        late.cancel()
        a._shutdown.set()
        loop.join(timeout=3.0)
        lease.join(timeout=3.0)


# ── what "confirmed" means, on the same path ────────────────────────────────────────────

@pytest.fixture
def short_budget(monkeypatch):
    # The same rule on a shorter clock: 40 looks, 0.01 s apart.
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.01)


def _adopted(a: Adapter, b: Nav2Binding, sent: list) -> _Handle:
    _assign(a, lease_ms=1)
    h = _Handle()
    b._on_goal_response(_Done(h), sent[0])                 # accepted and adopted
    time.sleep(0.01)                                       # the lease lapses
    return h


def test_self_stop_of_a_moving_robot_reports_stopped_only_once_rest_is_observed(short_budget):
    a, b, sent = _rig(vx=0.4)                              # driving
    h = _adopted(a, b, sent)
    rest_at: list[float] = []

    def decelerate() -> None:
        time.sleep(0.15)
        rest_at.append(time.monotonic())
        b.on_odom(_odom(0.0))
    threading.Thread(target=decelerate, daemon=True).start()
    a._tick_lease()                                        # self-stop, on this thread
    st = [(at, s) for at, s in _statuses(a) if s.action_id == "t-lease-expiry"]
    assert [s.state for _, s in st] == [pb.ACTION_STATE_RUNNING, pb.ACTION_STATE_STOPPED]
    assert rest_at and st[-1][0] >= rest_at[0]             # not before the robot was at rest
    assert st[-1][1].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    assert st[-1][1].message == ""                         # rest observed, nothing unanswered
    assert h.cancels == 1 and _zero_twists_only(b)


def test_self_stop_of_a_robot_never_observed_at_rest_is_failed_not_stopped(short_budget):
    a, b, sent = _rig(vx=0.4)                              # and it stays at 0.4 m/s
    h = _adopted(a, b, sent)
    a._tick_lease()
    st = [s for _, s in _statuses(a) if s.action_id == "t-lease-expiry"]
    assert [s.state for s in st] == [pb.ACTION_STATE_RUNNING, pb.ACTION_STATE_FAILED]
    assert st[-1].end_cause == pb.ACTION_END_CAUSE_LEASE_EXPIRED
    assert "rest was not observed" in st[-1].message
    assert h.cancels == 1 and _zero_twists_only(b)


def test_self_stop_of_a_robot_moving_with_its_goal_unanswered_is_failed(short_budget):
    a, b, sent = _rig(vx=0.4)
    _assign(a, lease_ms=1)                                 # the server never answers
    time.sleep(0.01)
    a._tick_lease()
    st = [s for _, s in _statuses(a) if s.action_id == "t-lease-expiry"]
    assert [s.state for s in st] == [pb.ACTION_STATE_RUNNING, pb.ACTION_STATE_FAILED]
    assert "not been answered" in st[-1].message and "not known to be stopped" in st[-1].message
    assert _zero_twists_only(b)


def test_estop_and_self_stop_give_the_same_answer_for_rest_with_a_goal_unanswered(short_budget):
    # One rule for both stops: odom at rest does not make a stop while a goal the server may
    # still accept is unanswered. The estop says FAILED; so does the self-stop.
    a, b, sent = _rig(vx=0.0)
    _assign(a, lease_ms=60_000)
    stub = _SafetyStub(a)
    a.run_safety_stream(stub)
    acks = [m.estop_ack.state for m in stub.acks]
    assert acks[-1] == pb.ESTOP_STATE_FAILED

    a, b, sent = _rig(vx=0.0)
    _assign(a, lease_ms=1)
    time.sleep(0.01)
    a._tick_lease()
    st = [s for _, s in _statuses(a) if s.action_id == "t-lease-expiry"]
    assert st[-1].state == pb.ACTION_STATE_FAILED


def test_self_stop_on_a_dead_transport_is_failed_and_says_why(short_budget):
    a, b, sent = _rig(vx=0.0)
    _adopted(a, b, sent)
    b._record_spin_exit(RuntimeError("executor died"))     # odom is frozen from here on
    a._tick_lease()
    st = [s for _, s in _statuses(a) if s.action_id == "t-lease-expiry"]
    assert [s.state for s in st] == [pb.ACTION_STATE_RUNNING, pb.ACTION_STATE_FAILED]
    assert "executor" in st[-1].message
