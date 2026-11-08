# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Teleop reaches the base only through the manual-drive action's gate, and a stop always wins.

The operator's teleop publishes on its own topic. The binding forwards a message to the base only
while a manual-drive action holds the robot's lease and is running; otherwise nothing is published.
Every stop path closes the gate under the lock a forwarded command is published under, before the
stop's own zero Twist, and only a new manual-drive assignment opens it again -- never a teleop
message. Run through the adapter, against the stand-in robot, so the assignment, the lease, the
renewal and every stop path are the shipped ones.
"""

from __future__ import annotations

import json
import threading
import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter

ROBOT = "amr-1"
ACTION = "manual-a"
DONE = (pb.ACTION_STATE_SUCCEEDED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED,
        pb.ACTION_STATE_STOPPED)


class StepClock:
    """The job's clock: simulated time that advances by each wait, with a real millisecond per
    wait so the test's thread gets a turn. The lease and the gate keep real monotonic time."""

    def __init__(self) -> None:
        self.t = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        with self._lock:
            self.t += seconds
        return event.wait(0.001)


def payload(**kw) -> bytes:
    doc = {"operation": md.OPERATION, "x": 1.0, "y": 0.0, "frame": "map", "radius_m": 0.25,
           "follows": "nav-a", **kw}
    return json.dumps(doc).encode()


def rig(**robot_kw):
    robot = md.SimulatedTeleopRobot(**robot_kw)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock())
    return a, robot


def drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def command(a: Adapter, command_id: int, **arm) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=command_id, robot_id=ROBOT, **arm)), None)


def assign(a: Adapter, body: bytes, *, action: str = ACTION, token: int = 1,
           lease_ms: int = 60_000, action_type: str = "Custom"):
    command(a, token, assign_action=pb.AssignAction(
        action_id=action, action_type=action_type, assignment_id=f"a-{action}",
        fencing_token=token, lease_generation=1, lease_duration_ms=lease_ms,
        payload_json=body))
    out = drain(a)
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"]
    assert len(res) == 1
    return res[0], out


def statuses(msgs, action: str = ACTION) -> list:
    return [m.action_status for m in msgs if m.WhichOneof("payload") == "action_status"
            and m.action_status.action_id == action]


def collect(a: Adapter, into: list, until, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        into.extend(drain(a))
        if until(into):
            return
        assert time.monotonic() < deadline, [s.message for s in statuses(into)]
        time.sleep(0.005)


def terminals(msgs, action: str = ACTION) -> list:
    return [s for s in statuses(msgs, action) if s.state in DONE]


def settle(a: Adapter, msgs: list, quiet_s: float = 0.15) -> None:
    """Let the job's thread run, then take everything said."""
    time.sleep(quiet_s)
    msgs.extend(drain(a))


# ── nothing running, nothing forwarded ──────────────────────────────────

def test_with_no_manual_drive_running_teleop_never_reaches_the_base() -> None:
    a, robot = rig()
    robot.teleop(0.1, 0.0, 0.3)
    assert robot.cmd_vel == []
    # Another action holding the lease is not a manual-drive: teleop stays outside.
    assert assign(a, b'{"x": 3, "y": 0}', action="nav-a", action_type="Navigate")[0].accepted
    before = list(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.3)
    assert robot.cmd_vel == before


def test_while_it_runs_teleop_is_forwarded_and_clamped() -> None:
    a, robot = rig()
    assert assign(a, payload())[0].accepted
    n = len(robot.cmd_vel)                      # opening holds the base: one zero
    robot.teleop(0.12, 0.0, 0.5)
    robot.teleop(0.5, 0.5, 5.0)
    robot.teleop(float("nan"), 0.0, float("inf"))
    (c1, c2, c3) = robot.cmd_vel[n:]
    assert c1 == (0.12, 0.0, 0.5)
    assert abs((c2[0] ** 2 + c2[1] ** 2) ** 0.5 - 0.20) < 1e-9 and c2[2] == 0.7
    assert c3 == md.ZERO


# ── no lease, no forwarding ─────────────────────────────────────────────

def test_teleop_stops_at_the_lease_deadline_before_the_lease_thread_looks() -> None:
    a, robot = rig()
    assert assign(a, payload(), lease_ms=60)[0].accepted
    robot.teleop(0.1, 0.0, 0.0)
    forwarded = len(robot.cmd_vel)
    time.sleep(0.09)                            # past the deadline; `_tick_lease` never ran
    robot.teleop(0.1, 0.0, 0.0)
    robot.teleop(0.0, 0.1, 0.2)
    assert len(robot.cmd_vel) == forwarded, robot.cmd_vel[forwarded:]


def test_a_renewal_of_this_action_extends_the_gate_and_another_does_not() -> None:
    a, robot = rig()
    assert assign(a, payload(), lease_ms=80)[0].accepted
    time.sleep(0.05)
    command(a, 2, renew_lease=pb.RenewActionLease(
        action_id="someone-else", lease_generation=1, lease_duration_ms=10_000))
    command(a, 3, renew_lease=pb.RenewActionLease(
        action_id=ACTION, lease_generation=1, lease_duration_ms=200))
    time.sleep(0.05)                            # past the first deadline, inside the renewed one
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel[n:] == [(0.1, 0.0, 0.0)]
    time.sleep(0.2)                             # past the renewed one
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n + 1


def test_a_manual_drive_with_no_lease_is_declined_and_nothing_opens() -> None:
    a, robot = rig()
    res, _ = assign(a, payload(), lease_ms=0)
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert "needs a lease" in res.message
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == []


def test_a_lapsed_lease_ends_it_once_and_closes_the_gate() -> None:
    a, robot = rig()
    assert assign(a, payload(), lease_ms=40)[0].accepted
    robot.teleop(0.1, 0.0, 0.0)
    time.sleep(0.06)
    a._tick_lease()
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    settle(a, msgs)
    assert [(s.state, s.end_cause) for s in terminals(msgs)] == [
        (pb.ACTION_STATE_STOPPED, pb.ACTION_END_CAUSE_LEASE_EXPIRED)]
    assert robot.cmd_vel[-1] == md.ZERO
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n


# ── a stop always wins, and teleop never clears one ─────────────────────

def test_an_estop_closes_the_gate_before_its_zero_and_teleop_never_reopens_it() -> None:
    a, robot = rig()
    assert assign(a, payload())[0].accepted
    robot.teleop(0.15, 0.0, 0.4)
    acks: list = []
    a._estop("e-1", acks.append)
    assert acks[-1].state == pb.ESTOP_STATE_STOPPED
    assert robot.cmd_vel[-1] == md.ZERO          # the stop's zero is the last thing published
    n = len(robot.cmd_vel)
    for _ in range(5):                           # the operator keeps pressing
        robot.teleop(0.15, 0.0, 0.4)
    assert len(robot.cmd_vel) == n
    assert not robot.teleop_open(ROBOT)
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    settle(a, msgs)
    end = terminals(msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED] and "stop" in end[0].message
    command(a, 9, renew_lease=pb.RenewActionLease(
        action_id=ACTION, lease_generation=1, lease_duration_ms=60_000))
    renew = [m.command_result.renew_lease for m in drain(a)
             if m.WhichOneof("payload") == "command_result"][-1]
    assert not renew.running
    robot.teleop(0.15, 0.0, 0.4)
    assert len(robot.cmd_vel) == n


def test_a_cancel_ends_it_once_as_cancelled_and_closes_the_gate() -> None:
    a, robot = rig()
    assert assign(a, payload())[0].accepted
    robot.teleop(0.1, 0.0, 0.0)
    command(a, 2, cancel_action=pb.CancelAction(action_id=ACTION, fencing_token=1,
                                                cause=pb.CANCEL_CAUSE_OPERATOR))
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    settle(a, msgs)
    assert [s.state for s in terminals(msgs)] == [pb.ACTION_STATE_CANCELLED]
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n and robot.cmd_vel[-1] == md.ZERO


def test_a_pause_ends_it_and_a_resume_does_not_reopen_teleop() -> None:
    a, robot = rig()
    assert assign(a, payload())[0].accepted
    command(a, 2, pause=pb.PauseCapabilities(require_stop_before_ack=True))
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    results = [m.command_result for m in msgs if m.WhichOneof("payload") == "command_result"]
    assert results[0].pause.paused
    assert [s.state for s in terminals(msgs)] == [pb.ACTION_STATE_FAILED]
    command(a, 3, resume=pb.ResumeCapabilities())
    assert not robot.teleop_open(ROBOT)
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n


def test_a_new_assignment_ends_the_drive_first_and_its_teleop_with_it() -> None:
    a, robot = rig()
    assert assign(a, payload())[0].accepted
    robot.teleop(0.1, 0.0, 0.0)
    assert assign(a, b'{"x": 1, "y": 2}', token=2, action="nav-b",
                  action_type="Navigate")[0].accepted
    assert robot.cmd_vel[-1] == md.ZERO and not robot.teleop_open(ROBOT)
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n
    time.sleep(0.05)
    assert statuses(drain(a)) == []


def test_with_the_estop_path_down_a_manual_drive_is_refused() -> None:
    a, robot = rig()
    a._safety_required = True                    # the SafetyStream is required and not up
    res, _ = assign(a, payload())
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == []


# ── after the end, nothing ──────────────────────────────────────────────

def test_in_the_circle_and_still_it_succeeds_and_teleop_then_publishes_nothing() -> None:
    a, robot = rig()
    assert assign(a, payload(settle_s=1.0))[0].accepted
    robot.place(0.9, 0.05)
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    end = terminals(msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED] and "in the circle" in end[0].message
    assert robot.cmd_vel[-1] == md.ZERO          # released: one zero, before the report
    n = len(robot.cmd_vel)
    robot.teleop(0.1, 0.0, 0.0)
    assert len(robot.cmd_vel) == n
    assert a._current_action == ""


def test_teleop_silence_after_motion_sends_one_zero() -> None:
    a, robot = rig()
    assert assign(a, payload(deadman_s=0.1))[0].accepted
    robot.teleop(0.1, 0.0, 0.0)
    n = len(robot.cmd_vel)
    time.sleep(0.3)
    assert robot.cmd_vel[n:] == [md.ZERO]


# ── a restart never resumes forwarding ──────────────────────────────────

def test_after_a_restart_the_reconcile_ends_it_and_teleop_stays_closed() -> None:
    a, robot = rig()                             # a fresh process: nothing held in memory
    a._reconcile_action_state(pb.RegisterAck(
        accepted=True, authoritative_action_state=pb.RobotActionState(
            action_id=ACTION, phase=pb.ROBOT_ACTION_PHASE_IN_PROGRESS, fencing_token=1)))
    msgs: list = []
    collect(a, msgs, lambda ms: terminals(ms))
    assert [s.state for s in terminals(msgs)] == [pb.ACTION_STATE_CANCELLED]
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == [md.ZERO]            # the reconcile's stop, and nothing forwarded


# ── the catalog ──────────────────────────────────────────────────────────────

def _custom(a: Adapter):
    snap = a._capabilities_snapshot()
    return [s for s in snap.supported_actions if s.action_type == "Custom"]


def test_a_binding_with_teleop_declares_manual_drive_and_one_without_does_not() -> None:
    a, _ = rig()
    (entry,) = _custom(a)
    op = next(p for p in entry.params if p.name == "operation")
    assert list(op.allowed) == [md.OPERATION]
    assert {p.name for p in entry.params if p.required} == {"operation", "x", "y", "frame",
                                                            "radius_m"}
    b, _ = rig(teleop_topic=None)
    assert _custom(b) == []
    res, _ = assign(b, payload())
    assert not res.accepted and "no teleop topic" in res.message


def test_with_an_explore_stack_one_custom_entry_names_both_operations() -> None:
    from fleet_adapter_ros2.sim_explore import SimulatedExploreStack
    a, _ = rig(explore=SimulatedExploreStack())
    (entry,) = _custom(a)
    op = next(p for p in entry.params if p.name == "operation")
    assert set(op.allowed) == {"explore-floor", md.OPERATION}
    assert [p.name for p in entry.params if p.required] == ["operation"]


def test_a_payload_naming_another_teleop_topic_is_declined() -> None:
    a, robot = rig()
    res, _ = assign(a, payload(teleop_topic="/some_other_teleop"))
    assert not res.accepted and "teleop_topic" in res.message
    robot.teleop(0.1, 0.0, 0.0)
    assert robot.cmd_vel == []
