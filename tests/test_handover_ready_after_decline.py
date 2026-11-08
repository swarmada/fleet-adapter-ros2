# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The operator's ready after a Nav2 decline, on the Nav2 binding, under a lease that is renewed.

On the robot on 2026-10-17 two long legs were declined, one by the navigation server's abort and
one by the progress watchdog's stall. Every ready press was logged "ignored: the handover of ...
is not waiting for one", the leg ended Failed when the 30 s bound passed, and the press lines
carried no time, so their order against the decline could not be read.

`TeleopGate.declined` is two halves: the drive is Declined, and the gate's lease deadline has not
passed. The tests here run the shape of that leg through the adapter on a ROS-free Nav2Binding,
on the binding's real monotonic clock: the lease is shorter than Nav2's drive, and the control
plane renews it every lease / 3 as RFC-0001 says ("The control plane renews the lease every
`leaseDurationSeconds / 3`"). A press while Nav2 drives is ignored and not queued; a press after
the decline is kept, the turn opens, teleop is forwarded and the leg ends Succeeded in the circle.
Nobody presses: Failed at the bound, as before. A lease that has really passed opens nothing: "The
robot may physically execute the action only while it holds an unexpired lease."

Every ready line now starts with its UTC time to the millisecond, and an ignored press names the
half of `declined` that failed, so the next field log answers which one it was.
"""

from __future__ import annotations

import json
import re
import threading
import time

import pytest
from conftest import mark_transport_live
from scan_room import Lidar, Plant
from test_handover_scan_rest import _laser
from test_manual_drive_gate import ROBOT, StepClock, collect, command, drain, statuses, terminals
from test_manual_drive_nav2 import TOPIC, Publisher, Twist, msg
from test_scan_rest import T  # noqa: F401 - the fixture: T from two stationary runs
from test_stall_reporting import Clock, _Done, _Handle, _limits
from test_stop_never_waits import _odom

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import operator_ready as opr
from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import _GS_ABORTED, Nav2Binding

ACTION = "handover-a"
TARGET = (2.0, 1.0)
LEASE_S = 1.5                    # renewed every LEASE_S / 3, as the control plane renews it
DRIVE_S = 2.2                    # Nav2 drives past the first lease before it declines
LOOK_S = sr.LAG_S + sr.WINDOW_S  # the veto's full look after the hold
UTC_MS = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z"
STAMP = re.compile(rf"^{UTC_MS} navigate-handover ready: ")
TURN_OPEN = re.compile(rf"^handover turn-open action=(\S+) turn=(\d+) ts=({UTC_MS})$")
TURN_CLOSED = re.compile(
    rf"^handover turn-closed action=(\S+) reason=(completed|timeout|failed|stopped) ts=({UTC_MS})$")
ADDRESS = re.compile(r"\d{1,3}(\.\d{1,3}){3}|:\d{2,5}\b")


@pytest.fixture(autouse=True)
def _own_state(tmp_path, monkeypatch):
    """Nothing here may read or write a state file in the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


class _Rig:
    """The adapter on a Nav2Binding with the operator source: a still room's scans at 10 Hz, a
    pose in `map`, and, unless ``renew`` is off, the control plane's renewals every lease / 3."""

    def __init__(self, threshold_m: float, *, renew: bool = True,
                 ready_s: float | None = 3600.0) -> None:
        b = mark_transport_live(Nav2Binding(
            ROBOT, stall_limits=_limits(window=DRIVE_S - 0.2), teleop_topic=TOPIC,
            odometry_source="command-derived", rest_scan_topic="/scan_raw",
            rest_scan_threshold_m=threshold_m, ready_topic=opr.READY_TOPIC))
        self.sent: list = []
        b._send_nav2_goal = lambda goal, epoch: self.sent.append((goal, epoch))
        b._server_ready = lambda: True
        b._cmd_vel, b._twist_type = Publisher(), Twist
        b.spawn(ROBOT)
        self.where = {"xy": (0.0, 0.0)}
        b.pose = lambda rid: (*self.where["xy"], 0)
        b.pose_frame = lambda rid: "map"
        b.on_odom(_odom(0.0))
        self.b = b
        self.a = Adapter(pb, None, b, ROBOT, job_clock=StepClock())
        if ready_s is not None:
            self.a._handover_ready_s = ready_s   # on the job's clock; the default is 30 s
        self.stop = threading.Event()
        self.renewed: list[bool] = []
        self._renew = renew
        threading.Thread(target=self._scans, daemon=True).start()

    def _scans(self) -> None:
        lidar, plant = Lidar(seed=96), Plant()
        while not self.stop.is_set():
            self.b.on_scan(_laser(lidar.cast(plant, time.monotonic())))
            time.sleep(0.1)

    def _renewals(self) -> None:
        n = 100
        while not self.stop.wait(LEASE_S / 3):
            n += 1
            command(self.a, n, renew_lease=pb.RenewActionLease(
                action_id=ACTION, lease_generation=1, lease_duration_ms=round(LEASE_S * 1000)))

    def assign(self, timeout_s: float = 300) -> list:
        body = json.dumps({"operation": ho.OPERATION, "x": TARGET[0], "y": TARGET[1],
                           "frame": "map", "radius_m": 0.3, "timeout_s": timeout_s}).encode()
        command(self.a, 1, assign_action=pb.AssignAction(
            action_id=ACTION, action_type="Custom", assignment_id="a-1", fencing_token=1,
            lease_generation=1, lease_duration_ms=round(LEASE_S * 1000), payload_json=body))
        msgs = drain(self.a)
        assert [m.command_result.assign_action.accepted for m in msgs
                if m.WhichOneof("payload") == "command_result"] == [True]
        if self._renew:
            threading.Thread(target=self._renewals, daemon=True).start()
        self.b._on_goal_response(_Done(_Handle()), self.sent[-1][1])
        self.b.on_feedback(3.0)                  # the server took the goal; Nav2 drives
        return msgs

    def decline(self, how: str, msgs: list) -> None:
        """Nav2 declines after DRIVE_S: the server aborts the goal, or the watchdog finds no
        progress in its window. Returns once the job says the base is held."""
        if how == "abort":
            with self.b._lock:                   # as `on_goal_result` records an ABORTED
                self.b._goal_status = _GS_ABORTED
                self.b._failure = "the navigation server aborted the goal"
        else:
            assert self.b.check_progress().startswith("stalled:")
        collect(self.a, msgs, waiting)

    def close(self) -> None:
        """Stop the scans and renewals, and end a job still running, so nothing it says later
        lands in another test."""
        self.stop.set()
        with self.a._lease_lock:
            self.a._halt_job()
        time.sleep(0.05)


@pytest.fixture
def rigs():
    made: list[_Rig] = []
    yield made
    for r in made:
        r.close()


def waiting(msgs) -> list:
    return [s for s in statuses(msgs, ACTION)
            if s.state == pb.ACTION_STATE_RUNNING and "press ready" in s.message]


def turned(msgs) -> bool:
    return any(s.message.startswith("operator's turn") for s in statuses(msgs, ACTION))


def renewals(msgs) -> list[bool]:
    return [m.command_result.renew_lease.renewed for m in msgs
            if m.WhichOneof("payload") == "command_result"
            and m.command_result.HasField("renew_lease")]


# ── a press after the decline opens the turn; the leg completes ──────────────

@pytest.mark.parametrize("how", ["abort", "stall"])
def test_a_ready_after_the_decline_opens_the_turn_and_the_leg_completes(T, rigs, how) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = r.assign()
    time.sleep(DRIVE_S)                          # past the first lease, inside the renewed one
    r.decline(how, msgs)
    assert ("aborted" if how == "abort" else "stalled") in waiting(msgs)[0].message
    time.sleep(LOOK_S + 0.3)                     # the veto's full look
    assert r.b.on_ready() == ""                  # kept for the held handover
    collect(r.a, msgs, turned)
    assert renewals(msgs) and all(renewals(msgs))
    n = len(r.b._cmd_vel.sent)
    r.b.on_teleop(msg(0.1, 0.0, 0.0))            # the operator's teleop reaches the base
    assert r.b._cmd_vel.sent[n:] == [(0.1, 0.0, 0.0)]
    r.where["xy"] = TARGET                       # driven into the circle, and still
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert "operator released" in end[0].message
    assert not r.b.teleop_open(ROBOT)


@pytest.mark.parametrize("how", ["abort", "stall"])
def test_a_ready_while_nav2_drives_is_ignored_and_one_after_the_decline_counts(
        T, rigs, how) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = r.assign()
    time.sleep(DRIVE_S / 2)
    assert "no declined handover is held" in r.b.on_ready()   # before the decline: not kept
    time.sleep(DRIVE_S / 2)
    r.decline(how, msgs)
    time.sleep(LOOK_S + 0.3)
    assert not turned(msgs + drain(r.a))         # the early press was never queued
    assert r.b.on_ready() == ""
    collect(r.a, msgs, turned)
    lines = r.b.ready_log()
    assert [("ready: ignored:" in x, "ready: kept for" in x) for x in lines] == [
        (True, False), (False, True)]


def test_nobody_presses_and_the_leg_fails_at_the_thirty_second_bound(T, rigs, capsys) -> None:  # noqa: F811
    r = _Rig(T, ready_s=None)                    # the adapter's own bound
    rigs.append(r)
    assert r.a._handover_ready_s == ho.READY_S == 30.0
    msgs = r.assign()
    time.sleep(DRIVE_S)
    r.decline("abort", msgs)
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert ("no operator confirmation within 30 s; the operator's turn never opened and the "
            "base is held") in end[0].message
    assert not turned(msgs) and not r.b.teleop_open(ROBOT)
    n = len(r.b._cmd_vel.sent)
    assert r.b.on_ready() != ""                  # after the end: nothing
    r.b.on_teleop(msg(0.1, 0.0, 0.0))
    assert all(c == md.ZERO for c in r.b._cmd_vel.sent[n:])
    assert turn_lines(capsys.readouterr().err) == []   # no turn opened, so none closed


def test_a_lease_that_really_passed_opens_nothing_and_the_press_says_so(T, rigs) -> None:  # noqa: F811
    r = _Rig(T, renew=False)                     # the control plane stops renewing
    rigs.append(r)
    msgs = r.assign()
    time.sleep(DRIVE_S)                          # the lease has passed; not yet looked at
    r.decline("abort", msgs)
    time.sleep(LOOK_S + 0.3)
    why = r.b.on_ready()
    assert "not waiting" in why and "the lease's deadline passed" in why
    n = len(r.b._cmd_vel.sent)
    r.b.on_teleop(msg(0.1, 0.0, 0.0))
    assert all(c == md.ZERO for c in r.b._cmd_vel.sent[n:])   # nothing under a dead lease
    r.a._tick_lease()                            # the lease thread's look
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.end_cause for s in end] == [pb.ACTION_END_CAUSE_LEASE_EXPIRED]
    assert not turned(msgs) and not r.b.teleop_open(ROBOT)


# ── which half failed, in the ignored press's own line ───────────────────────

def _held(clock=None, lease_s: float = 60.0):
    b = mark_transport_live(Nav2Binding(
        ROBOT, clock=clock, teleop_topic=TOPIC, rest_scan_topic="/scan_raw",
        rest_scan_threshold_m=0.0165, ready_topic=opr.READY_TOPIC))
    b._cmd_vel, b._twist_type = Publisher(), Twist
    doc = {"operation": ho.OPERATION, "x": TARGET[0], "y": TARGET[1], "frame": "map",
           "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    b.open_teleop(ROBOT, ACTION, drive, lease_s=lease_s)
    assert b.hold_for_handover(ROBOT, ACTION)    # held and armed; the job has not declined yet
    return b, drive


def test_an_ignored_press_names_the_drives_phase_when_it_is_not_declined_yet() -> None:
    b, _ = _held()
    why = b.on_ready()
    assert "not waiting" in why and "Navigating, not Declined" in why
    assert "lease" not in why
    assert "Navigating, not Declined" in b.ready_log()[-1]


def test_an_ignored_press_names_the_lease_deadline_when_it_has_passed() -> None:
    c = Clock()
    b, drive = _held(c, lease_s=30.0)
    drive.decline(c.t, "no valid path")
    c.t += 0.5
    assert b.on_ready() == ""                    # declined and inside the lease: kept
    b.take_ready(ROBOT, ACTION)
    c.t += 31.0
    why = b.on_ready()
    assert "the lease's deadline passed 1.5 s ago" in why and "Declined" not in why


def test_a_press_with_the_gate_closed_says_the_gate_is_closed() -> None:
    b, _ = _held()
    b.command_stop(ROBOT)
    assert "the gate is closed" in b.on_ready()


# ── every ready line carries its UTC time, to the millisecond ────────────────

def test_every_ready_line_starts_with_its_utc_time_to_the_millisecond() -> None:
    got: list[str] = []
    latch = opr.ReadyLatch(log=got.append, wall=lambda: 1792000000.0425)
    latch.press(1.0, lambda aid: False)          # no hold: ignored
    latch.arm(ACTION, 2.0)
    latch.press(3.0, lambda aid: True)           # kept
    assert got == latch.lines and len(got) == 2
    assert all(x.startswith("2026-10-14T17:46:40.042Z navigate-handover ready: ") for x in got)


@pytest.mark.parametrize("how", ["abort", "stall"])
def test_the_bindings_ready_lines_are_stamped(T, rigs, how) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = r.assign()
    r.b.on_ready()                               # while Nav2 drives
    time.sleep(DRIVE_S)
    r.decline(how, msgs)
    time.sleep(LOOK_S + 0.3)
    r.b.on_ready()
    collect(r.a, msgs, turned)
    lines = r.b.ready_log()
    assert len(lines) == 2 and all(STAMP.match(x) for x in lines), lines
    stamps = [x.split(" ", 1)[0] for x in lines]
    assert stamps == sorted(stamps)


# ── the turn's two machine-readable lines, each exactly once per turn ────────

def turn_lines(err: str) -> list[str]:
    return [x for x in err.splitlines() if x.startswith("handover turn-")]


def _one_turn(err: str, reason: str) -> None:
    lines = turn_lines(err)
    assert len(lines) == 2, lines
    opened, closed = TURN_OPEN.match(lines[0]), TURN_CLOSED.match(lines[1])
    assert opened and closed, lines
    assert opened.group(1) == closed.group(1) == ACTION and int(opened.group(2)) >= 1
    assert closed.group(2) == reason
    assert opened.group(3) <= closed.group(3)
    assert not any(ADDRESS.search(x.split(" ts=")[0]) for x in lines), lines


def _turn(r: _Rig, how: str = "abort", timeout_s: float = 300) -> list:
    msgs = r.assign(timeout_s)
    time.sleep(DRIVE_S)
    r.decline(how, msgs)
    time.sleep(LOOK_S + 0.3)
    assert r.b.on_ready() == ""
    collect(r.a, msgs, turned)
    return msgs


@pytest.mark.parametrize("how", ["abort", "stall"])
def test_a_completed_turn_says_open_and_closed_once_each(T, rigs, capsys, how) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = _turn(r, how)
    r.where["xy"] = TARGET
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    time.sleep(0.2)
    _one_turn(capsys.readouterr().err, "completed")


def test_a_stop_during_the_turn_closes_it_stopped_once(T, rigs, capsys) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = _turn(r)
    r.b.command_stop(ROBOT)                      # the stop path, as an estop takes it
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    time.sleep(0.2)
    _one_turn(capsys.readouterr().err, "stopped")


def test_a_superseding_assignment_closes_the_turn_stopped_once(T, rigs, capsys) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    _turn(r)
    command(r.a, 2, assign_action=pb.AssignAction(
        action_id="nav-b", action_type="Navigate", assignment_id="a-2", fencing_token=2,
        lease_generation=2, lease_duration_ms=60_000, payload_json=b'{"x": 0.5, "y": 0}'))
    time.sleep(0.5)                              # the halted job's thread ends
    _one_turn(capsys.readouterr().err, "stopped")


def test_the_turns_timeout_closes_it_timeout_once(T, rigs, capsys) -> None:  # noqa: F811
    r = _Rig(T)
    rigs.append(r)
    msgs = _turn(r, timeout_s=5)                 # on the job's clock
    collect(r.a, msgs, lambda ms: terminals(ms, ACTION))
    time.sleep(0.2)
    _one_turn(capsys.readouterr().err, "timeout")
