# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The navigate-handover's rest wait judged by the lidar's scan channel, for the handover only.

With ``--handover-rest-source scan`` the gate turns to the operator only once the scan shows the
robot at rest in a window that lies after the decline's hold, and the binding checks that again
under the lock every stop takes. Everything else of the wait is as before: a timeout, an unknown, a
moving, a stop of any scope or a lease that ends is Failed, and the gate never turns.

The estop's answer does not change with it: `is_stopped` and `rest_evidence_gap` never read the
scan, so on command-derived odometry an estop is FAILED with the scan showing rest.

The rest wait's own tests (`test_handover_waits_for_rest`) run here again, unchanged, on a stand-in
whose rest comes from scans: `ScanBrakingRobot` keeps the plant model's `is_stopped` true after the
hold, and only the scan -- cast from the plant's true pose, which drifts while it brakes -- says
whether it moves. The adapter refuses to start the scan source without a record that is evidence.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest
import test_handover_waits_for_rest as base
import test_mapview_target_state as mvt
from conftest import mark_transport_live
from scan_room import Lidar, Plant
from test_command_derived_odometry import STOP_S, _CommandDerivedBinding, _driving
from test_handover_waits_for_rest import WHY, presses
from test_manual_drive_gate import ROBOT, StepClock, collect, settle, terminals
from test_manual_drive_nav2 import binding, msg
from test_navigate_handover import ACTION, assign, payload, turned
from test_navigate_handover_nav2 import decline
from test_scan_rest import T  # noqa: F401 - the fixture: T from two stationary runs
from test_scan_rest_record import good_record
from test_stall_reporting import _main
from test_stop_never_waits import (  # noqa: F401 - ros_msgs is a fixture
    _ControlPlane,
    _odom,
    _Publisher,
    _SafetyStub,
    _Server,
    _Twist,
    ros_msgs,
)

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2 import scan_rest as sr
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

DRIFT = (0.1, 0.0, 0.0)     # m/s while the base brakes; the scan sees it, nothing else does


class ScanBrakingRobot(base.BrakingRobot):
    """The rest wait's braking stand-in, whose handover rest comes from scans alone.

    The plant model (`is_stopped`) reads at rest once held, braking or not, as command-derived
    odometry would. ``braking`` moves the TRUE plant, and the lidar casts scans from it at 10 Hz
    on the stand-in's clock, so only the scan can tell."""

    def __init__(self, threshold_m: float, **kw) -> None:
        self._plant = Plant()
        self._lidar = Lidar(seed=5)
        self._scan_lock = threading.Lock()
        self._next = None
        self._braking = False
        super().__init__(rest_channel=sr.RestChannel(threshold_m), **kw)
        self._next = self._clock()

    @property
    def braking(self) -> bool:
        return self._braking

    @braking.setter
    def braking(self, value: bool) -> None:
        if self._next is not None:
            self.pump()
        self._braking = value
        self._plant.true = DRIFT if value else (0.0, 0.0, 0.0)

    def is_stopped(self, robot_id: str) -> bool:
        return ho.SimulatedHandoverRobot.is_stopped(self, robot_id)

    def pump(self) -> None:
        """Every scan due by now, cast from where the plant truly was. The clock is read again
        for each scan: casting a backlog takes real time (about 9 ms a scan), and a lidar keeps
        publishing meanwhile, so the newest scan is never older than one period when this
        returns, however long the stand-in went unpumped."""
        with self._scan_lock:
            while self._next <= self._clock():
                self.feed_scan(self._lidar.cast(self._plant, self._next), self._next + 0.005)
                self._plant.step(0.1)
                self._next += 0.1

    def handover_at_rest(self, robot_id: str):
        self.pump()
        return super().handover_at_rest(robot_id)

    def hand_over_teleop(self, robot_id, action_id, t, why) -> bool:
        self.pump()
        return super().hand_over_teleop(robot_id, action_id, t, why)


@pytest.fixture
def scan_rig(monkeypatch, T):  # noqa: F811 - the imported fixture
    """`test_handover_waits_for_rest.rig`, with the scan as the handover's rest source."""
    def rig(rest_s: float = 3600.0, cls=None, **kw):
        robot = ScanBrakingRobot(T, **kw)
        robot.spawn(ROBOT)
        a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock())
        a._handover_rest_s = rest_s
        return a, robot
    monkeypatch.setattr(base, "rig", rig)
    return rig


# ── the rest wait's own tests, with the scan source ───────────────────────────

def test_with_the_scan_the_operator_cannot_drive_while_the_robot_may_still_move(scan_rig):
    base.test_the_operator_cannot_drive_while_the_robot_may_still_be_moving()


def test_with_the_scan_rest_not_shown_within_the_bound_is_failed(scan_rig):
    base.test_rest_not_confirmed_within_the_bound_ends_it_failed_and_never_opens()


@pytest.mark.parametrize("stop", sorted(base.STOPS))
def test_with_the_scan_a_stop_in_the_wait_wins(scan_rig, stop):
    base.test_a_stop_during_the_rest_wait_wins_and_teleop_never_opens(stop)


@pytest.mark.parametrize("stop", sorted(base.STOPS))
def test_with_the_scan_a_stop_between_the_rest_and_the_handover_wins(scan_rig, stop):
    base.test_a_stop_landing_between_the_rest_and_the_handover_wins(stop)


def test_with_the_scan_a_lease_deadline_in_the_wait_is_failed(scan_rig):
    base.test_a_lease_deadline_passing_during_the_wait_ends_it_failed_with_no_teleop()


def test_with_the_scan_a_lease_lapse_in_the_wait_has_no_teleop_and_its_report_ignores_the_scan(
        scan_rig):
    """The lapse (C4.2) ends it with no teleop. Its STOPPED or FAILED is the stop path's own
    answer, `is_stopped` -- here the plant model, at rest once held -- while the scan still says
    moving: a stop's report never reads the scan. (The rest wait's own row asserts FAILED because
    its stand-in's `is_stopped` brakes too.)"""
    a, robot = scan_rig()
    msgs = base.declined_moving(a, robot, lease_ms=80)
    assert presses(robot) == []
    time.sleep(0.12)
    a._tick_lease()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert robot.braking and not robot.handover_at_rest(ROBOT)[0]      # the scan: moving
    assert [(s.state, s.end_cause) for s in terminals(msgs, ACTION)] == [
        (pb.ACTION_STATE_STOPPED, pb.ACTION_END_CAUSE_LEASE_EXPIRED)]
    assert presses(robot) == [] and not turned(msgs) and not robot.teleop_open(ROBOT)


def test_with_the_scan_the_plant_model_alone_never_hands_over(scan_rig) -> None:
    """The plant model reads at rest the instant it is held. While the scan moves, the gate
    stays armed; the timeout message carries the scan's own answer."""
    a, robot = scan_rig(rest_s=2.0)
    msgs = base.declined_moving(a, robot)
    assert robot.is_stopped(ROBOT)                              # what odometry would say
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    end = terminals(msgs, ACTION)
    assert [s.state for s in end] == [pb.ACTION_STATE_FAILED]
    assert "scan:" in end[0].message and not turned(msgs) and presses(robot) == []


def test_with_the_scan_the_window_must_lie_after_the_hold(scan_rig) -> None:
    """The robot never moved: still, the gate turns only once a whole window of scans lies after
    the decline's hold. Time since the hold alone confirms nothing; the window is of scans."""
    a, robot = scan_rig()
    robot.decline_next(WHY)
    t0 = time.monotonic()
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, turned)
    assert time.monotonic() - t0 >= sr.WINDOW_S + sr.HOLD_MARGIN_S - 0.05


def test_with_the_scan_moving_at_the_handover_is_refused_under_the_lock(T) -> None:  # noqa: F811
    robot = ScanBrakingRobot(T)
    robot.spawn(ROBOT)
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    robot.open_teleop(ROBOT, ACTION, drive, lease_s=60.0)
    assert robot.hold_for_handover(ROBOT, ACTION)
    time.sleep(sr.WINDOW_S + sr.LAG_S + 0.4)
    robot.braking = True
    # A drift that starts after rest was shown reads moving once it has moved the scan by more
    # than T over the lag: up to LAG_S after it starts (the onset latency, in the report).
    time.sleep(sr.LAG_S + 0.15)
    assert not robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY)
    assert drive.phase != md.RUNNING and presses(robot) == []
    robot.braking = False
    time.sleep(sr.WINDOW_S + sr.LAG_S + 0.3)
    assert robot.hand_over_teleop(ROBOT, ACTION, 1.0, WHY) and drive.phase == md.RUNNING


# ── the Nav2 binding's scan channel ───────────────────────────────────────────

class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _laser(scan):
    from types import SimpleNamespace as NS
    sec = int(scan.stamp)
    return NS(header=NS(stamp=NS(sec=sec, nanosec=round((scan.stamp - sec) * 1e9))),
              ranges=list(scan.ranges), range_min=scan.range_min, range_max=scan.range_max,
              angle_min=scan.angle_min, angle_increment=scan.angle_increment)


def _armed_nav2_scan(T, source="command-derived"):  # noqa: F811
    clock = _Clock()
    b, pub = binding()
    b._odometry_source = source
    b._now = clock
    b._scan_rest = sr.RestChannel(T)
    b._rest_scan_topic = "/scan_raw"
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    drive = md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)
    b.open_teleop(ROBOT, ACTION, drive, lease_s=1e6)
    with b._lock:
        b._goal = task_map.GoalPose(x=2.0, y=1.0, yaw=0.0)
        b._holding, b._goal_status = False, 0
    decline(b)
    return b, pub, drive, clock


def _scans_for(b, clock, plant, lidar, seconds):
    for _ in range(round(seconds * 10)):
        b.on_scan(_laser(lidar.cast(plant, clock.t)))
        plant.step(0.1)
        clock.t += 0.1


def test_nav2_with_the_scan_hands_over_on_command_derived_odometry_only_at_rest(T) -> None:  # noqa: F811
    b, pub, drive, clock = _armed_nav2_scan(T)
    b.on_odom(_odom(0.0))
    assert b.handover_rest_gap(ROBOT) == ""
    assert b.hold_for_handover(ROBOT, ACTION)
    plant, lidar = Plant(true=DRIFT), Lidar(seed=81)
    _scans_for(b, clock, plant, lidar, 2.0)                    # held, and still drifting
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)
    ok, why = b.handover_at_rest(ROBOT)
    assert not ok and "moving" in why
    n = len(pub.sent)
    b.on_teleop(msg(0.1, 0.0, 0.0))
    assert len(pub.sent) == n
    plant.true = (0.0, 0.0, 0.0)
    _scans_for(b, clock, plant, lidar, 2.0)
    assert b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase == md.RUNNING


def test_nav2_with_the_scan_measured_odometry_at_rest_does_not_override_a_moving_scan(T) -> None:  # noqa: F811
    b, _, drive, clock = _armed_nav2_scan(T, source="measured")
    b.on_odom(_odom(0.0))                                       # odometry would confirm rest
    assert b.hold_for_handover(ROBOT, ACTION)
    _scans_for(b, clock, Plant(true=DRIFT), Lidar(seed=82), 3.0)
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY) and drive.phase != md.RUNNING


def test_nav2_with_the_scan_motion_on_another_channel_vetoes_it(T) -> None:  # noqa: F811
    b, _, _drive, clock = _armed_nav2_scan(T)
    assert b.hold_for_handover(ROBOT, ACTION)
    _scans_for(b, clock, Plant(), Lidar(seed=83), 2.5)
    b.on_odom(_odom(0.3))                                       # odometry shows motion: refute
    assert b.handover_at_rest(ROBOT) == (False, "odometry shows motion")
    b.on_odom(_odom(0.0))
    with b._lock:
        b._stop_pending.add(7)                                  # a goal that may still drive
    assert not b.handover_at_rest(ROBOT)[0]
    with b._lock:
        b._stop_pending.clear()
    assert b.handover_at_rest(ROBOT) == (True, "")


def test_nav2_with_the_scan_no_hold_no_scan_or_a_stale_scan_never_hands_over(T) -> None:  # noqa: F811
    b, _, _drive, clock = _armed_nav2_scan(T)
    assert not b.handover_at_rest(ROBOT)[0]                     # no scan at all
    _scans_for(b, clock, Plant(), Lidar(seed=84), 2.5)
    assert "hold" in b.handover_at_rest(ROBOT)[1]               # never commanded to hold
    with b._lock:
        b._handover_hold_at = clock.t - 5.0
    assert b.handover_at_rest(ROBOT) == (True, "")
    clock.t += 0.4                                              # the lidar went quiet
    ok, why = b.handover_at_rest(ROBOT)
    assert not ok and "old" in why
    assert not b.hand_over_teleop(ROBOT, ACTION, 5.0, WHY)


def test_nav2_without_the_scan_is_unchanged(T) -> None:  # noqa: F811
    b, _ = binding()
    b._odometry_source = "command-derived"
    b.on_scan(_laser(Lidar(seed=85).cast(Plant(), 1.0)))        # nothing subscribed: ignored
    assert b._scan_rest is None
    assert b.handover_rest_gap(ROBOT) == b.rest_evidence_gap(ROBOT) != ""
    with pytest.raises(ValueError, match="go together"):
        Nav2Binding(ROBOT, rest_scan_topic="/scan_raw")


# ── the estop's answer does not read the scan ─────────────────────────────────

def test_with_the_scan_on_and_showing_rest_an_estop_is_still_failed_on_command_derived(
        T, ros_msgs):  # noqa: F811
    server = _Server(up=True)
    b = mark_transport_live(_CommandDerivedBinding(
        ROBOT, odometry_source="command-derived", rest_scan_topic="/scan_raw",
        rest_scan_threshold_m=T))
    from types import SimpleNamespace as NS
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _Publisher(), _Twist
    b._cancel_goal_by_id = lambda _id: None
    b.start = lambda: None
    b.on_odom(_odom(0.0))
    with b._lock:
        b._handover_hold_at = time.monotonic() - 10.0
    stop = threading.Event()
    lidar, plant = Lidar(seed=86), Plant()

    def feed() -> None:                                         # a still room, at 10 Hz
        while not stop.is_set():
            b.on_scan(_laser(lidar.cast(plant, time.monotonic())))
            time.sleep(0.1)
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    a = Adapter(pb, None, b, ROBOT)
    cp = _ControlPlane()
    control = threading.Thread(target=a.run_control_stream, args=(cp,), daemon=True)
    control.start()
    try:
        assert cp.opened.wait(2.0)
        time.sleep(sr.LAG_S + sr.WINDOW_S + 0.3)
        assert b.handover_at_rest(ROBOT) == (True, "")          # the scan shows rest
        _driving(a, b, server, cp)
        safety = threading.Thread(target=a.run_safety_stream, args=(_SafetyStub(),),
                                  daemon=True)
        safety.start()
        deadline = time.time() + STOP_S + 1.0
        states, last = [], None
        while time.time() < deadline and (not states or states[-1] == pb.ESTOP_STATE_STOPPING):
            try:
                m = a._safety_outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if m.WhichOneof("payload") == "estop_ack":
                states.append(m.estop_ack.state)
                last = m.estop_ack
        assert states and states[-1] == pb.ESTOP_STATE_FAILED
        assert pb.ESTOP_STATE_STOPPED not in states and "command-derived" in last.message
        assert b.is_stopped(ROBOT) is False and b.rest_evidence_gap(ROBOT) != ""
    finally:
        stop.set()
        a._shutdown.set()
        cp.q.put(None)
        control.join(timeout=5.0)


# ── the circle: red follows the scan, not odometry ────────────────────────────

def test_red_follows_the_scans_at_rest_not_the_plant_models(T) -> None:  # noqa: F811
    a, robot, sink = mvt.rig(cls=ScanBrakingRobot, threshold_m=T)
    msgs = base.declined_moving(a, robot)
    settle(a, msgs, sr.LAG_S + sr.WINDOW_S + 0.5)
    assert robot.is_stopped(ROBOT)                              # odometry would say "at rest"
    assert mvt.states(sink) == [mvt.ROBOT_]                     # yellow: the scan says moving
    robot.come_to_rest()
    collect(a, msgs, turned)
    assert mvt.wait_states(sink, lambda s: s[-1:] == [mvt.OPERATOR]) == [mvt.ROBOT_, mvt.OPERATOR]


# ── the adapter: declared measured only by a record that is evidence ──────────

ROS2 = ("--binding", "ros2", "--stall-window", "5", "--stall-min-progress", "0.1",
        "--goal-response-bound", "1", "--odometry-source", "command-derived",
        "--teleop-topic", "/cmd_vel_teleop")


def _record_file(tmp_path, doc) -> str:
    p = tmp_path / "record.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_the_scan_source_with_a_good_record_runs_and_says_so(monkeypatch, capsys, tmp_path):
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--handover-rest-source", "scan",
                             "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration",
                             _record_file(tmp_path, good_record()))
    assert code is None and isinstance(robot, Nav2Binding)
    assert robot._scan_rest is not None and robot._scan_rest.threshold_m == 0.0165
    assert "declared measured by the calibration record md5" in err
    assert robot.is_stopped(ROBOT) is False and robot.rest_evidence_gap(ROBOT) != ""


def test_the_default_is_odometry_and_unchanged(monkeypatch, capsys):
    code, err, robot = _main(monkeypatch, capsys, *ROS2)
    assert code is None and robot._scan_rest is None
    assert "navigate-handover rest: the estop's own test" in err
    assert robot.handover_rest_gap(ROBOT) == robot.rest_evidence_gap(ROBOT) != ""


def _bad(**changes):
    doc = good_record()
    for k, v in changes.items():
        doc[k] = v
    return doc


def _with_miss():
    doc = good_record()
    doc["runs"][3]["misses"] = 1
    return doc


def _short_b():
    doc = good_record()
    doc["runs"][4]["motion_windows"] = 5
    return doc


def _other_definition():
    doc = good_record()
    doc["definition"]["lag_s"] = 0.3
    return doc


@pytest.mark.parametrize("extra,words", [
    (("--handover-rest-source", "scan", "--rest-scan-topic", "/scan_raw"),
     "needs --rest-scan-calibration"),
    (("--handover-rest-source", "scan", "--rest-scan-calibration", "{rec}"),
     "needs --rest-scan-topic"),
    (("--rest-scan-topic", "/scan_raw"), "--rest-scan-topic is used only with"),
    (("--rest-scan-calibration", "{rec}"), "--rest-scan-calibration is used only with"),
])
def test_the_scan_source_refuses_to_start_without_its_evidence(monkeypatch, capsys, tmp_path,
                                                               extra, words):
    rec = _record_file(tmp_path, good_record())
    code, err, robot = _main(monkeypatch, capsys, *ROS2,
                             *[x.format(rec=rec) for x in extra])
    assert code == 2 and robot is None and err.startswith("REFUSED: ") and words in err


@pytest.mark.parametrize("doc,words", [
    (_with_miss(), "has 1 misses"),
    (_short_b(), "fewer than 20"),
    (_other_definition(), "different definition"),
    (_bad(threshold_m=0.03), "is not 1.5 x s0"),
    ({"format": "nope"}, "not a scan-rest-calibration/1 record"),
])
def test_the_scan_source_refuses_a_record_that_is_not_evidence(monkeypatch, capsys, tmp_path,
                                                               doc, words):
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--handover-rest-source", "scan",
                             "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration",
                             _record_file(tmp_path, doc))
    assert code == 2 and robot is None
    assert "is not evidence the scan channel is measured" in err and words in err


def test_the_scan_source_refuses_a_missing_record_file(monkeypatch, capsys, tmp_path):
    code, err, _ = _main(monkeypatch, capsys, *ROS2, "--handover-rest-source", "scan",
                             "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration",
                             str(tmp_path / "absent.json"))
    assert code == 2 and "no calibration record" in err


def test_the_scan_source_refuses_the_simulated_binding_and_no_teleop(monkeypatch, capsys,
                                                                     tmp_path):
    rec = _record_file(tmp_path, good_record())
    code, err, _ = _main(monkeypatch, capsys, "--binding", "simulated", "--teleop-topic",
                         "/cmd_vel_teleop", "--handover-rest-source", "scan",
                         "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration", rec)
    assert code == 2 and "the simulated binding has no lidar" in err
    code, err, _ = _main(monkeypatch, capsys, *ROS2[:-2], "--handover-rest-source", "scan",
                         "--rest-scan-topic", "/scan_raw", "--rest-scan-calibration", rec)
    assert code == 2 and "needs --teleop-topic" in err
