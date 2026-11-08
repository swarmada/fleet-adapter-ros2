# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Nav2 / ROS 2 robot binding — the ROS 2 CLASS seam (no vendor-specific branches).

This binding wraps ``rclpy`` + the Nav2 stack behind the ``RobotBinding`` contract
(see robot.py). The telemetry side (sub-item 2) is implemented as a cache of the
latest odom/battery/diagnostics messages, mapped by ``telemetry.py``. The ROS
node + subscriptions are created in :meth:`start` (which requires a ROS 2 runtime);
the cache + mappings are exercised without ROS by feeding the ``on_*`` callbacks.

Task execution (sub-item 3): ``command_move`` translates a target into a
``nav2_msgs/action/NavigateToPose`` goal (via ``task.GoalPose``) and caches it;
the live action-client dispatch happens once :meth:`start` has a ROS 2 node.

Safety (sub-item 4): ``command_stop`` cancels the active Nav2 goal and holds
(shared by confirmed estop C5 and lease-expiry self-stop C4); ``is_stopped`` is
CONFIRMED from the odom twist being ~zero (never inferred from the command, a
timer, or silence — C5.3). The live goal dispatch/cancel + zero-``cmd_vel`` publish
are the only pieces that need a ROS 2 runtime; they are guarded behind a live node.
"""

from __future__ import annotations

import math
import sys
import threading
import time

from fleet_adapter_ros2 import manual_drive, operator_ready, scan_rest, telemetry
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.watchdog import (
    ODOMETRY_COMMAND_DERIVED,
    ODOMETRY_SOURCES,
    ProgressWatchdog,
    StallLimits,
    rest_evidence_gap,
)

# A feedback distance this far under the straight line from the robot to its goal cannot be
# the goal's own path length; it is stale (see on_feedback). 0.10 m covers the planner's grid
# rounding: on hardware every current feedback sat at or above the straight line.
STALE_FEEDBACK_MARGIN_M = 0.10

# A robot is CONFIRMED at rest when its measured odom twist is within sensor noise.
# Ground truth only — never a timer or the absence of a message (C5.3).
_EPS_LINEAR = 0.01   # m/s
_EPS_ANGULAR = 0.01  # rad/s

_GS_ABORTED = 6  # action_msgs/GoalStatus: the goal ended in failure

# How often the progress watchdog is evaluated on the executor. How promptly a stall is
# REPORTED, not what counts as one: a stall is declared at most one period after its window
# ends. The window and the minimum progress are the plant's (StallLimits), never this.
_WATCHDOG_PERIOD_S = 0.1

# The frame every goal is sent in, and the frame pose() reports in when TF can reach it.
MAP_FRAME = "map"

# The battery is reported missing (a Failed hardware entry, and said on stderr) once no
# reading has arrived for this long, whether since start or since the last reading.
BATTERY_SILENT_S = 10.0
# How often the battery topic's publishers are counted, for that report.
_BATTERY_GRAPH_PERIOD_S = 5.0
# How often start()'s timer asks TF for map -> odom. Localisation publishes it continuously
# (slam_toolbox every 20 ms, AMCL on each correction); 10 Hz follows an AMCL jump promptly.
_TF_PERIOD_S = 0.1


def _stamp_ns(msg) -> int | None:
    """``header.stamp`` of a stamped message as integer nanoseconds, or None if it has
    none. Integer, not float: the value is subtracted from its neighbours, and an epoch
    float spends most of its 53 bits on the seconds."""
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


class Nav2Binding:
    """ROS 2 / Nav2 implementation of ``RobotBinding`` (see robot.py).

    Constructible without ROS (for telemetry mapping/tests). :meth:`start` creates the
    rclpy node and subscribes to the topics — that requires a ROS 2 + Nav2 environment.
    Select it on the CLI with ``--binding ros2``.

    :meth:`start` and :meth:`stop` implement the ``RobotBinding`` lifecycle, which the
    adapter drives in ``adapter.bring_up_robot()``. **A harness whose verdict is about the
    adapter must go through that, not call :meth:`start` itself** — doing so is how an
    inert adapter stayed hidden for months: ``testbed/r1_smoke.py`` called ``start()`` on the
    adapter's behalf, so it exercised a live transport the shipped adapter never opened,
    and R1 reported eight passing assertions the whole time ``--binding ros2`` was inert.

    Calling :meth:`start` directly is still correct for a caller that is NOT asserting
    anything about the adapter — an instrument pointed at ROS 2 or at Nav2 rather than at
    this package (``testbed/tf_liveness.py``, ``testbed/r2_cancel_latency.py``,
    ``testbed/r2_transport_check.py``, ``testbed/r1_step5_probe.py``). Those need a live
    binding as apparatus; they do not claim the adapter brought one up.
    """

    def __init__(self, robot_id: str = "robot-conformance-1", floor: int = 0,
                 stall_limits: StallLimits | None = None, clock=None,
                 response_bound_s: float | None = None,
                 odometry_source: str | None = None, explore=None,
                 battery_mv_topic: str | None = None, battery_empty_mv: float | None = None,
                 battery_full_mv: float | None = None,
                 teleop_topic: str | None = None,
                 rest_scan_topic: str | None = None,
                 rest_scan_threshold_m: float | None = None,
                 ready_topic: str | None = None) -> None:
        """``stall_limits`` arms the progress watchdog (watchdog.py) on every goal. None
        leaves it unarmed, which only an instrument pointed at ROS 2 should want: the adapter
        refuses to start ``--binding ros2`` without it. ``clock`` is the watchdog's time
        source (default ``time.monotonic``), injectable for tests.

        ``response_bound_s`` is how long a goal SENT to the navigation server may go
        unanswered before it is abandoned (`_watch_answers`). None leaves goals unbounded,
        which, again, only an instrument should want: the adapter refuses to start
        ``--binding ros2`` without ``--goal-response-bound``.

        ``odometry_source`` declares what the odometry this binding confirms rest from
        actually is: ``"measured"`` (a sensor that observes the robot) or
        ``"command-derived"`` (the velocity command published back as odometry). Declared
        command-derived, :meth:`is_stopped` never returns True, because on such a base the
        twist reads zero the instant a stop is commanded whether or not the robot is moving,
        and CONFORMANCE.md C5.3 forbids reporting that as a confirmed stop (ADR-0071's
        platform precondition). None leaves it undeclared, which only an instrument should
        want: the adapter refuses to start ``--binding ros2`` without ``--odometry-source``,
        and :meth:`rest_evidence_gap` reports the same gap for an undeclared binding as for
        a command-derived one -- an independence nobody stated is not one this adapter may
        assume.

        ``battery_mv_topic`` names a ``std_msgs/UInt16`` topic that carries the pack's voltage
        in mV (a robot whose controller publishes that and no ``BatteryState``); the percent
        is then read between ``battery_empty_mv`` and ``battery_full_mv``, both required with
        it, since nothing on such a robot says what empty and full are. None keeps
        ``sensor_msgs/BatteryState`` on ``battery_state``. Either way a battery that never
        reported is None, never 0.

        ``teleop_topic`` names the ``geometry_msgs/Twist`` topic the operator's teleop publishes
        on. Its messages reach ``cmd_vel`` only through the manual-drive gate
        (`manual_drive.TeleopGate`), which every stop closes under this binding's lock. None
        (the default) subscribes to nothing and serves no manual-drive. It must not be
        ``cmd_vel``: teleop there would bypass the gate.

        ``rest_scan_topic`` names the ``sensor_msgs/LaserScan`` topic the navigate-handover's rest
        wait reads (`scan_rest`), with ``rest_scan_threshold_m``, the T of a calibration record
        the adapter has checked; both or neither. Then `handover_rest_gap` and
        `handover_at_rest` answer from the scan, for the handover only. `is_stopped` and
        `rest_evidence_gap` -- the estop's, the lease self-stop's and the cancel's answer --
        never read the scan. None: the handover's rest is the estop's, as before.

        ``ready_topic`` (with a scan channel) takes the handover's rest from the operator's ready
        on that topic (`operator_ready`, ``std_msgs/msg/Empty``), the scan a veto only: the gate
        turns on a ready kept for the held handover, after the scan's full look with no veto. It
        must be a topic of its own -- not cmd_vel, the teleop topic or the scan's. The estop's
        answer reads neither the ready nor the scan."""
        if teleop_topic is not None and teleop_topic.strip() in ("", *manual_drive.BASE_TOPICS):
            raise ValueError(f"teleop_topic must name a topic other than cmd_vel, got "
                             f"{teleop_topic!r}: forwarding is the gate")
        self._teleop_topic = teleop_topic
        if (rest_scan_topic is None) != (rest_scan_threshold_m is None):
            raise ValueError("rest_scan_topic and rest_scan_threshold_m go together: a scan "
                             "channel needs the threshold of a calibration record")
        self._rest_scan_topic = rest_scan_topic
        # The handover's rest channel, and when the decline's hold was commanded (this clock).
        self._scan_rest = (None if rest_scan_topic is None
                           else scan_rest.RestChannel(rest_scan_threshold_m))
        self._handover_hold_at: float | None = None
        if ready_topic is not None:
            if rest_scan_topic is None:
                raise ValueError("ready_topic needs rest_scan_topic: the operator confirms rest, "
                                 "and the veto needs the scan")
            if ready_topic.strip() in ("", *manual_drive.BASE_TOPICS, teleop_topic,
                                       rest_scan_topic):
                raise ValueError(f"ready_topic must name a topic of its own, got {ready_topic!r}: "
                                 f"not cmd_vel, the teleop topic or the scan's")
        self._ready_topic = ready_topic
        # The operator's ready, judged under _lock; None: no ready, the scan or odometry decides.
        self._ready = operator_ready.ReadyLatch() if ready_topic is not None else None
        if battery_mv_topic is not None:
            ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                     for v in (battery_empty_mv, battery_full_mv))
            if not ok or not battery_empty_mv < battery_full_mv:
                raise ValueError(f"battery_mv_topic needs battery_empty_mv below battery_full_mv, "
                                 f"got {battery_empty_mv!r} and {battery_full_mv!r}")
        self._battery_mv_topic = battery_mv_topic
        self._battery_span = (battery_empty_mv, battery_full_mv)
        self._now = clock or time.monotonic
        # The manual-drive gate. Read and changed only under `_lock`, the lock every stop takes
        # in `_halt_locked`, which closes it.
        self._teleop = manual_drive.TeleopGate(self._now)
        self._battery_t: float | None = None     # when the latest reading arrived
        self._battery_since = self._now()         # start() resets it: silence counts from there
        self._battery_publishers: int | None = None
        self._battery_said = ""
        if odometry_source is not None and odometry_source not in ODOMETRY_SOURCES:
            raise ValueError(f"odometry_source must be one of {', '.join(ODOMETRY_SOURCES)} "
                             f"or None, got {odometry_source!r}")
        if response_bound_s is not None and not (
                isinstance(response_bound_s, (int, float)) and not isinstance(response_bound_s, bool)
                and math.isfinite(response_bound_s) and response_bound_s > 0):
            raise ValueError(f"response_bound_s must be a finite number greater than 0, "
                             f"got {response_bound_s!r}")
        self._robot_id = robot_id
        self._floor = floor
        self._odometry_source = odometry_source
        self._lock = threading.Lock()
        self._position: telemetry.Position | None = None
        self._speed: tuple[float, float] | None = None  # (linear, angular); None until first odom
        # WHEN the reading above was taken, never WHETHER the robot is at rest: nothing in
        # is_stopped() reads these. `_odom_count` numbers odom messages in
        # receipt order, so a reader can prove it saw consecutive ones.
        self._odom_stamp: int | None = None  # header.stamp of that same message, ns
        self._odom_count = 0
        # The frame odometry is published in (its header.frame_id), and the latest
        # map -> that frame from TF. None = no map: pose() then answers in odom and says so.
        self._odom_frame = "odom"
        self._map_from_odom: telemetry.Position | None = None
        self._degraded_announced = False  # "no map" said, and not yet taken back
        self._battery = telemetry.Battery()
        self._explore = explore  # the exploration stack; see explore_stack()
        self._hardware: list[telemetry.Hardware] = []
        self._goal: task_map.GoalPose | None = None  # pending/active NavigateToPose target
        self._goal_handle = None  # rclpy action goal handle, set once start() sends it
        self._goal_status = 0  # action_msgs/GoalStatus of the active goal; 0 = in-flight
        self._nav_initial: float | None = None  # first distance_remaining (100% baseline)
        self._nav_distance: float | None = None  # latest distance_remaining from feedback
        self._holding = False  # a stop has been commanded; hold until a new goal
        self._node = None  # rclpy node, created in start()
        self._action_client = None   # NavigateToPose ActionClient (start())
        self._cmd_vel = None         # geometry_msgs/Twist publisher (halt on cancel)
        self._nav_action = None      # cached nav2_msgs/action/NavigateToPose type
        self._twist_type = None      # cached geometry_msgs/Twist type
        self._spin_thread = None     # rclpy executor thread
        self._stopping = False       # stop() asked the executor to end (a clean exit)
        # Set by the executor thread if it ends for any reason other than stop().
        # A dead executor means NO further ROS 2 callback of any kind will run, so
        # every reading below is frozen at its last value -- see transport_alive().
        self._spin_fault: BaseException | None = None
        self._spin_fault_raised = False  # tick() re-raises the fault exactly once
        # tick() reports a NEVER-STARTED transport exactly once, and separately from the
        # fault above. Separate flag, not a shared one: "never brought up" and "brought up
        # and then died" are different diagnoses with different remedies, and a binding
        # reported for one can still later need reporting for the other (a late start()
        # whose executor then dies).
        self._never_started_raised = False
        # Increments on every dispatch. A result callback carries the epoch it was
        # registered under, so a SUPERSEDED goal's result can be told apart from the
        # current goal's -- see on_goal_result.
        self._goal_epoch = 0
        # The progress watchdog, or None when unarmed; and why the binding itself ended the
        # current goal ("" when it did not), reported with the FAILED it causes.
        self._watchdog = (None if stall_limits is None else
                          ProgressWatchdog(stall_limits, **({"clock": clock} if clock else {})))
        self._failure = ""
        # A STOP REQUESTED BEFORE THE SERVER HAS ANSWERED A GOAL. `send_goal_async` returns
        # before the goal handle exists, so a stop in that gap has no handle to cancel; the
        # server then accepts the goal and the robot drives while the stop is reported done.
        # Every path that stops the robot (cancel, estop, lease self-stop, reconcile, pause,
        # the watchdog) goes through `_halt_locked`, which records the stop against every
        # goal still unanswered; `_on_goal_response` cancels such a goal the moment its
        # handle exists and never adopts it. One mechanism for all of them.
        self._unanswered: set[int] = set()    # epochs dispatched (sent, being sent or parked),
                                              # no goal response yet
        self._stop_pending: set[int] = set()  # of those, the ones a stop was recorded against
        self._settled = threading.Condition(self._lock)  # notified as _stop_pending drains
        # NOTHING THAT WAITS FOR THE SERVER RUNS ON A CALLER'S THREAD. A goal dispatched while
        # the navigation server is not up is PARKED here, not sent: `_send_when_ready`, on its
        # own thread, waits for the server holding no lock and then sends the newest parked
        # goal. A stop withdraws a parked goal outright -- it was never sent, so there is
        # nothing to cancel and nothing to wait for. `command_move` used to wait for the server
        # itself, with no bound, and the adapter called it holding the lock every stop needed.
        self._parked: tuple[task_map.GoalPose, int] | None = None
        self._sender: threading.Thread | None = None
        # A GOAL WHOSE ANSWER NEVER COMES. A goal response can be lost (the server crashed, or
        # the request went out while discovery was incomplete). Its epoch would then stay in
        # `_unanswered` for ever: every later stop would record it and none would settle, and if
        # the server did accept it the adapter would hold no handle to cancel it. So a SENT goal
        # has a bounded life, `response_bound_s`. Past it, `_watch_answers` ABANDONS it: the
        # goal is cancelled by the id it was sent with, the robot is stopped if it was the
        # current goal (and that goal is FAILED, with the reason), and the epoch leaves
        # `_unanswered` and `_stop_pending`, so no later stop waits on it. An answer that comes
        # after all is handled as a stop's is: accepted, it is cancelled on arrival.
        self._response_bound = response_bound_s
        self._sent_at: dict[int, float] = {}   # unanswered SENT goals: epoch -> when sent
        self._goal_ids: dict[int, object] = {}  # unanswered SENT goals: epoch -> its goal id
        self._abandoned: set[int] = set()      # epochs abandoned unanswered
        self._answer_watch: threading.Thread | None = None

    @property
    def stall_limits(self) -> StallLimits | None:
        return self._watchdog.limits if self._watchdog is not None else None

    # ── ROS 2 subscription callbacks (also the test entry points) ─────────────

    def on_odom(self, odom) -> None:
        pos = telemetry.position_from_odom(odom, floor=self._floor)
        speed = telemetry.speed_from_odom(odom)  # ground truth for is_stopped (C5.3)
        stamp = _stamp_ns(odom)  # record-only: evidence about WHEN, never WHETHER
        frame = getattr(getattr(odom, "header", None), "frame_id", "") or "odom"
        with self._lock:
            self._position = pos
            self._odom_frame = frame
            self._speed = speed
            self._odom_stamp = stamp
            self._odom_count += 1

    def on_scan(self, msg) -> None:
        """One ``sensor_msgs/LaserScan`` for the handover's rest channel. Its stamp is the
        header's; it is received now, on this binding's clock."""
        if self._scan_rest is None:
            return
        ns = _stamp_ns(msg)
        try:
            scan = scan_rest.Scan(
                stamp=(ns * 1e-9 if ns is not None else math.nan), ranges=list(msg.ranges),
                range_min=float(msg.range_min), range_max=float(msg.range_max),
                angle_min=float(msg.angle_min), angle_increment=float(msg.angle_increment))
        except (AttributeError, TypeError, ValueError):
            scan = scan_rest.Scan(math.nan, [], math.nan, math.nan, math.nan, math.nan)
        self._scan_rest.add(scan, self._now())

    def on_map_to_odom(self, tf) -> None:
        """The latest ``map -> odom`` from TF (a ``TransformStamped``), or None when a
        lookup found no ``map``. Called by start()'s TF timer; also the test entry point.

        Composing this with the odom pose is what puts pose() in the frame goals are sent
        in. The two differ by whatever localisation has corrected so far -- on hardware,
        0.088 m and -9.8 degrees after one 1 m move under SLAM -- and the difference grows
        with gyro drift, so reading odom as map is a drive to the wrong place."""
        transform = telemetry.transform_from_tf(tf) if tf is not None else None
        with self._lock:
            self._map_from_odom = transform
        self._announce_frame()

    def _announce_frame(self) -> None:
        """Say when pose() stops, and starts again, answering in ``map``. Once per
        transition, never per call: the fall-back to odom is a declared degradation,
        not a silent one, and not a flood either."""
        with self._lock:
            degraded = self._map_from_odom is None
            if degraded == self._degraded_announced:
                return
            self._degraded_announced = degraded
            odom_frame = self._odom_frame
        if degraded:
            print(f"Nav2Binding: no {MAP_FRAME} -> {odom_frame} transform; pose() now "
                  f"reports the {odom_frame!r} frame, which goals are NOT sent in",
                  file=sys.stderr, flush=True)
        else:
            print(f"Nav2Binding: {MAP_FRAME} -> {odom_frame} transform available; pose() "
                  f"reports the {MAP_FRAME!r} frame again", file=sys.stderr, flush=True)

    def on_battery(self, state) -> None:
        bat = telemetry.battery_from_state(state)
        with self._lock:
            self._battery = bat
            self._battery_t = self._now()

    def on_battery_mv(self, msg) -> None:
        """A ``std_msgs/UInt16`` reading in mV (``battery_mv_topic``)."""
        mv = float(msg.data)
        empty, full = self._battery_span
        pct = max(0, min(100, round(100.0 * (mv - empty) / (full - empty))))
        with self._lock:
            self._battery = telemetry.Battery(percent=pct, charging=None, voltage=mv / 1000.0)
            self._battery_t = self._now()

    def battery_source(self) -> str:
        if self._battery_mv_topic is not None:
            return f"{self._battery_mv_topic} (std_msgs/UInt16, mV)"
        return "battery_state (sensor_msgs/BatteryState)"

    def battery_age_s(self) -> float | None:
        """Seconds since the latest battery reading arrived; None if none ever has."""
        with self._lock:
            return None if self._battery_t is None else self._now() - self._battery_t

    def battery_fault(self) -> str:
        """Why the battery cannot be reported, or "": nothing received for
        ``BATTERY_SILENT_S``, since the binding started or since the last reading."""
        now = self._now()
        with self._lock:
            last, pubs = self._battery_t, self._battery_publishers
        pub = "" if pubs is None else f"; publishers on it: {pubs}"
        if last is None:
            if now - self._battery_since < BATTERY_SILENT_S:
                return ""
            return (f"no battery reading on {self.battery_source()} in "
                    f"{now - self._battery_since:.0f} s since start{pub}")
        if now - last > BATTERY_SILENT_S:
            return f"no battery reading on {self.battery_source()} for {now - last:.0f} s{pub}"
        return ""

    def on_diagnostics(self, diag) -> None:
        hw = telemetry.hardware_from_diagnostics(diag)
        with self._lock:
            self._hardware = hw

    # ── RobotBinding: telemetry accessors ─────────────────────────────────────

    def explore_stack(self):
        """The live exploration stack (`ros2_explore.Ros2ExploreStack`) when the adapter was
        given the battery thresholds and a site file to reach the robot's stack; None
        otherwise, and then explore-floor and Charge are not declared on this binding."""
        return self._explore

    def spawn(self, robot_id: str) -> None:
        self._robot_id = robot_id

    def _position_locked(self) -> telemetry.Position | None:
        p = self._position
        if p is None or self._map_from_odom is None:
            return p
        return telemetry.compose(self._map_from_odom, p)

    def pose(self, robot_id: str) -> tuple[float, float, int]:
        """The robot's position, in ``map`` whenever TF can reach it (see
        :meth:`pose_frame`); in the odometry frame, declared, when it cannot."""
        self._announce_frame()
        with self._lock:
            p = self._position_locked()
        return (p.x, p.y, p.floor) if p is not None else (0.0, 0.0, self._floor)

    def yaw(self, robot_id: str) -> float:
        """Heading in the same frame as :meth:`pose`."""
        self._announce_frame()
        with self._lock:
            p = self._position_locked()
        return p.yaw if p is not None else 0.0

    def pose_frame(self, robot_id: str) -> str:
        """The frame :meth:`pose` and :meth:`yaw` are answering in right now: ``map``, or
        the odometry frame when no ``map -> odom`` transform is available."""
        with self._lock:
            return MAP_FRAME if self._map_from_odom is not None else self._odom_frame

    def odom_frame(self, robot_id: str) -> str:
        """The odometry frame, from the odom messages' own header (``odom`` until one
        says otherwise). The TF lookup is ``map`` -> this frame."""
        with self._lock:
            return self._odom_frame

    def battery_percent(self, robot_id: str) -> int | None:
        """The charge in percent; None when it is unknown -- never read, or no percent in the
        reading -- which the adapter reports as an absent battery, never as 0."""
        with self._lock:
            return self._battery.percent

    def latest_battery(self, robot_id: str) -> telemetry.Battery:
        with self._lock:
            return self._battery

    def latest_hardware(self, robot_id: str) -> list[telemetry.Hardware]:
        """The diagnostics, and a ``battery`` entry Failed while no battery reading arrives:
        a topic nobody publishes is reported, never read as an empty battery."""
        fault = self.battery_fault()
        if fault != self._battery_said:
            self._battery_said = fault
            print(f"Nav2Binding: battery {'MISSING: ' + fault if fault else 'reporting again'}",
                  file=sys.stderr, flush=True)
        with self._lock:
            hw = list(self._hardware)
        if fault:
            hw.append(telemetry.Hardware("battery", telemetry.STATUS_FAILED, fault))
        return hw

    def odom_stamp(self, robot_id: str) -> tuple[int, int | None]:
        """``(count, stamp_ns)`` of the odom message :meth:`is_stopped` is currently
        judging: how many odom messages have been received, and the robot-clock
        ``header.stamp`` of the latest (None if it carried none). ``(0, None)`` before
        the first.

        A measurement seam, not a safety one. The stamp says WHEN a reading
        was taken and must never decide WHETHER the robot is at rest -- is_stopped() does
        not read it. The stamp is on the ROBOT's clock, so subtract it only from another
        stamp, never from this process's ``time.monotonic()``."""
        with self._lock:
            return self._odom_count, self._odom_stamp

    # ── transport liveness ────────────────────────────────────────────────────
    # The executor thread is the only thing feeding every accessor above. When it
    # dies they all keep answering — with the last value each cached before it did.

    def _record_spin_exit(self, exc: BaseException | None) -> None:
        """Called by the executor thread as it ends. ``exc`` is what ended it, or
        None if ``rclpy.spin()`` simply returned.

        A clean exit is only the one ``stop()`` asked for; anything else is a fault,
        including a normal return, because ``spin()`` returning means the executor is
        gone and no callback will ever run again.

        Separated from :meth:`start` so the failure path is reachable in tests without
        a ROS 2 runtime -- the same test seam ``on_goal_result`` is."""
        with self._lock:
            if self._stopping or self._spin_fault is not None:
                return
            self._spin_fault = exc if exc is not None else RuntimeError(
                "rclpy.spin() returned with no shutdown requested")

    def transport_fault(self) -> BaseException | None:
        """The exception that ended the rclpy executor thread, or None while the
        transport is healthy (and after a deliberate :meth:`stop`)."""
        with self._lock:
            return self._spin_fault

    def transport_alive(self) -> bool:
        """True only while the rclpy executor thread is running and has not faulted.

        False after :meth:`stop`, and False before :meth:`start` -- callers on the $0
        path have no transport at all, which is not the same as a BROKEN one; use
        :meth:`transport_fault` to tell those two apart."""
        with self._lock:
            if self._spin_fault is not None:
                return False
            thread = self._spin_thread
        return thread is not None and thread.is_alive()

    # ── RobotBinding: motion ──────────────────────────────────────────────────

    def command_move(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Record the NavigateToPose target. When a ROS 2 node is live (start()),
        the target is dispatched to the Nav2 action server; without one, it is only
        cached so the translation is unit-testable at $0."""
        with self._lock:
            self._goal = task_map.GoalPose(x=x, y=y, yaw=yaw)
            self._holding = False  # a new goal releases a prior hold
            self._goal_status = 0  # new goal in-flight → RUNNING
            self._nav_initial = None  # progress baseline reset for the new goal
            self._nav_distance = None
            self._goal_epoch += 1
            epoch = self._goal_epoch
            self._failure = ""
            if self._watchdog is not None:
                self._watchdog.arm()  # from dispatch: a goal that never reports is watched too
            goal = self._goal
            live = self._register_locked(epoch)
        if live:
            self._dispatch(goal, epoch)

    def _register_locked(self, epoch: int) -> bool:
        """Record a goal as unanswered in the same step that makes it the current goal, so a
        stop -- which takes only this lock, from any thread -- finds it however soon it comes.
        False without a live node: nothing will be sent."""
        if self._node is None:
            return False
        self._unanswered.add(epoch)
        return True

    def _dispatch(self, goal, epoch: int) -> None:
        """Send a goal without ever waiting: at once when the navigation server is up,
        otherwise parked for `_send_when_ready`. A send that raises sent nothing, so it leaves
        nothing for a stop to wait on, and raises to the caller as before."""
        if not self._server_ready():
            with self._settled:
                if self._withdrawn_locked(epoch):
                    return
                if self._parked is not None:
                    self._unanswered.discard(self._parked[1])  # replaced before it was sent
                self._parked = (goal, epoch)
                start = self._sender is None
                if start:
                    self._sender = threading.Thread(target=self._send_when_ready, daemon=True)
                self._settled.notify_all()
            if start:
                self._sender.start()
            return
        with self._settled:
            if self._parked is not None:
                self._unanswered.discard(self._parked[1])      # this one supersedes it
                self._parked = None
            if self._withdrawn_locked(epoch):
                return
        self._send(goal, epoch, raise_errors=True)

    def _withdrawn_locked(self, epoch: int) -> bool:
        """Whether a goal not yet sent has been overtaken -- a stop reached it, or a newer goal
        replaced it. If so it is withdrawn: never sent, so nothing is left for a stop to wait
        on. A goal that passes this and is then stopped before the send completes is still
        recorded against the stop, and cancelled on arrival."""
        if epoch in self._stop_pending or epoch != self._goal_epoch:
            self._unanswered.discard(epoch)
            self._stop_pending.discard(epoch)
            self._settled.notify_all()
            return True
        return False

    def _send(self, goal, epoch: int, raise_errors: bool) -> None:
        start = False
        if self._response_bound is not None:
            with self._settled:
                # From just before the send: the bound can only err towards abandoning sooner.
                self._sent_at[epoch] = time.monotonic()
                start = self._answer_watch is None
                if start:
                    self._answer_watch = threading.Thread(target=self._watch_answers, daemon=True)
                self._settled.notify_all()
            if start:
                self._answer_watch.start()
        try:
            self._send_nav2_goal(goal, epoch)
        except BaseException as exc:
            with self._settled:
                self._unanswered.discard(epoch)
                self._stop_pending.discard(epoch)
                self._sent_at.pop(epoch, None)
                self._goal_ids.pop(epoch, None)
                self._settled.notify_all()
            if raise_errors:
                raise
            print(f"goal dispatch failed; the goal was not sent ({exc!r})", file=sys.stderr)

    def _send_when_ready(self) -> None:
        """The sender thread: wait for the navigation server, holding nothing, then send the
        newest parked goal if nothing has withdrawn it meanwhile. The only place this binding
        waits for the server, so nothing a stop needs is ever held while it does."""
        while True:
            with self._settled:
                self._settled.wait_for(lambda: self._parked is not None or self._stopping)
                if self._stopping:
                    return
            ready = self._action_client.wait_for_server()
            with self._settled:
                if self._stopping:
                    return
                parked, self._parked = self._parked, None
                if parked is None:
                    continue                                   # a stop withdrew it
                goal, epoch = parked
                if not ready:                                  # rclpy is shutting down
                    self._unanswered.discard(epoch)
                    continue
                if self._withdrawn_locked(epoch):
                    continue
            self._send(goal, epoch, raise_errors=False)

    def _watch_answers(self) -> None:
        """The answer watch: abandon every sent goal still unanswered `response_bound_s` after
        it was sent. Waits on `_settled`, which releases the lock, so it holds nothing while it
        waits; the stop it commands is `command_stop`'s, made outside the lock."""
        bound = self._response_bound
        while True:
            with self._settled:
                while True:
                    if self._stopping:
                        return
                    now = time.monotonic()
                    due = [e for e, t in self._sent_at.items() if now - t >= bound]
                    if due:
                        break
                    left = [t + bound - now for t in self._sent_at.values()]
                    self._settled.wait(min(left) if left else None)
                halt, handle, goal_ids = self._abandon_locked(due)
            print(f"goal response overdue: abandoned {len(due)} goal(s) unanswered after "
                  f"{bound:g} s" + ("; the robot was stopped and the goal FAILED" if halt else ""),
                  file=sys.stderr)
            if self._node is not None:
                if halt:              # the base first: a cancel by id that fails cannot keep it
                    self._cancel_nav2_goal(handle)
                for goal_id in goal_ids:
                    try:
                        self._cancel_goal_by_id(goal_id)
                    except Exception as exc:  # noqa: BLE001 - the watch must outlive a fault
                        print(f"goal response overdue: cancelling an abandoned goal by its id "
                              f"raised {type(exc).__name__}: {exc}", file=sys.stderr)

    def _abandon_locked(self, due: list[int]):
        """Abandon the goals in `due`, under the lock. Returns ``(halt, handle, goal_ids)``:
        whether the robot must be stopped, the adopted handle that stop cancels, and the ids to
        cancel the abandoned goals by.

        * **The current goal, not stopped:** it is over. It is stopped as `command_stop`
          stops a goal, and FAILED with the reason, through the one terminal path.
        * **Already stopped** (recorded in `_stop_pending`): the stop that recorded it is no
          longer held up by it. Its zero Twist is sent again, as the goal may since have been
          accepted and driven off.
        * **Superseded, not stopped:** only cancelled by its id. The current goal is not touched."""
        bound = self._response_bound
        halt, handle, goal_ids = False, None, []
        for epoch in due:
            del self._sent_at[epoch]
            self._unanswered.discard(epoch)
            stopped = epoch in self._stop_pending
            self._stop_pending.discard(epoch)
            self._abandoned.add(epoch)
            goal_id = self._goal_ids.pop(epoch, None)
            if goal_id is not None:
                goal_ids.append(goal_id)
            if epoch == self._goal_epoch and not self._holding:
                self._goal = None
                handle = self._halt_locked(why="the navigation server never answered the goal; "
                                               "it was abandoned and the base held")
                halt = True
                self._goal_status = _GS_ABORTED
                self._failure = (
                    f"the navigation server did not answer the goal within {bound:g} s "
                    f"(--goal-response-bound): it was abandoned -- the robot was stopped and the "
                    f"goal cancelled by its id, and if it is accepted after all it is cancelled "
                    f"on arrival")
            elif stopped:
                halt = True
        self._settled.notify_all()
        return halt, handle, goal_ids

    def _server_ready(self) -> bool:  # pragma: no cover — ROS 2 runtime
        """Whether the navigation server has been discovered. A look, never a wait."""
        return self._action_client.server_is_ready()

    def on_feedback(self, distance_remaining: float,
                    pose: tuple[float, float] | None = None) -> None:
        """A NavigateToPose action FeedbackMessage: the remaining distance to the
        goal. The first (largest) value is the 100% baseline; later values give
        progress. Ignores a non-positive baseline (no usable distance).

        ``pose`` is the robot's (x, y) from the same feedback, in the goal's frame. A path
        can never be shorter than the straight line to its goal, so a distance more than
        STALE_FEEDBACK_MARGIN_M under that line is not this goal's (Nav2's first feedback
        for a new goal can still carry the previous plan's) and is ignored, for progress
        and for the watchdog alike: taken as the baseline, it makes every real distance
        look like no progress."""
        with self._lock:
            if pose is not None and self._goal is not None:
                straight = math.hypot(self._goal.x - pose[0], self._goal.y - pose[1])
                if distance_remaining < straight - STALE_FEEDBACK_MARGIN_M:
                    return
            if self._nav_initial is None and distance_remaining > 0:
                self._nav_initial = distance_remaining
            self._nav_distance = distance_remaining
            if self._watchdog is not None:
                self._watchdog.observe(distance_remaining)

    def task_progress(self, robot_id: str) -> int:
        """Advisory task progress (0-100) from the NavigateToPose feedback distance:
        ``round((1 - remaining/initial) * 100)``, clamped. 0 until feedback arrives."""
        with self._lock:
            initial, dist = self._nav_initial, self._nav_distance
        if initial is None or initial <= 0 or dist is None:
            return 0
        pct = int(round((1.0 - max(0.0, dist) / initial) * 100))
        return max(0, min(100, pct))

    def pending_goal(self, robot_id: str) -> task_map.GoalPose | None:
        """The last NavigateToPose target handed to :meth:`command_move`, or None
        after a stop cleared it (test seam)."""
        with self._lock:
            return self._goal

    def command_stop(self, robot_id: str) -> None:
        """Cancel the active Nav2 goal and hold. This is a REQUEST — confirmation
        that the robot is at rest comes only from :meth:`is_stopped` (odom). Shared
        by confirmed estop (C5) and lease-expiry self-stop (C4).

        The base is held and the goal cancelled whatever the exploration stack does: its halt
        is a step of its own, and one that raises is reported and never keeps the base from
        being told to stop, nor ends the caller's loop."""
        try:
            if self._explore is not None:
                self._explore.halt()  # the explorer and the return goal stop with the base
        except Exception as exc:  # noqa: BLE001 - a stop path: the base's own stop still runs
            print(f"stop: the exploration stack's halt raised {type(exc).__name__}: {exc}; "
                  f"holding the base and cancelling the goal regardless", file=sys.stderr)
        finally:
            self.command_hold(robot_id)

    def command_hold(self, robot_id: str) -> None:
        """Hold the base and cancel the Nav2 goal, then publish the zero Twist: the base's own
        stop, without the exploration stack. `command_stop` is the explore halt and this; the
        adapter's shutdown calls it first and stops the explorer as a step of its own after."""
        with self._lock:
            self._goal = None
            handle = self._halt_locked()
        if self._node is not None:
            self._cancel_nav2_goal(handle)

    def _halt_locked(self, *, close_gate: bool = True, why: str = manual_drive.STOP_WHY):
        """Hold, take the adopted goal handle for cancelling, and record the stop against
        every goal the server has not answered yet. Called under ``self._lock`` by every
        path that stops the robot, so none of them can miss a goal in flight. It closes the
        manual-drive gate too, here and under the same lock, so no teleop command forwarded
        after this can follow the stop's zero Twist. ``close_gate=False`` is the handover's
        alone: the base is held for the operator, whose gate it has just turned. ``why`` is
        what the gate's record says closed it."""
        if close_gate:
            self._teleop.close(why)
        self._holding = True
        handle, self._goal_handle = self._goal_handle, None
        if self._parked is not None:
            # Never sent: withdrawn here, with nothing to cancel and nothing to wait for.
            self._unanswered.discard(self._parked[1])
            self._parked = None
        self._stop_pending |= self._unanswered
        if self._watchdog is not None:
            self._watchdog.disarm()
        return handle

    def stop_settled(self, robot_id: str, timeout: float) -> bool:
        """Wait up to ``timeout`` s until no goal a stop was recorded against is still
        unanswered: each has been refused, or accepted and cancelled on arrival. True when
        that holds; False if a goal the server may still accept and drive remains. Nothing
        may report the robot stopped, or an action stopped safely, while this is False."""
        with self._settled:
            return self._settled.wait_for(lambda: not self._stop_pending, timeout)

    def command_pause(self, robot_id: str) -> None:
        """Pause navigation: cancel the active NavigateToPose goal so the base halts,
        but KEEP the target so :meth:`command_resume` can re-dispatch it (unlike
        command_stop, which drops the target). Confirmation the base is at rest is
        odom (:meth:`is_stopped`), never inferred."""
        with self._lock:
            # self._goal (target) intentionally KEPT for resume. Holding on purpose is not a
            # stall, so the watchdog is disarmed with the rest.
            handle = self._halt_locked(why=manual_drive.PAUSE_WHY)
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)

    def command_resume(self, robot_id: str) -> None:
        """Resume a paused navigation: re-dispatch the retained NavigateToPose target."""
        with self._lock:
            goal = self._goal
            self._holding = False
            self._goal_status = 0  # re-dispatched goal is in-flight again
            # The goal cancelled by command_pause will deliver a CANCELED result; it must
            # not be attributed to the goal being resumed here.
            self._goal_epoch += 1
            epoch = self._goal_epoch
            if goal is not None and self._watchdog is not None:
                self._watchdog.arm()  # a fresh window, and a fresh wait for feedback
            live = goal is not None and self._register_locked(epoch)
        if live:
            self._dispatch(goal, epoch)

    # ── manual-drive: teleop, only through the gate ─────────────────────────────

    def teleop_topic(self) -> str | None:
        """The topic teleop is forwarded from, or None: no manual-drive on this binding."""
        return self._teleop_topic

    def open_teleop(self, robot_id: str, action_id: str, drive, lease_s: float) -> None:
        """Open the gate for ``action_id``, for ``lease_s`` unless renewed. The adapter calls
        this in the step that grants the action's lease, and nothing else opens it. One source
        of motion: the goal is held and cancelled first, and its target dropped, so a resume
        has nothing to re-dispatch under teleop."""
        with self._lock:
            self._goal = None
            handle = self._halt_locked(why="a new action's lease")  # opened after, here
            self._teleop.open(action_id, drive, lease_s)
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)

    def extend_teleop(self, robot_id: str, action_id: str, lease_s: float) -> bool:
        with self._lock:
            return self._teleop.extend(action_id, lease_s)

    def hold_for_handover(self, robot_id: str, action_id: str) -> bool:
        """Nav2 declined a navigate-handover's goal: hold the base as a stop holds it, drop the
        target so a resume has nothing to re-dispatch, and publish one zero Twist, with the gate
        still armed and forwarding nothing -- or refuse, when a stop closed the gate first. The
        adapter's job then waits for rest before it asks for `hand_over_teleop`."""
        with self._lock:
            if not self._teleop.is_open(action_id):
                return False
            self._goal = None
            handle = self._halt_locked(close_gate=False)
            self._publish_twist_locked(manual_drive.ZERO)
            self._handover_hold_at = self._now()
            if self._ready is not None:
                self._ready.arm(action_id, self._handover_hold_at)
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)
        return True

    # ── the operator's ready (with a scan channel as its veto) ──────────────

    def handover_confirms_by_operator(self, robot_id: str) -> bool:
        return self._ready is not None

    def on_ready(self, action_id: str | None = None) -> str:
        """One press of the operator's ready, received now on this binding's clock: "" when it
        is kept for the held handover, otherwise why it is ignored (logged)."""
        if self._ready is None:
            return "this binding takes no operator's ready"
        with self._lock:
            return self._ready.press(self._now(), self._teleop.declined, action_id,
                                     self._teleop.why_not_declined)

    def take_ready(self, robot_id: str, action_id: str) -> float | None:
        if self._ready is None:
            return None
        with self._lock:
            return self._ready.take(action_id, self._teleop.declined)

    def ready_log(self) -> list[str]:
        return list(self._ready.lines) if self._ready is not None else []

    def handover_rest_gap(self, robot_id: str) -> str:
        """Why the handover's rest can never be confirmed on this binding, or "". With a scan
        channel: "" (its record was checked before the adapter started). Without one: the
        estop's gap, `rest_evidence_gap`."""
        if self._scan_rest is not None:
            return ""
        return self.rest_evidence_gap(robot_id)

    def handover_at_rest(self, robot_id: str) -> tuple[bool, str]:
        """(at rest for the handover, why not). Without a scan channel: `is_stopped`."""
        with self._lock:
            return self._handover_at_rest_locked()

    def _handover_at_rest_locked(self) -> tuple[bool, str]:
        if self._scan_rest is None:
            return (True, "") if self._at_rest_locked() else (False, "")
        # Refute wins: whatever the scan says, these veto it.
        if self._stop_pending:
            return False, "a goal sent before the hold is still unanswered and may drive"
        if self._spin_fault is not None:
            return False, "the ROS 2 executor has stopped, so no scan is current"
        if self._speed is not None and (self._speed[0] > _EPS_LINEAR
                                        or self._speed[1] > _EPS_ANGULAR):
            return False, "odometry shows motion"
        now = self._now()
        if self._ready is not None:
            # The operator confirms; the scan must first have had its full look since the hold.
            early = operator_ready.look_gap(now, self._handover_hold_at)
            if early:
                return False, early
        v = self._scan_rest.verdict(now, self._handover_hold_at)
        if not v.at_rest:
            return False, f"scan: {v}"
        return (True, operator_ready.no_veto(self._scan_rest)) if self._ready is not None \
            else (True, "")

    def hand_over_teleop(self, robot_id: str, action_id: str, t: float, why: str) -> bool:
        """Nav2 declined a navigate-handover's goal: turn the armed gate to the operator, or refuse
        (`TeleopGate.hand_over`). Under ``_lock``, so a stop that closed the gate -- every stop
        does, in `_halt_locked` -- wins, and so does a lease deadline that has passed. Refused
        too unless the robot is confirmed at rest under that same lock: with a scan channel,
        the scan shows rest in a window after the hold and nothing vetoes it; without one,
        odometry declared measured and its twist within noise (`_at_rest_locked`). On
        command-derived or undeclared odometry and no scan channel that never holds, so this
        binding never hands over there. Handed
        over: the base is held as a stop holds it, the target is dropped so a resume has nothing
        to re-dispatch under teleop, and one zero Twist goes out; the gate stays open."""
        with self._lock:
            if self._ready is not None and not self._ready.confirm(action_id):
                return False
            if self.handover_rest_gap(robot_id) or not self._handover_at_rest_locked()[0]:
                return False
            if not self._teleop.hand_over(action_id, t, why):
                return False
            self._goal = None
            handle = self._halt_locked(close_gate=False)
            self._publish_twist_locked(manual_drive.ZERO)
        if self._node is not None:  # pragma: no cover — requires the ROS 2 runtime
            self._cancel_nav2_goal(handle)
        return True

    def close_teleop(self, robot_id: str, action_id: str = "") -> None:
        """Close the gate (for ``action_id``, or whichever is open) and publish one zero."""
        with self._lock:
            if not self._teleop.is_open(action_id):
                return
            self._teleop.close()
            self._publish_twist_locked(manual_drive.ZERO)

    def teleop_open(self, robot_id: str, action_id: str = "") -> bool:
        with self._lock:
            return self._teleop.is_open(action_id)

    def teleop_log(self, robot_id: str, after: int = 0) -> list[dict]:
        """The gate's changes after ``after`` (`TeleopGate.changes`). Read only."""
        with self._lock:
            return self._teleop.changes(after)

    def teleop_deadman(self, robot_id: str, deadman_s: float) -> None:
        with self._lock:
            if self._teleop.idle(deadman_s):
                self._publish_twist_locked(manual_drive.ZERO)

    def on_teleop(self, msg) -> None:
        """One ``geometry_msgs/Twist`` on the teleop topic: published to ``cmd_vel`` as the gate
        admits it, or not at all."""
        try:
            vx, vy, wz = float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)
        except (AttributeError, TypeError, ValueError):
            return
        with self._lock:
            cmd = self._teleop.admit(vx, vy, wz)
            if cmd is not None:
                self._publish_twist_locked(cmd)

    def _publish_twist_locked(self, cmd) -> None:
        """One Twist on ``cmd_vel``, under ``_lock`` so it cannot follow a stop's zero. rclpy's
        publish queues and does not wait."""
        if self._cmd_vel is None or self._twist_type is None:
            return
        t = self._twist_type()
        t.linear.x, t.linear.y, t.angular.z = cmd
        self._cmd_vel.publish(t)

    def on_goal_result(self, status: int, *, epoch: int | None = None,
                       reason: str = "") -> None:
        """The NavigateToPose action's terminal ``action_msgs/GoalStatus`` (SUCCEEDED /
        ABORTED / CANCELED), delivered by the action client's result callback.

        ``epoch`` identifies the DISPATCH this result belongs to, and a result carrying a
        stale one is DROPPED. Without it, a superseded goal's result overwrites the status
        of the goal that replaced it, and the adapter then reports that terminal state
        against whatever action is currently assigned — observed as ``SUCCEEDED`` at
        ``progress=100`` for an action assigned 192 ms earlier whose target was 6 m away.
        On the wire that is indistinguishable from real completion, so the control plane
        marks the action done and is free to dispatch the next one while the robot is still
        driving toward the abandoned goal.

        ``epoch=None`` means "unidentified" and is accepted as-is, which keeps this usable
        as a direct test seam.
        """
        with self._lock:
            if epoch is not None and epoch != self._goal_epoch:
                return
            self._goal_status = status
            if status == _GS_ABORTED and not self._failure:
                # Why the goal failed, for the FAILED report: the server's own verdict.
                self._failure = reason or "the navigation server aborted the goal"
            if self._watchdog is not None:
                self._watchdog.disarm()  # the goal has ended; there is nothing left to watch

    def check_progress(self) -> str:
        """Evaluate the progress watchdog once; return the stall reason, or "".

        On a stall the goal is ended HERE, by the binding, the way ``command_stop`` ends
        one: the NavigateToPose goal is cancelled, a zero Twist is published, and nothing
        further is commanded. The goal is then FAILED -- the status an ABORTED goal has, so
        the adapter reports it through the one terminal path it already has -- and
        :meth:`task_failure` carries the reason.

        The epoch moves on first. The cancel makes the server deliver CANCELED for this goal,
        and that result must not overwrite FAILED: the adapter streams no CANCELLED from
        here, so the task would be left RUNNING again, which is the defect this fixes.

        Called every ``_WATCHDOG_PERIOD_S`` on the executor once :meth:`start` has run, so a
        stall is caught whoever is, or is not, calling :meth:`tick`.

        A STALL IS NOT A STOP. While a navigate-handover's gate is armed for this goal
        (`TeleopGate.navigating`), the stall is Nav2's decline, as an ABORTED goal is: the goal
        is cancelled and the base held, as `command_hold` holds it, and the gate is left armed,
        forwarding nothing, so the job holds, waits for rest and hands over as it does on an
        abort. No stop asked for it, and every stop path still closes the gate. With no handover
        armed, the stall ends the goal through ``command_stop``, as before."""
        with self._lock:
            if self._watchdog is None:
                return ""
            reason = self._watchdog.check()
            if not reason:
                return ""
            self._goal_epoch += 1
            self._goal_status = _GS_ABORTED
            self._failure = reason
            handing_over = self._teleop.navigating()
            if handing_over:
                self._goal = None
                handle = self._halt_locked(close_gate=False)
        if not handing_over:
            self.command_stop(self._robot_id)
        elif self._node is not None:
            self._cancel_nav2_goal(handle)
        return reason

    def task_failure(self, robot_id: str) -> str:
        """Why the binding itself ended the current goal (a stall), or "" if it did not."""
        with self._lock:
            return self._failure

    def task_state(self, robot_id: str) -> str:
        """Protocol ``TaskState`` for the active goal, derived from its Nav2 action
        ``GoalStatus`` — the task-status stream source (C6.4)."""
        with self._lock:
            status = self._goal_status
        return task_map.task_state_from_goal_status(status)

    def _send_nav2_goal(self, goal: task_map.GoalPose, epoch: int) -> None:  # pragma: no cover — ROS 2 runtime
        # Build a NavigateToPose goal (PoseStamped, yaw→quaternion) and dispatch it
        # via the action client; feedback → on_feedback (progress), result → on_goal_result.
        import uuid

        from geometry_msgs.msg import PoseStamped

        nav_goal = self._nav_action.Goal()
        pose = PoseStamped()
        pose.header.frame_id = getattr(goal, "frame", "") or "map"
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(goal.x)
        pose.pose.position.y = float(goal.y)
        half = float(goal.yaw) / 2.0  # yaw about +Z → quaternion (z, w)
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)
        nav_goal.pose = pose

        # The goal's id is chosen here, not by rclpy, so that a goal whose answer never comes
        # can still be cancelled by it (`_abandon_locked`).
        from unique_identifier_msgs.msg import UUID
        goal_id = UUID(uuid=list(uuid.uuid4().bytes))
        with self._lock:
            if self._response_bound is not None and epoch in self._sent_at:
                self._goal_ids[epoch] = goal_id
        # No wait_for_server() here: `_dispatch` sends only once the server is up, and parks
        # the goal for `_send_when_ready` otherwise, so this never waits on the caller's thread.
        send_future = self._action_client.send_goal_async(
            nav_goal, feedback_callback=self._on_nav_feedback, goal_uuid=goal_id)
        send_future.add_done_callback(lambda f: self._on_goal_response(f, epoch))

    def _on_goal_response(self, future, epoch: int) -> None:
        """The server's answer to a goal. Decided under the lock, in one step, so no stop
        can fall between "is a stop recorded?" and "adopt the handle":

        * a stop was recorded while it was unanswered, or it was abandoned unanswered
          (`_abandon_locked`): CANCELLED ON ARRIVAL -- the goal is cancelled and a zero Twist
          published, exactly as a stop cancels an adopted goal, and the handle is never
          adopted, so nothing it does is reported as the action's;
        * refused: nothing to cancel; reported ABORTED if it is still the current goal;
        * superseded by a newer goal (and not stopped): dropped, not adopted -- adopting it
          would make a later stop cancel the WRONG goal. The server preempts it;
        * otherwise adopted, and its result awaited."""
        handle = future.result()
        with self._settled:
            self._unanswered.discard(epoch)
            self._sent_at.pop(epoch, None)
            self._goal_ids.pop(epoch, None)
            stopped = epoch in self._stop_pending
            self._stop_pending.discard(epoch)
            abandoned = epoch in self._abandoned
            self._abandoned.discard(epoch)
            if abandoned:
                # Already ended here, and reported: accepted, it is cancelled on arrival;
                # refused, there is nothing left to do.
                verdict = "cancel" if handle.accepted else "late"
            elif not handle.accepted:
                verdict = "refused"
            elif stopped:
                verdict = "cancel"
            elif epoch != self._goal_epoch:
                verdict = "superseded"
            else:
                verdict = "adopt"
                self._goal_handle = handle
            self._settled.notify_all()
        if verdict == "refused":
            self.on_goal_result(_GS_ABORTED, epoch=epoch,
                                reason="the navigation server rejected the goal")
        elif verdict == "cancel":
            self._cancel_nav2_goal(handle)
        elif verdict == "adopt":
            handle.get_result_async().add_done_callback(
                lambda f: self.on_goal_result(f.result().status, epoch=epoch))

    def _on_nav_feedback(self, feedback_msg) -> None:  # pragma: no cover — ROS 2 runtime
        # NavigateToPose feedback carries distance_remaining → progress baseline, and the
        # robot's pose, which lets on_feedback refuse a distance that is not this goal's.
        fb = feedback_msg.feedback
        pos = fb.current_pose.pose.position
        self.on_feedback(float(fb.distance_remaining), pose=(float(pos.x), float(pos.y)))

    def _on_watchdog_timer(self) -> None:  # pragma: no cover — ROS 2 runtime
        reason = self.check_progress()
        if reason:
            import sys
            print(f"progress watchdog: goal cancelled and FAILED -- {reason}", file=sys.stderr)

    def _cancel_goal_by_id(self, goal_id) -> None:  # pragma: no cover — ROS 2 runtime
        # A CancelGoal request for the one goal sent with this id, for a goal whose handle
        # never came. Harmless if the server never received it: it answers "unknown goal".
        from rclpy.action.client import ClientGoalHandle
        ClientGoalHandle(self._action_client, goal_id, None).cancel_goal_async()

    def _cancel_nav2_goal(self, handle) -> None:
        # Cancel the NavigateToPose goal and publish a zero Twist to cmd_vel so the
        # base halts even before Nav2 tears the controller down. A cancel that raises is
        # reported and the zero Twist still goes out.
        try:
            if handle is not None:
                handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001 - a stop path: the zero Twist still goes out
            print(f"stop: cancelling the navigation goal raised {type(exc).__name__}: {exc}; "
                  f"publishing the zero Twist regardless", file=sys.stderr)
        finally:
            if self._cmd_vel is not None and self._twist_type is not None:
                self._cmd_vel.publish(self._twist_type())

    def tick(self, dt: float) -> None:
        """rclpy spins its own executor, so there is no model to advance -- but this
        is the binding's only guaranteed MAIN-THREAD callback (the adapter's telemetry
        loop drives it every tick), which makes it the one place a transport that is not
        working can be surfaced to a thread that is not itself broken.

        Raises on BOTH ways the transport can fail to deliver callbacks, each with its own
        diagnosis, each exactly once:

        * **the executor DIED** -- ``start()`` ran, the thread came up, and then ended.
          ``_record_spin_exit`` caught the exit, so ``transport_fault()`` holds the cause
          and it is re-raised from here, which is the half of this the first fix covered.
        * **the transport NEVER EXISTED** -- ``start()`` was never called, so there is no
          node and no thread was ever created, so there was no exit to record and
          ``_spin_fault`` stays ``None``. To a detector that raises off the recorded fault,
          that state is **byte-for-byte indistinguishable from perfect health** -- which is
          why the original detector covered the executor dying and was blind to the executor
          never existing.

        The second case is the one this method used to miss, and missing it was expensive:
        an inert binding raises NOTHING on any path. `pose()` answers the origin,
        `battery_percent()` answers 0, `command_move()` and `command_stop()` return `None`
        exactly as success does, and `is_stopped()` answers False forever. A control plane
        cannot tell that from a robot genuinely parked at the origin on a flat battery, and
        for months nothing did.

        Raising is deliberate, and the alternative is worse in both cases: telemetry that
        keeps flowing at full cadence carrying values no callback will ever update again.
        Publishing nothing is recoverable; publishing stale or invented readings as fresh is
        not. In the adapter the raise ends the telemetry thread, so the control plane sees a
        telemetry GAP -- a signal it can act on -- rather than a healthy-looking stream.

        ONCE, not on every tick, so a caller that logs and carries on is told rather than
        spammed; the state stays readable afterwards via :meth:`transport_alive` and
        :meth:`transport_fault`.

        A deliberate :meth:`stop` is silent, in both directions. `stop()` makes
        `rclpy.spin()` return on purpose and leaves no thread behind, so without this guard
        every clean shutdown would raise out of whichever tick landed next -- and the
        teardown path does still tick on its way out. `stop()` on a binding that was never
        started silences this too: the caller has said it is done, and nobody should be
        ticking a binding after asking for its teardown.

        NOTE what this does NOT do: it cannot make an un-started binding work, and it is not
        what stops the adapter shipping one -- `adapter.bring_up_robot()` is, by calling
        `start()` unconditionally on the path every run takes. This is the backstop for the
        NEXT caller, which may not be that path: a new entry point, a vendor's own harness,
        or a refactor that moves the lifecycle. The lesson of the inert adapter was that one
        which can be handed an inert binding and cannot tell will be handed one again, so the
        binding is where the telling belongs -- it is the only object that knows."""
        with self._lock:
            # Checked FIRST: a requested teardown is not a failure in either direction.
            if self._stopping:
                return
            fault = self._spin_fault
            report_dead = fault is not None and not self._spin_fault_raised
            if report_dead:
                self._spin_fault_raised = True
            # `_node is None` is the discriminator, chosen over `_spin_thread is None` and
            # over `not transport_alive()`. It is the EXACT condition every live path in this
            # class already guards on (`if self._node is not None`), so the raise fires in
            # precisely the state where goal dispatch, cancel and the zero-Twist halt
            # silently do nothing -- the detector and the silence now have one trigger
            # instead of two that can drift apart.
            #
            # `_spin_thread` would also be None between `start()` assigning it and starting
            # it, making a concurrent tick raise on a binding that is coming up correctly.
            # `transport_alive()` additionally folds in thread liveness, so it cannot tell
            # "never brought up" from "brought up and the thread is between states".
            report_unstarted = (fault is None
                                and self._node is None
                                and not self._never_started_raised)
            if report_unstarted:
                self._never_started_raised = True
        if report_dead:
            raise RuntimeError(
                "the ROS 2 executor thread is dead; this binding has stopped receiving "
                "odom, battery, diagnostics and action callbacks, and every reading it "
                f"reports is frozen at its last value ({fault!r})") from fault
        if report_unstarted:
            raise RuntimeError(
                "this Nav2Binding was NEVER STARTED: there is no rclpy node, no odom / "
                "battery / diagnostics subscription and no NavigateToPose action client, "
                "so no callback has ever run and none ever will. Every reading it reports "
                "is a default, not a measurement -- pose() is the origin, "
                "battery_percent() is 0, and is_stopped() is False forever, which can "
                "never confirm a stop (C5.3). Nothing raised to say so because there is no "
                "failed operation to raise from. The adapter brings the transport up in "
                "adapter.bring_up_robot(); a caller that constructs a Nav2Binding outside "
                "that path must call start() itself.")

    def rest_evidence_gap(self, robot_id: str) -> str:
        """Why no stop on this binding may be reported confirmed, or ``""`` when one may.

        ``""`` only when the odometry was declared ``measured``. A binding declared
        ``command-derived``, and one given no declaration at all, both answer with the
        sentence the adapter carries verbatim into the stop report, so the control plane and
        the operator are told what the platform cannot do instead of reading a bare failure.
        """
        return rest_evidence_gap(self._odometry_source)

    def is_stopped(self, robot_id: str) -> bool:
        """CONFIRMED at rest: the measured odom twist is within sensor noise. Returns
        False until odom has been observed — rest is never inferred from the stop
        command, a timer, or the absence of a message (C5.3).

        Returns False ALWAYS when the odometry was declared command-derived. That odom is
        the velocity command published back, so it reads zero the instant a stop is
        commanded and would answer True while the base is still braking, still slipping or
        still being pushed. There is nothing to wait for and nothing to sample: the channel
        cannot distinguish a robot that stopped from one that was told to
        (:meth:`rest_evidence_gap`)."""
        with self._lock:
            return self._at_rest_locked()

    def _at_rest_locked(self) -> bool:
        """:meth:`is_stopped`, for a caller that holds ``self._lock``."""
        if self._odometry_source == ODOMETRY_COMMAND_DERIVED:
            return False
        speed = self._speed
        if self._stop_pending:
            # A goal sent before the stop has not been answered: the server may still
            # accept it and drive. Refute wins -- whatever odom says now, rest cannot be
            # confirmed until that goal is refused or cancelled on arrival.
            return False
        if self._spin_fault is not None:
            # The executor is dead, so `speed` is whatever odom last said before it
            # died and nothing will ever update it. C5.3 forbids confirming rest
            # from silence, and a dead transport IS silence -- a robot that was at
            # rest when the thread died is not evidence it is at rest now.
            return False
        if speed is None:
            return False  # no odom yet → no evidence of rest
        linear, angular = speed
        return linear <= _EPS_LINEAR and angular <= _EPS_ANGULAR

    # ── ROS runtime ───────────────────────────────────────────────────────────

    def start(self) -> None:  # pragma: no cover — requires the ROS 2 runtime
        """Create the rclpy node, subscribe to odom/battery/diagnostics, open the
        NavigateToPose action client + cmd_vel publisher, and spin rclpy on a
        background thread. Requires a ROS 2 + Nav2 environment.

        NOTE: written to the rclpy / nav2_msgs API; validate against a live ROS 2 +
        Nav2 stack (it cannot be exercised without one). The `$0` path never reaches
        here — every dispatch is guarded by `self._node is not None`."""
        try:
            import atexit
            import sys
            import threading

            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from rclpy.executors import ExternalShutdownException
            from rclpy.node import Node
            from rclpy.signals import SignalHandlerOptions
            from rclpy.time import Time
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import BatteryState, LaserScan
            from std_msgs.msg import Empty, UInt16
            from tf2_ros import Buffer, TransformException, TransformListener
        except ImportError as exc:
            raise RuntimeError(
                "Nav2Binding.start() requires ROS 2 (rclpy) and the Nav2 stack. "
                "Use `--binding simulated` for a $0 run, or launch inside a sourced "
                "ROS 2 workspace."
            ) from exc

        if not rclpy.ok():
            # Without rclpy's own SIGINT and SIGTERM handlers: they shut the context down before
            # the adapter's handler runs, and its shutdown could then neither hold the base nor
            # cancel the goal ("publisher's context is invalid"). The adapter owns the signals,
            # and `stop()` shuts rclpy down, after the robot has been stopped.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = Node(f"swarmada_nav2_{self._robot_id}".replace("-", "_"))
        # Route each subscription to the existing (unit-tested) parse callbacks.
        node.create_subscription(Odometry, "odom", lambda m: self.on_odom(m), 10)
        if self._battery_mv_topic is not None:
            node.create_subscription(UInt16, self._battery_mv_topic,
                                     lambda m: self.on_battery_mv(m), 10)
        else:
            node.create_subscription(BatteryState, "battery_state",
                                     lambda m: self.on_battery(m), 10)
        self._battery_since = self._now()
        bat_topic = self._battery_mv_topic or "battery_state"

        def _count_battery_publishers() -> None:
            n = node.count_publishers(bat_topic)
            with self._lock:
                self._battery_publishers = n
        node.create_timer(_BATTERY_GRAPH_PERIOD_S, _count_battery_publishers)
        node.create_subscription(DiagnosticArray, "diagnostics", lambda m: self.on_diagnostics(m), 10)

        self._cmd_vel = node.create_publisher(Twist, "cmd_vel", 10)
        self._action_client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self._nav_action = NavigateToPose
        self._twist_type = Twist
        if self._rest_scan_topic is not None:
            # The handover's rest channel; sensor-data QoS takes a reliable or a best-effort
            # publisher alike.
            node.create_subscription(LaserScan, self._rest_scan_topic,
                                     lambda m: self.on_scan(m), qos_profile_sensor_data)
        if self._ready_topic is not None:
            # The operator's ready: a press is the arrival of an Empty; nothing in it is read.
            operator_ready.RosReadySource(node, self._ready_topic, Empty).attach(self.on_ready)
        if self._teleop_topic is not None:
            # Subscribed always, forwarded only through the gate (on_teleop).
            node.create_subscription(Twist, self._teleop_topic, lambda m: self.on_teleop(m), 10)
        self._node = node
        if self._watchdog is not None:
            node.create_timer(_WATCHDOG_PERIOD_S, self._on_watchdog_timer)

        # map -> odom, polled from TF into on_map_to_odom. A lookup that finds no map is
        # passed on as None, which pose() declares rather than hides.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        def _on_tf_timer() -> None:
            try:
                tf = self._tf_buffer.lookup_transform(MAP_FRAME, self.odom_frame(self._robot_id),
                                                      Time())
            except TransformException:
                tf = None
            self.on_map_to_odom(tf)

        node.create_timer(_TF_PERIOD_S, _on_tf_timer)

        def _spin() -> None:
            # Every exit from rclpy.spin() lands in _record_spin_exit, including a
            # normal return: whatever the reason, this node stops processing callbacks
            # at that instant and the binding goes blind. BaseException, not Exception,
            # because the thread is just as dead when it ends on one.
            try:
                rclpy.spin(node)
            except ExternalShutdownException:
                return                   # stop() shut the context down: the clean path
            except BaseException as exc:  # noqa: BLE001 - a boundary, not a handler
                print(f"rclpy spin thread exited: {exc!r}", file=sys.stderr)
                self._record_spin_exit(exc)
                return
            self._record_spin_exit(None)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()
        # Every caller -- the adapter's main(), r1/r2 smoke, the latency script -- creates a
        # binding and never tears it down. Registering here fixes all of them at once with
        # no Protocol change; stop() is idempotent so an explicit call remains fine.
        atexit.register(self.stop)

    def stop(self) -> None:  # pragma: no cover — requires the ROS 2 runtime
        """Shut the executor down and join the spin thread BEFORE the interpreter tears
        the C++ side out from under it.

        Without this, ``rclpy.spin`` runs on a daemon thread that the interpreter kills at
        exit while it is inside a C++ frame. The C++ runtime then unwinds a thread with no
        active exception, calls ``std::terminate``, and the process dies on SIGABRT --
        **exit 134 on a run whose work completed successfully**. Every smoke test, the
        latency script and both conformance containers ended that way.

        Idempotent, and safe to call when :meth:`start` was never called.

        This and :meth:`start` now IMPLEMENT the ``RobotBinding`` lifecycle rather than
        sitting beside it. That open question is answered: the Protocol
        declares both, so the adapter calls them on every binding without asking what kind
        it was handed. This docstring previously recorded the question as open, which is
        how the adapter came to construct this binding and never start it.
        """
        with self._settled:
            # Set BEFORE rclpy.shutdown() so the executor thread, which may end at any
            # moment after it, does not record its exit as a fault. Notified so the sender
            # thread, if it is waiting for a goal to park, ends too.
            self._stopping = True
            self._settled.notify_all()
        node, thread = self._node, self._spin_thread
        self._node, self._spin_thread = None, None
        if node is None:
            return
        import rclpy
        try:
            if rclpy.ok():
                rclpy.shutdown()   # makes rclpy.spin() return in the spin thread
        except Exception:          # noqa: BLE001 - teardown must not raise
            pass
        if thread is not None:
            thread.join(timeout=5.0)
        try:
            node.destroy_node()
        except Exception:          # noqa: BLE001 - teardown must not raise
            pass
