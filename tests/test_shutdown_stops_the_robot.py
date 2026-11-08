# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter stops the robot when it stops.

`main()` used to end with `robot.stop()` alone: rclpy torn down, and nothing else. The base was
not held, the navigation goal the binding sent was not cancelled, and an explore-floor segment's
explorer and the processes it launched (SLAM, the keep-out servers, the explorer) went on
running with nobody watching the battery or the bound. SIGTERM, which is how a container is
stopped, did not even reach the `finally`: the process died on the signal's default action.

Now every exit from `main()` -- the control stream closing, an exception out of its loop,
SIGTERM, SIGINT -- goes through the adapter's shutdown, which, in this order and each step on
its own so one that raises cannot skip the rest: holds the base and cancels the goal (a zero
Twist), stops the explorer, ends the running job, looks for rest, and ends the processes the
segment launched. rclpy is torn down after all of that, since every step needs it.

A dropped connection is not an exit: the adapter reconnects (RFC-0001, fleet-adapter protocol)
and the lease it holds is what stops the robot if the control plane does not come back (C4.2).
That bound is pinned here too.

Every test drives the shipped `main()`, with the control plane's stub and the channel replaced.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import threading
import time
from types import SimpleNamespace as NS

import grpc
import pytest
from conftest import mark_transport_live
from safety_call import SafetyCall
from test_explore_adapter import EXPLORE, StepClock
from test_ros2_explore import Clock, FakeRos, FakeShell, _site
from test_stop_never_waits import (  # noqa: F401 - ros_msgs is a fixture
    _Handle,
    _odom,
    _Publisher,
    _Twist,
    ros_msgs,
)
from test_unanswered_goal import _IdServer

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2 import explore, nav2_binding, ros2_explore
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
EXITS = ["stream-closes", "loop-raises", "sigterm", "sigint"]


class _RpcError(grpc.RpcError):
    def __init__(self, code) -> None:
        super().__init__(code.name)
        self._code = code

    def code(self):
        return self._code


class _Channel:
    def __init__(self) -> None:
        self.closed = threading.Event()

    def close(self) -> None:
        self.closed.set()


class _Stub:
    """The control plane's end of both streams, over a channel `main()` can close. A control
    stream answers the adapter's hello and registration as the control plane does, and reads
    from `q`: a message is delivered, None ends the stream cleanly, an exception is raised out
    of it. Closing the channel ends it as gRPC does, with CANCELLED. `opened` is set once the
    robot is registered, which the adapter does only once its SafetyStream is up."""

    def __init__(self, channel: _Channel) -> None:
        self.channel = channel
        self.q: queue.Queue = queue.Queue()
        self.opened = threading.Event()
        self.sessions = 0

    def ControlStream(self, requests):
        self.sessions += 1

        def read():
            with contextlib.suppress(Exception):  # the stream ended
                for m in requests:
                    kind = m.WhichOneof("payload")
                    if kind == "hello":
                        self.q.put(pb.ControlPlaneMessage(hello_ack=pb.HelloAck(accepted=True)))
                    elif kind == "register":
                        self.q.put(pb.ControlPlaneMessage(
                            register_ack=pb.RegisterAck(accepted=True)))
                        self.opened.set()
        threading.Thread(target=read, daemon=True).start()

        def stream():
            while True:
                if self.channel.closed.is_set():
                    raise _RpcError(grpc.StatusCode.CANCELLED)
                try:
                    m = self.q.get(timeout=0.02)
                except queue.Empty:
                    continue
                if m is None:
                    return
                if isinstance(m, BaseException):
                    raise m
                yield m
        return stream()

    def SafetyStream(self, requests):
        def stream():
            threading.Event().wait()
            yield from ()
        return SafetyCall(stream())

    def send(self, command_id: int, **command) -> None:
        self.q.put(pb.ControlPlaneMessage(command=pb.Command(
            command_id=command_id, robot_id=ROBOT, **command)))


class _ProcessDefault(BaseException):
    """What a signal the adapter installs no handler for does. The real default ends the
    process with no `finally` at all; raising is kinder than that, and still not a stop."""


@pytest.fixture
def signals():
    """SIGTERM and SIGINT as a process that installed nothing would have them, restored after."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def default(signum, frame):
        raise _ProcessDefault(signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, default)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def _patch_transport(monkeypatch) -> _Stub:
    channel = _Channel()
    stub = _Stub(channel)
    monkeypatch.setattr(adapter_mod, "open_control_channel", lambda args: channel)
    monkeypatch.setattr(pb_grpc, "FleetAdapterServiceStub", lambda ch: stub)
    return stub


def _main(monkeypatch, *argv: str):
    """Run the shipped main() on this thread; (exit code, what it raised)."""
    monkeypatch.setattr("sys.argv", ["adapter", "--robot-id", ROBOT, *argv])
    try:
        adapter_mod.main()
    except SystemExit as exc:
        return exc.code, None
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        return None, exc
    return None, None


def _until(cond, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.005)


def _exit_by(how: str, stub: _Stub) -> None:
    if how == "stream-closes":
        stub.q.put(None)
    elif how == "loop-raises":
        stub.q.put(RuntimeError("a bug in the control-stream loop"))
    elif how == "sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
    elif how == "sigint":
        os.kill(os.getpid(), signal.SIGINT)


def _expected_exit(how: str, code, raised) -> None:
    if how == "stream-closes":
        assert code is None and raised is None, (code, raised)
    elif how == "loop-raises":
        assert isinstance(raised, RuntimeError), (code, raised)
    else:
        assert not isinstance(raised, _ProcessDefault), (
            f"{how.upper()} reached the process default: the adapter installs no handler, so "
            f"the process dies on it without stopping the robot")
        signum = signal.SIGTERM if how == "sigterm" else signal.SIGINT
        assert code == 128 + signum and raised is None, (code, raised)


# ── the simulated binding: every exit path ───────────────────────────────────────────

class _Stack(SimulatedExploreStack):
    def __init__(self, log: list, **kw) -> None:
        super().__init__(**kw)
        self._log = log

    def stop_explorer(self) -> None:
        self._log.append("stop_explorer")
        super().stop_explorer()

    def end_segment(self) -> None:
        self._log.append("end_segment")
        super().end_segment()


class _Robot(SimulatedRobot):
    def __init__(self, log: list, **kw) -> None:
        super().__init__(**kw)
        self._log = log

    def command_hold(self, robot_id: str) -> None:
        self._log.append("hold")
        super().command_hold(robot_id)

    def command_stop(self, robot_id: str) -> None:
        self._log.append("command_stop")
        super().command_stop(robot_id)

    def stop(self) -> None:
        self._log.append("teardown")
        super().stop()


def _simulated(monkeypatch, log: list):
    clock = StepClock()
    stack = _Stack(log, clock=clock, explore_seconds=1e9, drain_mv_per_s=0.0)
    robot = _Robot(log, explore=stack)
    monkeypatch.setattr(adapter_mod, "SimulatedRobot", lambda engine=None: robot)
    monkeypatch.setattr(explore, "RealClock", lambda: clock)
    return robot, stack


def _run(monkeypatch, drive):
    """main() on this thread; `drive(stub)` on another, as the control plane and the operator."""
    stub = _patch_transport(monkeypatch)
    failures: list = []

    def driver():
        try:
            assert stub.opened.wait(5.0), "main() never registered the robot"
            drive(stub)
        except BaseException as exc:  # noqa: BLE001 - reported on the test's thread
            failures.append(exc)
            stub.q.put(None)
    t = threading.Thread(target=driver, daemon=True)
    t.start()
    code, raised = _main(monkeypatch, "--binding", "simulated")
    t.join(5.0)
    assert not failures, failures
    return code, raised


def _first(log: list, *names: str) -> int:
    hits = [i for i, n in enumerate(log) if n in names]
    assert hits, f"none of {names} happened: {log}"
    return hits[0]


@pytest.mark.parametrize("how", EXITS)
def test_every_exit_holds_the_base_before_rclpy_is_torn_down(monkeypatch, signals, how) -> None:
    log: list = []
    robot, _ = _simulated(monkeypatch, log)

    def drive(stub):
        stub.send(1, assign_action=pb.AssignAction(
            action_id="nav-1", assignment_id="a-nav-1", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=1, lease_generation=1,
            lease_duration_ms=60_000))
        _until(lambda: robot._moving.get(ROBOT), "the goal to be under way")
        _exit_by(how, stub)

    code, raised = _run(monkeypatch, drive)
    assert not robot._moving[ROBOT], f"{how}: the adapter exited and the robot is still moving"
    assert _first(log, "hold", "command_stop") < log.index("teardown"), log
    _expected_exit(how, code, raised)


@pytest.mark.parametrize("how", EXITS)
def test_every_exit_stops_the_explorer_and_ends_the_segment_in_order(
        monkeypatch, signals, how) -> None:
    log: list = []
    _, stack = _simulated(monkeypatch, log)

    def drive(stub):
        stub.send(1, assign_action=pb.AssignAction(
            action_id="explore-first", assignment_id="a-explore-first", action_type="Custom",
            payload_json=EXPLORE, fencing_token=1, lease_generation=1, lease_duration_ms=60_000))
        _until(stack.moving, "the segment to be exploring")
        _exit_by(how, stub)

    code, raised = _run(monkeypatch, drive)
    assert not stack.moving(), f"{how}: the adapter exited and the explorer is still driving"
    held = _first(log, "hold", "command_stop")
    assert held < _first(log, "stop_explorer") < _first(log, "end_segment") \
        < log.index("teardown"), log
    _expected_exit(how, code, raised)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_step_that_raises_does_not_skip_the_steps_after_it(monkeypatch, signals, capsys) -> None:
    log: list = []
    robot, stack = _simulated(monkeypatch, log)

    def broken(name):
        def step(*_a):
            log.append(name)
            raise RuntimeError(f"{name} broke")
        return step
    robot.command_hold = broken("hold")
    robot.command_stop = broken("command_stop")
    stack.stop_explorer = broken("stop_explorer")
    stack.end_segment = broken("end_segment")

    def drive(stub):
        stub.send(1, assign_action=pb.AssignAction(
            action_id="explore-first", assignment_id="a-explore-first", action_type="Custom",
            payload_json=EXPLORE, fencing_token=1, lease_generation=1, lease_duration_ms=60_000))
        _until(stack.moving, "the segment to be exploring")
        stub.q.put(None)

    code, raised = _run(monkeypatch, drive)
    assert code is None and raised is None, (code, raised)
    for step in ("stop_explorer", "end_segment", "teardown"):
        assert step in log, f"a failing earlier step skipped {step}: {log}"
    err = capsys.readouterr().err
    for name in ("hold", "stop_explorer", "end_segment"):
        assert f"{name} broke" in err, f"the failure of {name} was not reported:\n{err}"


def test_a_dropped_connection_reconnects_and_the_lease_stops_the_robot(
        monkeypatch, signals) -> None:
    """Not an exit: the adapter redials, and the lease it holds stops the robot within the
    lease's duration plus one lease tick, as C4.2 says, whether or not the control plane
    comes back."""
    log: list = []
    robot, _ = _simulated(monkeypatch, log)
    lease_s = 0.3
    stopped_after: list = []

    def drive(stub):
        stub.send(1, assign_action=pb.AssignAction(
            action_id="nav-1", assignment_id="a-nav-1", action_type="Navigate",
            payload_json=b'{"x": 3, "y": 0}', fencing_token=1, lease_generation=1,
            lease_duration_ms=int(lease_s * 1000)))
        _until(lambda: robot._moving.get(ROBOT), "the goal to be under way")
        t0 = time.monotonic()
        stub.q.put(_RpcError(grpc.StatusCode.UNAVAILABLE))
        _until(lambda: not robot._moving[ROBOT], "the lease to stop the robot", timeout=5.0)
        stopped_after.append(time.monotonic() - t0)
        _until(lambda: stub.sessions >= 2, "the adapter to reconnect")
        assert "teardown" not in log, "a dropped connection ended the adapter"
        stub.q.put(None)

    code, raised = _run(monkeypatch, drive)
    assert code is None and raised is None, (code, raised)
    assert stopped_after and stopped_after[0] <= lease_s + adapter_mod._LEASE_TICK_S + 0.25


# ── the ROS 2 binding: a goal in flight and a segment running ─────────────────────────

def test_on_the_nav2_binding_shutdown_holds_cancels_stops_the_explorer_and_ends_the_segment(
        monkeypatch, signals, ros_msgs) -> None:  # noqa: F811
    """The done-when: with the return goal in flight and a segment running on the ROS 2 stack,
    the shutdown holds the base, cancels the goal, publishes the zero Twist, tells the explorer
    to stop and kills what the segment launched -- in that order -- before rclpy is shut down."""
    log: list = []
    server = _IdServer()
    b = mark_transport_live(_nav2_binding(log))
    b._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    b._action_client = server
    b._nav_action = NS(Goal=lambda: NS(pose=None))
    b._cmd_vel, b._twist_type = _LogPublisher(log), _Twist
    b._cancel_goal_by_id = lambda goal_id: None
    b.start = lambda: None
    b.stop = lambda: log.append("teardown")
    b.on_odom(_odom(0.0))
    ros, shell = _LogRos(log), _LogShell(log)
    b._explore = ros2_explore.Ros2ExploreStack(
        binding=b, robot_id=ROBOT, site=_site(), shell=shell, ros=ros, clock=Clock(ros),
        limits=explore.BatteryLimits(hard_mv=7750, full_mv=8300, return_mv=7835))
    monkeypatch.setattr(nav2_binding, "Nav2Binding", lambda **kw: b)
    parked = threading.Event()

    def segment(job):  # the segment's body, parked on its way back to the mark
        job.stack.go_to_mark(1.0, 2.0, 0.0)
        parked.set()
        job.halted.wait()
        raise explore.Halted
    monkeypatch.setattr(explore.ExploreJob, "_run", segment)
    handle = _LogHandle(log)
    stub = _patch_transport(monkeypatch)

    def drive():
        assert stub.opened.wait(5.0), "the robot was never registered"
        stub.send(1, assign_action=pb.AssignAction(
            action_id="explore-first", assignment_id="a-explore-first", action_type="Custom",
            payload_json=EXPLORE, fencing_token=1, lease_generation=1, lease_duration_ms=60_000))
        assert parked.wait(5.0), "the segment never started"
        _until(lambda: server.futures, "the return goal to reach the server")
        server.futures[0].cb(NS(result=lambda: handle))
        _until(lambda: b._goal_handle is handle, "the return goal to be adopted")
        os.kill(os.getpid(), signal.SIGTERM)
    t = threading.Thread(target=drive, daemon=True)
    t.start()
    code, raised = _main(monkeypatch, "--binding", "ros2", "--stall-window", "5",
                         "--stall-min-progress", "0.05", "--goal-response-bound", "1",
                         "--odometry-source", "measured")
    t.join(5.0)
    assert not isinstance(raised, _ProcessDefault), "SIGTERM reached the process default"
    assert "teardown" in log, log
    assert b._holding, "the base was never held"
    assert handle.cancels == 1, "the return goal was not cancelled"
    order = [_first(log, "hold"), _first(log, "cancel"), _first(log, "zero_twist"),
             _first(log, "explore/resume=False"), _first(log, "kill"), log.index("teardown")]
    assert order == sorted(order), log
    kill = next(s for s, *_ in shell.runs if "kill -INT" in s)
    for name in ("lib/explore_lite/explore", "async_slam_toolbox_node", "keepout_servers"):
        assert name in kill, f"the segment's {name} was not ended"
    assert code == 128 + signal.SIGTERM, (code, raised)


def _nav2_binding(log: list):
    b = nav2_binding.Nav2Binding(ROBOT, response_bound_s=1.0, odometry_source="measured")
    halt = b._halt_locked

    def logged():
        log.append("hold")
        return halt()
    b._halt_locked = logged
    return b


class _LogPublisher(_Publisher):
    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def publish(self, msg) -> None:
        if msg.linear.x == 0.0 and msg.angular.z == 0.0:
            self._log.append("zero_twist")
        super().publish(msg)


class _LogHandle(_Handle):
    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def cancel_goal_async(self) -> None:
        self._log.append("cancel")
        super().cancel_goal_async()


class _LogRos(FakeRos):
    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def publish(self, topic, msg_type, fields):
        self._log.append(f"{topic}={fields.get('data')}")
        super().publish(topic, msg_type, fields)


class _LogShell(FakeShell):
    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def run(self, script, **kw):
        if "kill -INT" in script:
            self._log.append("kill")
        return super().run(script, **kw)



# ── rclpy must not end its context on the signal the adapter is stopping on ────────────
# rclpy's init installs SIGINT and SIGTERM handlers that shut every context down before the
# Python handler runs. On the stand-in the adapter's shutdown then failed at its first step:
# "RCLError: Failed to publish: publisher's context is invalid" -- no hold, no cancel, no zero
# Twist. The adapter owns the signals; the binding tears rclpy down itself, last.

def _fake_ros(monkeypatch) -> dict:
    """Every module `Nav2Binding.start()` imports, as mocks; returns what `rclpy.init` got."""
    import sys
    import types
    from unittest import mock

    got: dict = {}
    rclpy = mock.MagicMock(name="rclpy")
    rclpy.ok.return_value = False
    rclpy.init.side_effect = lambda *a, **kw: got.update(kw, called=True)
    rclpy.spin.side_effect = lambda node: threading.Event().wait()
    signals = types.ModuleType("rclpy.signals")
    signals.SignalHandlerOptions = NS(NO="NO", ALL="ALL", SIGINT="SIGINT", SIGTERM="SIGTERM")
    rclpy.signals = signals
    mods = {"rclpy": rclpy, "rclpy.signals": signals}
    for name in ("rclpy.action", "rclpy.executors", "rclpy.node", "rclpy.qos", "rclpy.time",
                 "diagnostic_msgs", "diagnostic_msgs.msg", "geometry_msgs", "geometry_msgs.msg",
                 "nav_msgs", "nav_msgs.msg", "nav2_msgs", "nav2_msgs.action", "sensor_msgs",
                 "sensor_msgs.msg", "std_msgs", "std_msgs.msg", "tf2_ros"):
        mods[name] = mock.MagicMock(name=name)
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    import atexit
    monkeypatch.setattr(atexit, "register", lambda *a, **k: None)
    return got


def test_the_binding_starts_rclpy_without_its_signal_handlers(monkeypatch) -> None:
    got = _fake_ros(monkeypatch)
    b = nav2_binding.Nav2Binding(ROBOT, odometry_source="measured")
    b.start()
    assert got.get("called"), "rclpy was never initialised"
    assert got.get("signal_handler_options") == "NO", (
        f"rclpy.init({got}) installs rclpy's own SIGINT/SIGTERM handlers, which invalidate the "
        f"context before the adapter's shutdown can hold the base")


# ── the safety stream ends quietly when the adapter is the one closing it ─────────────────
# Closing the channel on a signal ends the safety stream with CANCELLED. Its thread used to die
# with a traceback on every shutdown, and a traceback in the adapter's log is a no-go on the
# operator's sheet.

def test_the_safety_stream_ends_without_a_traceback_when_the_adapter_shuts_down(capsys) -> None:
    class _Closing:
        def SafetyStream(self, requests):
            def stream():
                raise _RpcError(grpc.StatusCode.CANCELLED)
                yield
            return SafetyCall(stream(), "refused")
    a = adapter_mod.fleet_adapter_ros2Adapter(pb, None, SimulatedRobot(), ROBOT)
    a._shutdown.set()
    a.run_safety_stream(_Closing())            # returns: nothing raised out of the thread's target
    assert "Traceback" not in capsys.readouterr().err
