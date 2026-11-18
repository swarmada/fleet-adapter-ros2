# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A fault on the safety path never ends a safety thread in silence.

The stop paths turn the transport's RuntimeError into a FAILED that says why. Anything else -- a
bug in the robot binding or in the adapter -- is not swallowed where it happens
(`test_the_wrapper_does_not_swallow_a_bug_in_the_safety_primitive`), and it used to end the thread
it reached:

* on the SafetyStream, the thread ended while the adapter still counted the stream as up: the
  estop got no answer, assignments went on being accepted, and no later estop was read;
* on the EdgeStream, the thread ended and no later edge estop was read;
* the SafetyStream-lost report's thread ended, and the lost action's FAILED was never sent;
* the simulated estop drill ended telemetry.

Now each of those boundaries answers what it owes -- the estop FAILED, the action FAILED -- logs the
fault at error level with its trace and counts it, and the SafetyStream is treated as lost: motion
refused and the robot held, then redialled. After a fault the redial waits back off (doubling from
_REDIAL_S to a cap, reset once a stream stays up long enough); an ordinary disconnect keeps the
fixed _REDIAL_S.
"""

from __future__ import annotations

import queue
import threading
import time
from itertools import pairwise
from types import SimpleNamespace as NS

import pytest
from safety_call import SafetyCall

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

ROBOT = "amr-1"


class _BrokenCount(SimulatedRobot):
    """A binding with a bug: `state_seq` raises something other than the transport's
    RuntimeError, so every confirmation of a stop raises it too."""

    def state_seq(self, robot_id: str) -> int:
        raise ValueError("binding bug: state_seq")


def _adapter() -> Adapter:
    robot = _BrokenCount()
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._safety_required = True              # as start_safety_stream sets it
    return a


class _EstopEveryStream:
    """A control plane whose every SafetyStream sends one estop and then stays open. It records
    each dial, and reads what the adapter sends back on each stream. `healthy_s` maps a dial
    number to how long that stream stays up before its estop.

    Its reader behaves as gRPC does: a message it pulls after the call was cancelled is dropped,
    not sent, and sending one takes a moment (SEND_S). So an answer queued and then cancelled
    straight away never arrives here, as it would not on the wire."""

    SEND_S = 0.02

    def __init__(self, healthy_s: dict | None = None) -> None:
        self.dials: list[float] = []
        self.acks: list = []
        self._hold = threading.Event()
        self.healthy_s = healthy_s or {}

    def SafetyStream(self, requests):  # the gRPC method name
        n = len(self.dials) + 1
        self.dials.append(time.monotonic())

        def stream():
            if n in self.healthy_s:
                time.sleep(self.healthy_s[n])
            yield pb.ControlPlaneSafetyMessage(
                robot_id=ROBOT, estop=pb.Estop(estop_id=f"e-{n}", reason="test"))
            self._hold.wait()
        call = SafetyCall(stream())

        def read() -> None:
            for m in requests:                      # ends when the adapter closes the stream
                time.sleep(self.SEND_S)
                if call.done():                     # cancelled: gRPC drops what it pulled
                    break
                self.acks.append((n, m.estop_ack))
        threading.Thread(target=read, daemon=True).start()
        return call

    def release(self) -> None:
        self._hold.set()


def _run_safety_stream(a: Adapter, cp: _EstopEveryStream, for_s: float) -> None:
    t = threading.Thread(target=a.run_safety_stream, args=(cp,), daemon=True)
    t.start()
    time.sleep(for_s)
    assert t.is_alive(), "the SafetyStream thread ended while the adapter runs"
    a._shutdown.set()
    cp.release()
    t.join(timeout=5.0)


def test_a_fault_in_an_estop_answers_it_failed_and_takes_the_safety_stream_down(capsys):
    a = _adapter()
    cp = _EstopEveryStream()
    downs: list = []
    down = a._safety_down

    def record_down(why, open_for) -> None:
        down(why, open_for)
        downs.append((why, a._safety_up.is_set(), a._no_estop_path()))
    a._safety_down = record_down
    _run_safety_stream(a, cp, for_s=1.2)
    first = [ack for n, ack in cp.acks if n == 1]
    assert first and first[-1].state == pb.ESTOP_STATE_FAILED, "the estop got no FAILED answer"
    assert "could not be confirmed" in first[-1].message
    assert "ValueError" in first[-1].message
    assert downs, "the faulted stream was not treated as lost"
    why, up, refusal = downs[0]
    assert "fault on the safety path" in why
    assert up is False and "SafetyStream is down" in refusal, "motion was not refused"
    assert len(cp.dials) >= 2, "the SafetyStream was not dialled again"
    assert a._safety_faults >= 1
    assert "SAFETY_PATH_FAULT count=1 where=estop e-1" in capsys.readouterr().err


def _gaps(dials: list) -> list:
    return [b - a for a, b in pairwise(dials)]


def test_a_binding_that_fails_every_estop_backs_off_to_the_cap(monkeypatch):
    """Every stream's estop faults and takes the stream down. The redial waits double from
    _REDIAL_S and stop at the cap, and every fault is counted, so a binding that fails every
    estop shows as a rising count and a slow redial, not a flapping stream. Scaled down: 0.05 s
    doubling to 0.4 s (in the adapter, 0.5 s to 5 s)."""
    monkeypatch.setattr(adapter_mod, "_REDIAL_S", 0.05)
    monkeypatch.setattr(adapter_mod, "_SAFETY_FAULT_WAIT_MAX_S", 0.4)
    a = _adapter()
    cp = _EstopEveryStream()
    _run_safety_stream(a, cp, for_s=2.4)
    gaps = _gaps(cp.dials)
    assert len(gaps) >= 6, f"only {len(gaps) + 1} dials"
    expected = [0.05, 0.1, 0.2, 0.4, 0.4]           # doubling, then held at the cap
    for got, want in zip(gaps, expected):
        assert want * 0.9 <= got <= want + 0.15, f"gaps {[round(g, 3) for g in gaps]}"
    assert max(gaps) <= 0.4 + 0.15, "the wait went past the cap"
    assert a._safety_faults >= len(cp.dials) - 1


def test_the_fault_backoff_resets_once_a_stream_stays_up(monkeypatch):
    """Three streams fault at once (waits 0.05, 0.1, 0.2 s); the fourth stays up past the reset
    time before its estop faults, so the next wait is back to _REDIAL_S."""
    monkeypatch.setattr(adapter_mod, "_REDIAL_S", 0.05)
    monkeypatch.setattr(adapter_mod, "_SAFETY_FAULT_WAIT_MAX_S", 0.4)
    monkeypatch.setattr(adapter_mod, "_SAFETY_FAULT_RESET_S", 0.3)
    a = _adapter()
    cp = _EstopEveryStream(healthy_s={4: 0.35})
    _run_safety_stream(a, cp, for_s=1.6)
    gaps = _gaps(cp.dials)
    shown = [round(g, 3) for g in gaps]
    assert len(gaps) >= 4, f"only {len(gaps) + 1} dials"
    assert gaps[2] >= 0.2 * 0.9, f"gaps {shown}"            # backed off to 0.2 s
    # The fourth stream was up 0.35 s, then faulted: the wait after it is _REDIAL_S again
    # (0.35 + 0.05 s), not the next step of the back-off (0.35 + 0.4 s).
    assert gaps[3] < 0.35 + 0.15, f"gaps {shown}"


def test_an_ordinary_disconnect_keeps_the_fixed_redial_wait(monkeypatch):
    """A stream that simply ends, with no fault, is redialled after _REDIAL_S every time: the
    back-off is for faults only."""
    monkeypatch.setattr(adapter_mod, "_REDIAL_S", 0.05)
    monkeypatch.setattr(adapter_mod, "_SAFETY_FAULT_WAIT_MAX_S", 0.4)
    a = _adapter()
    dials: list = []

    class _EndsAtOnce:
        def SafetyStream(self, requests):  # the gRPC method name
            dials.append(time.monotonic())
            return SafetyCall(iter(()))

    t = threading.Thread(target=a.run_safety_stream, args=(_EndsAtOnce(),), daemon=True)
    t.start()
    time.sleep(0.8)
    a._shutdown.set()
    t.join(timeout=5.0)
    gaps = _gaps(dials)
    # Fixed at 0.05 s (plus the time to dial and take the stream down); a back-off would reach
    # 0.2 s by the third gap.
    assert len(gaps) >= 5 and max(gaps) < 0.15, f"gaps {[round(g, 3) for g in gaps]}"
    assert a._safety_faults == 0


class _FailsAfterStopping(SimulatedRobot):
    """A binding whose first look at rest says moving, so the estop's STOPPING receipt goes out,
    and whose next look raises: a fault after the receipt."""

    def __init__(self) -> None:
        super().__init__()
        self.looks = 0

    def is_stopped(self, robot_id: str) -> bool:
        self.looks += 1
        if self.looks == 1:
            return False
        raise ValueError("binding bug: is_stopped")


def test_a_fault_after_the_stopping_receipt_still_reaches_the_control_plane_as_failed():
    """C5.6: a STOPPING is followed by a terminal answer. The FAILED is let out before the
    faulted stream is cancelled, so it reaches the control plane."""
    robot = _FailsAfterStopping()
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._safety_required = True
    cp = _EstopEveryStream()
    _run_safety_stream(a, cp, for_s=0.4)
    states = [ack.state for n, ack in cp.acks if n == 1]
    assert states == [pb.ESTOP_STATE_STOPPING, pb.ESTOP_STATE_FAILED], states


def test_motion_is_refused_for_the_whole_wait_after_a_fault():
    a = _adapter()
    cp = _EstopEveryStream()
    t = threading.Thread(target=a.run_safety_stream, args=(cp,), daemon=True)
    t.start()
    end = time.monotonic() + 3.0
    while a._safety_faults == 0 and time.monotonic() < end:
        time.sleep(0.005)
    assert a._safety_faults >= 1
    time.sleep(0.15)                                # inside the first 0.5 s wait
    try:
        assert a._safety_up.is_set() is False
        assert "SafetyStream is down" in a._no_estop_path()
        assert len(cp.dials) == 1
    finally:
        a._shutdown.set()
        cp.release()
        t.join(timeout=5.0)


class _StopRaises(SimulatedRobot):
    """A binding whose stop raises `fail` times, then goes out (None: always raises)."""

    def __init__(self, fail: int | None = None) -> None:
        super().__init__()
        self.fail, self.stops = fail, 0
        self.senders: list = []

    def command_stop(self, robot_id: str) -> None:
        self.stops += 1
        self.senders.append(threading.get_ident())
        if self.fail is None or self.stops <= self.fail:
            raise ValueError("binding bug: command_stop")
        super().command_stop(robot_id)


def _lose_the_stream_with_a_raising_stop(robot: _StopRaises, lease: bool = True,
                                         cancelling: bool = False):
    """An action running (under a lease, or without one; or being cancelled), a job running,
    and the SafetyStream lost."""
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT)
    a._safety_required = True
    a._current_action, a._current_fencing_token = "t-1", 1
    if cancelling:
        a._halted_action = "t-1"   # a cancel stopped it and is still confirming rest
    if lease:
        a._lease.grant(ROBOT, 60.0, 1)
    job = NS(halted=threading.Event(), action_id="t-1")
    a._job = job
    a._safety_up.set()
    a._safety_down("test: the stream ended", 10.0)  # must not raise
    return a, job


def _action_ends(a: Adapter, wait_s: float) -> list:
    """Every action_status the adapter sends within `wait_s`."""
    out, end = [], time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            m = a._outbox.get(timeout=0.02)
        except queue.Empty:
            continue
        if m.WhichOneof("payload") == "action_status":
            out.append(m.action_status)
    return out


def _fast_retries(monkeypatch) -> None:
    """The stop budget 40 x 5 ms (0.2 s), and the later retries every 20 ms."""
    monkeypatch.setattr(adapter_mod, "_ESTOP_TICK_S", 0.005)
    monkeypatch.setattr(adapter_mod, "_SAFETY_STOP_RETRY_S", 0.02)


_RETRY_THREADS = ("safety-lost-report", "estop-stop-retry")


@pytest.fixture(autouse=True)
def _retries_end_with_their_test(monkeypatch):
    """A test's stop retry ends before the next test starts. A retry ended by its adapter's
    shutdown logs SAFETY_STOP_RETRY_ENDED as it goes, and a late one landed in the next test's
    captured stderr, failing a test that asserts no retry of its own ended. This waits up to 2 s
    for the retry threads the test started. It depends on `monkeypatch`, so it is torn down
    first, while the test's fast retry interval still holds."""
    before = set(threading.enumerate())
    yield
    end = time.monotonic() + 2.0
    while time.monotonic() < end and any(
            t.is_alive() and t.name in _RETRY_THREADS and t not in before
            for t in threading.enumerate()):
        time.sleep(0.01)


def test_a_stop_that_raises_once_is_tried_again_and_reported_stopped(monkeypatch):
    _fast_retries(monkeypatch)
    robot = _StopRaises(fail=1)
    a, job = _lose_the_stream_with_a_raising_stop(robot)
    assert a._safety_up.is_set() is False and job.halted.is_set()
    assert a._current_action == ""
    assert ROBOT not in a._lease._deadline, "the lease is released at once"
    ends = _action_ends(a, 1.0)
    assert len(ends) == 1 and ends[0].state == pb.ACTION_STATE_FAILED
    assert ends[0].message.startswith("stopped after 2 attempts (the first: "), ends[0].message
    assert ends[0].message.endswith("no estop could reach the robot"), ends[0].message  # at rest
    assert robot.stops == 2


def test_a_stop_that_keeps_raising_is_reported_at_the_budget_and_tried_until_it_goes_out(
        monkeypatch, capsys):
    """No lease: nothing but the adapter's own retry will stop this robot. At the end of the stop
    budget the action is reported FAILED saying the robot could not be stopped and that the stop
    is still being commanded -- never "stopped". The stop goes on being tried; when it finally
    goes out that is logged, and the action is not reported a second time."""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, job = _lose_the_stream_with_a_raising_stop(robot, lease=False)
    assert job.halted.is_set(), "the running job was not ended"
    ends = _action_ends(a, 0.6)
    assert len(ends) == 1 and ends[0].state == pb.ACTION_STATE_FAILED
    assert "the robot could not be stopped within 0.2 s" in ends[0].message, ends[0].message
    assert "keeps commanding the stop every 0.02 s" in ends[0].message
    assert not ends[0].message.startswith("stopped")
    tried = robot.stops
    time.sleep(0.2)
    assert robot.stops > tried, "the stop is no longer tried after the budget"
    robot.fail = robot.stops + 2                       # it goes out on the third from now
    assert _action_ends(a, 0.5) == [], "the action was reported a second time"
    assert "SAFETY_PATH_RECOVERED: the stop went out" in capsys.readouterr().err


def test_a_stop_retry_waits_for_the_lease_lock_and_gives_way_to_work_current_since(
        monkeypatch):
    """The check for new work and the stop go together under `_lease_lock`, which an assignment
    holds while it checks and dispatches: while the test holds it as an assignment would, no
    stop goes out, and once new work is current the old stop is not tried again. (An assignment
    is refused while the stop is failing; new work can be current only through one that passed
    that check just before the loss.)"""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, _ = _lose_the_stream_with_a_raising_stop(robot, lease=False)
    time.sleep(0.05)                                   # inside the budget, retries running
    with a._lease_lock:                                # an assignment checks, then dispatches
        held_at = robot.stops
        time.sleep(0.1)
        assert robot.stops == held_at, "a stop went out while the assignment held the lock"
        a._current_action = "t-new"
    time.sleep(0.3)
    assert robot.stops == held_at, "the old stop was tried on the new work"
    ends = _action_ends(a, 0.3)
    assert len(ends) == 1 and "then new work took the robot" in ends[0].message, ends
    assert a._stop_failing == "", "the refusal outlived the retry"


def _control_results(a: Adapter, wait_s: float) -> tuple[list, list]:
    """The action_status and command_result messages the adapter sends within `wait_s`."""
    ends, results, end = [], [], time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            m = a._outbox.get(timeout=0.02)
        except queue.Empty:
            continue
        kind = m.WhichOneof("payload")
        if kind == "action_status":
            ends.append(m.action_status)
        elif kind == "command_result":
            results.append(m.command_result)
    return ends, results


def test_a_cancel_during_the_retries_does_not_end_the_action_a_second_time(monkeypatch):
    """The action was ended when the stream was lost; a cancel for it arriving while the stop is
    still being tried is answered, and stops the robot, but reports no second end."""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, _ = _lose_the_stream_with_a_raising_stop(robot, lease=False)
    time.sleep(0.05)                                   # inside the budget, retries running
    robot.fail = robot.stops                           # the stop goes out from now on
    a._on_cancel(pb.CancelAction(action_id="t-1", cause=pb.CANCEL_CAUSE_OPERATOR),
                 pb.CommandResult(command_id=2, robot_id=ROBOT))
    ends, results = _control_results(a, 1.0)
    assert len(ends) == 1, [e.message for e in ends]
    assert ends[0].state == pb.ACTION_STATE_FAILED
    assert ends[0].message.startswith("stopped after "), ends[0].message
    assert [r.WhichOneof("result") for r in results] == ["cancel_action"]


def test_a_renewal_during_the_retries_reports_the_action_not_running_and_the_retry_goes_on(
        monkeypatch):
    """A renewal of the lost action while its stop is still being tried: the action is reported
    not running (C4.4), the retry goes on, and the action is not reported ended a second time.
    (Whether the renewal recreates the released lease is the lease monitor's rule, not this
    path's: it renews any robot whose generation is not stale.)"""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, _ = _lose_the_stream_with_a_raising_stop(robot)
    assert ROBOT not in a._lease._deadline
    try:
        time.sleep(0.05)                               # inside the budget, retries running
        result = pb.CommandResult(command_id=3, robot_id=ROBOT)
        a._on_renew(pb.RenewActionLease(action_id="t-1", lease_generation=1,
                                        lease_duration_ms=60_000), result)
        assert result.renew_lease.running is False
        tried = robot.stops
        time.sleep(0.1)
        assert robot.stops > tried, "the renewal stopped the retries"
        ends = _action_ends(a, 0.4)
        assert len(ends) == 1, [e.message for e in ends]
        assert "the robot could not be stopped" in ends[0].message, ends[0].message
    finally:
        a._shutdown.set()   # ends the retry


def _navigate(action_id: str, token: int):
    return pb.AssignAction(action_id=action_id, action_type="Navigate",
                           payload_json=b'{"x": 3, "y": 4}', fencing_token=token,
                           lease_generation=1, lease_duration_ms=60_000)


def test_assignments_and_resumes_are_refused_while_the_stop_is_failing(monkeypatch):
    """The SafetyStream is back, but the stop it lost still raises: no new work and no resume
    until a stop goes out. Then work is accepted again."""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, _ = _lose_the_stream_with_a_raising_stop(robot, lease=False)
    a._safety_up.set()                                 # the stream is back
    try:
        time.sleep(0.05)
        res = pb.CommandResult(command_id=4, robot_id=ROBOT)
        a._on_assign(_navigate("t-2", 2), res)
        assert not res.assign_action.accepted
        assert res.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
        assert "the robot's stop is failing" in res.assign_action.message, res.assign_action
        res = pb.CommandResult(command_id=5, robot_id=ROBOT)
        a._on_resume(type(pb.Command().resume)(), res)
        assert not res.resume.resumed and "the robot's stop is failing" in res.resume.message
        a._handle_command(pb.Command(command_id=7, robot_id=ROBOT, validate_action=type(
            pb.Command().validate_action)(action_type="Navigate", payload_json=b'{"x": 3, "y": 4}')))
        _, results = _control_results(a, 0.1)
        validate = [r.validate_action for r in results if r.WhichOneof("result") == "validate_action"]
        assert len(validate) == 1 and not validate[0].servable, results
        assert "the robot's stop is failing" in validate[0].message
        time.sleep(0.3)                                # past the 0.2 s budget: the later retries
        res = pb.CommandResult(command_id=8, robot_id=ROBOT)
        a._on_assign(_navigate("t-2", 2), res)
        assert not res.assign_action.accepted, "work was accepted once the budget ended"
        assert "the robot's stop is failing" in res.assign_action.message
        robot.fail = robot.stops                       # the stop goes out from now on
        end = time.monotonic() + 1.0
        while a._stop_failing and time.monotonic() < end:
            time.sleep(0.01)
        res = pb.CommandResult(command_id=6, robot_id=ROBOT)
        a._on_assign(_navigate("t-2", 2), res)
        assert res.assign_action.accepted, res.assign_action.message
    finally:
        a._shutdown.set()


@pytest.mark.parametrize("cancel_ends", ["still confirming", "confirmed", "not confirmed"])
def test_a_cancel_in_progress_at_the_loss_does_not_end_the_retry(monkeypatch, capsys,
                                                                 cancel_ends):
    """A cancel that is still confirming rest keeps its action current; that is not new work,
    so the stop that raised is still tried again, and the cancel reports the action's end. When
    the cancel finishes -- rest confirmed (the action cleared), or not (the action kept) -- the
    retry still goes on: neither is new work."""
    _fast_retries(monkeypatch)
    robot = _StopRaises()
    a, _ = _lose_the_stream_with_a_raising_stop(robot, lease=False, cancelling=True)
    try:
        time.sleep(0.05)
        if cancel_ends == "confirmed":
            with a._lease_lock:                        # as the cancel's report does
                a._current_action = ""
        elif cancel_ends == "not confirmed":           # the cancel ends the action FAILED, kept
            a._end_on_cancel("t-1", adapter_mod.task_map.TASK_FAILED, 1,
                             pb.CANCEL_CAUSE_OPERATOR, message="test: rest not confirmed")
        tried = robot.stops
        time.sleep(0.3)                                # through the budget and into the later retries
        assert robot.stops > tried + 3, f"the stop was tried {robot.stops - tried} more times"
        assert a._stop_failing, "the retry gave way to the action being cancelled"
        ends = _action_ends(a, 0.1)
        want = 1 if cancel_ends == "not confirmed" else 0   # the cancel's own end, no other
        assert len(ends) == want, [e.message for e in ends]
        assert "SAFETY_STOP_RETRY_ENDED" not in capsys.readouterr().err
    finally:
        a._shutdown.set()


def test_a_second_loss_takes_the_retry_over(monkeypatch):
    """The stream comes back and is lost again while the first loss's stop is still raising:
    one retry goes on, not two, and the first loss's action says the second took over."""
    _fast_retries(monkeypatch)
    robot = _StopRaises()

    def retries() -> set:
        return {t for t in threading.enumerate() if t.name == "safety-lost-report"}

    before = retries()
    a, _ = _lose_the_stream_with_a_raising_stop(robot, lease=False)
    try:
        time.sleep(0.05)
        a._safety_up.set()
        a._safety_down("test: lost again", 10.0)
        time.sleep(0.1)
        assert len(retries() - before) == 1, retries() - before
        sent = len(robot.senders)
        time.sleep(0.15)
        assert len(set(robot.senders[sent:])) == 1, "more than one thread is sending the stop"
        ends = _action_ends(a, 0.3)
        assert len(ends) == 1 and "the SafetyStream was lost again" in ends[0].message, ends
        assert a._stop_failing, "the second loss's retry is not running"
    finally:
        a._shutdown.set()


def test_a_fault_dialling_the_safety_stream_keeps_motion_refused_and_redials():
    """A fault in the dial itself (`stub.SafetyStream` raises): nothing to drain, the thread
    survives, the fault is counted, motion stays refused, and the dial is tried again."""
    a = _adapter()
    dials: list = []

    class _DialRaises:
        def SafetyStream(self, requests):  # the gRPC method name
            dials.append(time.monotonic())
            raise ValueError("stub bug: SafetyStream")

    t = threading.Thread(target=a.run_safety_stream, args=(_DialRaises(),), daemon=True)
    t.start()
    time.sleep(1.2)
    try:
        assert t.is_alive(), "the SafetyStream thread ended"
        assert len(dials) >= 2
        assert a._safety_faults >= 2
        assert "SafetyStream is down" in a._no_estop_path()
    finally:
        a._shutdown.set()
        t.join(timeout=5.0)


class _EdgeTwoEstops:
    """An edge node that sends two estops, and reads what the adapter sends back on its own
    thread, so a test never blocks waiting for an answer that does not come."""

    def __init__(self) -> None:
        self.sent: list = []

    def EdgeStream(self, requests):  # the gRPC method name
        def read() -> None:
            for m in requests:
                self.sent.append(m)
        threading.Thread(target=read, daemon=True).start()
        return iter([pb.EdgeControlMessage(estop=pb.Estop(estop_id="edge-1", reason="test")),
                     pb.EdgeControlMessage(estop=pb.Estop(estop_id="edge-2", reason="test"))])

    def acks(self, want: int, timeout: float = 2.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            acks = {m.estop_ack.estop_id: m.estop_ack for m in list(self.sent)
                    if m.WhichOneof("msg") == "estop_ack"}
            if len(acks) >= want:
                return acks
            time.sleep(0.01)
        return {m.estop_ack.estop_id: m.estop_ack for m in list(self.sent)
                if m.WhichOneof("msg") == "estop_ack"}


def test_a_fault_in_an_edge_estop_answers_it_failed_and_the_next_edge_estop_is_still_read():
    a = _adapter()
    edge = _EdgeTwoEstops()
    a._pb_grpc = NS(EdgeServiceStub=lambda channel: edge)
    a._edge_addr.put("localhost:1")
    a._edge_loop()                                   # returns when the edge stream ends
    acks = edge.acks(want=2)
    assert set(acks) == {"edge-1", "edge-2"}, f"answered: {sorted(acks)}"
    assert all(ack.state == pb.ESTOP_STATE_FAILED for ack in acks.values())
    assert a._safety_faults == 2


def test_a_fault_in_the_safety_stream_lost_report_still_reports_the_action_failed():
    a = _adapter()
    a._current_action, a._current_fencing_token = "t-1", 1
    a._safety_up.set()
    a._safety_down("test: the stream ended", 10.0)
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        try:
            m = a._outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        if m.WhichOneof("payload") == "action_status":
            assert m.action_status.state == pb.ACTION_STATE_FAILED
            assert "safety path raised ValueError" in m.action_status.message
            assert a._safety_faults == 1
            return
    raise AssertionError("the lost action's FAILED status was never sent")


def test_a_fault_in_the_estop_drill_answers_it_failed_and_telemetry_goes_on():
    a = _adapter()
    a._sim = NS(estop_due=lambda: True)
    a._safety_outbox = queue.Queue()
    a._current_action = "t-1"
    a._maybe_estop_drill()                           # must not raise into the telemetry loop
    acks = [a._safety_outbox.get_nowait().estop_ack for _ in range(a._safety_outbox.qsize())]
    assert acks and acks[-1].state == pb.ESTOP_STATE_FAILED
    assert a._current_action == ""
    assert a._safety_faults == 1
