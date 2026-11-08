# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""explore-floor and Charge through the adapter: the catalog, admission, and the stop paths.

A binding with an exploration stack declares both actions; the Nav2 binding, which has none
yet, declares exactly what it did before. What the catalog declares is what assignment and
validate_action enforce. A segment runs on its own thread and reports through the adapter,
and cancel, estop and lease expiry end it exactly as they end a Navigate: the stop goes to the
binding, the binding stops the explorer and the return goal, the stop path reports -- and the
job says nothing more. Pause and resume reach the job.
"""

from __future__ import annotations

import json
import threading
import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
EXPLORE = (b'{"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2}')


class StepClock:
    """Simulated time that advances by each wait, with a real millisecond per wait so another
    thread (the test's) gets a turn while a segment runs."""

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


def _rig(tmp_path, **stack_kw):
    clock = StepClock()
    stack = SimulatedExploreStack(clock=clock, **stack_kw)
    robot = SimulatedRobot(explore=stack)
    robot.spawn(ROBOT)
    a = Adapter(pb, None, robot, ROBOT, job_clock=clock,
                artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab")
    return a, robot, stack


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _command(a: Adapter, command_id: int, **arm) -> None:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=command_id, robot_id=ROBOT, **arm)), None)


def _assign(a: Adapter, action_type: str, payload: bytes, token: int = 1,
            action: str = "explore-first", lease_ms: int = 60_000):
    _command(a, token, assign_action=pb.AssignAction(
        action_id=action, action_type=action_type, assignment_id=f"a-{action}",
        fencing_token=token, lease_generation=1, lease_duration_ms=lease_ms,
        payload_json=payload))
    out = _drain(a)
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"]
    assert len(res) == 1
    return res[0], out


def _statuses(msgs, action: str) -> list:
    return [m.action_status for m in msgs if m.WhichOneof("payload") == "action_status"
            and m.action_status.action_id == action]


def _until(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _collect(a: Adapter, into: list, until, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        into.extend(_drain(a))
        if until(into):
            return
        assert time.monotonic() < deadline, [m.action_status.message for m in into
                                             if m.WhichOneof("payload") == "action_status"]
        time.sleep(0.005)


def _terminal(a: Adapter, action: str, msgs: list):
    done = (pb.ACTION_STATE_SUCCEEDED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED,
            pb.ACTION_STATE_STOPPED)
    _collect(a, msgs, lambda ms: any(s.state in done for s in _statuses(ms, action)))
    return [s for s in _statuses(msgs, action) if s.state in done]


# ── the catalog ─────────────────────────────────────────────────────────────

def test_a_binding_with_a_stack_declares_explore_floor_and_charge(tmp_path) -> None:
    a, _, _ = _rig(tmp_path)
    snap = a._capabilities_snapshot()
    assert [s.action_type for s in snap.supported_actions] == ["Navigate", "Custom", "Charge"]
    custom = {p.name: p for p in snap.supported_actions[1].params}
    assert custom["operation"].kind == "enum" and list(custom["operation"].allowed) == [
        "explore-floor"] and custom["operation"].required
    assert custom["bound_ref"].required and custom["bound_ref"].kind == "string"
    cap = custom["max_speed_mps"]
    assert (cap.kind, cap.unit, cap.required, cap.max) == ("number", "m/s", True,
                                                           explore.MAX_SPEED_MPS)
    assert custom["resume"].kind == "bool" and not custom["resume"].required
    assert {"keepouts_ref", "battery_return_mv", "exhausted_after_recoveries",
            "time_box_s"} <= set(custom)
    assert {r.name for r in snap.supported_actions[1].results} == {
        "map-ref", "map-sha256", "covered-m2"}
    charge = snap.supported_actions[2]
    assert [p.name for p in charge.params] == ["full_mv"] and not charge.params[0].required
    assert "Never moves the robot" in charge.description


def test_the_nav2_binding_without_a_stack_declares_what_it_did() -> None:
    snap = Adapter(pb, None, Nav2Binding(), ROBOT)._capabilities_snapshot()
    assert [s.action_type for s in snap.supported_actions] == ["Navigate"]


# ── admission: what the catalog says is what is enforced ────────────────────

def test_an_explore_above_the_speed_cap_is_declined_and_nothing_moves(tmp_path) -> None:
    a, robot, stack = _rig(tmp_path)
    res, out = _assign(a, "Custom", EXPLORE.replace(b"0.2}", b"0.25}"))
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert "max_speed_mps" in res.message and "above" in res.message
    assert stack.log == [] and robot.is_stopped(ROBOT) and a._current_action == ""
    assert _statuses(out, "explore-first") == []


def test_an_operation_not_declared_is_declined(tmp_path) -> None:
    a, _, stack = _rig(tmp_path)
    res, _ = _assign(a, "Custom", EXPLORE.replace(b"explore-floor", b"map-floor"))
    assert not res.accepted and "map-floor" in res.message and stack.log == []


def test_validate_action_answers_from_the_same_check(tmp_path) -> None:
    a, _, _ = _rig(tmp_path)
    _command(a, 1, validate_action=pb.ValidateAction(action_type="Custom", payload_json=EXPLORE))
    ok = _drain(a)[0].command_result.validate_action
    _command(a, 2, validate_action=pb.ValidateAction(
        action_type="Custom", payload_json=EXPLORE.replace(b"0.2}", b"0.3}")))
    no = _drain(a)[0].command_result.validate_action
    assert ok.servable and not no.servable and "max_speed_mps" in no.message


# ── a segment through the adapter ───────────────────────────────────────────

def test_a_segment_is_accepted_runs_and_ends_succeeded_with_its_map(tmp_path) -> None:
    a, robot, _ = _rig(tmp_path)
    res, out = _assign(a, "Custom", EXPLORE)
    assert res.accepted and res.accepted_fencing_token == 1
    msgs = list(out)
    end = _terminal(a, "explore-first", msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert "map-ref=floor-maps/lab/explore-first" in end[0].message
    assert end[0].fencing_token == 1
    running = [s for s in _statuses(msgs, "explore-first") if s.state == pb.ACTION_STATE_RUNNING]
    assert any("bound proven" in s.message for s in running)
    assert (tmp_path / "lab" / "explore-first" / "MANIFEST.sha256").is_file()
    assert a._current_action == "" and ROBOT not in a._lease._deadline
    assert robot.is_stopped(ROBOT)


def test_a_cancel_stops_the_explorer_and_the_segment_says_nothing_after(tmp_path) -> None:
    a, robot, stack = _rig(tmp_path, explore_seconds=1e9, drain_mv_per_s=0.0)
    assert _assign(a, "Custom", EXPLORE)[0].accepted
    _until(lambda: stack.moving())
    _command(a, 2, cancel_action=pb.CancelAction(action_id="explore-first", fencing_token=1,
                                                  cause=pb.CANCEL_CAUSE_OPERATOR))
    msgs: list = []
    end = _terminal(a, "explore-first", msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_CANCELLED]
    assert "halt" in stack.log and not stack.moving() and robot.is_stopped(ROBOT)
    time.sleep(0.05)
    msgs.extend(_drain(a))
    assert [s.state for s in _statuses(msgs, "explore-first")
            if s.state != pb.ACTION_STATE_RUNNING] == [pb.ACTION_STATE_CANCELLED]


def test_a_lapsed_lease_stops_a_segment_as_it_stops_a_navigate(tmp_path) -> None:
    a, robot, stack = _rig(tmp_path, explore_seconds=1e9, drain_mv_per_s=0.0)
    assert _assign(a, "Custom", EXPLORE, lease_ms=50)[0].accepted
    _until(lambda: stack.moving())
    time.sleep(0.1)
    a._tick_lease()
    msgs: list = []
    end = _terminal(a, "explore-first", msgs)
    assert [(s.state, s.end_cause) for s in end] == [
        (pb.ACTION_STATE_STOPPED, pb.ACTION_END_CAUSE_LEASE_EXPIRED)]
    assert not stack.moving() and robot.is_stopped(ROBOT)


def test_an_estop_stops_the_explorer_and_is_confirmed_from_the_robot(tmp_path) -> None:
    a, _, stack = _rig(tmp_path, explore_seconds=1e9, drain_mv_per_s=0.0)
    assert _assign(a, "Custom", EXPLORE)[0].accepted
    _until(lambda: stack.moving())
    acks: list = []
    a._estop("e-1", acks.append)
    assert acks[-1].state == pb.ESTOP_STATE_STOPPED
    assert "halt" in stack.log and not stack.moving()


def test_pause_and_resume_reach_the_segment(tmp_path) -> None:
    a, _, stack = _rig(tmp_path, explore_seconds=1e9, drain_mv_per_s=0.0)
    assert _assign(a, "Custom", EXPLORE)[0].accepted
    _until(lambda: stack.moving())
    _command(a, 2, pause=pb.PauseCapabilities(require_stop_before_ack=True))
    assert _drain(a)[-1].command_result.pause.paused
    assert not stack.moving()
    _command(a, 3, resume=pb.ResumeCapabilities())
    assert _drain(a)[-1].command_result.resume.resumed
    _until(lambda: stack.log.count("start_explorer") == 2 and stack.moving())


def test_a_new_assignment_ends_the_running_segment_first(tmp_path) -> None:
    a, _, stack = _rig(tmp_path, explore_seconds=1e9, drain_mv_per_s=0.0)
    assert _assign(a, "Custom", EXPLORE)[0].accepted
    _until(lambda: stack.moving())
    assert _assign(a, "Navigate", b'{"x": 1, "y": 2}', token=2, action="t-2")[0].accepted
    assert not stack.moving() and a._current_action == "t-2"
    time.sleep(0.05)
    assert _statuses(_drain(a), "explore-first") == []


# ── Charge ──────────────────────────────────────────────────────────────────

def test_charge_through_the_adapter_waits_and_never_moves(tmp_path) -> None:
    a, robot, stack = _rig(tmp_path, start_mv=8000)
    res, out = _assign(a, "Charge", b"", action="charge")
    assert res.accepted
    msgs = list(out)
    seen_moving = []
    _collect(a, msgs, lambda ms: (seen_moving.append(not robot.is_stopped(ROBOT)) or any(
        s.state == pb.ACTION_STATE_SUCCEEDED for s in _statuses(ms, "charge"))))
    assert not any(seen_moving)
    running = [s.message for s in _statuses(msgs, "charge") if s.state == pb.ACTION_STATE_RUNNING]
    assert any(m.startswith("waiting for charge") and "mV" in m for m in running)
    for motion in ("start_explorer", "go_to_mark", "start_slam"):
        assert motion not in stack.log


# ── the keep-out set the adapter obeys, for Navigate and explore-floor ──────

LEDGE = {"name": "floor-room-a-keepout-1", "reason": "operator keep-out, a ledge",
         "polygon": [{"x": 1.2, "y": 1.2}, {"x": 1.8, "y": 1.2}, {"x": 1.8, "y": 1.8},
                     {"x": 1.2, "y": 1.8}]}


def _set(tmp_path, name: str = "zones-applied") -> str:
    doc = {"theta0_deg": 0.0, "source_map": "lab/explore-first", "keepouts": [LEDGE]}
    ref, digest = explore.ArtifactDir(str(tmp_path), "floor-maps").save(
        "lab", name, {"keepouts.json": json.dumps(doc).encode()})
    return f"{ref}@sha256:{digest}"


def _guarded(tmp_path, keepouts_ref: str):
    robot = SimulatedRobot(explore=None)
    robot.spawn(ROBOT)
    return Adapter(pb, None, robot, ROBOT, namespace="lab", keepouts_ref=keepouts_ref,
                   artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps")), robot


def test_a_navigate_goal_inside_a_keep_out_is_declined_naming_the_zone(tmp_path) -> None:
    a, robot = _guarded(tmp_path, _set(tmp_path))
    res, out = _assign(a, "Navigate", b'{"x": 1.5, "y": 1.5}', action="t-1")
    assert not res.accepted and res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert "floor-room-a-keepout-1" in res.message and "a ledge" in res.message
    assert robot.is_stopped(ROBOT) and a._current_action == ""
    assert _statuses(out, "t-1") == []
    assert _assign(a, "Navigate", b'{"x": 3.0, "y": 1.5}', token=2, action="t-2")[0].accepted


def test_a_navigate_under_a_keep_out_set_that_cannot_be_proven_is_declined(tmp_path) -> None:
    ref = _set(tmp_path)
    (tmp_path / "lab" / "zones-applied" / "keepouts.json").write_text('{"keepouts": []}')
    a, robot = _guarded(tmp_path, ref)
    res, _ = _assign(a, "Navigate", b'{"x": 3.0, "y": 1.5}', action="t-1")
    assert not res.accepted and "keepouts.json hashes to" in res.message
    assert robot.is_stopped(ROBOT)


def test_validate_action_answers_the_geometry_too(tmp_path) -> None:
    a, _ = _guarded(tmp_path, _set(tmp_path))
    _command(a, 1, validate_action=pb.ValidateAction(action_type="Navigate",
                                                     payload_json=b'{"x": 1.5, "y": 1.5}'))
    res = _drain(a)[0].command_result.validate_action
    assert not res.servable and "floor-room-a-keepout-1" in res.message


def test_an_explore_naming_a_keep_out_set_that_is_missing_is_declined(tmp_path) -> None:
    a, _, stack = _rig(tmp_path)
    ref = "floor-maps/lab/zones-applied@sha256:" + "c" * 64
    payload = json.dumps({**json.loads(EXPLORE), "keepouts_ref": ref}).encode()
    res, _ = _assign(a, "Custom", payload)
    assert not res.accepted and "missing" in res.message and stack.log == []
