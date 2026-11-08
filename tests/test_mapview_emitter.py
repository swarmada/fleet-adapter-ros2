# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter's map-view feed: the contract emitted while an explore-floor segment runs.

A feed reads three accessors of the exploration stack and commands nothing. The segment runs,
reports and stops exactly as it does without one: the same reports reach the control plane,
and cancel, estop and a lapsed lease end it as before, with the feed ending itself after.
A sink that fails or hangs changes nothing the adapter does.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
import urllib.error
from pathlib import Path

import pytest

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import _resolve_mapview
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import emitter as mv
from fleet_adapter_ros2.robot import SimulatedRobot
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

ROBOT = "amr-1"
EXPLORE = b'{"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2}'
FAST = {"pose_period_s": 0.01, "map_check_s": 0.02, "map_repeat_s": 0.2}
TOUR = [(3.0, 0.0), (3.0, 2.5), (-1.0, 2.5)]


class StepClock:
    """Simulated time that advances by each wait (as in test_explore_adapter)."""

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


class ListSink:
    def __init__(self) -> None:
        self.msgs: list[dict] = []
        self._lock = threading.Lock()

    def put(self, m: dict) -> None:
        c.check(m)
        with self._lock:
            self.msgs.append(json.loads(json.dumps(m)))

    def kinds(self) -> list[str]:
        with self._lock:
            return [m["kind"] for m in self.msgs]

    def last_status(self):
        with self._lock:
            return next((m for m in reversed(self.msgs) if m["kind"] == "SessionStatus"), None)


class RaisingSink:
    def put(self, m: dict) -> None:
        raise RuntimeError("the sink is broken")


class HangingSink:
    """A sink whose put never returns: the feed's thread hangs, and nothing else may."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def put(self, m: dict) -> None:
        self.entered.set()
        self.release.wait(30)


def _rig(tmp_path, sink=None, **stack_kw):
    clock = StepClock()
    stack = SimulatedExploreStack(clock=clock, **stack_kw)
    robot = SimulatedRobot(explore=stack)
    robot.spawn(ROBOT)
    emitter = mv.MapviewEmitter(sink, ROBOT, simulated=True, **FAST) if sink is not None else None
    a = Adapter(pb, None, robot, ROBOT, job_clock=clock, mapview=emitter,
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


def _assign(a: Adapter, lease_ms: int = 60_000) -> list:
    _command(a, 1, assign_action=pb.AssignAction(
        action_id="explore-first", action_type="Custom", assignment_id="a-explore-first",
        fencing_token=1, lease_generation=1, lease_duration_ms=lease_ms, payload_json=EXPLORE))
    out = _drain(a)
    assert [m.command_result.assign_action.accepted for m in out
            if m.WhichOneof("payload") == "command_result"] == [True]
    return out


def _statuses(msgs) -> list:
    return [m.action_status for m in msgs if m.WhichOneof("payload") == "action_status"]


def _until(cond, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _run_to_end(a: Adapter, msgs: list, timeout: float = 15.0) -> list:
    done = (pb.ACTION_STATE_SUCCEEDED, pb.ACTION_STATE_FAILED, pb.ACTION_STATE_CANCELLED,
            pb.ACTION_STATE_STOPPED)
    deadline = time.monotonic() + timeout
    while not any(s.state in done for s in _statuses(msgs)):
        assert time.monotonic() < deadline, "timed out"
        msgs.extend(_drain(a))
        time.sleep(0.005)
    return [s for s in _statuses(msgs) if s.state in done]


def _feed_thread_gone(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while any(t.name.startswith("mapview-map@") for t in threading.enumerate()):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


# ── what a feed may touch ───────────────────────────────────────────────────

def test_a_feed_calls_the_read_accessors_and_nothing_else():
    """Run a feed over a stack that answers only READS: any other attribute it asks for fails."""
    real = SimulatedExploreStack(tour=TOUR)
    real.start_slam(None)
    real.start_explorer()
    asked: list[str] = []

    class OnlyReads:
        def __getattr__(self, name):
            asked.append(name)
            if name not in mv.READS:
                raise AssertionError(f"the feed asked the stack for {name}")
            return getattr(real, name)

    sink = ListSink()
    feed = mv.SegmentFeed(stack=OnlyReads(), sink=sink, robot="sim-a", action_id="lab/e1",
                          simulated=True, **FAST)
    ticks = iter(range(40))
    feed.start(alive=lambda: next(ticks, None) is not None)
    feed.join(10)
    assert set(asked) == set(mv.READS)
    kinds = sink.kinds()
    assert {"SessionStatus", "Overlays", "PoseSample", "MapSnapshot"} <= set(kinds)
    assert sink.last_status()["phase"] == "stopped"


def test_the_feed_source_names_only_the_read_accessors():
    """The same, by reading the source: every name the feed reads through is in READS, and the
    stack is reached only through `_read`."""
    tree = ast.parse(Path(mv.__file__).read_text())
    reads, direct = set(), []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_read" and node.args
                and isinstance(node.args[0], ast.Constant)):
            reads.add(node.args[0].value)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "_stack"):
            direct.append(node.attr)
    assert reads == set(mv.READS) and direct == []


# ── a segment with a feed, through the adapter ─────────────────────────────

def test_a_segment_emits_the_contract_and_ends_with_its_final_report(tmp_path):
    sink = ListSink()
    a, robot, _ = _rig(tmp_path, sink, tour=TOUR, explore_seconds=40.0)
    msgs = _assign(a)
    end = _run_to_end(a, msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert _feed_thread_gone()
    st = c.ViewState()
    for m in sink.msgs:
        assert st.accept(m), m["kind"]
    assert st.refused == {"frame": 0, "contract": 0, "version": 0}
    assert st.frame == "map@explore-first"
    assert st.status["phase"] == "stopped" and st.status["source"] == "explore-floor"
    assert st.status["text"] == end[0].message[:c.MAX_TEXT] and st.status["simulated"] is True
    assert st.map is not None and st.keepout_grids, "the map and the served keep-out are shown"
    assert len({(round(p[1], 1), round(p[2], 1)) for p in st.poses}) > 5, "the robot moved"
    assert robot.is_stopped(ROBOT)


def test_the_reports_reaching_the_control_plane_are_the_same_with_and_without_a_feed(tmp_path):
    def run(sink, where):
        a, _, _ = _rig(where, sink)
        msgs = _assign(a)
        _run_to_end(a, msgs)
        # The manifest digest covers segment.json, which records when it was saved: two runs
        # differ there with or without a feed, so it is masked.
        return [(s.state, re.sub(r"map-sha256=[0-9a-f]{64}", "map-sha256=*", s.message),
                 s.progress_pct) for s in _statuses(msgs)]

    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    assert run(ListSink(), tmp_path / "with") == run(None, tmp_path / "without")


def _stop_run(tmp_path, sink, stop):
    """One stop path of test_explore_adapter on a running segment: what it ended with."""
    a, robot, stack = _rig(tmp_path, sink, explore_seconds=1e9, drain_mv_per_s=0.0, tour=TOUR)
    _assign(a, lease_ms=50 if stop == "lease" else 60_000)
    _until(lambda: stack.moving())
    if sink is not None:
        _until(lambda: "PoseSample" in sink.kinds())
    msgs: list = []
    acks: list = []
    t0 = time.monotonic()
    if stop == "cancel":
        _command(a, 2, cancel_action=pb.CancelAction(action_id="explore-first", fencing_token=1,
                                                      cause=pb.CANCEL_CAUSE_OPERATOR))
        end = [(s.state, s.end_cause) for s in _run_to_end(a, msgs)]
    elif stop == "estop":
        a._estop("e-1", acks.append)
        end = [ack.state for ack in acks[-1:]]
    else:
        time.sleep(0.1)
        a._tick_lease()
        end = [(s.state, s.end_cause) for s in _run_to_end(a, msgs)]
    took = time.monotonic() - t0
    time.sleep(0.05)
    msgs.extend(_drain(a))
    states = [s.state for s in _statuses(msgs)]
    ends = [i for i, st in enumerate(states) if st != pb.ACTION_STATE_RUNNING]
    after = states[ends[0] + 1:] if ends else []        # anything said after the end
    return {"end": end, "halted": "halt" in stack.log, "moving": stack.moving(),
            "stopped": robot.is_stopped(ROBOT), "running_after": after}, took


@pytest.mark.parametrize("stop", ["cancel", "estop", "lease"])
def test_a_stop_ends_a_segment_with_a_feed_exactly_as_without_one(tmp_path, stop):
    """The stop paths of test_explore_adapter, run with and without a feed: the same end, the
    explorer stopped, nothing said after it; and the feed ends itself saying the segment ended."""
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    sink = ListSink()
    with_feed, took = _stop_run(tmp_path / "with", sink, stop)
    without, _ = _stop_run(tmp_path / "without", None, stop)
    assert with_feed == without
    assert with_feed["halted"] and not with_feed["moving"] and with_feed["stopped"]
    if stop == "estop":
        assert with_feed["end"] == [pb.ESTOP_STATE_STOPPED] and took < 2.0
    assert _feed_thread_gone()
    assert sink.last_status()["phase"] == "stopped"
    assert "ended without a final report" in sink.last_status()["text"]


@pytest.mark.parametrize("sink_kind", ["raising", "hanging"])
def test_a_sink_that_fails_or_hangs_changes_nothing_the_adapter_does(tmp_path, sink_kind):
    sink = RaisingSink() if sink_kind == "raising" else HangingSink()
    a, robot, _ = _rig(tmp_path, sink)
    msgs = _assign(a)
    end = _run_to_end(a, msgs)
    assert [s.state for s in end] == [pb.ACTION_STATE_SUCCEEDED]
    assert robot.is_stopped(ROBOT)
    if sink_kind == "hanging":
        assert sink.entered.is_set()
        # An estop with the feed's thread hung in its sink is answered as ever.
        acks: list = []
        a._estop("e-2", acks.append)
        assert acks[-1].state == pb.ESTOP_STATE_STOPPED
        sink.release.set()


def test_charge_gets_no_feed(tmp_path):
    sink = ListSink()
    a, _, stack = _rig(tmp_path, sink)
    stack.set_battery(8000.0)
    _command(a, 1, assign_action=pb.AssignAction(
        action_id="charge-1", action_type="Charge", assignment_id="a-charge-1", fencing_token=1,
        lease_generation=1, lease_duration_ms=60_000, payload_json=b"{}"))
    _run_to_end(a, _drain(a))
    assert sink.msgs == []


# ── the HTTP sink ───────────────────────────────────────────────────────────

class FakeOpener:
    def __init__(self, fail: int = 0) -> None:
        self.fail = fail
        self.bodies: list[list[dict]] = []
        self.headers: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, req, timeout):
        with self.lock:
            if self.fail > 0:
                self.fail -= 1
                raise urllib.error.URLError("refused")
            self.bodies.append([json.loads(ln) for ln in req.data.decode().splitlines()])
            self.headers.append(dict(req.header_items()))

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"{}"
        return Resp()


def _msg(kind: str, i: int) -> dict:
    if kind == "PoseSample":
        return c.message(kind, "sim-a", "map@t", float(i), x=float(i), y=0.0, yaw_deg=0.0,
                         source="sim")
    if kind == "MapSnapshot":
        return c.message(kind, "sim-a", "map@t", float(i), seq=i,
                         **c.encode_grid([[0, i % 100]], 0.1, (0, 0)))
    return c.message(kind, "sim-a", "map@t", float(i), source="explore-floor", phase="running",
                     text=f"t{i}", area_free_m2=None)


def test_the_http_sink_posts_json_lines_with_the_token_and_retries_what_failed():
    opener = FakeOpener(fail=2)
    sink = mv.HttpSink("http://map-view:8080", "s3cret", opener=opener, max_backoff_s=0.05)
    for i in range(5):
        sink.put(_msg("PoseSample", i))
    _until(lambda: sum(len(b) for b in opener.bodies) == 5, timeout=5)
    sink.close()
    assert [m["x"] for b in opener.bodies for m in b] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert opener.headers[0]["Authorization"] == "Bearer s3cret"
    assert opener.headers[0]["Content-type"] == "application/x-ndjson"
    assert sink.failed_posts == 2 and sink.dropped == 0


def test_a_full_queue_drops_the_oldest_poses_and_keeps_the_newest_of_each_kind():
    gate = threading.Event()

    def blocked(req, timeout):
        gate.wait(5)
        raise urllib.error.URLError("down")

    sink = mv.HttpSink("http://map-view:8080", opener=blocked, max_queue=5, max_backoff_s=0.05)
    sink.put(_msg("PoseSample", 0))          # taken by the poster, which blocks on it
    time.sleep(0.05)
    for i, kind in enumerate(["MapSnapshot", "SessionStatus", "PoseSample", "PoseSample",
                              "PoseSample", "PoseSample", "MapSnapshot"], start=1):
        sink.put(_msg(kind, i))
    with sink._cv:
        kept = [(m["kind"], m["stamp"]) for m in sink._q]
    gate.set()
    sink.close(0.5)
    assert len(kept) == 5 and sink.dropped >= 3
    assert ("MapSnapshot", 7.0) in kept and ("SessionStatus", 2.0) in kept
    assert ("PoseSample", 6.0) in kept and ("MapSnapshot", 1.0) not in kept
    assert ("PoseSample", 3.0) not in kept


# ── the command line ────────────────────────────────────────────────────────

class Args:
    def __init__(self, **kw) -> None:
        self.mapview_url = kw.get("url")
        self.mapview_file = kw.get("file")
        self.mapview_robot = kw.get("robot")
        self.mapview_session = kw.get("session")
        self.robot_id = kw.get("robot_id", "sim-robot-001")
        self.binding = kw.get("binding", "simulated")


@pytest.mark.parametrize("kw,why", [
    ({"url": "http://a", "file": "/tmp/x"}, "give one"),
    ({"url": "map-view:8080"}, "http:// or https://"),
    ({"url": "http://a", "robot": "10.0.0.7"}, "not a local alias"),
    ({"robot": "sim-a"}, "only with"),
    ({"session": "s1"}, "only with"),
    ({"url": "http://a", "session": "s 1"}, "--mapview-session"),
])
def test_the_mapview_flags_are_refused_before_anything_connects(kw, why):
    emitter, refusal = _resolve_mapview(Args(**kw))
    assert emitter is None and why in refusal


def test_no_mapview_flag_means_no_feed_and_no_refusal():
    assert _resolve_mapview(Args()) == (None, "")


def test_a_file_feed_is_built_from_the_flags(tmp_path):
    emitter, refusal = _resolve_mapview(Args(file=str(tmp_path / "m.jsonl")))
    assert refusal == "" and emitter.robot == "sim-robot-001" and emitter.simulated
    emitter.close()
