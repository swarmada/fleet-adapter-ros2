# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The circle's state in the map view, `Overlays.target.state`, from the handover job alone.

A navigate-handover's feed writes the target it drives to, with its state:

- ``robot`` (yellow) while Nav2 drives, and while a declined goal waits for confirmed rest;
- ``operator`` (red) only once the gate has turned to the operator, after that wait;
- ``reached`` (green) when the action Succeeded, then ``done`` when it was the last target;
- ``stopped`` (grey) on a stop of any scope, a lease loss, or a Failed handover. It is never left
  for this action: only a new action, with a new circle, starts again at ``robot``.

A stop wins at once. ``robot`` and ``operator`` are never true together: one action, one lease
holder, one state. The states come from the job, its gate and whether the adapter still holds it,
never from the pose. `target.state` is an optional field added to `mapview/0`; a reader written
for the schema before it (the frozen `contract.py` of 24a732a) ignores it.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest
from test_handover_waits_for_rest import BrakingRobot, declined_moving, presses
from test_manual_drive_gate import ROBOT, StepClock, collect, settle, terminals
from test_navigate_handover import ACTION, STOPS, assign, payload, turned

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import emitter

ROBOT_, OPERATOR, REACHED, STOPPED, DONE = "robot", "operator", "reached", "stopped", "done"
OLD_READER = Path(__file__).parent / "fixtures" / "mapview0_reader_24a732a.py"


class ListSink:
    def __init__(self) -> None:
        self.msgs: list[dict] = []
        self._lock = threading.Lock()

    def put(self, m: dict) -> None:
        with self._lock:
            self.msgs.append(json.loads(c.dumps(m)))   # as a reader receives it

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.msgs)


def rig(rest_s: float = 3600.0, cls=BrakingRobot, **kw):
    robot = cls(**kw)
    robot.spawn(ROBOT)
    sink = ListSink()
    mv = emitter.MapviewEmitter(sink, ROBOT, simulated=True, session="s1")
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock(), mapview=mv)
    a._handover_rest_s = rest_s
    return a, robot, sink


def states(sink: ListSink, action: str = ACTION) -> list[str]:
    """The circle's states in the order written, repeats folded."""
    out: list[str] = []
    for m in sink.snapshot():
        t = m.get("target") if m["kind"] == "Overlays" else None
        if t and t.get("label") == action and (not out or out[-1] != t["state"]):
            out.append(t["state"])
    return out


def wait_states(sink: ListSink, until, timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while not until(states(sink)):
        assert time.monotonic() < deadline, states(sink)
        time.sleep(0.01)
    return states(sink)


def lawful(seq: list[str]) -> bool:
    """robot, then at most one operator, then reached (and done) or stopped -- and nothing after."""
    allowed = {(): {ROBOT_}, (ROBOT_,): {OPERATOR, REACHED, STOPPED},
               (ROBOT_, OPERATOR): {REACHED, STOPPED}, (ROBOT_, REACHED): {DONE},
               (ROBOT_, OPERATOR, REACHED): {DONE}}
    for i, s in enumerate(seq):
        if s not in allowed.get(tuple(seq[:i]), set()):
            return False
    return True


# ── the table, end to end through the adapter ─────────────────────────────

def test_nav2_reaches_it_robot_then_reached_in_the_sessions_frame() -> None:
    a, _, sink = rig()
    assert assign(a, payload())[0].accepted
    seq = wait_states(sink, lambda s: s[-1:] == [REACHED])
    assert seq == [ROBOT_, REACHED]
    msgs = sink.snapshot()
    assert {m["frame"] for m in msgs} == {"map@s1"}
    assert {m["source"] for m in msgs if m["kind"] == "SessionStatus"} == {"navigate-handover"}
    (t,) = {(m["target"]["x"], m["target"]["y"]) for m in msgs if m["kind"] == "Overlays"}
    assert t == (2.0, 1.0)


def test_the_last_target_reached_is_done() -> None:
    a, _, sink = rig()
    assert assign(a, payload(last=True))[0].accepted
    assert wait_states(sink, lambda s: s[-1:] == [DONE]) == [ROBOT_, REACHED, DONE]


def test_red_only_after_the_rest_wait_and_green_when_the_operator_brings_it_in() -> None:
    a, robot, sink = rig()
    msgs = declined_moving(a, robot)
    settle(a, msgs, 0.3)
    assert states(sink) == [ROBOT_]                  # declined, held, still moving: not red
    robot.come_to_rest()
    collect(a, msgs, turned)
    assert wait_states(sink, lambda s: s[-1:] == [OPERATOR]) == [ROBOT_, OPERATOR]
    robot.place(2.1, 1.05)
    assert wait_states(sink, lambda s: s[-1:] == [REACHED]) == [ROBOT_, OPERATOR, REACHED]


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_in_the_operators_turn_is_grey_at_once_and_stays(stop) -> None:
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, turned)
    wait_states(sink, lambda s: s[-1:] == [OPERATOR])
    STOPS[stop](a)
    t0 = time.monotonic()
    seq = wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=2.0)
    assert time.monotonic() - t0 < 1.0               # within a few of the feed's 0.1 s looks
    robot.place(2.1, 1.05)                           # in the circle now: no green after a stop
    settle(a, msgs, 0.5)
    assert states(sink) == seq == [ROBOT_, OPERATOR, STOPPED]


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_while_nav2_drives_is_grey(stop) -> None:
    a, robot, sink = rig()
    robot.stall_next()
    assert assign(a, payload())[0].accepted
    wait_states(sink, lambda s: s == [ROBOT_])
    STOPS[stop](a)
    assert wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=2.0) == [ROBOT_, STOPPED]


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_in_the_rest_wait_is_grey_and_never_red(stop) -> None:
    a, robot, sink = rig()
    msgs = declined_moving(a, robot)
    settle(a, msgs, 0.1)
    STOPS[stop](a)
    robot.come_to_rest()
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    settle(a, msgs, 0.3)
    assert wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=2.0) == [ROBOT_, STOPPED]


def test_a_lease_lapse_in_the_operators_turn_is_grey() -> None:
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload(), lease_ms=150)
    assert res.accepted
    collect(a, msgs, turned)
    wait_states(sink, lambda s: s[-1:] == [OPERATOR])
    time.sleep(0.2)
    a._tick_lease()
    assert wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=2.0) == [
        ROBOT_, OPERATOR, STOPPED]


def test_a_lease_deadline_passed_in_the_rest_wait_is_grey() -> None:
    a, robot, sink = rig()
    declined_moving(a, robot, lease_ms=80)
    time.sleep(0.12)
    robot.come_to_rest()
    assert wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=3.0) == [ROBOT_, STOPPED]
    assert presses(robot) == []


def test_rest_not_confirmed_in_time_is_grey() -> None:
    a, robot, sink = rig(rest_s=2.0)
    declined_moving(a, robot)
    assert wait_states(sink, lambda s: s[-1:] == [STOPPED], timeout=5.0) == [ROBOT_, STOPPED]


def test_only_a_new_action_leaves_grey_with_its_own_circle() -> None:
    a, robot, sink = rig()
    robot.stall_next()
    assert assign(a, payload())[0].accepted
    wait_states(sink, lambda s: s == [ROBOT_])
    STOPS["cancel"](a)
    wait_states(sink, lambda s: s[-1:] == [STOPPED])
    assert assign(a, payload(x=1.0, y=0.5), action="handover-b", token=2)[0].accepted
    deadline = time.monotonic() + 10
    while states(sink, "handover-b")[-1:] != [REACHED]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert states(sink) == [ROBOT_, STOPPED]                     # the first stays grey
    assert states(sink, "handover-b") == [ROBOT_, REACHED]


def test_every_sequence_is_lawful_robot_and_operator_never_together() -> None:
    for seq in ([ROBOT_, REACHED], [ROBOT_, OPERATOR, REACHED, DONE], [ROBOT_, STOPPED],
                [ROBOT_, OPERATOR, STOPPED]):
        assert lawful(seq)
    for seq in ([OPERATOR], [ROBOT_, OPERATOR, ROBOT_], [ROBOT_, STOPPED, OPERATOR],
                [ROBOT_, STOPPED, ROBOT_], [ROBOT_, REACHED, STOPPED]):
        assert not lawful(seq)


def test_the_feed_writes_only_lawful_sequences_over_many_runs() -> None:
    seen = []
    for i, stop in enumerate(["estop", "pause", "cancel", None, None]):
        a, robot, sink = rig()
        robot.decline_next("the navigation server aborted the goal")
        res, msgs = assign(a, payload(last=i == 4))
        assert res.accepted
        collect(a, msgs, turned)
        if stop:
            STOPS[stop](a)
        else:
            robot.place(2.1, 1.05)
        final = ([STOPPED], [DONE]) if i == 4 or stop else ([REACHED],)
        seq = wait_states(sink, lambda s, final=final: s[-1:] in final, timeout=5.0)
        settle(a, msgs, 0.3)
        assert lawful(states(sink)), states(sink)
        seen.append(tuple(seq))
    assert len(set(seen)) == 3


def test_robot_is_written_first_even_when_the_feed_thread_starts_after_the_gate_turned(
        monkeypatch) -> None:
    """The feed's thread may first run after the declined goal has come to rest and the gate has
    turned to the operator. `robot` is already written, when the feed is created, so the circle
    never starts at `operator`."""
    from fleet_adapter_ros2.mapview.handover_feed import HandoverFeed

    release, started = threading.Event(), []
    real_start = HandoverFeed.start

    def late_start(self, alive) -> None:
        def later() -> None:
            assert release.wait(10.0)
            real_start(self, alive)
            started.append(self)
        threading.Thread(target=later, daemon=True).start()

    monkeypatch.setattr(HandoverFeed, "start", late_start)
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, turned)                         # the gate has turned; no sampling thread yet
    assert not started
    assert states(sink) == [ROBOT_]
    release.set()
    assert wait_states(sink, lambda s: s[-1:] == [OPERATOR]) == [ROBOT_, OPERATOR]
    robot.place(2.1, 1.05)
    assert wait_states(sink, lambda s: s[-1:] == [REACHED]) == [ROBOT_, OPERATOR, REACHED]
    assert lawful(states(sink))


# ── the field in the contract ─────────────────────────────────────────────

def _target(state=None) -> dict:
    t = {"label": "target-1", "x": 2.0, "y": 1.0}
    if state is not None:
        t["state"] = state
    return c.message("Overlays", ROBOT, "map@s1", 1.0, keepouts=[], zones=[], target=t)


def _old_reader():
    spec = importlib.util.spec_from_file_location("mapview0_reader_24a732a", OLD_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_old_reader_is_the_frozen_24a732a_contract() -> None:
    old = _old_reader()
    assert old.VERSION == c.VERSION == "mapview/0"     # the field did not change the version
    assert not hasattr(old, "TARGET_STATES")


@pytest.mark.parametrize("state", [ROBOT_, OPERATOR, REACHED, STOPPED, DONE])
def test_a_reader_written_for_the_old_schema_ignores_target_state(state) -> None:
    old = _old_reader()
    m = json.loads(c.dumps(_target(state)))
    assert old.check(m) is m
    view = old.ViewState()
    assert view.accept(m, now=0.0)
    assert view.refused == {"frame": 0, "contract": 0, "version": 0}
    t = view.overlays["target"]
    assert (t["label"], t["x"], t["y"]) == ("target-1", 2.0, 1.0)   # what it knew, read as before


def test_the_old_reader_accepts_every_message_a_handover_feed_writes() -> None:
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload(last=True))
    assert res.accepted
    collect(a, msgs, turned)
    robot.place(2.1, 1.05)
    wait_states(sink, lambda s: s[-1:] == [DONE])
    old = _old_reader().ViewState()
    written = sink.snapshot()
    assert written and all(old.accept(m, now=0.0) for m in written)
    assert old.refused == {"frame": 0, "contract": 0, "version": 0}


def test_the_new_reader_knows_the_five_states_and_refuses_any_other() -> None:
    for s in (ROBOT_, OPERATOR, REACHED, STOPPED, DONE):
        c.check(_target(s))
    c.check(_target())                                  # optional: absent is as before
    assert c.TARGET_STATES == (ROBOT_, OPERATOR, REACHED, STOPPED, DONE)
    for bad in ("yellow", "", "ROBOT", 3, None):
        m = {**_target(), "target": {"label": "target-1", "x": 2.0, "y": 1.0, "state": bad}}
        with pytest.raises(c.ContractError):
            c.check(m)


def test_the_contract_text_declares_the_field_optional_with_a_changelog_line() -> None:
    text = (Path(c.__file__).parent / "CONTRACT.md").read_text()
    assert "`state`" in text and "optional" in text.lower()
    assert "## Changes" in text and "target.state" in text.split("## Changes", 1)[1]
    assert "mapview/1" not in text.split("## Changes", 1)[1]
