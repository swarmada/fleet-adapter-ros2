# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The baton in the map view: who holds the robot, from the gate itself, on `target.baton`.

A viewer that shows a switchover -- the fleet's Nav2 handing the robot to a person's keys and the
person handing it back -- must show what the adapter did, not what a message suggests. So the
feed carries, on the target of each navigate-handover and each manual-drive:

- every teleop-gate change, as the gate made it (`TeleopGate` records it under the binding's own
  lock): ``armed`` (open for the action, forwarding nothing: Nav2 drives, or the declined goal
  waits for rest), ``open`` (the operator's keys reach the base) and ``closed``, each with the
  action, the reason and the wall-clock time;
- every in-circle / still tick the job judged (distance to the centre, radius, inside or not,
  still for how long against ``settle_s``), from the drive's own `on_pose`;
- the turn's start (the lease granted, the feed made) and its end (the outcome and the message).

The gate log is a read: recording it adds nothing a stop waits for, and the feed only reads it.
"""

from __future__ import annotations

import importlib.util
import json
import time

from test_handover_waits_for_rest import BrakingRobot
from test_manual_drive_gate import ROBOT, StepClock, collect, terminals
from test_manual_drive_nav2 import binding
from test_mapview_target_state import OLD_READER, ListSink
from test_navigate_handover import ACTION, assign, payload, turned

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import handover as ho
from fleet_adapter_ros2 import manual_drive as md
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import emitter

MANUAL = "manual-a"


def nav_drive() -> md.ManualDrive:
    doc = {"operation": ho.OPERATION, "x": 2.0, "y": 1.0, "frame": "map", "radius_m": 0.3}
    return md.ManualDrive(ho.parse_payload(doc), started_at=0.0, phase=md.NAVIGATING)


def man_drive(**kw) -> md.ManualDrive:
    doc = {"operation": md.OPERATION, "x": 1.0, "y": 0.0, "frame": "map", "radius_m": 0.25, **kw}
    return md.ManualDrive(md.parse_payload(doc), started_at=0.0)


def gates(log) -> list[tuple[str, str]]:
    return [(e["gate"], e["action"]) for e in log]


# ── the gate's own record ────────────────────────────────────────────────

def test_the_gate_records_armed_open_and_closed_with_the_action_and_reason() -> None:
    g = md.TeleopGate(clock=lambda: 10.0, wall=lambda: 1000.5)
    g.open("nav-a", nav_drive(), 60.0)
    assert g.hand_over("nav-a", 1.0, "no valid path")
    g.close("a stop")
    log = g.changes()
    assert gates(log) == [("armed", "nav-a"), ("open", "nav-a"), ("closed", "nav-a")]
    assert "nav2 drives" in log[0]["reason"]
    assert "no valid path" in log[1]["reason"]
    assert log[2]["reason"] == "a stop"
    assert [e["seq"] for e in log] == [1, 2, 3] and all(e["stamp"] == 1000.5 for e in log)
    assert g.changes(after=2) == log[2:]


def test_a_manual_drive_opens_to_the_operator_at_once_and_closes_with_the_drives_end() -> None:
    g = md.TeleopGate(clock=lambda: 10.0)
    d = man_drive(settle_s=1.0)
    g.open(MANUAL, d, 60.0)
    for t in (1.0, 2.0, 3.5):                         # inside, still from 2.0: released at 3.5
        d.on_pose(t, 1.0, 0.0, "map")
    assert d.phase == md.SUCCEEDED
    g.close()
    log = g.changes()
    assert gates(log) == [("open", MANUAL), ("closed", MANUAL)]
    assert "operator released" in log[1]["reason"]


def test_closing_a_closed_gate_records_nothing() -> None:
    g = md.TeleopGate()
    g.close("nothing open")
    assert g.changes() == []


def test_the_nav2_binding_records_a_stop_closing_the_gate() -> None:
    b, _ = binding()
    b.open_teleop(ROBOT, "nav-a", nav_drive(), lease_s=60)
    b.command_stop(ROBOT)
    log = b.teleop_log(ROBOT)
    assert gates(log) == [("armed", "nav-a"), ("closed", "nav-a")]
    assert "stop" in log[1]["reason"]


# ── the drive's in-circle / still judgement ──────────────────────────────

def test_each_pose_judged_leaves_the_circle_tick() -> None:
    d = man_drive(settle_s=1.0)
    d.on_pose(1.0, 3.0, 0.0, "map")
    t = d.circle
    assert t["inside"] is False and abs(t["distance_m"] - 2.0) < 1e-9
    assert t["radius_m"] == 0.25 and t["still_s"] == 0.0 and t["settle_s"] == 1.0
    d.on_pose(2.0, 1.1, 0.0, "map")
    d.on_pose(2.5, 1.1, 0.0, "map")
    d.on_pose(3.0, 1.1, 0.0, "map")
    t = d.circle
    assert t["inside"] is True and abs(t["still_s"] - 0.5) < 1e-9


# ── the feed carries it all on target.baton ──────────────────────────────

def rig(robot_cls=BrakingRobot, **kw):
    robot = robot_cls(**kw)
    robot.spawn(ROBOT)
    sink = ListSink()
    mv = emitter.MapviewEmitter(sink, ROBOT, simulated=True, session="s1")
    a = Adapter(pb, None, robot, ROBOT, job_clock=StepClock(), mapview=mv)
    a._handover_rest_s = 3600.0
    return a, robot, sink


def batons(sink: ListSink, label: str) -> list[dict]:
    return [m["target"]["baton"] for m in sink.snapshot()
            if m["kind"] == "Overlays" and (m.get("target") or {}).get("label") == label
            and "baton" in m["target"]]


def wait_for(sink: ListSink, label: str, until, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        bs = batons(sink, label)
        if bs and until(bs[-1]):
            return bs[-1]
        assert time.monotonic() < deadline, bs[-1:] or "no baton"
        time.sleep(0.01)


def test_a_handover_turn_shows_armed_then_open_then_closed_and_its_start_and_end() -> None:
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, turned)
    b = wait_for(sink, ACTION, lambda b: b["gate"] == "open")
    assert b["operation"] == ho.OPERATION and b["action"] == ACTION
    assert b["phase"] == md.RUNNING                       # the drive's own phase
    assert [e["gate"] for e in b["gateLog"]] == ["armed", "open"]
    assert "aborted the goal" in b["gateLog"][1]["reason"]
    assert b["turn"]["startedAt"] > 0 and b["turn"]["endedAt"] is None
    robot.place(2.0, 1.0)                                  # the operator drives into the circle
    b = wait_for(sink, ACTION, lambda b: (b.get("circle") or {}).get("inside") is True)
    assert b["circle"]["radius_m"] == 0.3 and b["circle"]["distance_m"] <= 0.3
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    b = wait_for(sink, ACTION, lambda b: b["turn"]["endedAt"] is not None)
    assert b["turn"]["outcome"] == "succeeded"
    assert "operator released" in b["turn"]["message"]
    assert [e["gate"] for e in b["gateLog"]] == ["armed", "open", "closed"]
    assert b["gate"] == "closed"
    assert b["turn"]["endedAt"] >= b["gateLog"][-1]["stamp"] >= b["turn"]["startedAt"]


def test_a_fleet_turn_nav2_reaches_shows_armed_then_closed() -> None:
    a, _, sink = rig()
    res, msgs = assign(a, payload())
    assert res.accepted
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    first = batons(sink, ACTION)[0]
    assert first["phase"] == md.NAVIGATING
    b = wait_for(sink, ACTION, lambda b: b["turn"]["endedAt"] is not None)
    assert b["phase"] == md.SUCCEEDED
    assert [e["gate"] for e in b["gateLog"]] == ["armed", "closed"]
    assert b["turn"]["outcome"] == "succeeded" and "nav2 reached" in b["turn"]["message"]


def test_a_manual_drive_has_a_feed_of_its_own() -> None:
    a, robot, sink = rig()
    body = json.dumps({"operation": md.OPERATION, "x": 1.0, "y": 0.0, "frame": "map",
                       "radius_m": 0.25, "settle_s": 1.0}).encode()
    res, msgs = assign(a, body, action=MANUAL)
    assert res.accepted
    b = wait_for(sink, MANUAL, lambda b: b["gate"] == "open")
    assert b["operation"] == md.OPERATION
    # A person drives a manual-drive: a guided source, so no viewer calls it autonomous.
    sources = {m["source"] for m in sink.snapshot() if m["kind"] == "SessionStatus"}
    assert sources == {f"guided:{md.OPERATION}"}
    robot.place(0.9, 0.05)
    collect(a, msgs, lambda ms: terminals(ms, MANUAL))
    b = wait_for(sink, MANUAL, lambda b: b["turn"]["endedAt"] is not None)
    assert [e["gate"] for e in b["gateLog"]] == ["open", "closed"]
    assert b["turn"]["outcome"] == "succeeded"
    targets = [m["target"] for m in sink.snapshot() if m["kind"] == "Overlays"
               and (m.get("target") or {}).get("label") == MANUAL]
    assert targets[0]["state"] == "operator" and targets[-1]["state"] == "reached"


def test_every_message_checks_and_an_old_reader_still_reads_the_target() -> None:
    a, robot, sink = rig()
    robot.decline_next("the navigation server aborted the goal")
    _, msgs = assign(a, payload(last=True))
    collect(a, msgs, turned)
    robot.place(2.0, 1.0)
    collect(a, msgs, lambda ms: terminals(ms, ACTION))
    wait_for(sink, ACTION, lambda b: b["turn"]["endedAt"] is not None)
    spec = importlib.util.spec_from_file_location("old_reader", OLD_READER)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for m in sink.snapshot():
        c.check(m)
        old.check(json.loads(c.dumps(m)))
