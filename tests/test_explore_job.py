# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""`Custom` explore-floor and `Charge`, as jobs, against the simulated exploration stack.

One explore-floor segment is: the bound resolved; the speed cap read live and proven within the
payload's; SLAM started (or the saved graph loaded); the keep-out mask turned into this boot's
frame, loaded, the keep-out servers brought up and the costmap reset once; the bound proven in
the live costmap -- and only then the explorer, the first motion. At the battery threshold, the
time box or the explorer's completion: stop the explorer, wait for it to pause plus the settle,
return to the mark, save the map, the pose graph and the boot frame's delta at a path keyed by
the action id, and end SUCCEEDED with the coverage, the exhausted areas, the map reference and
its digest. Every read and bring-up is bounded: a dropped reply is tried again, a wrong answer
is not.

`Charge` waits, and reports the battery while it does, until the battery reads full AND the
cable is out (the IMU streaming again). It never moves the robot.

The clock is fake: waits advance it and never sleep, so a segment runs in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading

import pytest

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.t += seconds
        return event.is_set()


class Emits:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, int, str]] = []

    def __call__(self, state: str, message: str, pct: int = 0, end_cause: str = "") -> None:
        # The job reports the adapter's ActionState names; the short name reads better here.
        self.updates.append((state.removeprefix("ACTION_STATE_"), message, pct, end_cause))

    @property
    def last(self):
        return self.updates[-1]

    def messages(self) -> list[str]:
        return [u[1] for u in self.updates]


DOC = {"operation": "explore-floor", "bound_ref": "sim-room", "max_speed_mps": 0.2}


def _job(tmp_path, stack=None, doc=None, action_id="explore-first", **kw):
    # One clock for the job and the stack: a stack on a clock nobody advances never finishes.
    clock = kw.pop("clock", None) or (stack.clock if stack is not None else FakeClock())
    stack = stack or SimulatedExploreStack(clock=clock)
    emits = Emits()
    job = explore.ExploreJob(
        action_id=action_id, doc={**DOC, **(doc or {})}, stack=stack,
        limits=stack.battery_limits(), artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"),
        namespace="lab", clock=clock, emit=emits)
    return job, stack, emits, clock


# ── validation: what the catalog declares is what is enforced ───────────────

LIMITS = explore.BatteryLimits(hard_mv=7750, full_mv=8300, return_mv=7835)


@pytest.mark.parametrize("doc, words", [
    ({"operation": "map-floor", "bound_ref": "r", "max_speed_mps": 0.2}, ["operation", "map-floor"]),
    ({"bound_ref": "r", "max_speed_mps": 0.2}, ["no operation"]),
    ({"operation": "explore-floor", "max_speed_mps": 0.2}, ["no bound_ref"]),
    ({"operation": "explore-floor", "bound_ref": "r"}, ["no max_speed_mps"]),
    ({"operation": "explore-floor", "bound_ref": "r", "max_speed_mps": 0.21}, ["above", "0.2"]),
    ({"operation": "explore-floor", "bound_ref": "r", "max_speed_mps": 3}, ["above"]),
    ({"operation": "explore-floor", "bound_ref": "r", "max_speed_mps": "0.1"}, ["number"]),
    ({"operation": "explore-floor", "bound_ref": "r", "max_speed_mps": float("nan")},
     ["number"]),
    ({"operation": "explore-floor", "bound_ref": "", "max_speed_mps": 0.2}, ["bound_ref"]),
    ({**DOC, "resume": "yes"}, ["resume", "true or false"]),
    ({**DOC, "exhausted_after_recoveries": 2.5}, ["whole number"]),
    ({**DOC, "exhausted_after_recoveries": 0}, ["below"]),
    ({**DOC, "time_box_s": 0}, ["below"]),
    ({**DOC, "battery_return_mv": 7700}, ["battery_return_mv", "hard stop", "7750"]),
    ({**DOC, "battery_return_mv": 8400}, ["battery_return_mv", "full", "8300"]),
])
def test_an_explore_payload_outside_the_catalog_is_refused_with_the_reason(doc, words) -> None:
    why = explore.refusal("Custom", doc, LIMITS)
    assert why
    for w in words:
        assert w in why


def test_no_payload_can_ask_for_more_than_the_speed_cap() -> None:
    cap = next(p for p in explore.EXPLORE_PARAMS if p.name == "max_speed_mps")
    assert cap.max == explore.MAX_SPEED_MPS == 0.20 and cap.required
    for v in (0.2000001, 0.25, 1, 1e9, float("inf")):
        assert explore.refusal("Custom", {**DOC, "max_speed_mps": v}, LIMITS)
    assert explore.refusal("Custom", {**DOC, "max_speed_mps": 0.2}, LIMITS) == ""


def test_a_complete_payload_is_servable_and_unknown_keys_are_left_alone() -> None:
    # The conformance probe carries a fixture target beside the declared parameters.
    assert explore.refusal("Custom", {**DOC, "x": 1.0, "y": 2.0}, LIMITS) == ""
    assert explore.refusal("Custom", {**DOC, "resume": True,
                                      "keepouts_ref": "floor-maps/lab/k@sha256:" + "a" * 64,
                                      "exhausted_after_recoveries": 3, "time_box_s": 600,
                                      "battery_return_mv": 7900}, LIMITS) == ""


def test_with_no_return_threshold_anywhere_explore_is_refused() -> None:
    limits = explore.BatteryLimits(hard_mv=7750, full_mv=8300, return_mv=None)
    assert "battery_return_mv" in explore.refusal("Custom", DOC, limits)
    assert explore.refusal("Custom", {**DOC, "battery_return_mv": 7900}, limits) == ""


def test_charge_takes_an_optional_full_threshold_above_the_hard_stop() -> None:
    assert explore.refusal("Charge", {}, LIMITS) == ""
    assert explore.refusal("Charge", {"full_mv": 8200}, LIMITS) == ""
    assert "hard stop" in explore.refusal("Charge", {"full_mv": 7700}, LIMITS)


# ── one segment, end to end ─────────────────────────────────────────────────

def test_a_fresh_segment_explores_returns_saves_and_reports_the_map(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    state, message, pct, _ = emits.last
    assert state == "SUCCEEDED", emits.updates
    ref = re.search(r"map-ref=(\S+)", message).group(1)
    digest = re.search(r"map-sha256=([0-9a-f]{64})", message).group(1)
    assert ref == "floor-maps/lab/explore-first"
    saved = tmp_path / "lab" / "explore-first"
    manifest = (saved / "MANIFEST.sha256").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == digest
    listed = {ln.split("  ")[1] for ln in manifest.decode().splitlines()}
    assert listed == {p.name for p in saved.iterdir()} - {"MANIFEST.sha256"}
    assert {"map.yaml", "map.pgm", "frame-delta.json", "segment.json"} <= listed
    assert any(n.endswith(".posegraph") for n in listed)       # the pose graph is saved too
    for line in manifest.decode().splitlines():
        want, name = line.split("  ")
        assert hashlib.sha256((saved / name).read_bytes()).hexdigest() == want
    assert "frontier: none-left" in message and "exhausted: none" in message
    covered = re.search(r"covered ([\d.]+) of ([\d.]+) m2 \((\d+)%\)", message)
    assert int(covered.group(3)) == pct > 0


def test_the_phases_are_named_in_order(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    text = " | ".join(emits.messages())
    order = ["speed cap", "SLAM", "bound proven", "exploring", "returning", "saving",
             "segment done"]
    positions = [text.find(w) for w in order]
    assert all(p >= 0 for p in positions), text
    assert positions == sorted(positions), text


def test_progress_is_the_covered_area_against_the_bounds(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    running = [u[2] for u in emits.updates if u[0] == "RUNNING"]
    assert running == sorted(running) and running[-1] > 0
    assert emits.last[2] <= 100


def test_no_motion_before_the_bound_is_proven(tmp_path) -> None:
    job, stack, _, _ = _job(tmp_path)
    job.run()
    first_motion = stack.log.index("start_explorer")
    for step in ("read_param", "start_slam", "load_keepout_mask", "activate_keepout",
                 "reset_costmap", "read_costmap"):
        assert stack.log.index(step) < first_motion, (step, stack.log)


# ── refused before any motion, with the reason ──────────────────────────────

def _assert_no_motion(stack) -> None:
    assert "start_explorer" not in stack.log and "go_to_mark" not in stack.log
    assert not stack.moving()


def test_an_unknown_bound_ends_failed_before_anything_moves(tmp_path) -> None:
    job, stack, emits, _ = _job(tmp_path, doc={"bound_ref": "no-such-room"})
    job.run()
    state, message, _, cause = emits.last
    assert state == "FAILED" and cause == "ACTION_END_CAUSE_EXECUTION_FAILURE"
    assert "no-such-room" in message and "no motion" in message
    _assert_no_motion(stack)


def test_a_bound_not_in_force_ends_failed_with_the_grading(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), keepout_leak=True, keepout_leak_fixed_by=None)
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "bound not in force" in message and "not lethal" in message
    assert "no motion" in message
    _assert_no_motion(stack)


def test_a_keep_out_that_needs_one_restart_is_restarted_once(tmp_path) -> None:
    # The lifecycle reply was lost and the filter never came up; one restart brought it.
    stack = SimulatedExploreStack(clock=FakeClock(), drop={"activate_keepout": 99})
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    assert emits.last[0] == "SUCCEEDED", emits.updates
    assert stack.log.count("restart_keepout") == 1


def test_a_live_speed_cap_above_the_payloads_ends_failed_and_is_not_read_again(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), params={
        ("/controller_server", "FollowPath.max_vel_x"): "0.35"})
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "MISMATCH" in message and "0.35" in message
    assert stack.log.count("read_param") == 1
    _assert_no_motion(stack)


def test_a_cap_below_the_payloads_passes_and_a_lower_payload_is_not_raised(tmp_path) -> None:
    # The adapter never sets the cap; it proves the live one is within what was asked.
    job, _, emits, _ = _job(tmp_path, doc={"max_speed_mps": 0.1})
    job.run()
    assert emits.last[0] == "FAILED" and "0.2" in emits.last[1]


def test_a_dropped_speed_cap_reply_is_read_again_and_the_segment_goes_on(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), drop={"read_param": 2})
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    assert emits.last[0] == "SUCCEEDED"
    assert stack.log.count("read_param") >= 3


def test_a_speed_cap_never_answered_ends_failed_after_the_ruled_retries(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), drop={"read_param": 99})
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "NO ANSWER after 4 attempt(s)" in message
    _assert_no_motion(stack)


def test_a_slam_start_never_answered_ends_failed(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), drop={"start_slam": 99})
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    assert emits.last[0] == "FAILED" and "SLAM" in emits.last[1]
    _assert_no_motion(stack)


# ── the boot frame ──────────────────────────────────────────────────────────

def test_the_mask_is_turned_by_this_boots_delta_and_the_delta_is_saved(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=-73.85)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    assert emits.last[0] == "SUCCEEDED"
    assert stack.loaded_mask_delta_deg == pytest.approx(-73.85 - stack.bound_theta0_deg)
    delta = json.loads((tmp_path / "lab" / "explore-first" / "frame-delta.json").read_text())
    assert delta["dyaw_rad"] == pytest.approx(
        (-73.85 - stack.bound_theta0_deg) * 3.141592653589793 / 180)
    assert (delta["dx_m"], delta["dy_m"]) == (0.0, 0.0)


# ── the end of a segment ────────────────────────────────────────────────────

def test_the_battery_threshold_stops_the_explorer_then_waits_then_returns(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, start_mv=7900,
                                  drain_mv_per_s=1.0)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"battery_return_mv": 7835})
    job.run()
    assert emits.last[0] == "SUCCEEDED" and "frontier: remaining" in emits.last[1]
    i_stop, i_mark = stack.log.index("stop_explorer"), stack.log.index("go_to_mark")
    assert i_stop < i_mark
    assert stack.mark_sent_at - stack.explorer_stopped_at >= 2.0


def test_the_time_box_ends_a_segment_like_the_battery(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock, doc={"time_box_s": 30})
    job.run()
    assert emits.last[0] == "SUCCEEDED" and "frontier: remaining" in emits.last[1]
    assert any("time box" in m for m in emits.messages())


def test_the_explorers_completion_does_not_stop_it_again(tmp_path) -> None:
    job, stack, _, _ = _job(tmp_path)
    job.run()
    assert "stop_explorer" not in stack.log


def test_the_hard_stop_ends_the_segment_failed_where_it_stands(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, drain_mv_per_s=0.0)
    # The pack falls below the hard stop once the robot moves: before motion a reading that low
    # is refused outright (see the battery tests below).
    stack.on_explorer_started = lambda: stack.set_battery(stack.battery_limits().hard_mv - 50)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "hard stop" in message and "power off" in message
    assert "go_to_mark" not in stack.log and not stack.moving()


def test_a_failed_return_saves_where_it_stands_and_ends_failed(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), mark_result="ABORTED")
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "return to the mark failed (ABORTED)" in message
    assert "map-ref=floor-maps/lab/explore-first" in message


# ── exhausted areas ─────────────────────────────────────────────────────────

def test_an_area_with_n_recoveries_is_skipped_named_and_reported(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, recoveries=[(5, 1.2, 0.3), (6, 1.4, 0.6),
                                                           (7, 1.9, 0.1), (8, -2.0, 0.5)])
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"exhausted_after_recoveries": 3})
    job.run()
    name = "area x 1.0..2.0 m, y 0.0..1.0 m"
    assert stack.skipped == [name]
    assert f"exhausted: {name}" in emits.last[1]
    assert any("exhausted" in m and name in m for m in emits.messages()[:-1])
    seg = json.loads((tmp_path / "lab" / "explore-first" / "segment.json").read_text())
    assert seg["exhausted"] == [name]


def _exhausting_stack(clock, boot_yaw_deg=0.0):
    # Three recoveries in one square metre, and an explorer that would run for hours: a segment
    # keeps the area out and goes on, so each one here is ended by its time box.
    return SimulatedExploreStack(clock=clock, explore_seconds=10_000, boot_yaw_deg=boot_yaw_deg,
                                 recoveries=[(5, 1.2, 0.3), (6, 1.4, 0.6), (7, 1.9, 0.1)])


AREA = "area x 1.0..2.0 m, y 0.0..1.0 m"


def test_the_next_segment_keeps_an_exhausted_area_out_and_says_so(tmp_path) -> None:
    clock = FakeClock()
    stack = _exhausting_stack(clock, boot_yaw_deg=30.0)
    # The area is kept out of the running segment and exploring goes on; a time box ends it.
    first, _, e1, _ = _job(tmp_path, stack=stack, clock=clock,
                           doc={"exhausted_after_recoveries": 3, "time_box_s": 20})
    first.run()
    assert e1.last[0] == "SUCCEEDED", e1.updates
    stack.reboot(boot_yaw_deg=-60.0)
    second, _, e2, _ = _job(tmp_path, stack=stack, clock=clock, action_id="explore-resume",
                            doc={"resume": True, "time_box_s": 20})
    second.run()
    assert e2.last[0] == "SUCCEEDED", e2.updates
    # The resumed graph is in the first boot's frame, where the area was named.
    assert stack.loaded_mask_keep_at(1.5, 0.5) is True
    assert stack.loaded_mask_keep_at(0.5, 0.5) is False
    assert any("kept out" in m and AREA in m for m in e2.messages()[:-1])
    assert f"kept out: {AREA}" in e2.last[1]
    # More restrictive, never less: a segment with nothing new exhausted carries it forward.
    seg = json.loads((tmp_path / "lab" / "explore-resume" / "segment.json").read_text())
    assert [a["name"] for a in seg["exhausted_areas"]] == [AREA]
    stack.reboot(boot_yaw_deg=10.0)
    third, _, e3, _ = _job(tmp_path, stack=stack, clock=clock, action_id="explore-resume-2",
                           doc={"resume": True, "time_box_s": 20})
    third.run()
    assert e3.last[0] == "SUCCEEDED", e3.updates
    assert stack.loaded_mask_keep_at(1.5, 0.5) is True


def test_recoveries_beside_the_mark_never_exhaust_its_area(tmp_path) -> None:
    # On a real robot: it stopped 0.75 m from the mark and the area exhausted was the square with
    # the mark as its corner, so every resume would start against its own keep-out.
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, recoveries=[
        (5, 0.63, -0.41), (6, 0.65, -0.44), (7, 0.60, -0.45), (8, 0.2, 0.1)])
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock,
                                doc={"exhausted_after_recoveries": 3, "time_box_s": 20})
    job.run()
    assert emits.last[0] == "SUCCEEDED", emits.updates
    assert stack.skipped == []
    assert "exhausted: none" in emits.last[1]
    assert any("start mark" in m and "area x 0.0..1.0 m, y -1.0..0.0 m" in m
               for m in emits.messages())
    seg = json.loads((tmp_path / "lab" / "explore-first" / "segment.json").read_text())
    assert seg["exhausted"] == [] and seg["exhausted_areas"] == []


@pytest.mark.parametrize("polygon", [
    [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],      # the mark inside
    [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, -1.0]],        # the mark as a corner, as a robot wrote it
    [[0.2, 0.2], [1.2, 0.2], [1.2, 1.2], [0.2, 1.2]],          # 0.28 m from the mark
])
def test_an_earlier_area_at_the_mark_is_neither_kept_out_nor_carried_forward(
        tmp_path, polygon) -> None:
    clock = FakeClock()
    stack = _exhausting_stack(clock)
    # The area is kept out of the running segment and exploring goes on; a time box ends it.
    first, _, e1, _ = _job(tmp_path, stack=stack, clock=clock,
                           doc={"exhausted_after_recoveries": 3, "time_box_s": 20})
    first.run()
    assert e1.last[0] == "SUCCEEDED"
    # Move the saved area to the mark, as an adapter that counted raw recoveries wrote it.
    path = tmp_path / "lab" / "explore-first" / "segment.json"
    seg = json.loads(path.read_text())
    seg["exhausted_areas"][0]["polygon"] = polygon
    path.write_text(json.dumps(seg))
    stack.reboot()
    second, _, e2, _ = _job(tmp_path, stack=stack, clock=clock, action_id="explore-resume",
                            doc={"resume": True, "time_box_s": 20})
    second.run()
    assert e2.last[0] == "SUCCEEDED", e2.updates
    assert any("not kept out" in m and AREA in m and "start mark" in m for m in e2.messages())
    assert f"kept out: {AREA}" not in e2.last[1]
    assert stack.loaded_mask_keep_at(0.3, -0.3) is False
    # The simulated explorer has its three recoveries at x 1..2 again: that area is exhausted
    # afresh, and nothing at the mark is written.
    resumed = json.loads((tmp_path / "lab" / "explore-resume" / "segment.json").read_text())
    assert [a["reason"] for a in resumed["exhausted_areas"]] == [
        "3 recovery episodes in segment explore-resume"]
    for a in resumed["exhausted_areas"]:
        assert steps.distance_to_polygon(0.0, 0.0, a["polygon"]) >= steps.MARK_CLEAR_M


# ── resume ──────────────────────────────────────────────────────────────────

def test_resume_loads_the_saved_graph_of_the_same_bound_first(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000, boot_yaw_deg=30.0)
    first, _, e1, _ = _job(tmp_path, stack=stack, clock=clock, doc={"time_box_s": 20})
    first.run()
    assert e1.last[0] == "SUCCEEDED"
    stack.reboot(boot_yaw_deg=-60.0)
    second, _, e2, _ = _job(tmp_path, stack=stack, clock=clock, action_id="explore-resume",
                            doc={"resume": True, "time_box_s": 20})
    second.run()
    assert e2.last[0] == "SUCCEEDED", e2.updates
    assert stack.slam_started_with == "sim-graph:lab/explore-first"
    # The graph's frame is the first boot's, so the first boot's delta is kept.
    assert stack.loaded_mask_delta_deg == pytest.approx(30.0 - stack.bound_theta0_deg)
    assert "SLAM" in " ".join(e2.messages()) and "explore-first" in " ".join(e2.messages())


def test_resume_with_no_saved_segment_ends_failed_before_anything_moves(tmp_path) -> None:
    job, stack, emits, _ = _job(tmp_path, doc={"resume": True})
    job.run()
    assert emits.last[0] == "FAILED" and "no saved segment" in emits.last[1]
    _assert_no_motion(stack)


# ── stops ───────────────────────────────────────────────────────────────────

def test_a_halt_ends_the_job_without_a_report_and_stops_the_explorer(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)
    stack.on_explorer_started = stack.halt           # a stop arrives once it moves
    job.run()
    assert all(u[0] == "RUNNING" for u in emits.updates)
    assert not stack.moving() and job.halted.is_set()


def test_pause_holds_and_resume_restarts_the_explorer(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=60)
    job, stack, emits, _ = _job(tmp_path, stack=stack, clock=clock)

    def pause_then_resume() -> None:
        stack.on_explorer_started = None             # once
        job.pause()
        assert not stack.moving()
        job.resume()
    stack.on_explorer_started = pause_then_resume
    job.run()
    assert emits.last[0] == "SUCCEEDED"
    assert stack.log.count("start_explorer") == 2


# ── the artifact path ───────────────────────────────────────────────────────

@pytest.mark.parametrize("action_id, ns, want", [
    ("explore-first", "lab", ("lab", "explore-first")),
    ("other/explore-first", "lab", ("other", "explore-first")),
    ("../etc", "lab", None), ("Explore", "lab", None), ("a/b/c", "lab", None),
    ("", "lab", None),
])
def test_the_path_is_the_action_id_and_nothing_else(action_id, ns, want) -> None:
    assert explore.split_action_id(action_id, ns) == want


def test_an_action_id_that_is_no_path_ends_failed_before_anything_moves(tmp_path) -> None:
    job, stack, emits, _ = _job(tmp_path, action_id="../escape")
    job.run()
    assert emits.last[0] == "FAILED" and "action id" in emits.last[1]
    _assert_no_motion(stack)


# ── Charge ──────────────────────────────────────────────────────────────────

def _charge(stack, clock, doc=None):
    emits = Emits()
    job = explore.ChargeJob(action_id="charge", doc=doc or {}, stack=stack,
                            limits=stack.battery_limits(), clock=clock, emit=emits)
    return job, emits


def test_charge_waits_reports_the_battery_and_ends_when_full_and_the_cable_is_out() -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, start_mv=7900)
    job, emits = _charge(stack, clock)
    job.run()
    state, message, pct, _ = emits.last
    assert state == "SUCCEEDED" and pct == 100
    assert "full" in message and "cable out" in message
    running = [u for u in emits.updates if u[0] == "RUNNING"]
    assert running[0][1].startswith("waiting for charge") and "mV" in running[0][1]
    assert any("waiting for the cable" in u[1] for u in running)


def test_charge_never_moves_the_robot() -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, start_mv=7900)
    job, _ = _charge(stack, clock)
    job.run()
    for motion in ("start_explorer", "go_to_mark", "start_slam"):
        assert motion not in stack.log
    assert not stack.moving()


def test_full_with_the_cable_still_in_is_not_done() -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, unplug_after_full_s=None)
    stack.set_battery(stack.battery_limits().full_mv + 50)
    stack.plug()                                      # full, and the cable still in
    job, emits = _charge(stack, clock)
    halt_at = 120.0
    stack.on_tick = lambda: job.halted.set() if clock.t >= halt_at else None
    job.run()
    assert all(u[0] == "RUNNING" for u in emits.updates)
    assert "IMU silent" in emits.updates[-1][1]


def test_no_battery_reading_is_said_and_never_taken_for_empty_or_full() -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, battery_reading=False)
    job, emits = _charge(stack, clock)
    stack.on_tick = lambda: job.halted.set() if clock.t >= 30 else None
    job.run()
    assert emits.updates and all("no battery reading" in u[1] for u in emits.updates)


# ── ALIGNED, before anything moves ──────────────────────────────────────────

def test_a_boot_off_the_mark_fails_aligned_with_the_score_and_no_motion(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), misaligned_m=0.6)
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "not aligned" in message and "want >= 80%" in message
    assert "no motion" in message and "load_keepout_mask" not in stack.log
    _assert_no_motion(stack)


def test_a_bound_with_no_saved_map_to_align_to_is_refused(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock(), bound_has_ref_map=False)
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    assert emits.last[0] == "FAILED" and "no saved map" in emits.last[1]
    _assert_no_motion(stack)


def test_a_resume_is_aligned_against_the_saved_segments_own_map(tmp_path) -> None:
    clock = FakeClock()
    stack = SimulatedExploreStack(clock=clock, explore_seconds=10_000)
    first, _, _, _ = _job(tmp_path, stack=stack, clock=clock, doc={"time_box_s": 20})
    first.run()
    (tmp_path / "lab" / "explore-first" / "map.pgm").unlink()
    stack.reboot()
    second, _, e2, _ = _job(tmp_path, stack=stack, clock=clock, action_id="explore-resume",
                            doc={"resume": True, "time_box_s": 20})
    second.run()
    assert e2.last[0] == "FAILED" and "map.pgm" in e2.last[1] and "no motion" in e2.last[1]


def test_aligned_is_reported_before_the_bound(tmp_path) -> None:
    job, _, emits, _ = _job(tmp_path)
    job.run()
    text = " | ".join(emits.messages())
    assert 0 <= text.find("aligned") < text.find("bound proven"), text


# ── the keep-out set: one source of geometry, proven by its digest ──────────

LEDGE = {"name": "floor-room-a-keepout-1", "reason": "operator keep-out, a ledge",
         "polygon": [{"x": 1.0, "y": 0.5}, {"x": 1.6, "y": 0.5}, {"x": 1.6, "y": 1.1},
                     {"x": 1.0, "y": 1.1}]}


def _keepout_set(tmp_path, keepouts=(LEDGE,), theta0=None, name="zones-applied",
                 extra=None) -> str:
    doc = {"theta0_deg": explore_sim_theta0() if theta0 is None else theta0,
           "source_map": "lab/explore-first", "zone_set_sha256": "ab" * 32,
           "keepouts": list(keepouts)}
    files = {"keepouts.json": json.dumps(doc).encode(), **(extra or {})}
    ref, digest = explore.ArtifactDir(str(tmp_path), "floor-maps").save("lab", name, files)
    return f"{ref}@sha256:{digest}"


def explore_sim_theta0() -> float:
    return SimulatedExploreStack(clock=FakeClock()).bound_theta0_deg


def _arts(tmp_path):
    return explore.ArtifactDir(str(tmp_path), "floor-maps")


def test_a_keep_out_set_is_read_only_when_its_digest_matches(tmp_path) -> None:
    ref = _keepout_set(tmp_path)
    kset = explore.load_keepout_set(_arts(tmp_path), ref)
    assert [k.name for k in kset.keepouts] == ["floor-room-a-keepout-1"]
    assert kset.zone_set_sha256 == "ab" * 32


@pytest.mark.parametrize("mutate, words", [
    (lambda d, ref: ref.replace("zones-applied", "zones-gone"), ["missing"]),
    (lambda d, ref: ref[:-64] + "0" * 64, ["MANIFEST.sha256 hashes to", "not the"]),
    (lambda d, ref: (d / "keepouts.json").write_text('{"keepouts": []}') and ref,
     ["keepouts.json hashes to"]),
    (lambda d, ref: (d / "stray.txt").write_text("x") and ref, ["not listed", "stray.txt"]),
    (lambda d, ref: ref.replace("floor-maps/", "other-store/"), ["store", "other-store"]),
])
def test_a_keep_out_set_that_is_not_the_one_referenced_is_refused_by_name(
        tmp_path, mutate, words) -> None:
    ref = mutate(tmp_path / "lab" / "zones-applied", _keepout_set(tmp_path))
    with pytest.raises(LookupError) as exc:
        explore.load_keepout_set(_arts(tmp_path), ref)
    for w in words:
        assert w in str(exc.value)


@pytest.mark.parametrize("ref", ["zones-applied", "floor-maps/lab/zones-applied",
                                 "floor-maps/lab/zones-applied@sha256:xyz",
                                 "floor-maps/lab/Zones@sha256:" + "a" * 64])
def test_a_keepouts_ref_without_a_digest_is_refused_at_admission(ref) -> None:
    why = explore.refusal("Custom", {**DOC, "keepouts_ref": ref}, LIMITS)
    assert "keepouts_ref" in why and "@sha256:" in why


def test_a_segment_with_a_missing_keep_out_set_ends_failed_before_motion(tmp_path) -> None:
    ref = "floor-maps/lab/zones-applied@sha256:" + "c" * 64
    job, stack, emits, _ = _job(tmp_path, doc={"keepouts_ref": ref})
    job.run()
    assert emits.last[0] == "FAILED" and "missing" in emits.last[1] and "no motion" in emits.last[1]
    _assert_no_motion(stack)


def test_the_referenced_keep_outs_are_drawn_into_the_loaded_mask(tmp_path) -> None:
    ref = _keepout_set(tmp_path)
    job, stack, emits, _ = _job(tmp_path, doc={"keepouts_ref": ref, "time_box_s": 20})
    job.run()
    assert emits.last[0] == "SUCCEEDED", emits.updates
    # The set's frame is the bound's here (same theta0) and this boot's delta is 0 - theta0.
    d = math.radians(stack.loaded_mask_delta_deg)
    x, y = 1.3, 0.8
    lx, ly = math.cos(d) * x - math.sin(d) * y, math.sin(d) * x + math.cos(d) * y
    assert stack.loaded_mask_keep_at(lx, ly) is True
    assert stack.loaded_mask_keep_at(0.0, 0.0) is False
    assert "floor-room-a-keepout-1" in " ".join(emits.messages())


def test_a_mark_inside_a_keep_out_is_refused_naming_the_zone(tmp_path) -> None:
    around_mark = {**LEDGE, "polygon": [{"x": -0.3, "y": -0.3}, {"x": 0.3, "y": -0.3},
                                        {"x": 0.3, "y": 0.3}, {"x": -0.3, "y": 0.3}]}
    ref = _keepout_set(tmp_path, keepouts=(around_mark,))
    job, stack, emits, _ = _job(tmp_path, doc={"keepouts_ref": ref})
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "floor-room-a-keepout-1" in message
    assert "operator keep-out, a ledge" in message and "no motion" in message
    _assert_no_motion(stack)


def test_a_goal_inside_a_keep_out_is_named_and_outside_is_not(tmp_path) -> None:
    kset = explore.load_keepout_set(_arts(tmp_path), _keepout_set(tmp_path))
    why = explore.goal_refusal(1.3, 0.8, kset)
    assert "floor-room-a-keepout-1" in why and "a ledge" in why
    assert explore.goal_refusal(3.0, 3.0, kset) == ""


def test_a_segment_ends_its_robot_side_processes_however_it_ends(tmp_path) -> None:
    for doc in ({}, {"bound_ref": "no-such-room"}):
        job, stack, _, _ = _job(tmp_path, doc=doc)
        job.run()
        assert stack.log[-1] == "end_segment", (doc, stack.log[-3:])
