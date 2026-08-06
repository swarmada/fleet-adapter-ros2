# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Intermediate task progress (beyond C1-C8): the adapter streams a RUNNING
ActionStatusUpdate carrying progressPct derived from NavigateToPose feedback distance,
throttled to ≥5-point advances, so the control-plane FleetTask.status.progressPct
climbs during a task. Terminal 100% is the completion-stream's job."""

from __future__ import annotations

from fleet_adapter_ros2 import task
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding
from fleet_adapter_ros2.robot import SimulatedRobot


# ── binding: distance-remaining → progress % (pure) ───────────────────────────

def test_task_progress_from_feedback_distance() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_move("amr-1", 10.0, 0.0)
    assert b.task_progress("amr-1") == 0   # no feedback yet
    b.on_feedback(10.0)                     # baseline (100% distance)
    assert b.task_progress("amr-1") == 0
    b.on_feedback(5.0)
    assert b.task_progress("amr-1") == 50
    b.on_feedback(1.0)
    assert b.task_progress("amr-1") == 90
    b.on_feedback(0.0)
    assert b.task_progress("amr-1") == 100


def test_task_progress_resets_on_new_goal() -> None:
    b = Nav2Binding(robot_id="amr-1")
    b.command_move("amr-1", 10.0, 0.0)
    b.on_feedback(10.0)
    b.on_feedback(2.0)
    assert b.task_progress("amr-1") == 80
    b.command_move("amr-1", 20.0, 0.0)  # new goal → baseline reset
    assert b.task_progress("amr-1") == 0


# ── adapter: throttled RUNNING progress streaming (fake pb) ───────────────────

class _Pb:
    ACTION_STATE_RUNNING = 1

    class ActionStatusUpdate:
        def __init__(self, action_id="", state=0, progress_pct=0):
            self.action_id, self.state, self.progress_pct = action_id, state, progress_pct
            self.fencing_token = None

    class AdapterMessage:
        def __init__(self, action_status=None, **_):
            self.action_status = action_status


class _ProgressBinding:
    def __init__(self) -> None:
        self.pct = 0

    def task_progress(self, robot_id: str) -> int:
        return self.pct


def _drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_running_progress_streams_on_advance_and_throttles() -> None:
    b = _ProgressBinding()
    a = Adapter(_Pb(), None, b, "amr-1")
    a._current_action = "t1"
    a._last_progress = 0
    a._current_fencing_token = 3

    b.pct = 3
    a._maybe_emit_running_progress()
    assert _drain(a._outbox) == []          # <5-point advance → throttled

    b.pct = 20
    a._maybe_emit_running_progress()
    msgs = _drain(a._outbox)
    assert len(msgs) == 1
    upd = msgs[0].action_status
    assert upd.state == _Pb.ACTION_STATE_RUNNING and upd.progress_pct == 20
    assert upd.fencing_token == 3 and a._last_progress == 20

    b.pct = 100
    a._maybe_emit_running_progress()
    assert _drain(a._outbox) == []          # 100 is the terminal-stream's job


def test_running_progress_noop_without_task_progress() -> None:
    a = Adapter(_Pb(), None, SimulatedRobot(), "amr-1")  # sim has no task_progress
    a._current_action = "t1"
    a._maybe_emit_running_progress()
    assert _drain(a._outbox) == []
