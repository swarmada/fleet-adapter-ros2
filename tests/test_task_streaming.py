# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Terminal task-status streaming (C6.4): the adapter streams a SUCCEEDED / FAILED
ActionStatusUpdate derived from the Nav2 goal status, once, then clears the task. A
tiny fake pb records the emitted update so this stays proto-free."""

from __future__ import annotations

from fleet_adapter_ros2 import task
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot


class _Pb:
    ACTION_STATE_RUNNING = 1
    ACTION_STATE_SUCCEEDED = 2
    ACTION_STATE_FAILED = 3

    class ActionStatusUpdate:
        def __init__(self, action_id="", state=0, progress_pct=0):
            self.action_id, self.state, self.progress_pct = action_id, state, progress_pct
            self.fencing_token = None

    class AdapterMessage:
        def __init__(self, action_status=None, **_):
            self.action_status = action_status


class _Binding:
    def __init__(self, state: str) -> None:
        self._state = state

    def task_state(self, robot_id: str) -> str:
        return self._state


def _adapter(binding) -> Adapter:
    return Adapter(_Pb(), None, binding, "amr-1")


def _drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_streams_succeeded_once_then_clears() -> None:
    a = _adapter(_Binding(task.TASK_SUCCEEDED))
    a._current_action = "t1"
    a._last_task_state = task.TASK_RUNNING
    a._current_fencing_token = 9

    a._maybe_emit_task_progress()
    msgs = _drain(a._outbox)
    assert len(msgs) == 1
    upd = msgs[0].action_status
    assert upd.action_id == "t1" and upd.state == _Pb.ACTION_STATE_SUCCEEDED
    assert upd.progress_pct == 100 and upd.fencing_token == 9  # C6.4 token echoed
    assert a._current_action == ""

    a._maybe_emit_task_progress()  # done → no re-emit
    assert _drain(a._outbox) == []


def test_streams_failed() -> None:
    a = _adapter(_Binding(task.TASK_FAILED))
    a._current_action = "t1"
    a._last_task_state = task.TASK_RUNNING
    a._maybe_emit_task_progress()
    assert _drain(a._outbox)[0].action_status.state == _Pb.ACTION_STATE_FAILED


def test_cancelled_goal_is_not_streamed_here() -> None:
    # A CANCELED goal is reported via the cancel path, not the completion stream.
    a = _adapter(_Binding(task.TASK_CANCELLED))
    a._current_action = "t1"
    a._last_task_state = task.TASK_RUNNING
    a._maybe_emit_task_progress()
    assert _drain(a._outbox) == []


def test_binding_without_goal_state_is_noop() -> None:
    a = _adapter(SimulatedRobot())  # no task_state → nothing to stream
    a._current_action = "t1"
    a._maybe_emit_task_progress()
    assert _drain(a._outbox) == []
