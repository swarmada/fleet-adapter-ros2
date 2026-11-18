# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A Charge or a segment that supersedes what ran holds the base first.

A Navigate that supersedes sends its own goal, and the navigation server ends the one before
it. A Charge or a segment sends none when it starts: the adapter halted the superseded segment,
which tells its explorer to stop, and nothing more. The binding was not held, so nothing
cancelled what was still driving: a superseded Navigate's goal, a segment's return to its mark,
or an explorer goal accepted late. The robot drove during the Charge, or while the new segment
was still checking it may start. Now the base is held, as a stop holds it, in the step that
makes the job current: the goal is cancelled and its target dropped, and every goal the server
shows running is cancelled by its id until a segment releases the hold for its own explorer.
"""

from __future__ import annotations

import json

import pytest
from test_explore_adapter import StepClock, _assign
from test_ros2_explore import _stack as _ros2_stack

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter

ROBOT = "amr-1"
SEGMENT = json.dumps({"operation": "explore-floor", "bound_ref": "sim-room",
                      "max_speed_mps": 0.2}).encode()
NEXT = {"Charge": b"", "Custom": SEGMENT}
EXPLORERS = b"\x07" * 16              # a goal id the superseded segment's explorer sent
LATE = b"\x0b" * 16                   # one of its goals, accepted after the new job started


class _Handle:
    """The Navigate goal's handle, as the action client gives it."""

    def __init__(self) -> None:
        self.cancels = 0

    def cancel_goal_async(self) -> None:
        self.cancels += 1


def _rig(tmp_path, monkeypatch):
    """The Nav2 binding, with the ROS 2 exploration stack, through the adapter. The jobs' own
    threads are not run: what is tested is the step that makes a job current."""
    monkeypatch.setattr(explore.ChargeJob, "run", lambda self: None)
    monkeypatch.setattr(explore.ExploreJob, "run", lambda self: None)
    st, _ros, _shell, b, _sent = _ros2_stack(tmp_path)
    b._explore = st
    b._status_sub = object()          # subscribed to the server's goal status (start())
    b.cancelled = []                  # the goal ids cancelled by id
    b._cancel_running_goal = b.cancelled.append
    a = Adapter(pb, None, b, ROBOT, job_clock=StepClock(),
                artifacts=explore.ArtifactDir(str(tmp_path), "floor-maps"), namespace="lab")
    return a, b


@pytest.mark.parametrize("new", ["Charge", "Custom"])
def test_a_job_that_supersedes_a_segment_cancels_its_explorers_goals(
        tmp_path, monkeypatch, new) -> None:
    a, b = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Custom", SEGMENT, token=1, action="seg-1")[0].accepted
    assert b.release_hold_for_explorer(ROBOT, a._job.may_move)    # its explorer drives
    b.on_nav_status([(EXPLORERS, 2)])
    assert b.cancelled == []
    assert _assign(a, new, NEXT[new], token=2, action="next")[0].accepted
    b.on_nav_status([(EXPLORERS, 2), (LATE, 1)])
    assert {EXPLORERS, LATE} <= set(b.cancelled), (
        "goals still driving when the segment was superseded were not cancelled")


@pytest.mark.parametrize("new", ["Charge", "Custom"])
def test_a_job_that_supersedes_a_navigate_cancels_its_goal(tmp_path, monkeypatch, new) -> None:
    a, b = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Navigate", b'{"x": 1, "y": 2}', token=1, action="t-1")[0].accepted
    handle = b._goal_handle = _Handle()                       # the server accepted it
    assert _assign(a, new, NEXT[new], token=2, action="next")[0].accepted
    assert handle.cancels == 1, "the superseded Navigate's goal was never cancelled"
    assert b.pending_goal(ROBOT) is None


def test_a_segment_still_releases_the_hold_for_its_own_explorer(tmp_path, monkeypatch) -> None:
    a, b = _rig(tmp_path, monkeypatch)
    assert _assign(a, "Charge", b"", token=1, action="charge")[0].accepted
    assert _assign(a, "Custom", SEGMENT, token=2, action="seg-2")[0].accepted
    assert b.release_hold_for_explorer(ROBOT, a._job.may_move)
    b.cancelled.clear()
    b.on_nav_status([(EXPLORERS, 2)])
    assert b.cancelled == []
