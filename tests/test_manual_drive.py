# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The manual-drive action's own rules: its payload, its forwarding filter, how it completes.

Ported from the prototype the action was designed with, test for test (16). What changed in the
port: the payload is the AssignAction's JSON (the control plane decodes the FleetAction's base64
`payload.raw` before it reaches the adapter), the target is `x`/`y` as for Navigate, and the frame
is `map`, the one frame a Navigate goal is sent in. Synthetic poses and commands only.
"""

from __future__ import annotations

import json
import math

import pytest

from fleet_adapter_ros2 import manual_drive as md

F = md.FRAME


def raw(**fields) -> bytes:
    return json.dumps({"operation": md.OPERATION, **fields}).encode()


def task(**kw):
    return md.ManualDrive(md.parse_payload(raw(**{"x": 1.0, "y": 0.0, "frame": F,
                                                  "radius_m": 0.25, **kw})), started_at=0.0)


def drive(t: md.ManualDrive, pts, dt=0.2, t0=0.0):
    for i, (x, y) in enumerate(pts):
        t.on_pose(t0 + i * dt, x, y, F)
    return t


# -- the payload -----------------------------------------------------------------------------------------

def test_the_sketched_payload_parses():
    p = md.parse_payload(raw(x=0.47, y=-0.96, frame=F, radius_m=0.25, settle_s=1.0,
                             max_speed_mps=0.2, teleop_topic="/cmd_vel_teleop", follows="nav-a"))
    assert p.target == (0.47, -0.96) and p.follows == "nav-a" and p.max_speed_mps == 0.2


@pytest.mark.parametrize("bad,match", [
    ({"max_speed_mps": 0.25}, "max_speed_mps"),          # above the robot's 0.20 cap
    ({"max_speed_mps": 0}, "max_speed_mps"),
    ({"radius_m": 0.01}, "radius_m"),
    ({"teleop_topic": "/cmd_vel"}, "teleop_topic"),      # forwarding must stay the gate
    ({"max_turn_rps": 2.0}, "max_turn_rps"),
    ({"timeout_s": float("nan")}, "non-finite"),
])
def test_unsafe_payloads_are_refused(bad, match):
    with pytest.raises(md.PayloadError, match=match):
        md.parse_payload(raw(**{"x": 1, "y": 0, "frame": F, "radius_m": 0.25, **bad}))


def test_a_missing_field_or_another_operation_is_refused():
    with pytest.raises(md.PayloadError, match="frame"):
        md.parse_payload(raw(x=1, y=0, radius_m=0.25))
    other = json.dumps({"operation": "navigate", "x": 1, "y": 0, "frame": F,
                        "radius_m": 0.25}).encode()
    with pytest.raises(md.PayloadError, match="operation"):
        md.parse_payload(other)
    with pytest.raises(md.PayloadError, match="JSON"):
        md.parse_payload(b"not json!")


# -- forwarding --------------------------------------------------------------------------------------------

def test_commands_are_forwarded_and_clamped_while_running():
    t = task()
    assert t.filter_cmd(0.12, 0.0, 0.5) == (0.12, 0.0, 0.5)
    vx, vy, wz = t.filter_cmd(0.5, 0.5, 5.0)
    assert math.hypot(vx, vy) == pytest.approx(0.20) and wz == pytest.approx(0.7)
    assert t.filter_cmd(float("nan"), 0.0, float("inf")) == (0.0, 0.0, 0.0)   # a non-finite command is no command


def test_after_the_end_every_command_becomes_zero():
    t = drive(task(), [(1.0, 0.0)] * 8)                     # still in the circle 1.4 s
    assert t.phase == md.SUCCEEDED
    assert t.filter_cmd(0.12, 0.1, 0.5) == md.ZERO


# -- completion ----------------------------------------------------------------------------------------------

def test_succeeds_only_inside_the_circle_and_still_for_settle_s():
    t = drive(task(settle_s=1.0), [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (0.85, 0.0)])
    assert t.phase == md.RUNNING                             # arrived but still moving
    t = drive(t, [(0.85, 0.0)] * 5, t0=0.8)
    assert t.phase == md.RUNNING                             # still for 0.8 s only
    t.on_pose(2.0, 0.85, 0.0, F)
    assert t.phase == md.SUCCEEDED and "in the circle" in t.message


def test_moving_inside_the_circle_resets_the_settle_clock():
    t = task(settle_s=1.0)
    for i, x in enumerate([0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]):   # a nudge at i=4
        t.on_pose(i * 0.2, x, 0.0, F)
    assert t.phase == md.RUNNING
    t.on_pose(9 * 0.2 + 0.4, 1.0, 0.0, F)
    assert t.phase == md.SUCCEEDED


def test_still_outside_the_circle_never_succeeds():
    t = drive(task(), [(0.5, 0.0)] * 30)
    assert t.phase == md.RUNNING


def test_a_pose_in_another_frame_ends_it_failed():
    t = task()
    t.on_pose(0.0, 1.0, 0.0, "odom")                       # no map -> odom: cannot be judged
    assert t.phase == md.FAILED and t.filter_cmd(0.1, 0, 0) == md.ZERO


def test_timeout_and_abort_end_it_failed_and_stop_forwarding():
    t = task(timeout_s=60)
    t.tick(59.9)
    assert t.forwarding
    t.tick(60.0)
    assert t.phase == md.FAILED and "timeout" in t.message and t.filter_cmd(0.1, 0, 0) == md.ZERO
    t2 = task()
    t2.abort(1.0, "the switch")
    assert t2.phase == md.FAILED and not t2.forwarding


def test_an_ended_action_does_not_change_again():
    t = drive(task(), [(1.0, 0.0)] * 8)
    t.tick(10_000)
    t.abort(10_001, "late")
    t.on_pose(10_002, 5.0, 5.0, "other")
    assert t.phase == md.SUCCEEDED and len([e for e in t.events if "Succeeded" in e[1]]) == 1
