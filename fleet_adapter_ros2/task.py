# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""FleetTask → Nav2 task translation (pure, proto-free, rclpy-free).

Two mappings, each a plain function over duck-typed / primitive inputs so they are
unit-testable with no ROS install and no generated protobufs; the adapter converts
the returned dataclasses to protobuf:

* :func:`goal_from_assign` — an ``AssignAction`` (its ``destination`` string and raw
  ``payload_json`` bytes) → a :class:`GoalPose`, i.e. the target of a
  ``nav2_msgs/action/NavigateToPose`` goal.
* :func:`task_state_from_goal_status` — an ``action_msgs/GoalStatus`` code from the
  running Nav2 goal → the protocol's ``TaskState`` name, so :class:`TaskUpdate`
  (converted to ``ActionStatusUpdate``) reflects the live goal.

This adapter covers the ROS 2 CLASS: the translation is generic (coordinates in
the frame, or a named location the map server resolves) — no vendor branches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# protocol ActionState enum names (bound to the proto enum by the adapter).
TASK_RUNNING = "ACTION_STATE_RUNNING"
TASK_SUCCEEDED = "ACTION_STATE_SUCCEEDED"
TASK_FAILED = "ACTION_STATE_FAILED"
TASK_CANCELLED = "ACTION_STATE_CANCELLED"
# A clean commanded stop — NOT a failure. Reported when the assignment lease
# lapses (C4.2) or an estop safe-holds the action, so the control plane can tell
# "stopped on purpose" from "crashed" and from silence.
TASK_STOPPED = "ACTION_STATE_STOPPED"

# action_msgs/msg/GoalStatus status codes (the Nav2 goal's in-flight/terminal state).
_GS_ACCEPTED, _GS_EXECUTING = 1, 2
_GS_CANCELED, _GS_SUCCEEDED, _GS_ABORTED = 5, 4, 6

_DEFAULT_FRAME = "map"


@dataclass
class GoalPose:
    """A NavigateToPose target. ``label`` is set (with coords 0,0) when the task
    names a location instead of coordinates — the Nav2 map server resolves it."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    frame: str = _DEFAULT_FRAME
    label: str = ""


@dataclass
class TaskUpdate:
    """What the adapter reports as a ``ActionStatusUpdate`` (numeric + bounded only)."""

    action_id: str
    state: str                       # one of the TASK_* names above
    fencing_token: int | None = None  # echoed for C6.4 (superseded-assignment detection)
    progress_pct: int = 0            # 0–100; 0 if unknown
    message: str = ""


def _coords_from_string(text: str) -> tuple[float, float, float] | None:
    """Parse ``"x,y"`` or ``"x,y,yaw"`` (a coordinate destination) → (x, y, yaw)."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    x, y = nums[0], nums[1]
    yaw = nums[2] if len(nums) == 3 else 0.0
    return x, y, yaw


def goal_from_assign(destination: str, payload_json: bytes | str = b"") -> GoalPose:
    """``AssignAction`` → :class:`GoalPose`.

    Target resolution order (generic to the ROS 2 class, no vendor branches):

    1. ``payload_json`` object with numeric ``x``/``y`` (+ optional ``yaw``,
       ``frame``) — explicit coordinates from ``FleetTask.spec.payload``.
    2. ``destination`` parsed as ``"x,y"`` / ``"x,y,yaw"`` — a coordinate string.
    3. otherwise ``destination`` is a **named** location (``label``); the Nav2 map
       server / waypoint DB resolves it at goal time. Coordinates default to 0,0.
    """
    frame = _DEFAULT_FRAME
    payload: dict = {}
    if payload_json:
        try:
            raw = payload_json.decode() if isinstance(payload_json, bytes) else payload_json
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                payload = loaded
        except (ValueError, UnicodeDecodeError):
            payload = {}  # opaque/garbage payload → fall through to destination

    if isinstance(payload.get("frame"), str):
        frame = payload["frame"]

    if _is_number(payload.get("x")) and _is_number(payload.get("y")):
        yaw = float(payload["yaw"]) if _is_number(payload.get("yaw")) else 0.0
        return GoalPose(x=float(payload["x"]), y=float(payload["y"]), yaw=yaw,
                        frame=frame, label=str(destination or ""))

    if destination:
        coords = _coords_from_string(destination)
        if coords is not None:
            x, y, yaw = coords
            return GoalPose(x=x, y=y, yaw=yaw, frame=frame)
        return GoalPose(frame=frame, label=destination)  # named location

    return GoalPose(frame=frame)  # no target given → origin of the frame


def task_state_from_goal_status(status: int) -> str:
    """``action_msgs/GoalStatus`` code → protocol ``TaskState`` name.

    A cancelled goal maps to CANCELLED (not FAILED) so a confirmed-estop or
    cancel-task cancellation is reported distinctly from a navigation failure.
    """
    if status == _GS_SUCCEEDED:
        return TASK_SUCCEEDED
    if status == _GS_CANCELED:
        return TASK_CANCELLED
    if status == _GS_ABORTED:
        return TASK_FAILED
    return TASK_RUNNING  # ACCEPTED / EXECUTING / anything in-flight


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
