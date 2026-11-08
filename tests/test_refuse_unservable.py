# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter refuses what it cannot serve, and does exactly what its catalog says.

The CapabilitiesSnapshot advertises one action, `Navigate`, with `x` and `y` required and `yaw`
optional. An assignment outside that is DECLINED `ASSIGN_ACTION_REJECTION_INVALID_ACTION`
("unparseable / unsupported"), never run as navigation:

* an `action_type` the catalog does not list (RFC-0001: an adapter "MUST NOT treat an
  action_type it does not recognise as a nearest-neighbour verb; it rejects the assignment");
* a `Navigate` whose payload lacks a required parameter, or carries one that is not a finite
  number: no coordinate is ever defaulted, so a missing target is not the map origin;
* a named destination, which this adapter cannot resolve;
* a payload `frame` other than `map`: the goal is always sent in `map`, so any other frame
  would be a drive to the wrong place. An absent frame is `map`.

An EMPTY or absent `action_type` (proto3's unset; a control plane never sends one, since
`FleetAction.spec.type` is a required enum) is declined for the same reason as a type the
catalog does not list: which verb was meant is never guessed.

`validate_action` answers from the same check, as the reference simulation adapter answers it
from the same list it advertises. Driven through `_handle_control_plane` with the real protobuf
messages; the no-motion tests spy on a ROS-free Nav2Binding's goal dispatch and /cmd_vel.
"""

from __future__ import annotations

import pytest
from fleet_adapter.v1 import fleet_adapter_pb2 as pb

from conftest import mark_transport_live
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"


class _Publisher:
    """Stands in for the /cmd_vel publisher; records every Twist published."""

    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


def _spied() -> tuple[Adapter, Nav2Binding, list, _Publisher]:
    """An adapter on a ROS-free Nav2Binding whose two motion outputs are recorded: the
    NavigateToPose dispatch (`_send_nav2_goal`) and the /cmd_vel publisher."""
    b = mark_transport_live(Nav2Binding())
    sent: list = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(goal)
    b._server_ready = lambda: True                        # the server is up: goals go at once
    cmd_vel = _Publisher()
    b._cmd_vel, b._twist_type = cmd_vel, object
    return Adapter(pb, None, b, ROBOT), b, sent, cmd_vel


def _drain(a: Adapter) -> list:
    out = []
    while not a._outbox.empty():
        out.append(a._outbox.get_nowait())
    return out


def _assign(a: Adapter, action_type: str, payload: bytes = b"", destination: str = "",
            token: int = 1, action: str = "t-1") -> tuple[pb.AssignActionResult, list]:
    task = pb.AssignAction(action_id=action, action_type=action_type, assignment_id=f"a-{action}",
                           fencing_token=token, lease_generation=1, lease_duration_ms=5000,
                           payload_json=payload, destination=destination)
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=token, robot_id=ROBOT, assign_action=task)), None)
    out = _drain(a)
    results = [m.command_result for m in out if m.WhichOneof("payload") == "command_result"]
    assert len(results) == 1 and not results[0].unsupported
    return results[0].assign_action, out


def _assert_invalid(res: pb.AssignActionResult, *words: str) -> None:
    assert res.accepted is False
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    for w in words:
        assert w in res.message


# ── an action type the catalog does not list ────────────────────────────────

@pytest.mark.parametrize("action_type", ["Charge", "PickUp", "DropOff", "Patrol", "Clean",
                                         "Custom", "External", "navigate", " Navigate"])
def test_an_action_type_outside_the_catalog_is_declined_invalid_action(action_type) -> None:
    a, b, sent, cmd_vel = _spied()
    res, _ = _assign(a, action_type, b'{"x": 4, "y": 5}')
    _assert_invalid(res, "unsupported action type")
    assert sent == [] and cmd_vel.published == [] and b.pending_goal(ROBOT) is None


# ── an unstated (empty) type ────────────────────────────────────────────────

def test_an_empty_type_is_declined_like_any_type_not_listed() -> None:
    # A complete target does not make it servable: the type is what says what to do with it.
    a, b, sent, cmd_vel = _spied()
    _assert_invalid(_assign(a, "", b'{"x": 4, "y": 5}')[0], "unsupported action type")
    assert sent == [] and cmd_vel.published == [] and b.pending_goal(ROBOT) is None


def test_an_absent_type_is_declined_like_any_type_not_listed() -> None:
    # The field never set at all, as a peer that predates it would send.
    a, b, sent, cmd_vel = _spied()
    task = pb.AssignAction(action_id="t-1", assignment_id="a-t-1", fencing_token=1,
                           lease_generation=1, lease_duration_ms=5000,
                           payload_json=b'{"x": 4, "y": 5}')
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=task)), None)
    out = _drain(a)
    res = [m.command_result.assign_action for m in out
           if m.WhichOneof("payload") == "command_result"][0]
    _assert_invalid(res, "unsupported action type", "none stated")
    assert sent == [] and cmd_vel.published == [] and b.pending_goal(ROBOT) is None
    assert a._current_action == "" and ROBOT not in a._lease._deadline
    assert [m for m in out if m.WhichOneof("payload") == "action_status"] == []


def test_an_empty_type_is_declined_even_with_a_one_entry_catalog(monkeypatch) -> None:
    # Whatever the catalog holds, a sole entry is not a default verb.
    monkeypatch.setattr(adapter_mod, "_SUPPORTED_ACTIONS", [
        ("Patrol", ["navigation.2d"], [("x", "number", "m", True), ("y", "number", "m", True)]),
    ])
    a, _, sent, _ = _spied()
    _assert_invalid(_assign(a, "", b'{"x": 4, "y": 5}')[0], "unsupported action type")
    assert sent == []


# ── a Navigate without a target it can reach ────────────────────────────────

@pytest.mark.parametrize("payload, word", [
    (b"", "x"),
    (b"{}", "x"),
    (b'{"y": 2}', "x"),
    (b'{"x": 1}', "y"),
    (b'{"x": "1", "y": 2}', "x"),
    (b'{"x": true, "y": 2}', "x"),
    (b'{"x": 1, "y": NaN}', "y"),
    (b'{"x": Infinity, "y": 2}', "x"),
    (b'{"x": 1, "y": 2, "yaw": "north"}', "yaw"),
    (b'{"location": "dock-3"}', "x"),
    (b"not json", "JSON object"),
    (b"[1, 2]", "JSON object"),
])
def test_a_navigate_without_a_usable_target_is_declined_not_defaulted(payload, word) -> None:
    a, b, sent, cmd_vel = _spied()
    res, _ = _assign(a, "Navigate", payload)
    _assert_invalid(res, word)
    assert sent == [] and cmd_vel.published == [] and b.pending_goal(ROBOT) is None


def test_a_named_destination_is_declined_it_cannot_be_resolved() -> None:
    a, b, sent, cmd_vel = _spied()
    res, _ = _assign(a, "Navigate", destination="dock-3")
    _assert_invalid(res, "destination")
    assert sent == [] and b.pending_goal(ROBOT) is None


def test_a_destination_beside_coordinates_is_still_declined() -> None:
    # Two targets, one of which this adapter cannot read: refusing is the only answer that
    # cannot drive the robot somewhere nobody asked for.
    a, _, sent, _ = _spied()
    res, _ = _assign(a, "Navigate", b'{"x": 1, "y": 2}', destination="dock-3")
    _assert_invalid(res, "destination")
    assert sent == []


# ── only the map frame ──────────────────────────────────────────────────────

@pytest.mark.parametrize("frame, named", [
    (b'"odom"', '"odom"'),
    (b'"base_footprint"', '"base_footprint"'),
    (b'"base_link"', '"base_link"'),
    (b'"Map"', '"Map"'),
    (b'"/map"', '"/map"'),
    (b'""', '""'),
    (b"null", "null"),
    (b"5", "5"),
])
def test_a_frame_other_than_map_is_declined_and_named(frame, named) -> None:
    # Before the fix `{"x": 1, "y": 2, "frame": "odom"}` was accepted and driven to (1, 2) in
    # `map`. On hardware odom drifts from map: a drive to the wrong place.
    a, b, sent, cmd_vel = _spied()
    res, out = _assign(a, "Navigate", b'{"x": 1, "y": 2, "frame": ' + frame + b"}")
    _assert_invalid(res, "frame " + named, "map")
    assert sent == []                       # no NavigateToPose goal
    assert cmd_vel.published == []          # no /cmd_vel
    assert b.pending_goal(ROBOT) is None
    assert a._current_action == "" and ROBOT not in a._lease._deadline
    assert [m for m in out if m.WhichOneof("payload") == "action_status"] == []


@pytest.mark.parametrize("payload", [
    b'{"x": 1, "y": 2}',
    b'{"x": 1, "y": 2, "frame": "map"}',
])
def test_the_map_frame_stated_or_absent_is_served_in_map(payload) -> None:
    a, _, sent, _ = _spied()
    res, _ = _assign(a, "Navigate", payload)
    assert res.accepted is True
    assert [(g.x, g.y, g.frame) for g in sent] == [(1.0, 2.0, "map")]


def test_validate_action_declines_a_frame_other_than_map() -> None:
    a, _, sent, _ = _spied()
    res = _validate(a, "Navigate", b'{"x": 1, "y": 2, "frame": "odom"}')
    assert res.validate_action.servable is False
    assert '"odom"' in res.validate_action.message
    assert _validate(a, "Navigate", b'{"x": 1, "y": 2, "frame": "map"}').validate_action.servable
    assert sent == []


# ── the (0, 0) reproduction, after the fix ──────────────────────────────────

@pytest.mark.parametrize("action_type, payload, destination", [
    ("Navigate", b'{"location": "dock-3"}', ""),
    ("Navigate", b"", "dock-3"),
    ("Charge", b"", ""),
    ("PickUp", b'{"x": 4, "y": 5}', ""),
])
def test_the_four_cases_that_drove_to_the_origin_are_now_declined(action_type, payload,
                                                                  destination) -> None:
    # Before the fix every one was accepted: the first three with a goal of (0, 0) in `map`,
    # the fourth as a PickUp run as navigation.
    a, b, sent, cmd_vel = _spied()
    res, _ = _assign(a, action_type, payload, destination)
    assert res.accepted is False
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_INVALID_ACTION
    assert b.pending_goal(ROBOT) is None
    assert sent == [] and cmd_vel.published == []


# ── a declined assignment moves nothing and leaves nothing behind ───────────

def test_a_declined_assignment_publishes_no_goal_no_cmd_vel_and_takes_no_lease() -> None:
    a, b, sent, cmd_vel = _spied()
    res, out = _assign(a, "Charge")
    assert res.accepted is False
    assert sent == []                       # no NavigateToPose goal
    assert cmd_vel.published == []          # no /cmd_vel
    assert b.pending_goal(ROBOT) is None
    assert a._current_action == ""
    assert ROBOT not in a._lease._deadline
    assert [m for m in out if m.WhichOneof("payload") == "action_status"] == []


def test_a_decline_leaves_the_navigation_already_running_untouched() -> None:
    a, b, sent, cmd_vel = _spied()
    assert _assign(a, "Navigate", b'{"x": 1, "y": 2}', token=1, action="t-1")[0].accepted
    res, _ = _assign(a, "Charge", token=2, action="t-2")
    assert res.accepted is False
    assert a._current_action == "t-1"
    assert len(sent) == 1 and (sent[0].x, sent[0].y) == (1.0, 2.0)
    assert cmd_vel.published == []  # t-1 was not stopped either


def test_the_spy_sees_a_served_navigate() -> None:
    # The no-motion assertions above mean something only if the spy records real dispatch.
    a, b, sent, _ = _spied()
    res, _ = _assign(a, "Navigate", b'{"x": 3.5, "y": -1, "yaw": 1.5}')
    assert res.accepted is True
    assert [(g.x, g.y, g.yaw) for g in sent] == [(3.5, -1.0, 1.5)]


def test_a_stale_token_is_still_refused_stale_first() -> None:
    a, _, _, _ = _spied()
    assert _assign(a, "Navigate", b'{"x": 1, "y": 2}', token=5)[0].accepted
    res, _ = _assign(a, "Charge", token=3, action="t-old")
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_STALE_FENCING_TOKEN


def test_an_operating_rule_is_still_refused_as_a_rule_first() -> None:
    # The rule is about the zone and binds whatever the action; C17 grades that reason.
    a, _, sent, _ = _spied()
    task = pb.AssignAction(action_id="t-c", action_type="Custom", assignment_id="a-c",
                           fencing_token=1, operating_constraints=[
                               pb.EffectiveOperatingConstraint(
                                   subject="camera", mode=pb.OPERATING_CONSTRAINT_MODE_DENIED)])
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=1, robot_id=ROBOT, assign_action=task)), None)
    res = [m.command_result.assign_action for m in _drain(a)
           if m.WhichOneof("payload") == "command_result"][0]
    assert res.rejection == pb.ASSIGN_ACTION_REJECTION_OPERATING_CONSTRAINT_UNMET
    assert sent == []


# ── validate_action answers from the same check ─────────────────────────────

def _validate(a: Adapter, action_type: str, payload: bytes = b"") -> pb.CommandResult:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=9, robot_id=ROBOT,
        validate_action=pb.ValidateAction(action_type=action_type, payload_json=payload))), None)
    return [m.command_result for m in _drain(a) if m.WhichOneof("payload") == "command_result"][0]


@pytest.mark.parametrize("action_type, payload, servable", [
    ("Navigate", b'{"x": 1, "y": 2}', True),
    ("Navigate", b'{"x": 1, "y": 2, "yaw": 0.5}', True),
    ("Navigate", b"", False),
    ("Navigate", b'{"x": 1}', False),
    ("Charge", b'{"x": 1, "y": 2}', False),
    ("", b'{"x": 1, "y": 2}', False),
    ("", b"", False),
])
def test_validate_action_agrees_with_what_assignment_would_do(action_type, payload,
                                                              servable) -> None:
    a, _, sent, _ = _spied()
    res = _validate(a, action_type, payload)
    assert res.unsupported is False and res.WhichOneof("result") == "validate_action"
    assert res.validate_action.servable is servable
    assert (res.validate_action.message == "") is servable
    assert sent == []  # validation is pure inspection: it moves nothing
    assigned, _ = _assign(a, action_type, payload)
    assert assigned.accepted is servable


# ── one source for the catalog and the acceptance path ──────────────────────

def test_the_catalog_and_the_acceptance_path_read_one_list(monkeypatch) -> None:
    # Swap the ONE list for another; the advertised catalog and what is accepted must both
    # follow it, with no second place to update.
    monkeypatch.setattr(adapter_mod, "_SUPPORTED_ACTIONS", [
        ("Patrol", ["navigation.2d"], [("x", "number", "m", True), ("y", "number", "m", True)]),
    ])
    a, _, sent, _ = _spied()
    snap = a._capabilities_snapshot()
    assert [s.action_type for s in snap.supported_actions] == ["Patrol"]
    assert _assign(a, "Patrol", b'{"x": 1, "y": 2}')[0].accepted is True
    _assert_invalid(_assign(a, "Navigate", b'{"x": 1, "y": 2}', token=2, action="t-2")[0],
                    "unsupported action type")
