# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter acts only for the robot it serves.

One ControlStream carries the commands for every Robot bound to an adapter, and
`Command.robot_id` selects which one (RFC-0001 #fleet-adapter-protocol-the-controlstream-service).
This adapter serves exactly one. It used to run every command whatever robot it named: an
assignment for a second Robot object bound to the same adapter sent a Nav2 goal on this robot,
and a cancel for it stopped this robot's running action.

Now every command that acts on the robot, or vouches that it can take work, is refused when it
names another robot, with the refusal the contract already defines for its arm, and nothing
reaches the robot, the fence or the lease. An estop on the SafetyStream for another robot is
answered ESTOP_STATE_FAILED for that id (this adapter cannot confirm that robot at rest), and
this robot is not stopped by it. An estop for this robot is unchanged, and still always stops it.
The same rule as the reference simulation adapter.
"""

from __future__ import annotations

import threading

from safety_call import SafetyCall
from test_refuse_unservable import ROBOT, _drain, _spied

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

OTHER = "amr-1-c"  # a robot this adapter does not serve


def _cmd(a: Adapter, robot_id: str, cid: int, **command) -> list:
    a._handle_control_plane(pb.ControlPlaneMessage(command=pb.Command(
        command_id=cid, robot_id=robot_id, **command)), None)
    return [m.command_result for m in _drain(a) if m.WhichOneof("payload") == "command_result"]


def _nav(action: str, token: int = 1) -> pb.AssignAction:
    return pb.AssignAction(action_id=action, action_type="Navigate", assignment_id=f"a-{action}",
                           fencing_token=token, lease_generation=1, lease_duration_ms=5000,
                           payload_json=b'{"x": 1.0, "y": 2.0}')


def _running() -> tuple:
    """An adapter whose robot is running `t-mine`, with every stop it commands recorded."""
    a, b, sent, cmd_vel = _spied()
    ok = _cmd(a, ROBOT, 1, assign_action=_nav("t-mine"))
    assert ok[0].assign_action.accepted and a._current_action == "t-mine"
    stops: list = []
    for name in ("command_stop", "command_pause", "command_resume"):
        real = getattr(b, name)
        setattr(b, name, lambda rid, n=name, f=real: (stops.append((n, rid)), f(rid))[1])
    return a, b, sent, cmd_vel, stops


def _one(res: list, robot_id: str = OTHER) -> pb.CommandResult:
    assert len(res) == 1, res
    assert res[0].robot_id == robot_id   # echoed, so the dispatcher can correlate it
    return res[0]


def test_assign_for_an_unserved_robot_is_refused_and_sends_no_goal():
    a, _b, sent, cmd_vel = _spied()
    r = _one(_cmd(a, OTHER, 1, assign_action=_nav("t-other")))
    assert not r.assign_action.accepted
    assert r.assign_action.rejection == pb.ASSIGN_ACTION_REJECTION_ROBOT_UNAVAILABLE
    assert OTHER in r.assign_action.message
    assert sent == [], f"a Nav2 goal was sent on {ROBOT!r}: {sent}"
    assert cmd_vel.published == []
    assert a._current_action == ""


def test_cancel_for_an_unserved_robot_is_not_acknowledged_and_stops_nothing():
    a, _b, _sent, cmd_vel, stops = _running()
    twists = len(cmd_vel.published)
    r = _one(_cmd(a, OTHER, 2, cancel_action=pb.CancelAction(action_id="t-mine", fencing_token=1)))
    assert not r.cancel_action.acknowledged
    assert stops == [], f"a cancel addressed to {OTHER!r} commanded {stops}"
    assert len(cmd_vel.published) == twists
    assert a._current_action == "t-mine"


def test_pause_for_an_unserved_robot_is_refused_and_holds_nothing():
    a, _b, _sent, _cmd_vel, stops = _running()
    r = _one(_cmd(a, OTHER, 2, pause=pb.PauseCapabilities(reason="test")))
    assert not r.pause.paused
    assert stops == []


def test_resume_for_an_unserved_robot_is_refused_and_moves_nothing():
    a, _b, _sent, _cmd_vel, stops = _running()
    r = _one(_cmd(a, OTHER, 2, resume=pb.ResumeCapabilities(reason="test")))
    assert not r.resume.resumed
    assert stops == []


def test_lease_renewal_for_an_unserved_robot_is_refused_and_leaves_the_lease():
    a, _b, _sent, _cmd_vel, _stops = _running()
    lease = (dict(a._lease._deadline), dict(a._lease._generation))
    r = _one(_cmd(a, OTHER, 2, renew_lease=pb.RenewActionLease(
        action_id="t-mine", lease_generation=1, lease_duration_ms=60_000)))
    assert not r.renew_lease.renewed
    assert not r.renew_lease.running
    assert (a._lease._deadline, a._lease._generation) == lease


def test_validate_for_an_unserved_robot_does_not_vouch_for_it():
    a, _b, _sent, _cmd_vel = _spied()
    r = _one(_cmd(a, OTHER, 1, validate_action=pb.ValidateAction(
        action_type="Navigate", payload_json=b'{"x": 1.0, "y": 2.0}')))
    assert not r.validate_action.servable


def test_zone_admission_for_an_unserved_robot_gets_no_admission():
    # This adapter declines zone_admission for every robot; one for another id stays declined.
    a, _b, _sent, _cmd_vel = _spied()
    r = _one(_cmd(a, OTHER, 1, zone_admission=pb.ZoneAdmission(zone_name="z", admit=True)))
    assert not r.zone_admission.acknowledged


def test_commands_for_the_served_robot_still_run():
    a, _b, sent, _cmd_vel = _spied()
    r = _one(_cmd(a, ROBOT, 1, assign_action=_nav("t-mine")), ROBOT)
    assert r.assign_action.accepted and len(sent) == 1


class _OneEstop:
    """A control plane that sends one Estop for `robot_id` down the SafetyStream, then ends the
    adapter's SafetyStream loop, keeping the acks in `acks`."""

    def __init__(self, adapter: Adapter, robot_id: str) -> None:
        self.adapter, self.robot_id, self.acks = adapter, robot_id, []

    def SafetyStream(self, requests):
        def stream():
            yield pb.ControlPlaneSafetyMessage(
                robot_id=self.robot_id, estop=pb.Estop(estop_id="e-1", reason="test"))
            outbox = self.adapter._safety_outbox
            while not outbox.empty():
                m = outbox.get_nowait()
                if m is not None:
                    self.acks.append(m)
            self.adapter._shutdown.set()
        return SafetyCall(stream())


def _estop(monkeypatch, robot_id: str) -> tuple[list, list]:
    stopped: list = []
    real = adapter_mod.command_estop
    monkeypatch.setattr(adapter_mod, "command_estop",
                        lambda robot, rid: (stopped.append(rid), real(robot, rid))[1])
    a = Adapter(pb, None, SimulatedRobot(), ROBOT)
    cp = _OneEstop(a, robot_id)
    t = threading.Thread(target=a.run_safety_stream, args=(cp,), daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    return cp.acks, stopped


def test_estop_for_an_unserved_robot_fails_for_that_id_and_does_not_stop_this_robot(monkeypatch):
    acks, stopped = _estop(monkeypatch, OTHER)
    assert stopped == [], f"an estop addressed to {OTHER!r} stopped {stopped}"
    assert len(acks) == 1, acks
    assert acks[0].robot_id == OTHER
    assert acks[0].estop_ack.estop_id == "e-1"
    assert acks[0].estop_ack.state == pb.ESTOP_STATE_FAILED
    assert acks[0].estop_ack.stop_initiated_at == 0   # no stop was commanded
    assert OTHER in acks[0].estop_ack.message


def test_estop_for_this_robot_still_stops_it(monkeypatch):
    acks, stopped = _estop(monkeypatch, ROBOT)
    assert stopped == [ROBOT]
    assert acks and all(m.robot_id == ROBOT for m in acks)
    assert acks[-1].estop_ack.state == pb.ESTOP_STATE_STOPPED
