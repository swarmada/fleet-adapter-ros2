# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The CANCELLED report after a confirmed cancel carries the executing fencing token.

The terminal ActionStatusUpdate for a confirmed cancel went out with no token, so the control
plane could not tell it from a report of a superseded assignment (C6.4). The unconfirmed path
(FAILED) already echoed the token; the confirmed one now does too. The report a re-registration
makes with no fencing mark known stays token-less: there is no token to echo.
"""

from __future__ import annotations

import queue
import time

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from test_cancel_confirms_rest import _accepted, _cancel
from test_stop_never_waits import _odom
from test_unanswered_goal import rig, ros_msgs  # noqa: F401 - fixtures


def _until_cancel_answered(a, action_id: str, within: float) -> list:
    """Every ActionStatusUpdate for the action sent up to the cancel's CommandResult. (The
    shared `_result` helper discards what it reads past, and the end is sent before the answer.)"""
    out, deadline = [], time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            m = a._outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        kind = m.WhichOneof("payload")
        if kind == "action_status" and m.action_status.action_id == action_id:
            out.append(m.action_status)
        elif kind == "command_result" and m.command_result.WhichOneof("result") == "cancel_action":
            return out
    raise AssertionError("the cancel was never answered")


def test_the_cancelled_report_carries_the_executing_token(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)                            # t-1 executing at token 1
    _cancel(cp, token=1)
    updates = _until_cancel_answered(a, "t-1", 1.0)        # rest confirmed at once
    cancelled = [u for u in updates if u.state == pb.ACTION_STATE_CANCELLED]
    assert len(cancelled) == 1
    assert cancelled[0].HasField("fencing_token") and cancelled[0].fencing_token == 1


def test_the_failed_report_of_an_unconfirmed_cancel_still_carries_the_token(rig):  # noqa: F811
    a, b, server, cp = rig
    _accepted(a, b, server, cp)
    b.on_odom(_odom(0.4))                                  # never reaches rest
    _cancel(cp, token=1)
    updates = _until_cancel_answered(a, "t-1", 3.0)
    failed = [u for u in updates if u.state == pb.ACTION_STATE_FAILED]
    assert len(failed) == 1 and failed[0].fencing_token == 1
