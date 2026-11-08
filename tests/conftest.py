# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Make the adapter package importable under pytest without an editable install
(handy for a $0 run; a real deployment `pip install -e .` makes this a no-op)."""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── ROS-free "the transport is up" fixture ───────────────────────────────────

def mark_transport_live(binding):
    """Make a ROS-free ``Nav2Binding`` present as though ``start()`` had run, and
    return it.

    Needed because ``tick()`` now REFUSES to tick a binding whose transport was never
    brought up: an un-started binding answers every accessor with a default
    and confirms nothing, and before this it was indistinguishable from a healthy one.
    That refusal is right in production and wrong for a unit-test fixture that stands in
    for a live robot — and some of those fixtures reach ``tick()`` indirectly, via
    ``swarmada_sdk.safety.confirm_estop``, whose polling loop ticks.

    Sets BOTH flags the binding answers liveness questions from, so the fixture is
    consistent across all of them:

    * ``_node`` — what ``tick()`` discriminates on.
    * ``_spin_thread`` — what ``transport_alive()`` reports from.

    Setting only one would be worse than not having this helper. A binding that ticks
    silently while ``transport_alive()`` says False is a state no real binding can be in,
    and a fixture in it would let a test pass for a reason the production code could never
    reproduce.

    Neither stand-in has behaviour. The node is a bare marker and is never called: every
    live dispatch path is additionally guarded on ``_cmd_vel`` / ``_action_client`` / a goal
    handle, all of which stay None, so goal send, goal cancel and the zero-Twist publish
    remain no-ops. The thread stand-in answers ``is_alive()`` and nothing else, which is
    all ``transport_alive()`` asks of it. So the fixture says "the transport is up" without
    claiming a ROS 2 runtime that is not there.

    Reaching into named underscore-prefixed flags with a reason is the existing convention
    in these tests — ``_record_spin_exit`` and ``_stopping`` are used the same way, for the same
    purpose: reaching a lifecycle state that a ROS-free test cannot reach honestly any
    other way.
    """
    class _AliveThread:
        """Stands in for the rclpy executor thread. `transport_alive()` calls is_alive()."""

        @staticmethod
        def is_alive() -> bool:
            return True

    binding._node = object()          # a marker, never called; see above
    binding._spin_thread = _AliveThread()
    return binding
