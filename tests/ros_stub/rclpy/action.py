# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""An action client that accepts and answers at once. See ../README.md."""

from __future__ import annotations


class _Handle:
    accepted = True

    def __init__(self) -> None:
        self.cancelled = False

    def cancel_goal_async(self):
        self.cancelled = True
        return _Future(None)


class _Future:
    def __init__(self, value) -> None:
        self._value = value

    def add_done_callback(self, fn):
        fn(self)            # a server with nothing else to do answers on the spot

    def result(self):
        return self._value


class ActionClient:
    def __init__(self, node, action_type, action_name: str) -> None:
        self.node = node
        self.action_name = action_name
        self.sent = []

    def wait_for_server(self, timeout_sec: float = 0.0) -> bool:
        return True

    def send_goal_async(self, goal):
        self.sent.append(goal)
        return _Future(_Handle())
