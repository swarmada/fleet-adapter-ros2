# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A node that holds subscriptions and a clock. See ../README.md."""

from __future__ import annotations


class _Stamp:
    def to_msg(self):
        return self


class _Now:
    def to_msg(self):
        return _Stamp()


class _Clock:
    def now(self):
        return _Now()


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self._subs = []
        self.destroyed = False

    def create_subscription(self, msg_type, topic, callback, qos):
        self._subs.append((msg_type, callback))
        return object()

    def get_clock(self):
        return _Clock()

    def _deliver(self) -> None:
        """One message per subscription, so the probe can read a pose and send a placed goal."""
        for msg_type, callback in self._subs:
            callback(msg_type())

    def destroy_node(self) -> None:
        import rclpy
        rclpy._note_teardown("a node was destroyed")
        self.destroyed = True
