# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Pose types, with the fields the probe reads and writes. See ../README.md."""

from __future__ import annotations


class Point:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class Quaternion:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0
        self.w = 1.0            # facing +x, so a goal placed ahead is ahead


class Pose:
    def __init__(self) -> None:
        self.position = Point()
        self.orientation = Quaternion()


class Header:
    def __init__(self) -> None:
        self.frame_id = ""
        self.stamp = None


class PoseStamped:
    def __init__(self) -> None:
        self.header = Header()
        self.pose = Pose()
