# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Odometry, carrying the one pose the probe places its goals at. See ../README.md."""

from __future__ import annotations

from geometry_msgs.msg import Pose


class _PoseWithCovariance:
    def __init__(self) -> None:
        self.pose = Pose()


class Odometry:
    def __init__(self) -> None:
        self.pose = _PoseWithCovariance()
