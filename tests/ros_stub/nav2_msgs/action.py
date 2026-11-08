# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The NavigateToPose goal the probe fills in. See ../README.md."""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped


class NavigateToPose:
    class Goal:
        def __init__(self) -> None:
            self.pose = PoseStamped()
