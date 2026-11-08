# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The simulated stack's tour: what a map viewer sees the simulated robot do.

Without a tour the robot stays on the mark and the known radius grows, as before. With one,
the robot is shown driving the tour while it explores, never faster than the tour's speed,
the map becomes known round where it has been, and the return brings it back to the mark.
"""

from __future__ import annotations

import itertools
import math
import threading

from fleet_adapter_ros2.sim_explore import SimulatedExploreStack


class ManualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.t += seconds
        return event.is_set()


def _known(stack) -> int:
    return sum(v != -1 for v in stack.read_map().data)


def test_without_a_tour_the_robot_stays_on_the_mark():
    clock = ManualClock()
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=12.5)
    stack.start_slam(None)
    stack.start_explorer()
    for _ in range(20):
        clock.t += 1.0
        assert stack.robot_pose() == (0.0, 0.0, 12.5)


def test_with_a_tour_the_robot_drives_it_within_its_speed_and_the_map_follows():
    clock = ManualClock()
    tour = [(3.0, 0.0), (3.0, 2.5), (-1.0, 2.5)]
    stack = SimulatedExploreStack(clock=clock, boot_yaw_deg=12.5, tour=tour, tour_speed_mps=0.18,
                                  explore_seconds=200.0, return_seconds=4.0)
    stack.start_slam(None)
    assert stack.robot_pose() == (0.0, 0.0, 12.5)          # on the mark, at the boot's yaw
    known0 = _known(stack)
    stack.start_explorer()
    poses = []
    for _ in range(60):
        clock.t += 1.0
        poses.append(stack.robot_pose())
    steps = [math.dist(a[:2], b[:2]) for a, b in itertools.pairwise(poses)]
    assert max(steps) <= 0.18 + 1e-9 and sum(steps) > 8.0
    assert max(p[0] for p in poses) > 2.9 and max(p[1] for p in poses) > 2.4
    assert _known(stack) > known0
    stack.stop_explorer()
    stack.go_to_mark(0.0, 0.0, 12.5)
    clock.t += 2.0
    mid = stack.robot_pose()
    assert 0.0 < math.hypot(mid[0], mid[1]) < math.hypot(poses[-1][0], poses[-1][1])
    clock.t += 2.5
    assert stack.mark_result() == "SUCCEEDED"
    assert stack.robot_pose() == (0.0, 0.0, 12.5)
