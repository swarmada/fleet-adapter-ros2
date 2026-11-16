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


# ── "the odom topic keeps publishing after a stop" ───────────────────────────

def odom_msg(vx: float = 0.0, wz: float = 0.0, x: float = 0.0, y: float = 0.0,
             yaw: float = 0.0, frame: str = "odom"):
    """A ROS-free ``nav_msgs/Odometry`` with this twist and pose, as ``Nav2Binding.on_odom``
    reads one."""
    import math
    from types import SimpleNamespace as NS
    return NS(header=NS(frame_id=frame),
              pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0),
                              orientation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2),
                                             w=math.cos(yaw / 2)))),
              twist=NS(twist=NS(linear=NS(x=vx, y=0.0, z=0.0),
                                angular=NS(x=0.0, y=0.0, z=wz))))


def odom_after_stop(binding, vx: float | None = None, wz: float | None = None):
    """Make a ROS-free ``Nav2Binding`` stand in for a robot whose odom topic keeps publishing
    once it is told to stop, and return it.

    Every stop is confirmed only from an odom message counted after the stop was commanded
    (``state_seq``; ``swarmada_sdk.safety.await_estop_confirmation``). A test that fed one
    odom message before the stop and none after is the case that confirmation now refuses, and
    rightly. A test that MEANS a live robot, at rest or moving after the stop, says so with this.

    Only after ``command_stop``, ``command_hold`` or ``command_pause`` has RETURNED: from then
    on each ``tick()`` first delivers one odom message, as the topic would between two looks.
    Nothing is delivered before the stop, or while the stop is being commanded, so the report
    that confirms a stop is never one received before it or during it.

    Every look therefore finds a new message, so a stop that can be confirmed is confirmed at
    its first look. A real topic publishes at its own rate and almost never lands between the
    stop and that look; `ThreadedOdom` below is that case.

    The message carries the robot's latest pose, frame and twist -- the topic goes on saying
    what it last said, moving or at rest, including what the test feeds later -- unless ``vx``
    (and ``wz``) give the twist. Nothing is delivered while the binding has had no odom at all.
    """
    armed = [False]
    for name in ("command_stop", "command_hold", "command_pause"):
        original = getattr(binding, name)

        def stop(robot_id, _original=original):
            result = _original(robot_id)
            armed[0] = True               # returned: a stop that raised arms nothing
            return result
        setattr(binding, name, stop)
    tick = binding.tick

    def tick_with_odom(dt):
        if armed[0]:
            with binding._lock:
                speed, pos, frame = binding._speed, binding._position, binding._odom_frame
            if speed is not None and pos is not None:     # None: no odom yet, nothing to say
                lin, ang = (vx or 0.0, wz or 0.0) if (vx, wz) != (None, None) else speed
                binding.on_odom(odom_msg(lin, ang, pos.x, pos.y, pos.yaw, frame))
        return tick(dt)
    binding.tick = tick_with_odom
    return binding


class ThreadedOdom:
    """A live odom topic for a ROS-free ``Nav2Binding``: one message every ``1/hz`` s from a thread
    of its own, as a robot publishes them, whatever the adapter is doing at the time.

    The plant behind it cruises at ``cruise`` m/s until a stop (``command_stop``,
    ``command_hold`` or ``command_pause``) has returned, then brakes to rest over ``brake_s``, as
    a real base does. So the first look at a stop sees a robot still moving, and rest is
    confirmed only from a message published once the base has stopped. 30 Hz is the Gazebo
    rig's measured odom rate. ``close()`` ends the thread.
    """

    def __init__(self, binding, hz: float = 30.0, cruise: float = 0.4,
                 brake_s: float = 0.3) -> None:
        import threading
        import time
        self._time = time
        self.binding, self.period, self.cruise, self.brake_s = binding, 1.0 / hz, cruise, brake_s
        self.stopped_at: float | None = None
        self._end = threading.Event()
        for name in ("command_stop", "command_hold", "command_pause"):
            original = getattr(binding, name)

            def stop(robot_id, _original=original):
                result = _original(robot_id)
                if self.stopped_at is None:
                    self.stopped_at = time.monotonic()
                return result
            setattr(binding, name, stop)
        binding.on_odom(odom_msg(cruise))
        self._thread = threading.Thread(target=self._run, daemon=True, name="threaded-odom")
        self._thread.start()

    def speed(self) -> float:
        if self.stopped_at is None:
            return self.cruise
        braked = (self._time.monotonic() - self.stopped_at) / self.brake_s
        return max(0.0, self.cruise * (1.0 - braked))

    def _run(self) -> None:
        while not self._end.wait(self.period):
            self.binding.on_odom(odom_msg(self.speed()))

    def close(self) -> None:
        self._end.set()
        self._thread.join(timeout=2.0)
