# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Enough of `rclpy` to run the goal-response probe's ROS path. See ../README.md.

WHAT THIS IMITATES is one thing: what happens when a node is destroyed, or a context shut
down, while a spin thread is still inside `spin_once()`. On a real runtime that spinner is
down in rcl holding the wait set it was handed, the teardown frees it, and the fault is
reported from C++ -- on a daemon thread with no Python handler above it, and at a point in
context destruction the process cannot continue from. It ends on `terminate called without an
active exception`, exit 134, with everything the probe measured already printed and written.

Two things about that shape are easy to get wrong in a stand-in, and both are load-bearing:

  * `spin_once` BLOCKS for its timeout. That wait is the window. A spinner that has passed its
    stop check sits in it, and a teardown that does not join has exactly that long to free the
    node underneath it. A stub returning at once holds the window shut and passes against the
    ordering it exists to catch.
  * The abort lands at EXIT, not where the fault is. The spinner is a daemon thread: the main
    thread prints, writes its JSON, returns 0, and the interpreter kills the spinner on the way
    out -- so timing alone never reaches it, and the run comes back 0 however wrong the order
    was. What kills the real process is context destruction finding state still in use. So the
    fault is recorded when the teardown lands on a spinner mid-wait, and acted on at exit.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading

_state = {"up": False, "in_spin": 0, "faulted": ""}
_lock = threading.Lock()


def init(*args, **kwargs) -> None:
    _state.update(up=True, in_spin=0, faulted="")


def ok() -> bool:
    return _state["up"]


def _note_teardown(what: str) -> None:
    """Record a teardown that landed while a spinner was inside `spin_once()`."""
    with _lock:
        if _state["in_spin"] > 0 and not _state["faulted"]:
            _state["faulted"] = what


def shutdown(*args, **kwargs) -> None:
    _note_teardown("the context was shut down")
    _state["up"] = False


def spin_once(node, timeout_sec: float = 0.0) -> None:
    """Wait like rcl does, then deliver. Reading the context AFTER the wait is deliberate."""
    with _lock:
        _state["in_spin"] += 1
    try:
        _sleep(max(0.0, timeout_sec))
        if not _state["up"] or getattr(node, "destroyed", False):
            _abort("spun a node whose context was gone")
        node._deliver()
    finally:
        with _lock:
            _state["in_spin"] -= 1


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _abort(why: str) -> None:
    sys.stderr.write(f"terminate called without an active exception ({why})\n")
    sys.stderr.flush()
    os._exit(134)


@atexit.register
def _abort_if_torn_down_mid_spin() -> None:
    if _state["faulted"]:
        _abort(_state["faulted"] + " while a spin thread was still in spin_once()")
