"""An edge estop's local latch is kept on the robot's own disk, and the adapter does not start
while it is there. Removing the file on the robot is the latch's only reset.

RFC-0001 (safety, *Edge / Headless Emergency Stop*, *The local latch*): the latch survives the
adapter's restart, and only a local reset at the robot lifts it. Before, the latch was kept in the
adapter process only, so any restart lifted it, a supervisor's included, and the restarted adapter
accepted work with nobody having confirmed the robot safe.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time

import pytest
from test_edge_estop_latch import _NeverAtRest
from test_estop_stop_retry import _adapter, _edge_estop, _navigate, _Robot, _until

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod

ROBOT = "amr-1"
ADAPTER = pathlib.Path(adapter_mod.__file__)


def _start(latch: pathlib.Path, log: pathlib.Path, started: str | None) -> tuple:
    """Start the adapter as a robot's supervisor would, against no control plane. Returns its
    exit code, or None if it was still running once `started` appeared in its stderr (it is then
    killed), and its stderr."""
    with open(log, "w") as err:
        p = subprocess.Popen([sys.executable, str(ADAPTER), "--endpoint", "localhost:1",
                              "--binding", "simulated", "--robot-id", ROBOT,
                              "--edge-latch-file", str(latch)],
                             stdout=subprocess.DEVNULL, stderr=err)
    end = time.monotonic() + 30.0
    rc = None
    while time.monotonic() < end:
        rc = p.poll()
        if rc is not None or (started and started in log.read_text()):
            break
        time.sleep(0.05)
    if rc is None:
        p.kill()
        p.wait(5.0)
    return rc, log.read_text()


def test_an_edge_estop_latches_across_a_restart_until_it_is_cleared_at_the_robot(tmp_path):
    latch = tmp_path / "state" / "edge-latch"
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file = str(latch)
    try:
        _edge_estop(a, "edge-1")
        assert _until(latch.exists, 2.0), "the latch was not written to the robot's disk"
        assert "edge estop edge-1" in latch.read_text()
    finally:
        a._shutdown.set()
    rc, err = _start(latch, tmp_path / "start-1.log", None)          # a restart
    assert rc == 2, f"the adapter started with the latch set: {err[-600:]}"
    assert "REFUSED: an edge estop set this robot's local latch" in err, err[-600:]
    assert f"rm {latch}" in err and "edge estop edge-1" in err
    latch.unlink()                                                    # the local reset
    rc, err = _start(latch, tmp_path / "start-2.log", "control stream:")
    assert rc is None and "REFUSED" not in err, f"cleared, the start was refused: {err[-600:]}"
    b = _adapter(_Robot(fail=0))                                      # and it takes work again
    b._edge_latch_file = str(latch)
    try:
        res = pb.CommandResult(command_id=1, robot_id=ROBOT)
        b._on_assign(_navigate(), res)
        assert res.assign_action.accepted, res.assign_action
    finally:
        b._shutdown.set()


def test_main_gives_the_adapter_its_latch_file(monkeypatch, tmp_path):
    seen: dict = {}

    class _Stop(Exception):
        pass

    def adapter(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop
    monkeypatch.setattr(adapter_mod, "fleet_adapter_ros2Adapter", adapter)
    monkeypatch.setattr(sys, "argv", ["fleet-adapter-ros2", "--binding", "simulated",
                                      "--robot-id", ROBOT, "--edge-latch-file",
                                      str(tmp_path / "latch-{robot_id}")])
    with pytest.raises(_Stop):
        adapter_mod.main()
    assert seen["edge_latch_file"] == str(tmp_path / f"latch-{ROBOT}")


def test_the_default_latch_file_is_the_robots_own():
    path = adapter_mod._edge_latch_path(adapter_mod.EDGE_LATCH_FILE_DEFAULT, "fleet/amr-1")
    assert path.endswith("/.local/state/swarmada/edge-latch-fleet_amr-1") and "~" not in path


def test_a_latch_file_that_cannot_be_checked_refuses_the_start(tmp_path):
    (tmp_path / "f").write_text("")
    why = adapter_mod._edge_latch_refusal(str(tmp_path / "f" / "edge-latch"))
    assert "cannot be checked" in why and "treated as set" in why, why
    assert adapter_mod._edge_latch_refusal(str(tmp_path / "absent")) == ""


def test_a_latch_that_cannot_be_written_is_said_and_still_refuses_work(tmp_path, capsys):
    (tmp_path / "f").write_text("")
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file = str(tmp_path / "f" / "edge-latch")
    try:
        _edge_estop(a, "edge-1")
        assert _until(lambda: "EDGE_LATCH_NOT_PERSISTED" in capsys.readouterr().err, 2.0)
        assert "edge estop edge-1 set this robot's local latch" in a._no_estop_path()
    finally:
        a._shutdown.set()


def test_the_latch_is_written_while_the_estop_is_still_confirming_and_a_dispatch_holds_the_lock(
        tmp_path):
    """Written by a thread of its own as the latch is set: not after the estop's answer (a goal
    driving keeps that 2 s away), and not behind the lease lock."""
    latch = tmp_path / "edge-latch"
    a = _adapter(_NeverAtRest(fail=0))
    a._edge_latch_file = str(latch)
    a._lease_lock.acquire()                          # a dispatch holds it
    try:
        threading.Thread(target=_edge_estop, args=(a, "edge-1"), daemon=True).start()
        assert _until(latch.exists, 0.5), "the latch waited for the answer or the lease lock"
    finally:
        a._lease_lock.release()
        a._shutdown.set()


def test_shutdown_writes_a_latch_not_written_yet(tmp_path):
    """An exit right after the edge estop: the latch is written before the adapter exits."""
    latch = tmp_path / "edge-latch"
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file = str(latch)
    a._edge_latched = "edge-1"                       # set, and no writer has run yet
    a.shutdown()
    assert latch.exists() and "edge estop edge-1" in latch.read_text()


def test_the_latch_files_directory_is_synced_after_the_rename(monkeypatch, tmp_path):
    """Without it, a power loss right after the edge stop can drop the rename, and a reboot
    starts the adapter as if no latch were set."""
    latch = tmp_path / "state" / "edge-latch"
    events: list = []
    real_open, real_fsync, real_replace = adapter_mod.os.open, adapter_mod.os.fsync, adapter_mod.os.replace
    opened: dict = {}

    def record_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        opened[fd] = str(path)
        return fd

    def record_fsync(fd):
        events.append(("fsync", opened.get(fd, "the file")))
        return real_fsync(fd)

    def record_replace(src, dst):
        events.append(("replace", str(dst)))
        return real_replace(src, dst)
    monkeypatch.setattr(adapter_mod.os, "open", record_open)
    monkeypatch.setattr(adapter_mod.os, "fsync", record_fsync)
    monkeypatch.setattr(adapter_mod.os, "replace", record_replace)
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file, a._edge_latched = str(latch), "edge-1"
    a._persist_edge_latch()
    monkeypatch.undo()
    assert latch.exists()
    after = events[events.index(("replace", str(latch))) + 1:]
    assert ("fsync", str(latch.parent)) in after, events


def test_an_empty_latch_file_is_said_at_start(monkeypatch, capsys):
    seen: dict = {}

    class _Stop(Exception):
        pass

    def adapter(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop
    monkeypatch.setattr(adapter_mod, "fleet_adapter_ros2Adapter", adapter)
    monkeypatch.setattr(sys, "argv", ["fleet-adapter-ros2", "--binding", "simulated",
                                      "--robot-id", ROBOT, "--edge-latch-file", ""])
    with pytest.raises(_Stop):
        adapter_mod.main()
    assert seen["edge_latch_file"] is None
    assert "edge estop latch: NOT kept on disk" in capsys.readouterr().err


def test_the_directories_the_write_creates_are_synced_in_their_parents(monkeypatch, tmp_path):
    """On a fresh robot the first write creates the latch's directories: each new entry is
    synced in its parent, so a power loss cannot drop a new directory with the file in it."""
    latch = tmp_path / "a" / "b" / "edge-latch"
    synced: list = []
    real_open, real_fsync = adapter_mod.os.open, adapter_mod.os.fsync
    opened: dict = {}

    def record_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        opened[fd] = str(path)
        return fd

    def record_fsync(fd):
        if fd in opened:
            synced.append(opened[fd])
        return real_fsync(fd)
    monkeypatch.setattr(adapter_mod.os, "open", record_open)
    monkeypatch.setattr(adapter_mod.os, "fsync", record_fsync)
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file, a._edge_latched = str(latch), "edge-1"
    a._persist_edge_latch()
    monkeypatch.undo()
    assert latch.exists()
    assert synced == [str(tmp_path / "a" / "b"), str(tmp_path / "a"), str(tmp_path)], synced


def test_an_edge_estop_during_shutdown_is_written_before_the_exit(tmp_path):
    """The latch set while shutdown looks for rest, after its steps were listed, is written by
    its last step."""
    latch = tmp_path / "edge-latch"
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file = str(latch)
    look = a._report_rest_at_shutdown

    def estop_while_looking(since):
        a._edge_latched = "edge-9"                   # set, and no writer has run yet
        look(since)
    a._report_rest_at_shutdown = estop_while_looking
    a.shutdown()
    assert latch.exists() and "edge estop edge-9" in latch.read_text()


def test_a_retried_write_syncs_the_directories_a_failed_write_created(monkeypatch, tmp_path):
    """A first write that creates the latch's directories and then fails leaves them existing;
    the retry (the estop's thread, or shutdown) must still sync each new entry in its parent, or
    a power loss can drop them with the file, and the reboot starts unlatched."""
    latch = tmp_path / "a" / "b" / "edge-latch"
    synced: list = []
    real_open, real_fsync, real_replace = adapter_mod.os.open, adapter_mod.os.fsync, adapter_mod.os.replace
    opened: dict = {}
    fail = [True]

    def record_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        opened[fd] = str(path)
        return fd

    def record_fsync(fd):
        if fd in opened:
            synced.append(opened[fd])
        return real_fsync(fd)

    def replace_once_failing(src, dst):
        if fail[0]:
            fail[0] = False
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)
    monkeypatch.setattr(adapter_mod.os, "open", record_open)
    monkeypatch.setattr(adapter_mod.os, "fsync", record_fsync)
    monkeypatch.setattr(adapter_mod.os, "replace", replace_once_failing)
    a = _adapter(_Robot(fail=0))
    a._edge_latch_file, a._edge_latched = str(latch), "edge-1"
    a._persist_edge_latch()                          # makes a/ and a/b/, then fails
    assert not latch.exists() and (tmp_path / "a" / "b").is_dir() and synced == []
    a._persist_edge_latch()                          # the retry
    monkeypatch.undo()
    assert latch.exists()
    assert synced == [str(tmp_path / "a" / "b"), str(tmp_path / "a"), str(tmp_path)], synced
