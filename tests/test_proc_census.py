# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A teardown check must tell RUNNING, DEFUNCT and ABSENT apart.

Two robot runs on 2026-09-18 -- the R1 interval run, and the rest-channel run that
followed it -- both counted launched processes by name after tearing them down by PGID,
read 0, and reported the robot clean while each group's leader sat `<defunct>`. These
tests pin one arm each, against the robot's own captured tables where they exist and
against real processes on this host, and pin the reason a name count cannot be the check.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

_TESTBED = pathlib.Path(__file__).resolve().parents[1] / "testbed"
if str(_TESTBED) not in sys.path:
    sys.path.insert(0, str(_TESTBED))

import proc_census as pc  # noqa: E402

HEADER = "    PID    PPID    PGID STAT COMMAND"

# The four PGIDs the count-by-name check read as 0, as `ps` printed them in the MentorPi
# container at 01:08:45Z on 2026-09-18 after the interval run's teardown (rows verbatim,
# ppid column added: the capture's awk selected by PGID, and all four had been reparented
# to PID 1),
# plus the vendor launch that teardown did not touch.
INTERVAL_RUN_AFTER = "\n".join([
    HEADER,
    "     31       1      31 Ssl  /usr/bin/python3 /opt/ros/humble/bin/ros2 launch bringup bringup.launch.py",
    "  19687       1   19687 Zs   [python3] <defunct>",
    "  21716       1   21716 Zs   [ros2] <defunct>",
    "  21747       1   21747 Zs   [ros2] <defunct>",
    "  28433       1   28433 Zs   [ros2] <defunct>",
])

# The rest-channel run's rf2o group before and after `kill -INT -24063`, from the robot's
# own process table captured on 2026-09-18. The node name in the row below is that run's
# own, as captured.
REST_RUN_BEFORE = "\n".join([
    HEADER,
    "  24063       1   24063 Ss   /usr/bin/python3 /opt/ros/humble/bin/ros2 run "
    "rf2o_laser_odometry rf2o_laser_odometry_node --ros-args -r __node:=rc1_rf2o",
    "  24064   24063   24063 Sl   /home/ubuntu/third_party_ros2/third_party_ws/install/"
    "rf2o_laser_odometry/lib/rf2o_laser_odometry/rf2o_laser_odometry_node",
])
REST_RUN_AFTER = "\n".join([HEADER, "  24063       1   24063 Zs   [ros2] <defunct>"])


# ── the three arms, on the robot's own tables ────────────────────────────────

def test_running_arm_rest_run_before_teardown() -> None:
    procs = pc.parse_ps(REST_RUN_BEFORE)
    v = pc.by_pgid(procs, 24063)
    assert v.state == pc.RUNNING
    assert [p.pid for p in v.running] == [24063, 24064]
    assert pc.by_name(procs, "rf2o").state == pc.RUNNING


def test_defunct_arm_rest_run_after_teardown_where_the_name_check_said_none_left() -> None:
    procs = pc.parse_ps(REST_RUN_AFTER)
    assert pc.by_pgid(procs, 24063).state == pc.DEFUNCT
    # What that run's `grep -E "rf2o|..." || echo "none left"` saw. The name query reports
    # the same zero -- and refuses to call it ABSENT.
    v = pc.by_name(procs, "rf2o")
    assert v.state == "NO LIVE MATCH"
    assert v.state != pc.ABSENT and "PGID" in v.note


@pytest.mark.parametrize("pgid", [19687, 21716, 21747, 28433])
def test_defunct_arm_every_torn_down_group_is_defunct_not_absent(pgid) -> None:
    v = pc.by_pgid(pc.parse_ps(INTERVAL_RUN_AFTER), pgid)
    assert v.state == pc.DEFUNCT
    assert len(v.defunct) == 1 and not v.running


def test_absent_arm_a_reaped_group() -> None:
    assert pc.by_pgid(pc.parse_ps(INTERVAL_RUN_AFTER), 24063).state == pc.ABSENT


def test_the_census_lists_every_zombie_even_unqueried() -> None:
    procs = pc.parse_ps(INTERVAL_RUN_AFTER)
    text = pc.render([pc.by_name(procs, "slam_toolbox")], procs)
    assert "defunct census: 4 defunct row(s)" in text
    for pgid in ("19687", "21716", "21747", "28433"):
        assert f"pgid {pgid}" in text


# ── why a name count is not the check ────────────────────────────────────────

def test_a_name_count_reads_zero_in_all_three_states_and_the_pgid_does_not() -> None:
    """The defect, pinned. The interval run's `slam_toolbox 0` is the same number whether
    the group is gone, defunct, or running under a name the pattern misses."""
    running_renamed = "\n".join([HEADER, "  21716       1   21716 Sl   /usr/bin/async_slam_node"])
    tables = {pc.RUNNING: running_renamed, pc.DEFUNCT: INTERVAL_RUN_AFTER,
              pc.ABSENT: "\n".join([HEADER, "     31       1      31 Ssl  ros2 launch"])}
    for expected, table in tables.items():
        procs = pc.parse_ps(table)
        assert len(pc.by_name(procs, "slam_toolbox").running) == 0      # the old check
        assert pc.by_pgid(procs, 21716).state == expected               # this one


def test_a_live_member_of_a_group_whose_leader_is_defunct_is_running() -> None:
    table = "\n".join([HEADER, "  21716       1   21716 Zs   [ros2] <defunct>",
                       "  21720       1   21716 Sl   /opt/ros/humble/lib/nav2_bt_navigator"])
    v = pc.by_pgid(pc.parse_ps(table), 21716)
    assert v.state == pc.RUNNING and len(v.running) == 1 and len(v.defunct) == 1


# ── exit codes and refusal ───────────────────────────────────────────────────

@pytest.mark.parametrize("table,argv,code", [
    (REST_RUN_BEFORE, ["--pgid", "24063"], 1),
    (REST_RUN_AFTER, ["--pgid", "24063"], 3),
    (INTERVAL_RUN_AFTER, ["--pgid", "24063", "--name", "rf2o"], 0),
    (INTERVAL_RUN_AFTER, ["--pgid", "24063", "--pgid", "21716"], 3),
])
def test_exit_code_separates_the_three(table, argv, code, capsys) -> None:
    assert pc.main(argv, io.StringIO(table)) == code
    assert "defunct census" in capsys.readouterr().out


def test_refuses_a_table_that_is_not_the_documented_columns(capsys) -> None:
    # The rest-channel capture's own column order (stat,pid,ppid,cmd): reading it as
    # pid,ppid,pgid,stat would print a confident, wrong census.
    wrong = "STAT PID PPID CMD\nZs 24063 1 [ros2] <defunct>"
    assert pc.main(["--pgid", "24063"], io.StringIO(wrong)) == 2
    assert "is not `ps -eo pid,ppid,pgid,stat,args`" in capsys.readouterr().err


# ── the three arms, on real processes on this host ───────────────────────────

@pytest.mark.skipif(shutil.which("ps") is None, reason="no ps on this host")
def test_each_arm_against_real_processes() -> None:
    """A group leader left unreaped (exactly the robot's shape: own session, parent that
    never waits), a live group, and a group that was reaped."""
    live = subprocess.Popen(["sleep", "30"], start_new_session=True)
    zombie = subprocess.Popen(["sh", "-c", "exit 0"], start_new_session=True)
    reaped = subprocess.Popen(["sh", "-c", "exit 0"], start_new_session=True)
    reaped.wait()
    try:
        deadline = time.monotonic() + 5.0
        while True:
            procs = pc.parse_ps(subprocess.run(pc.PS_COMMAND.split(), capture_output=True,
                                               text=True, check=True).stdout)
            if pc.by_pgid(procs, zombie.pid).state == pc.DEFUNCT or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        assert pc.by_pgid(procs, live.pid).state == pc.RUNNING
        assert pc.by_pgid(procs, zombie.pid).state == pc.DEFUNCT
        assert pc.by_pgid(procs, reaped.pid).state == pc.ABSENT
    finally:
        live.kill()
        live.wait()
        zombie.wait()


# ── the verdict is a visible line, through the runbook's own command ────────────

# What both robot runbooks now run after a teardown, from the Mac, with the robot's
# process table on stdin. The container is named because it is the process the
# teardown kills by PID; the PGID is the launch's.
RUNBOOK_ARGV = ["--pgid", "21716", "--name", "component_container_isolated"]


def _run_as_the_runbook_does(table: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_TESTBED / "proc_census.py"), *RUNBOOK_ARGV],
                          input=table, capture_output=True, text=True, check=False)


def test_a_defunct_group_fails_the_teardown_visibly() -> None:
    """The interval run's table: the launch group's leader is `<defunct>`. The old check
    printed `slam_toolbox 0`; the census must end on a FAIL line naming the group, and exit
    non-zero -- not leave the verdict to a reader who knows the exit codes."""
    r = _run_as_the_runbook_does(INTERVAL_RUN_AFTER)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 3
    assert last.startswith("FAIL  TEARDOWN NOT CLEAN: PGID 21716 DEFUNCT")
    assert "Do not record this teardown as clean" in last


def test_a_surviving_group_fails_the_teardown_visibly() -> None:
    running = "\n".join([HEADER, "  21716       1   21716 Sl   /opt/ros/humble/bin/ros2 launch",
                          "  21790   21716   21716 Sl   /opt/ros/humble/lib/rclcpp_components/"
                          "component_container_isolated --ros-args"])
    r = _run_as_the_runbook_does(running)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1
    assert last.startswith("FAIL  TEARDOWN NOT CLEAN: PGID 21716 RUNNING")
    assert "name /component_container_isolated/ RUNNING" in last


def test_a_clean_teardown_says_so_last() -> None:
    clean = "\n".join([HEADER, "     31       1      31 Ssl  ros2 launch bringup"])
    r = _run_as_the_runbook_does(clean)
    assert r.returncode == 0
    assert r.stdout.strip().splitlines()[-1].startswith("PASS  teardown clean: PGID 21716")


def test_an_unreadable_table_is_not_a_pass() -> None:
    r = _run_as_the_runbook_does("")
    assert r.returncode == 2
    assert r.stdout.strip().splitlines()[-1].startswith("FAIL  TEARDOWN NOT VERIFIED")
