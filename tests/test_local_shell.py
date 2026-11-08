# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter inside the robot's container runs the same scripts, locally.

Placed in the container, the adapter has rclpy and the robot's DDS graph at hand, and no ssh
or ``docker exec`` to reach anything: it is already where the scripts run. ``--shell local``
runs them there, in the same three forms -- ``bash -s`` with the script on stdin, ``bash -c``
when stdin carries a file, ``bash -c`` detached for a launch -- and with the same text as over
ssh, so the ssh-mode proofs of each script still hold. Two things change, and both are pinned:

* each script starts from a clean environment (``env -i``, the DDS settings kept), because the
  adapter's own PYTHONPATH and its protobuf must not reach a ROS launch, as ``docker exec``
  never let them;
* a launch is detached under ``setsid -f`` with its output discarded, so it outlives the call
  and the adapter's process group.

In the container there is no host, user or container to reach, so those site keys drop out;
the bounds are read from ``<workdir>/bounds``.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import threading

import pytest
from test_ros2_explore import (
    ROS2,
    SITE,
    Clock,
    FakeRos,
    _bound_dir,
    _saved_files,
    _site,
    _site_file,
)
from test_stall_reporting import _main

from fleet_adapter_ros2 import explore, ros2_explore
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2.nav2_binding import Nav2Binding

LOCAL_SITE = {k: SITE[k] for k in ros2_explore.SiteConfig.LOCAL_KEYS}


class Recorder:
    """A subprocess.run stand-in that records argv and stdin and answers from rules keyed by a
    substring of the SCRIPT TEXT, so both modes get the same answers."""

    def __init__(self, rules=()) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.rules = list(rules)

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        text = script_of(argv, kw)
        out = b""
        for needle, answer in self.rules:
            if needle in text:
                out = answer
                break

        class R:
            returncode, stdout, stderr = 0, out, b""
        return R()


def script_of(argv, kw) -> str:
    """The script a call runs, whichever mode built it."""
    if argv[0] == "ssh":
        remote = shlex.split(argv[2])
        return remote[-1] if remote[-2] == "-c" else kw["input"].decode()
    return argv[-1] if argv[-2] == "-c" else kw["input"].decode()


def detached(argv) -> bool:
    if argv[0] == "ssh":
        return shlex.split(argv[2])[:4] == ["sudo", "docker", "exec", "-d"]
    return argv[0] == "setsid"


def _local_stack(runner, environ=None, tmp_path=None):
    ros = FakeRos()
    ros.answers["/controller_server/get_parameters"] = {"values": [{"type": 3,
                                                                    "double_value": 0.2}]}
    ros.answers["/slam_toolbox/serialize_map"] = {"result": 0}
    values = dict(LOCAL_SITE)
    if tmp_path is not None:
        values["SWARMADA_ROBOT_WORKDIR"] = str(tmp_path)
    site = ros2_explore.SiteConfig(values, mode="local")
    shell = ros2_explore.RobotShell(site, runner=runner, environ=environ or {})
    st = ros2_explore.Ros2ExploreStack(
        binding=Nav2Binding(), robot_id="amr-1", site=site, shell=shell, ros=ros,
        clock=Clock(ros), limits=explore.BatteryLimits(7750, 8300, 7835))
    st.arm(threading.Event())
    return st, shell


# ── the forms ───────────────────────────────────────────────────────────────

def test_a_local_script_goes_on_stdin_to_bash_s_from_a_clean_environment() -> None:
    rec = Recorder()
    shell = ros2_explore.RobotShell(_site(), runner=rec, mode="local",
                                    environ={"HOME": "/home/r", "PYTHONPATH": "/deps",
                                             "ROS_DOMAIN_ID": "7", "SECRET_THING": "x"})
    assert shell.run("pgrep -fc explore") == (0, b"", b"")
    argv, kw = rec.calls[0]
    assert argv[:2] == ["env", "-i"] and argv[-2:] == ["bash", "-s"]
    assert kw["input"] == b"pgrep -fc explore"
    assert "HOME=/home/r" in argv and "ROS_DOMAIN_ID=7" in argv
    assert not any(a.startswith(("PYTHONPATH=", "SECRET_THING=")) for a in argv)
    assert "ssh" not in argv and "docker" not in argv


def test_a_local_file_upload_is_bash_c_with_the_file_on_stdin() -> None:
    rec = Recorder()
    shell = ros2_explore.RobotShell(_site(), runner=rec, mode="local", environ={})
    shell.run("mkdir -p /w && cat > /w/f", stdin=b"xyz")
    argv, kw = rec.calls[0]
    assert argv[-3:] == ["bash", "-c", "mkdir -p /w && cat > /w/f"] and kw["input"] == b"xyz"


def test_a_local_launch_is_detached_under_setsid_with_its_output_discarded() -> None:
    rec = Recorder()
    shell = ros2_explore.RobotShell(_site(), runner=rec, mode="local", environ={})
    assert shell.run("exec ros2 launch x", detach=True) == (0, b"", b"")
    argv, kw = rec.calls[0]
    assert argv[:4] == ["setsid", "-f", "env", "-i"]
    assert argv[-3:] == ["bash", "-c", "exec ros2 launch x"]
    for stream in ("stdin", "stdout", "stderr"):
        assert kw[stream] is subprocess.DEVNULL
    assert "capture_output" not in kw


def test_a_local_shell_that_cannot_run_is_no_answer() -> None:
    def runner(argv, **kw):
        raise OSError("no bash")
    assert ros2_explore.RobotShell(_site(), runner=runner, mode="local").run("true") is None


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        ros2_explore.RobotShell(_site(), mode="docker")


# ── the same script text in both modes ──────────────────────────────────────

def _drive(st) -> None:
    """One pass over every stack operation that reaches the shell."""
    st.read_param("/controller_server", "FollowPath.max_vel_x")
    st.start_slam("/work/kit/out/map/lab__explore-first")
    st.load_keepout_mask(steps.Mask(width=2, height=1, res=0.1, ox=0.0, oy=0.0,
                                    keep=[True, False]), 0.0)
    st.activate_keepout()
    st.restart_keepout()
    st.start_explorer()
    st.save_map("lab/explore-first")
    st.end_segment()


def test_the_local_and_ssh_modes_run_the_same_script_text() -> None:
    base = "/work/kit/out/map/"
    files = _saved_files()
    rules = [("map_saver_cli", b"map_saver rc=0\n"),
             ("md5sum " + base, "".join(f"{hashlib.md5(d).hexdigest()}  {base}{n}\n"
                                        for n, d in files.items()).encode()),
             *((f"cat {base}{n}", d) for n, d in files.items()),
             ("grep -c", b"1\n")]
    texts = {}
    for mode in ros2_explore.SHELL_MODES:
        rec = Recorder(rules)
        ros = FakeRos()
        ros.answers["/slam_toolbox/serialize_map"] = {"result": 0}
        site = ros2_explore.SiteConfig(SITE, mode=mode)
        st = ros2_explore.Ros2ExploreStack(
            binding=Nav2Binding(), robot_id="amr-1", site=site,
            shell=ros2_explore.RobotShell(site, runner=rec, environ={}), ros=ros,
            clock=Clock(ros), limits=explore.BatteryLimits(7750, 8300, 7835))
        st.arm(threading.Event())
        _drive(st)
        texts[mode] = [(script_of(a, kw), detached(a)) for a, kw in rec.calls]
    assert len(texts["local"]) > 10
    assert texts["local"] == texts["ssh"]


# ── the site keys ───────────────────────────────────────────────────────────

def test_in_the_container_only_the_work_directory_env_script_and_overlay_are_needed() -> None:
    site = ros2_explore.SiteConfig(LOCAL_SITE, mode="local")
    assert site.workdir == SITE["SWARMADA_ROBOT_WORKDIR"]
    assert site.bounds_dir == SITE["SWARMADA_ROBOT_WORKDIR"] + "/bounds"
    assert site.host == site.ssh_user == site.container == site.container_user == ""


def test_in_the_container_a_bounds_directory_named_for_another_host_is_not_read() -> None:
    site = ros2_explore.SiteConfig({**LOCAL_SITE, "SWARMADA_BOUNDS_DIR": "/elsewhere"},
                                   mode="local")
    assert site.bounds_dir == SITE["SWARMADA_ROBOT_WORKDIR"] + "/bounds"


def test_over_ssh_every_key_is_still_needed() -> None:
    with pytest.raises(ros2_explore.SiteError, match="SWARMADA_ROBOT_HOST"):
        ros2_explore.SiteConfig(LOCAL_SITE, mode="ssh")


def test_a_local_site_missing_a_key_names_only_that_key() -> None:
    partial = {k: v for k, v in LOCAL_SITE.items() if k != "SWARMADA_ROBOT_OVERLAY"}
    with pytest.raises(ros2_explore.SiteError) as exc:
        ros2_explore.SiteConfig(partial, mode="local")
    assert "SWARMADA_ROBOT_OVERLAY" in str(exc.value)
    assert "SWARMADA_ROBOT_HOST" not in str(exc.value)
    for v in LOCAL_SITE.values():
        assert v not in str(exc.value)


def test_the_keys_can_come_from_the_environment() -> None:
    site = ros2_explore.SiteConfig.load("env", "local", environ={**LOCAL_SITE, "OTHER": "1"})
    assert site.env_script == SITE["SWARMADA_ROBOT_ENV_SCRIPT"]


def test_in_the_container_the_bound_is_read_from_the_work_directory(tmp_path) -> None:
    (tmp_path / "bounds").mkdir()
    _bound_dir(tmp_path / "bounds")
    st, _ = _local_stack(Recorder(), tmp_path=tmp_path)
    assert st.load_bound("room-1").theta0_deg == -60.639


# ── start-up ────────────────────────────────────────────────────────────────

def test_shell_local_starts_with_only_the_local_keys(monkeypatch, capsys, tmp_path) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--shell", "local",
                             "--site-env", _site_file(tmp_path, LOCAL_SITE),
                             "--battery-hard-mv", "7750", "--battery-full-mv", "8300")
    assert code is None
    stack = robot.explore_stack()
    assert stack.shell.mode == "local" and stack.site.mode == "local"
    assert "from inside its container (--shell local)" in err
    for v in LOCAL_SITE.values():
        assert v not in err


def test_shell_local_reads_the_keys_from_its_environment(monkeypatch, capsys) -> None:
    for k, v in LOCAL_SITE.items():
        monkeypatch.setenv(k, v)
    code, _, robot = _main(monkeypatch, capsys, *ROS2, "--shell", "local", "--site-env", "env",
                           "--battery-hard-mv", "7750", "--battery-full-mv", "8300")
    assert code is None and robot.explore_stack().site.workdir == LOCAL_SITE[
        "SWARMADA_ROBOT_WORKDIR"]


def test_the_default_shell_is_still_ssh(monkeypatch, capsys, tmp_path) -> None:
    code, _, robot = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                             "--battery-hard-mv", "7750", "--battery-full-mv", "8300")
    assert code is None and robot.explore_stack().shell.mode == "ssh"


# ── for real, on this machine's own bash ────────────────────────────────────

@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("pgrep") is None,
                    reason="needs bash and pgrep")
def test_a_real_local_script_sees_none_of_the_adapters_environment(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/adapter/deps")
    shell = ros2_explore.RobotShell(_site(), mode="local")
    r = shell.run('echo "pp=[$PYTHONPATH] path=[$PATH]"')
    assert r is not None and r[0] == 0
    out = r[1].decode()
    assert "pp=[]" in out and "/adapter/deps" not in out


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("pgrep") is None,
                    reason="needs bash and pgrep")
def test_a_real_local_pgrep_does_not_count_its_own_shell() -> None:
    # The script is on stdin, so no process's command line holds the pattern: nothing matches.
    marker = f"no-such-process-{os.getpid()}-marker"
    r = ros2_explore.RobotShell(_site(), mode="local").run(f"pgrep -f {marker}; echo rc=$?")
    assert r is not None and r[1].decode().split() == ["rc=1"], r
