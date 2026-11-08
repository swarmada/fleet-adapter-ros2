# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The live exploration stack of the ROS 2 binding, against fakes of everything it touches.

`Ros2ExploreStack` holds the logic; two narrow seams hold everything that needs a robot:

* `RobotShell` starts and stops the robot-side processes the way the supervised runs did --
  `ssh <user>@<robot> sudo docker exec [-d|-i] -u <user> <container> bash -c '<script>'`, the
  script sourcing the robot's environment and `exec`ing `ros2 launch` / `ros2 run` -- and moves
  files, md5 at each hop. Every site value comes from the site file read at run time.
* `RosFacade` is the few rclpy calls the stack makes: a service call, the latest or drained
  messages of a topic, a publish, a publisher count, a transform.

Here both are fakes, so these tests pin what the stack asks and how it judges each answer. What
they cannot show -- that the real facade reaches a real graph, that the launch commands start
real processes -- is for the stand-in and the robot.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
from conftest import mark_transport_live

from fleet_adapter_ros2 import explore, ros2_explore
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2 import task as task_map
from fleet_adapter_ros2.nav2_binding import Nav2Binding

ROBOT = "amr-1"

SITE = {
    "SWARMADA_ROBOT_HOST": "robot.test", "SWARMADA_ROBOT_SSH_USER": "op-7f",
    "SWARMADA_ROBOT_CONTAINER": "ctr-9q", "SWARMADA_ROBOT_CONTAINER_USER": "run-3k",
    "SWARMADA_ROBOT_WORKDIR": "/work/kit", "SWARMADA_ROBOT_ENV_SCRIPT": "/work/kit/env.sh",
    "SWARMADA_ROBOT_OVERLAY": "/work/ws/overlay.sh", "SWARMADA_BOUNDS_DIR": "/bounds",
}


def _site(**over) -> ros2_explore.SiteConfig:
    return ros2_explore.SiteConfig({**SITE, **over})


# ── fakes ───────────────────────────────────────────────────────────────────

class FakeShell:
    """Records every remote script; answers from a list of rules (substring -> answer)."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, bool, bytes | None]] = []
        self.rules: list[tuple[str, object]] = []
        self.files: dict[str, bytes] = {}

    def on(self, needle: str, answer) -> None:
        self.rules.insert(0, (needle, answer))

    def run(self, script: str, *, detach: bool = False, stdin: bytes | None = None,
            timeout: float = 60.0):
        self.runs.append((script, detach, stdin))
        if stdin is not None and script.startswith("mkdir -p") and "cat >" in script:
            self.files[script.split("cat > ")[1].strip()] = stdin
        for needle, answer in self.rules:
            if needle in script:
                return answer(script) if callable(answer) else answer
        return (0, b"", b"")

    def scripts(self) -> list[str]:
        return [r[0] for r in self.runs]


class FakeRos:
    def __init__(self) -> None:
        self.t = 0.0
        self.calls: list[tuple[str, str, dict]] = []
        self.answers: dict[str, object] = {}
        self.published: list[tuple[str, dict]] = []
        self.latest_msgs: dict[str, tuple[dict, float]] = {}
        self.queues: dict[str, list[dict]] = {}
        self.windows: dict[str, list[tuple[float, dict]]] = {}
        self.publishers: dict[str, int | None] = {}
        self.tf = (0.0, 0.0, 30.0)

    def now(self) -> float:
        return self.t

    def call(self, service, srv_type, fields, timeout=10.0):
        self.calls.append((service, srv_type, fields))
        a = self.answers.get(service)
        return a(fields) if callable(a) else a

    def latest(self, topic):
        return self.latest_msgs.get(topic)

    def drain(self, topic):
        q, self.queues[topic] = self.queues.get(topic, []), []
        return q

    def recent(self, topic, window_s):
        return [(t, m) for t, m in self.windows.get(topic, []) if self.t - t <= window_s]

    def publish(self, topic, msg_type, fields):
        self.published.append((topic, fields))

    def publisher_count(self, topic):
        return self.publishers.get(topic, 0)

    def transform(self, target, source):
        return self.tf


class Clock:
    def __init__(self, ros: FakeRos) -> None:
        self.ros = ros

    def now(self) -> float:
        return self.ros.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.ros.t += seconds
        return event.is_set()


def keepout_filter_answers(local: dict | None = None):
    """A `get_parameters` answer for a costmap carrying the keep-out filter, as the robot's
    overlay gives both planning costmaps; ``local`` overrides a parameter's value."""
    values = {"filters": {"type": 9, "string_array_value": ["keepout_filter"]},
              "keepout_filter.plugin": {"type": 4,
                                        "string_value": "nav2_costmap_2d::KeepoutFilter"},
              "keepout_filter.enabled": {"type": 1, "bool_value": True},
              "keepout_filter.filter_info_topic": {"type": 4,
                                                   "string_value": "/costmap_filter_info"},
              **(local or {})}
    return lambda fields: {"values": [values.get(n, {"type": 0}) for n in fields["names"]]}


def give_both_costmaps_the_keepout_filter(ros) -> None:
    for node in ("/global_costmap/global_costmap", "/local_costmap/local_costmap"):
        ros.answers[f"{node}/get_parameters"] = keepout_filter_answers()


def _stack(tmp_path=None, **kw):
    ros, shell = FakeRos(), FakeShell()
    b = mark_transport_live(Nav2Binding())
    sent = []
    b._send_nav2_goal = lambda goal, epoch: sent.append(goal)
    b._server_ready = lambda: True
    b.spawn(ROBOT)
    site = _site(SWARMADA_BOUNDS_DIR=str(tmp_path)) if tmp_path else _site()
    st = ros2_explore.Ros2ExploreStack(
        binding=b, robot_id=ROBOT, site=site, shell=shell, ros=ros, clock=Clock(ros),
        limits=explore.BatteryLimits(hard_mv=7750, full_mv=8300, return_mv=7835), **kw)
    st.arm(threading.Event())
    return st, ros, shell, b, sent


# ── the site file ───────────────────────────────────────────────────────────

def test_the_site_file_is_read_at_run_time_and_quotes_and_comments_are_handled(tmp_path) -> None:
    f = tmp_path / "site.env"
    f.write_text("# a comment\n" + "\n".join(f'{k}="{v}"' for k, v in SITE.items())
                 + "\nexport OTHER=1\n")
    cfg = ros2_explore.SiteConfig.load(str(f))
    assert cfg.host == "robot.test" and cfg.workdir == "/work/kit"


def test_a_site_file_missing_keys_names_the_keys_and_never_a_value(tmp_path) -> None:
    f = tmp_path / "site.env"
    f.write_text("SWARMADA_ROBOT_HOST=secret-host\n")
    with pytest.raises(ros2_explore.SiteError) as exc:
        ros2_explore.SiteConfig.load(str(f))
    assert "SWARMADA_ROBOT_CONTAINER" in str(exc.value)
    assert "secret-host" not in str(exc.value)


def test_no_site_value_is_written_into_the_code() -> None:
    src = Path(ros2_explore.__file__).read_text()
    for v in ("MentorPi", "pi@", "/tmp/hwc", "ubuntu"):
        assert v not in src


# ── the shell: the supervised runs' mechanism ───────────────────────────────

def test_the_shell_runs_docker_exec_over_ssh_with_the_site_values() -> None:
    seen = []

    def runner(argv, **kw):
        seen.append((argv, kw))

        class R:
            returncode, stdout, stderr = 0, b"ok", b""
        return R()
    shell = ros2_explore.RobotShell(_site(), runner=runner)
    assert shell.run("echo 'a b'", detach=True) == (0, b"ok", b"")
    argv = seen[0][0]
    assert argv[:2] == ["ssh", "op-7f@robot.test"]
    assert argv[2] == "sudo docker exec -d -u run-3k ctr-9q bash -c 'echo '\"'\"'a b'\"'\"''"
    shell.run("cat > f", stdin=b"xyz")
    assert "-i -u run-3k ctr-9q bash -c" in seen[1][0][2] and seen[1][1]["input"] == b"xyz"
    # Any other script goes on stdin to `bash -s`, so a pgrep in it cannot match its own shell.
    shell.run("pgrep -fc explore")
    assert seen[2][0][2] == "sudo docker exec -i -u run-3k ctr-9q bash -s"
    assert seen[2][1]["input"] == b"pgrep -fc explore"


def test_an_ssh_that_cannot_run_is_no_answer_not_an_answer() -> None:
    def runner(argv, **kw):
        raise OSError("no route")
    assert ros2_explore.RobotShell(_site(), runner=runner).run("true") is None


# ── reads ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, want", [
    ({"type": 3, "double_value": 0.2}, "0.2"),
    ({"type": 8, "double_array_value": [0.2, 0.0, 1.0]}, "[0.2, 0.0, 1.0]"),
    ({"type": 2, "integer_value": 3}, "3"),
    ({"type": 0}, "not set"),
])
def test_a_parameter_reads_as_ros2_param_get_prints_it(value, want) -> None:
    st, ros, *_ = _stack()
    ros.answers["/controller_server/get_parameters"] = {"values": [value]}
    assert st.read_param("/controller_server", "FollowPath.max_vel_x") == want
    assert ros.calls[-1] == ("/controller_server/get_parameters",
                             "rcl_interfaces/srv/GetParameters",
                             {"names": ["FollowPath.max_vel_x"]})


def test_a_parameter_read_with_no_reply_is_no_answer() -> None:
    st, *_ = _stack()
    assert st.read_param("/controller_server", "FollowPath.max_vel_x") is None


# ── SLAM ────────────────────────────────────────────────────────────────────

def test_slam_starts_with_slam_toolbox_s_own_launch_detached_and_is_up_when_one_runs() -> None:
    st, ros, shell, *_ = _stack()
    shell.on("pgrep -fc async_slam_toolbox_node", (0, b"1\n", b""))
    assert st.start_slam(None) is True
    launch = [r for r in shell.runs if "online_async_launch.py" in r[0]]
    assert len(launch) == 1 and launch[0][1] is True
    assert "source /work/kit/env.sh && exec ros2 launch slam_toolbox online_async_launch.py" \
        in launch[0][0] and "slam_params_file:=/work/kit/map-kit/slam_params.yaml" in launch[0][0]
    assert ros.t >= ros2_explore.SLAM_SETTLE_S


def test_slam_not_up_yet_is_no_answer_and_a_retry_does_not_launch_again() -> None:
    st, _, shell, *_ = _stack()
    shell.on("pgrep -fc async_slam_toolbox_node", (0, b"0\n", b""))
    assert st.start_slam(None) is None
    assert st.start_slam(None) is None
    assert sum("online_async_launch.py" in s for s in shell.scripts()) == 1


def test_two_slam_processes_are_refused() -> None:
    st, _, shell, *_ = _stack()
    shell.on("pgrep -fc async_slam_toolbox_node", (0, b"2\n", b""))
    assert st.start_slam(None) is False


def test_a_resume_rewrites_the_resume_parameters_to_the_saved_graph() -> None:
    st, _, shell, *_ = _stack()
    shell.on("pgrep -fc async_slam_toolbox_node", (0, b"1\n", b""))
    shell.on("grep -c", (0, b"1\n", b""))
    assert st.start_slam("/work/kit/out/map/lab__explore-first") is True
    sed = next(s for s in shell.scripts() if s.startswith("sed "))
    assert "/work/kit/map-kit/slam_params_resume.yaml" in sed
    assert "/work/kit/out/map/lab__explore-first" in sed
    launch = next(s for s in shell.scripts() if "online_async_launch.py" in s)
    assert "slam_params_file:=/work/kit/map-kit/slam_params_resume-live.yaml" in launch


@pytest.mark.parametrize("graph", ["../etc/passwd", "/work/kit/out/map/a b", "$(reboot)",
                                   "/elsewhere/out/map/x"])
def test_a_graph_reference_that_is_not_a_saved_graph_is_refused(graph) -> None:
    st, _, shell, *_ = _stack()
    assert st.start_slam(graph) is False and shell.runs == []


def test_the_pose_is_the_map_to_base_footprint_transform() -> None:
    st, ros, *_ = _stack()
    ros.tf = (0.1, -0.2, -73.85)
    assert st.robot_pose() == (0.1, -0.2, -73.85)
    ros.tf = None
    assert st.robot_pose() is None


# ── the keep-out ────────────────────────────────────────────────────────────

def _mask() -> steps.Mask:
    return steps.Mask(width=4, height=3, res=0.1, ox=-0.2, oy=-0.1,
                      keep=[True] * 4 + [True, False, False, True] + [True] * 4)


def test_the_mask_is_pushed_and_its_md5_proven_on_the_robot() -> None:
    st, _, shell, *_ = _stack()
    m = _mask()
    pgm, yaml = steps.mask_to_pgm(m), steps.mask_yaml(m, "keepout.pgm")
    md5 = f"{hashlib.md5(pgm).hexdigest()}  keepout.pgm\n{hashlib.md5(yaml).hexdigest()}  " \
          f"keepout.yaml\n".encode()
    shell.on("md5sum keepout.pgm keepout.yaml", (0, md5, b""))
    assert st.load_keepout_mask(m, 12.5) is True
    assert shell.files["/work/kit/hws3/mask-live/keepout.pgm"] == pgm
    assert shell.files["/work/kit/hws3/mask-live/keepout.yaml"] == yaml


def test_mask_bytes_that_differ_on_the_robot_are_refused() -> None:
    st, _, shell, *_ = _stack()
    shell.on("md5sum keepout.pgm keepout.yaml", (0, b"0" * 32 + b"  keepout.pgm\n", b""))
    assert st.load_keepout_mask(_mask(), 0.0) is False


KEEPOUT_NODES = ("keepout_mask_server", "keepout_filter_info_server")
UNCONFIGURED, INACTIVE, ACTIVE = 1, 2, 3
CONFIGURE, ACTIVATE = 1, 3


class Lifecycle:
    """The keep-out servers' lifecycle services. ``drop`` names how many change_state replies
    to lose per node; ``applies`` says whether a lost reply's transition still happened (the
    robot's case: the server did it, the reply never arrived)."""

    def __init__(self, ros: FakeRos, state: int = ACTIVE, drop: int = 0,
                 applies: bool = True) -> None:
        self.state = dict.fromkeys(KEEPOUT_NODES, state)
        self.drop = dict.fromkeys(KEEPOUT_NODES, drop)
        self.applies, self.changes = applies, []
        for node in KEEPOUT_NODES:
            ros.answers[f"/{node}/get_state"] = (
                lambda f, n=node: {"current_state": {"id": self.state[n], "label": ""}})
            ros.answers[f"/{node}/change_state"] = lambda f, n=node: self._change(n, f)

    def _change(self, node: str, fields: dict):
        t = fields["transition"]["id"]
        self.changes.append((node, t))
        ok = (self.state[node], t) in ((UNCONFIGURED, CONFIGURE), (INACTIVE, ACTIVATE))
        if self.drop[node]:
            self.drop[node] -= 1
            if ok and self.applies:
                self.state[node] += 1
            return None
        if ok:
            self.state[node] += 1
        return {"success": ok}


def test_the_keep_out_servers_come_up_when_both_are_active_and_the_filter_info_publishes() -> None:
    st, ros, shell, *_ = _stack()
    Lifecycle(ros)
    ros.publishers["/costmap_filter_info"] = 0
    assert st.activate_keepout() is None                    # the lost-reply case
    assert st.activate_keepout() is None                    # read again, not relaunched
    assert sum("keepout_servers.launch.py" in s for s in shell.scripts()) == 1
    ros.publishers["/costmap_filter_info"] = 1
    assert st.activate_keepout() is True
    launch = next(r for r in shell.runs if "keepout_servers.launch.py" in r[0])
    assert launch[1] and "mask:=/work/kit/hws3/mask-live/keepout.yaml" in launch[0]


def test_a_filter_info_publisher_alone_is_not_up_while_a_server_is_not_active() -> None:
    # The robot's failure: the info server's activate timed out in its lifecycle manager, so
    # the server was configured -- its publisher counted -- and never active, and the costmap
    # never got the mask. The adapter must not read that as up.
    st, ros, *_ = _stack()
    life = Lifecycle(ros, drop=99, applies=False)
    life.state["keepout_filter_info_server"] = INACTIVE
    ros.publishers["/costmap_filter_info"] = 1
    assert st.activate_keepout() is None
    activates = [c for c in life.changes if c == ("keepout_filter_info_server", ACTIVATE)]
    assert len(activates) == 3                              # bounded: three tries, then no answer
    assert life.state["keepout_filter_info_server"] == INACTIVE


@pytest.mark.parametrize("applies", [True, False])
def test_a_dropped_change_state_reply_is_followed_by_a_state_read_and_a_retry(applies) -> None:
    st, ros, *_ = _stack()
    life = Lifecycle(ros, state=UNCONFIGURED, drop=1, applies=applies)
    ros.publishers["/costmap_filter_info"] = 1
    t0 = ros.t
    assert st.activate_keepout() is True
    assert life.state == dict.fromkeys(KEEPOUT_NODES, ACTIVE)
    for node in KEEPOUT_NODES:
        # The first reply lost: the state is read again before anything else is asked.
        first = next(i for i, c in enumerate(ros.calls) if c[0] == f"/{node}/change_state")
        assert ros.calls[first + 1][0] == f"/{node}/get_state"
    assert ros.t - t0 >= ros2_explore.KEEPOUT_SETTLE_S + 1.0   # it waited before reading again


def test_a_restart_stops_every_keep_out_process_by_name_then_launches_again() -> None:
    st, ros, shell, *_ = _stack()
    Lifecycle(ros)
    ros.publishers["/costmap_filter_info"] = 1
    assert st.restart_keepout() is True
    kill = next(s for s in shell.scripts() if "pgrep -f" in s)
    for name in ("keepout_servers.launch.py", "__node:=keepout_", "__node:=lifecycle_manager_keepout"):
        assert name in kill
    assert "kill -INT" in kill and "kill -9" in kill
    assert shell.scripts().index(kill) < max(
        i for i, s in enumerate(shell.scripts()) if "keepout_servers.launch.py mask:=" in s)


def _grid_msg(stamp=0.0):
    return {"info": {"width": 2, "height": 1, "resolution": 0.1,
                     "origin": {"position": {"x": -0.1, "y": 0.0}}}, "data": [-1, 100]}


def test_the_costmap_is_reset_and_read_only_once_it_is_newer_than_the_reset() -> None:
    st, ros, *_ = _stack()
    ros.answers["/global_costmap/clear_entirely_global_costmap"] = {}
    ros.latest_msgs["/global_costmap/costmap"] = (_grid_msg(), 1.0)
    ros.t = 10.0
    assert st.reset_costmap() is True
    assert st.read_costmap() is None                        # the one from before the reset
    ros.latest_msgs["/global_costmap/costmap"] = (_grid_msg(), ros.t + 1)
    g = st.read_costmap()
    assert (g.width, g.height, g.res, g.ox, g.data) == (2, 1, 0.1, -0.1, [-1, 100])


def test_the_served_mask_is_read_off_its_topic_and_only_once_it_is_newer_than_the_load(
        ) -> None:
    # The hand keep-out is proven on the mask the server SERVES, not on the file the adapter
    # wrote: a mask published before the last load (or reload) is not that mask.
    st, ros, shell, *_ = _stack()
    ros.latest_msgs["/keepout_filter_mask"] = (_grid_msg(), 1.0)
    ros.t = 10.0
    mask = steps.Mask(2, 1, 0.1, -0.1, 0.0, [False, True])
    shell.on("md5sum keepout.pgm", lambda s: (0, "".join(
        f"{hashlib.md5(d).hexdigest()}  {n.rsplit('/', 1)[1]}\n"
        for n, d in shell.files.items()).encode(), b""))
    assert st.load_keepout_mask(mask, 0.0) is True
    assert st.read_keepout_mask() is None                   # served before this load
    ros.latest_msgs["/keepout_filter_mask"] = (_grid_msg(), ros.t + 1)
    g = st.read_keepout_mask()
    assert (g.width, g.height, g.res, g.ox, g.data) == (2, 1, 0.1, -0.1, [-1, 100])
    ros.t = 20.0
    ros.answers["/keepout_mask_server/load_map"] = {"result": 0}
    assert st.reload_keepout_mask() is True
    assert st.read_keepout_mask() is None                   # served before the reload


def _path_msg(frame, points):
    return {"header": {"frame_id": frame},
            "poses": [{"pose": {"position": {"x": x, "y": y}}} for x, y in points]}


def test_the_controllers_local_plan_is_placed_in_the_map_from_its_own_frame() -> None:
    # TEB publishes its trajectory in the local costmap's frame, odom on the robot.
    st, ros, *_ = _stack()
    ros.tf = (1.0, 2.0, 90.0)                       # map -> odom
    ros.queues["/local_plan"] = [_path_msg("odom", [(1.0, 0.0), (0.0, 1.0)]),
                                 _path_msg("map", [(0.5, 0.5)]), _path_msg("odom", [])]
    plans = st.local_plans()
    assert len(plans) == 2
    assert [tuple(round(c, 9) for c in p) for p in plans[0]] == [(1.0, 3.0), (0.0, 2.0)]
    assert plans[1] == [(0.5, 0.5)]
    ros.tf = None                                   # no map -> odom: cannot be placed
    ros.queues["/local_plan"] = [_path_msg("odom", [(1.0, 0.0)])]
    assert st.local_plans() == [None]


def test_a_reset_with_no_reply_is_no_answer() -> None:
    st, *_ = _stack()
    assert st.reset_costmap() is None


def test_the_map_is_the_latest_map() -> None:
    st, ros, *_ = _stack()
    assert st.read_map() is None
    ros.latest_msgs["/map"] = (_grid_msg(), 0.0)
    assert st.read_map().data == [-1, 100]


# ── the explorer ────────────────────────────────────────────────────────────

def test_the_explorer_starts_as_the_supervised_run_started_it() -> None:
    st, _, shell, *_ = _stack()
    counts = iter([b"0\n", b"1\n"])
    shell.on("pgrep -fc lib/explore_lite/explore", lambda s: (0, next(counts), b""))
    assert st.start_explorer() is True
    launch = next(r for r in shell.runs if "ros2 run explore_lite explore" in r[0])
    assert launch[1] is True
    assert "source /work/kit/env.sh && source /work/ws/overlay.sh && exec ros2 run " \
           "explore_lite explore --ros-args --params-file /work/kit/hws3/explore_params.yaml" \
           in launch[0]


def test_a_running_explorer_is_resumed_not_launched_again() -> None:
    st, ros, shell, *_ = _stack()
    shell.on("pgrep -fc lib/explore_lite/explore", (0, b"1\n", b""))
    assert st.start_explorer() is True
    assert not any("ros2 run explore_lite" in s for s in shell.scripts())
    assert ros.published[-1] == ("explore/resume", {"data": True})


def test_stop_halt_and_hold_all_tell_the_explorer_to_stop() -> None:
    st, ros, _, b, _ = _stack()
    ev = threading.Event()
    st.arm(ev)
    st.stop_explorer()
    assert ros.published[-1] == ("explore/resume", {"data": False})
    b.command_move(ROBOT, 1.0, 0.0, 0.0)
    st.hold()
    assert ros.published[-1] == ("explore/resume", {"data": False})
    assert b.pending_goal(ROBOT) is not None               # held: the goal is kept to resume
    st.halt()
    assert ev.is_set()


def test_the_explorer_status_is_the_latest_status() -> None:
    st, ros, *_ = _stack()
    assert st.explorer_status() == ""
    ros.latest_msgs["explore/status"] = ({"status": "exploration_complete"}, 0.0)
    assert st.explorer_status() == "exploration_complete"


def _fb(goal: int, n: int, x: float, y: float) -> dict:
    return {"goal_id": {"uuid": [goal] * 16}, "feedback": {
        "number_of_recoveries": n,
        "current_pose": {"pose": {"position": {"x": x, "y": y}}}}}


FEEDBACK = "/navigate_to_pose/_action/feedback"


def test_recoveries_are_read_from_the_navigation_feedback_where_they_happened() -> None:
    st, ros, *_ = _stack()
    # The robot moves 1.9 m between the two rises, so they are two episodes, each where it began.
    ros.queues[FEEDBACK] = [_fb(1, 0, 0.0, 0.0), _fb(1, 1, 1.2, 0.3), _fb(1, 1, 1.2, 0.3),
                            _fb(2, 0, 2.0, 2.0), _fb(1, 3, 1.4, 0.6)]
    assert st.recoveries() == [(1.2, 0.3), (1.4, 0.6)]
    assert st.recoveries() == []


def test_one_recovery_episode_is_one_recovery_however_far_nav2_s_count_rises() -> None:
    # One stop as a real robot made it: TEB "not feasible", controller abort, clear the local
    # costmap, clear the global one, spin -- number_of_recoveries 0 -> 3 inside one second, the
    # robot where it stopped. Then the explorer's next goal (a new goal id, its count from 0)
    # while it is still there: the same stop, not a second one.
    st, ros, *_ = _stack()
    ros.queues[FEEDBACK] = [_fb(1, 0, 0.60, -0.40), _fb(1, 1, 0.63, -0.41),
                            _fb(1, 2, 0.63, -0.41), _fb(1, 3, 0.63, -0.41)]
    assert st.recoveries() == [(0.63, -0.41)]
    for k, n in enumerate((0, 1, 2, 3)):
        ros.t += 1.5
        ros.queues[FEEDBACK] = [_fb(2, n, 0.65, -0.44 - 0.01 * k)]
        assert st.recoveries() == []
    counter = steps.AreaCounter(3)
    assert counter.add(0.63, -0.41) is None and counter.exhausted == []


def test_recovery_episodes_are_told_apart_by_progress_or_a_quiet_spell() -> None:
    st, ros, *_ = _stack()
    counter, got = steps.AreaCounter(3), []

    def feed(*msgs, after_s=0.5):
        ros.t += after_s
        ros.queues[FEEDBACK] = list(msgs)
        for x, y in st.recoveries():
            got.append((x, y))
            counter.add(x, y)
    feed(_fb(1, 0, 2.2, 1.2), _fb(1, 1, 2.2, 1.3), _fb(1, 2, 2.2, 1.3))       # episode 1
    feed(_fb(1, 2, 2.6, 1.6), after_s=3.0)                                    # 0.5 m on
    feed(_fb(1, 3, 2.6, 1.6), _fb(1, 4, 2.6, 1.6))                           # episode 2
    feed(_fb(1, 4, 2.6, 1.6), after_s=15.0)                                   # quiet, in place
    feed(_fb(1, 5, 2.6, 1.6), _fb(1, 6, 2.6, 1.6))                           # episode 3
    assert got == [(2.2, 1.3), (2.6, 1.6), (2.6, 1.6)]
    assert counter.exhausted == ["area x 2.0..3.0 m, y 1.0..2.0 m"]


def test_a_skipped_area_is_recorded() -> None:
    st, *_ = _stack()
    st.skip_area("area x 1.0..2.0 m, y 0.0..1.0 m")
    assert st.skipped == ["area x 1.0..2.0 m, y 0.0..1.0 m"]


# ── the return ──────────────────────────────────────────────────────────────

def test_the_return_is_a_goal_through_the_binding_and_its_result_is_the_bindings() -> None:
    st, _, _, b, sent = _stack()
    st.go_to_mark(0.0, 0.0, 90.0)
    assert sent and sent[-1].yaw == pytest.approx(1.5707963, rel=1e-6)
    assert st.mark_result() is None
    b.task_state = lambda rid: task_map.TASK_SUCCEEDED
    assert st.mark_result() == "SUCCEEDED"
    b.task_state = lambda rid: task_map.TASK_FAILED
    b.task_failure = lambda rid: "aborted by the server"
    assert st.mark_result() == "aborted by the server"


# ── save and copy back ──────────────────────────────────────────────────────

def _saved_files():
    return {"lab__explore-first.pgm": b"P5\n1 1\n255\n\xfe",
            "lab__explore-first.yaml": b"image: lab__explore-first.pgm\nresolution: 0.05\n",
            "lab__explore-first.posegraph": b"graph", "lab__explore-first.data": b"data"}


def _serve_saved(shell, files, corrupt: str | None = None):
    base = "/work/kit/out/map/"
    shell.on("map_saver_cli", (0, b"map_saver rc=0\n", b""))
    shell.on("md5sum " + base, (0, "".join(
        f"{hashlib.md5(d).hexdigest()}  {base}{n}\n" for n, d in files.items()).encode(), b""))
    for n, d in files.items():
        shell.on(f"cat {base}{n}", (0, d + (b"!" if n == corrupt else b""), b""))


def test_the_map_and_graph_are_saved_and_copied_back_md5_at_each_hop() -> None:
    st, ros, shell, *_ = _stack()
    ros.answers["/slam_toolbox/serialize_map"] = {"result": 0}
    _serve_saved(shell, _saved_files())
    saved = st.save_map("lab/explore-first")
    assert set(saved.files) == {"map.pgm", "map.yaml", "map.posegraph", "map.data"}
    assert saved.files["map.yaml"].startswith(b"image: map.pgm\n")
    assert saved.graph_ref == "/work/kit/out/map/lab__explore-first"
    assert ros.calls[-1] == ("/slam_toolbox/serialize_map", "slam_toolbox/srv/SerializePoseGraph",
                             {"filename": "/work/kit/out/map/lab__explore-first"})
    saver = next(s for s in shell.scripts() if "map_saver_cli" in s)
    assert "-f /work/kit/out/map/lab__explore-first" in saver


def test_a_copy_whose_md5_differs_is_no_answer_so_it_is_copied_again() -> None:
    st, ros, shell, *_ = _stack()
    ros.answers["/slam_toolbox/serialize_map"] = {"result": 0}
    _serve_saved(shell, _saved_files(), corrupt="lab__explore-first.posegraph")
    assert st.save_map("lab/explore-first") is None


def test_a_graph_that_did_not_serialize_is_no_answer() -> None:
    st, ros, shell, *_ = _stack()
    ros.answers["/slam_toolbox/serialize_map"] = {"result": 1}
    _serve_saved(shell, _saved_files())
    assert st.save_map("lab/explore-first") is None


# ── the end of a segment ────────────────────────────────────────────────────

def test_the_end_of_a_segment_stops_every_process_it_started_by_name() -> None:
    st, _, shell, *_ = _stack()
    st.end_segment()
    kill = shell.scripts()[-1]
    for name in ("lib/explore_lite/explore", "keepout_servers.launch.py", "__node:=keepout_",
                 "__node:=lifecycle_manager_keepout", "online_async_launch.py",
                 "async_slam_toolbox_node"):
        assert name in kill
    assert "kill -INT" in kill and "kill -9" in kill


# ── battery and IMU ─────────────────────────────────────────────────────────

class _Bat:
    def __init__(self, voltage):
        self.voltage = voltage


def test_the_battery_is_the_bindings_reading_in_mv_and_never_a_default() -> None:
    st, _, _, b, _ = _stack()
    b.on_battery(type("S", (), {"percentage": float("nan"), "power_supply_status": 0,
                                "voltage": 8.115})())
    assert st.battery_mv() == pytest.approx(8115.0)
    b.on_battery(type("S", (), {"percentage": float("nan"), "power_supply_status": 0,
                                "voltage": float("nan")})())
    assert st.battery_mv() is None


def _imu(wz: float) -> dict:
    return {"angular_velocity": {"x": 0.0, "y": 0.0, "z": wz}}


def test_the_imu_is_streaming_only_when_it_reports_and_its_gyro_moves() -> None:
    st, ros, *_ = _stack()
    ros.t = 10.0
    assert st.imu_streaming() is False                     # silent: nothing at all
    ros.windows["imu"] = [(9.0 + i * 0.1, _imu(0.0)) for i in range(10)]
    assert st.imu_streaming() is False                     # frozen: every sample the same
    ros.windows["imu"] = [(9.0 + i * 0.1, _imu(0.001 * (i % 3))) for i in range(10)]
    assert st.imu_streaming() is True


# ── bounds, on the adapter host ─────────────────────────────────────────────

def _bound_dir(tmp_path, name="room-1"):
    d = tmp_path / name
    d.mkdir()
    m = _mask()
    (d / "keepout.pgm").write_bytes(steps.mask_to_pgm(m))
    (d / "keepout.yaml").write_bytes(steps.mask_yaml(m, "keepout.pgm"))
    g = steps.Grid(width=4, height=3, res=0.1, ox=-0.2, oy=-0.1, data=[100] * 4 + [100, 0, 0, 100]
                   + [100] * 4)
    (d / "map.pgm").write_bytes(steps.grid_to_pgm(g))
    (d / "map.yaml").write_text("image: map.pgm\nresolution: 0.1\norigin: [-0.2, -0.1, 0]\n")
    (d / "bound.json").write_text('{"theta0_deg": -60.639}')
    return d


def test_a_bound_is_read_from_the_bounds_directory_with_its_saved_map(tmp_path) -> None:
    _bound_dir(tmp_path)
    st, *_ = _stack(tmp_path)
    b = st.load_bound("room-1")
    assert b.theta0_deg == -60.639 and b.mask.keep == _mask().keep
    assert b.ref_map is not None and b.ref_map.data[5] == 0


@pytest.mark.parametrize("ref", ["../room-1", "room-1/..", "Room 1", "room-2"])
def test_a_bound_outside_the_directory_or_absent_is_refused(tmp_path, ref) -> None:
    _bound_dir(tmp_path)
    st, *_ = _stack(tmp_path)
    with pytest.raises(LookupError):
        st.load_bound(ref)


# ── the binding declares the actions only with a stack ──────────────────────

def test_a_ros2_binding_given_a_stack_declares_it() -> None:
    st, _, _, b, _ = _stack()
    b._explore = st
    assert b.explore_stack() is st


# ── start-up: --binding ros2 declares the actions only when all of it resolves ──

from test_stall_reporting import _main

ROS2 = ("--binding", "ros2", "--stall-window", "5", "--stall-min-progress", "0.05",
        "--goal-response-bound", "1", "--odometry-source", "measured")


def _site_file(tmp_path, values=SITE) -> str:
    f = tmp_path / "site.env"
    f.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
    return str(f)


def test_ros2_with_battery_thresholds_and_a_site_file_has_the_stack(
        monkeypatch, capsys, tmp_path) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                             "--battery-hard-mv", "7750", "--battery-return-mv", "7835",
                             "--battery-full-mv", "8300")
    assert code is None and isinstance(robot, Nav2Binding)
    stack = robot.explore_stack()
    assert isinstance(stack, ros2_explore.Ros2ExploreStack)
    assert stack.battery_limits() == explore.BatteryLimits(7750, 8300, 7835)
    assert "explore-floor and Charge: declared" in err
    for v in SITE.values():
        assert v not in err                             # no site value is printed


def test_ros2_without_battery_thresholds_declares_navigate_only_and_says_why(
        monkeypatch, capsys) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2)
    assert code is None and robot.explore_stack() is None
    assert "explore-floor and Charge: not declared" in err and "--battery-hard-mv" in err


def test_ros2_with_one_battery_threshold_refuses_to_start(monkeypatch, capsys, tmp_path) -> None:
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                             "--battery-hard-mv", "7750")
    assert code == 2 and robot is None and "--battery-full-mv" in err


def test_ros2_with_an_incomplete_site_file_refuses_naming_keys(
        monkeypatch, capsys, tmp_path) -> None:
    partial = {k: v for k, v in SITE.items() if k != "SWARMADA_ROBOT_CONTAINER"}
    code, err, robot = _main(monkeypatch, capsys, *ROS2, "--site-env",
                             _site_file(tmp_path, partial), "--battery-hard-mv", "7750",
                             "--battery-full-mv", "8300")
    assert code == 2 and robot is None and "SWARMADA_ROBOT_CONTAINER" in err
    assert "robot.test" not in err


def test_a_return_threshold_outside_hard_and_full_refuses_to_start(
        monkeypatch, capsys, tmp_path) -> None:
    code, err, _ = _main(monkeypatch, capsys, *ROS2, "--site-env", _site_file(tmp_path),
                         "--battery-hard-mv", "7750", "--battery-return-mv", "7700",
                         "--battery-full-mv", "8300")
    assert code == 2 and "--battery-return-mv" in err


def _path(*points) -> dict:
    return {"poses": [{"pose": {"position": {"x": x, "y": y, "z": 0.0}}} for x, y in points]}


def test_plans_are_read_from_the_planners_path_topic_each_once() -> None:
    st, ros, *_ = _stack()
    ros.queues["/plan"] = [_path((0.0, 0.0), (0.5, 0.25)), _path((1.0, 1.0))]
    assert st.plans() == [[(0.0, 0.0), (0.5, 0.25)], [(1.0, 1.0)]]
    assert st.plans() == []


def test_the_planners_path_topic_is_subscribed_and_queued() -> None:
    facade = ros2_explore.RclpyFacade(binding=None)
    assert facade.topics["/plan"][0] == "nav_msgs/msg/Path"
    assert "/plan" in facade.queued


def test_a_reload_asks_the_mask_server_to_load_the_live_mask_again() -> None:
    st, ros, *_ = _stack()
    ros.answers["/keepout_mask_server/load_map"] = {"result": 0}
    assert st.reload_keepout_mask() is True
    service, srv_type, fields = ros.calls[-1]
    assert service == "/keepout_mask_server/load_map" and srv_type == "nav2_msgs/srv/LoadMap"
    assert fields == {"map_url": "/work/kit/hws3/mask-live/keepout.yaml"}


@pytest.mark.parametrize("answer, want", [(None, None), ({"result": 1}, False),
                                          ({"result": 255}, False)])
def test_a_reload_lost_or_refused_is_said_as_such(answer, want) -> None:
    st, ros, *_ = _stack()
    ros.answers["/keepout_mask_server/load_map"] = answer
    assert st.reload_keepout_mask() is want
