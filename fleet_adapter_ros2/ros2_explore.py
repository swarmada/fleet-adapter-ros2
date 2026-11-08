# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The live exploration stack of the ROS 2 binding: SLAM, the keep-out servers, the costmap,
the explorer, the battery and the IMU of a real robot, behind `explore.ExploreStack`.

Three layers, so the logic is tested and only the thinnest layer needs a robot:

* `SiteConfig` -- where the robot is and how its stack is reached. Every value comes from the
  site file (``~/.config/swarmada/site.env`` by default), read at run time; none is written in
  this code, and a missing key is named, never guessed.
* `RobotShell` -- the robot-side processes, started and stopped the way the supervised
  explorations started them: ``bash -c 'source <env> && exec ros2 launch ...'`` for a launch,
  and every other script on stdin to ``bash -s`` (so a ``pgrep -f`` in it cannot match its own
  shell). Files move through it too, and are compared by md5 at each end. Two modes run the
  same script text: ``ssh`` reaches the robot's container with ``ssh <user>@<robot> sudo
  docker exec [-d|-i] -u <user> <container>``; ``local`` is the adapter inside that container,
  and runs each script in a clean environment (``env -i``), a launch detached under
  ``setsid``, so nothing of the adapter's own Python environment reaches a ROS launch.
* `RosFacade` -- the rclpy calls: a service call, the latest, drained or recent messages of a
  topic, a publish, a publisher count and a transform. `RclpyFacade` is the real one, on the
  binding's node; it is the only code here that needs a ROS 2 runtime.

`Ros2ExploreStack` answers every read with None when its reply was lost and every call with
None (lost) or False (refused), so the jobs' bounded retries and exact verdicts apply to it
unchanged. The robot-side files it runs -- the SLAM parameters, the keep-out servers' launch
file, the explorer's parameters and its overlay build -- are staged in the robot's work
directory beforehand, as they were for the supervised runs; this stack does not stage them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar, Protocol

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2 import explore_steps as steps
from fleet_adapter_ros2 import task as task_map

# How long each robot-side start is given before it is looked at, as the supervised runs waited.
SLAM_SETTLE_S = 20.0
KEEPOUT_SETTLE_S = 12.0
EXPLORER_SETTLE_S = 5.0
RESET_SETTLE_S = 4.0
# The keep-out servers, by the names their launch gives them, and how a lifecycle transition the
# adapter asks of one is tried: under SLAM and Nav2 load a change_state reply is computed and
# not delivered, and the server's lifecycle manager then gives up on it. The state is read
# again after each try, since a lost reply may be a transition that happened.
KEEPOUT_NODES = ("keepout_mask_server", "keepout_filter_info_server")
LIFECYCLE_TRIES = 3
LIFECYCLE_BACKOFF_S = (1.0, 2.0, 4.0)
_UNCONFIGURED, _INACTIVE, _ACTIVE, _FINALIZED = 1, 2, 3, 4
_NEXT_TRANSITION = {_UNCONFIGURED: (1, "configure"), _INACTIVE: (3, "activate")}
# The IMU streams when it has reported this often in the window and its gyro is not frozen: a
# silent gyro is what a boot gate fails, and what a robot on its charging cable shows.
IMU_WINDOW_S = 2.0
IMU_MIN_SAMPLES = 5


class SiteError(Exception):
    """The site file cannot say how to reach the robot. Names keys, never values."""


SHELL_MODES = ("ssh", "local")

#: ``--site-env`` reads the keys from the adapter's own environment instead of a file.
SITE_FROM_ENV = "env"


class SiteConfig:
    """How the robot and its stack are reached, from the site file. Keys are code; values are
    the site's and live only in the file.

    Which keys are needed depends on where the adapter runs. Over ssh (``mode="ssh"``) it
    reaches the container from another host, and reads the bounds on that host. Inside the
    container (``mode="local"``) there is no host, user or container to reach, and the bounds
    are read from ``<workdir>/bounds``, where the operator stages them beside the rest of the
    kit; a bounds directory named for another host would not exist here."""

    KEYS: ClassVar[dict[str, str]] = {
        "SWARMADA_ROBOT_HOST": "host",
        "SWARMADA_ROBOT_SSH_USER": "ssh_user",
        "SWARMADA_ROBOT_CONTAINER": "container",
        "SWARMADA_ROBOT_CONTAINER_USER": "container_user",
        "SWARMADA_ROBOT_WORKDIR": "workdir",
        "SWARMADA_ROBOT_ENV_SCRIPT": "env_script",
        "SWARMADA_ROBOT_OVERLAY": "overlay",
        "SWARMADA_BOUNDS_DIR": "bounds_dir",
    }
    LOCAL_KEYS: ClassVar[tuple[str, ...]] = (
        "SWARMADA_ROBOT_WORKDIR", "SWARMADA_ROBOT_ENV_SCRIPT", "SWARMADA_ROBOT_OVERLAY")

    def __init__(self, values: dict[str, str], mode: str = "ssh") -> None:
        if mode not in SHELL_MODES:
            raise ValueError(f"mode must be one of {', '.join(SHELL_MODES)}, got {mode!r}")
        needed = self.LOCAL_KEYS if mode == "local" else tuple(self.KEYS)
        missing = [k for k in needed if not values.get(k)]
        if missing:
            raise SiteError("the site file does not set " + ", ".join(missing)
                            + "; the robot's stack cannot be reached without them")
        self.mode = mode
        for key, attr in self.KEYS.items():
            setattr(self, attr, values.get(key, "") if key in needed else "")
        if mode == "local":
            self.bounds_dir = f"{self.workdir}/bounds"

    @classmethod
    def load(cls, path: str, mode: str = "ssh", environ=None) -> SiteConfig:
        if path == SITE_FROM_ENV:
            env = os.environ if environ is None else environ
            return cls({k: env[k] for k in cls.KEYS if env.get(k)}, mode)
        try:
            text = Path(path).expanduser().read_text()
        except OSError as exc:
            raise SiteError(f"the site file cannot be read ({exc.strerror})") from exc
        values = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ").strip()
            key, sep, value = line.partition("=")
            if sep:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                values[key.strip()] = value
        return cls(values, mode)


# What a local script's clean environment keeps from the adapter's: the DDS settings, so a
# launch joins the graph the adapter is on. The robot's environment script sets the rest.
_LOCAL_ENV_KEEP = ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION",
                   "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE", "LANG")
_LOCAL_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class RobotShell:
    """Scripts in the robot's container: over ssh, or -- the adapter inside the container --
    locally. ``run`` answers (returncode, stdout, stderr), or None when the shell itself
    could not run or timed out -- no answer, never an answer.

    Both modes run the same script text in the same form: ``bash -s`` with the script on
    stdin, ``bash -c`` when stdin carries a file, and ``bash -c`` detached for a launch. A
    local script starts from ``env -i``, as ``docker exec`` starts from the container's own
    environment: the adapter's PYTHONPATH and its protobuf must not reach a ROS launch. A
    local launch is detached with ``setsid -f`` and its output discarded (the script
    redirects its own), so it outlives the call and the adapter's process group."""

    def __init__(self, site: SiteConfig, runner=subprocess.run, mode: str | None = None,
                 environ=None) -> None:
        self.site, self._runner = site, runner
        self.mode = mode or getattr(site, "mode", "ssh")
        if self.mode not in SHELL_MODES:
            raise ValueError(f"mode must be one of {', '.join(SHELL_MODES)}, got {self.mode!r}")
        self._environ = os.environ if environ is None else environ

    def _clean_env(self) -> list[str]:
        home = self._environ.get("HOME", "/")
        keep = [f"{k}={self._environ[k]}" for k in _LOCAL_ENV_KEEP if self._environ.get(k)]
        return ["env", "-i", f"HOME={home}", f"PATH={_LOCAL_PATH}", *keep]

    def run(self, script: str, *, detach: bool = False, stdin: bytes | None = None,
            timeout: float = 60.0):
        if detach or stdin is not None:
            form, data = ["bash", "-c", script], stdin
        else:
            form, data = ["bash", "-s"], script.encode()
        try:
            if self.mode == "local":
                if detach:
                    r = self._runner(["setsid", "-f", *self._clean_env(), *form],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=timeout)
                    return r.returncode, b"", b""
                r = self._runner([*self._clean_env(), *form], input=data, capture_output=True,
                                 timeout=timeout)
            else:
                q = shlex.quote
                head = (f"sudo docker exec {'-d' if detach else '-i'} "
                        f"-u {q(self.site.container_user)} {q(self.site.container)}")
                remote = " ".join([head, *form[:2], *(q(a) for a in form[2:])])
                r = self._runner(["ssh", f"{self.site.ssh_user}@{self.site.host}", remote],
                                 input=data, capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.returncode, r.stdout, r.stderr


class RosFacade(Protocol):
    def now(self) -> float: ...
    def call(self, service: str, srv_type: str, fields: dict, timeout: float = 10.0): ...
    def latest(self, topic: str): ...
    def drain(self, topic: str) -> list: ...
    def recent(self, topic: str, window_s: float) -> list: ...
    def publish(self, topic: str, msg_type: str, fields: dict) -> None: ...
    def publisher_count(self, topic: str) -> int | None: ...
    def transform(self, target: str, source: str): ...


_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")
_FEEDBACK = "/navigate_to_pose/_action/feedback"
# The planner server publishes every global path it computes here: the explorer's goals, the
# return, and any goal sent to the navigator by anyone else.
_PLAN = "/plan"
# The controller's local trajectory: TEB (the controller here) publishes it on every control
# cycle, in its local costmap's frame (odom on the robot, as its recorded runs of 2026-09-26 show).
_LOCAL_PLAN = "/local_plan"
# The keep-out mask server (a second map_server) publishes the mask it serves here, latched.
_MASK = "/keepout_filter_mask"


def _grid(msg: dict) -> steps.Grid:
    info = msg["info"]
    pos = info["origin"]["position"]
    return steps.Grid(width=int(info["width"]), height=int(info["height"]),
                      res=float(info["resolution"]), ox=float(pos["x"]), oy=float(pos["y"]),
                      data=[int(v) for v in msg["data"]])


def _param_text(value: dict) -> str:
    """A ParameterValue as `ros2 param get` prints it."""
    kind = value.get("type", 0)
    if kind == 0:
        return "not set"
    field = {1: "bool_value", 2: "integer_value", 3: "double_value", 4: "string_value",
             6: "bool_array_value", 7: "integer_array_value", 8: "double_array_value",
             9: "string_array_value"}.get(kind)
    v = value.get(field)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


class Ros2ExploreStack:
    """`explore.ExploreStack` on a real robot: see the module docstring."""

    def __init__(self, *, binding, robot_id: str, site: SiteConfig, shell: RobotShell,
                 ros: RosFacade, limits: explore.BatteryLimits, clock=None,
                 imu_topic: str = "imu") -> None:
        self.binding, self.robot_id, self.site = binding, robot_id, site
        self.shell, self.ros, self.limits = shell, ros, limits
        self.clock = clock or explore.RealClock()
        self.imu_topic = imu_topic
        w = site.workdir
        self._paths = {
            "slam_fresh": f"{w}/map-kit/slam_params.yaml",
            "slam_resume": f"{w}/map-kit/slam_params_resume.yaml",
            "slam_live": f"{w}/map-kit/slam_params_resume-live.yaml",
            "mask_live": f"{w}/hws3/mask-live",
            "keepout_launch": f"{w}/hws3/keepout_servers.launch.py",
            "explore_params": f"{w}/hws3/explore_params.yaml",
            "out": f"{w}/out", "maps": f"{w}/out/map",
        }
        self._graph = re.compile("^" + re.escape(self._paths["maps"])
                                 + r"/[a-z0-9][-a-z0-9._]*$")
        self._halted: threading.Event | None = None
        self._never = threading.Event()
        self._slam_launched = self._keepout_launched = False
        self._reset_at: float | None = None
        self._mask_at: float | None = None
        self._episodes = steps.RecoveryEpisodes()
        self.skipped: list[str] = []

    # ── helpers ─────────────────────────────────────────────────────────────

    def _wait(self, seconds: float) -> bool:
        return self.clock.wait(self._halted or self._never, seconds)

    def _sourced(self, command: str, overlay: bool = False) -> str:
        env = f"source {self.site.env_script} && "
        if overlay:
            env += f"source {self.site.overlay} && "
        return env + command

    def _count(self, pattern: str) -> int | None:
        r = self.shell.run(f"pgrep -fc {pattern}")
        if r is None:
            return None
        try:
            return int(r[1].decode().strip() or "0")
        except ValueError:
            return None

    def _kill(self, names: str) -> None:
        self.shell.run(f'P="{names}"\n'
                       'for p in $(pgrep -f "$P"); do kill -INT $p 2>/dev/null; done; sleep 4\n'
                       'for p in $(pgrep -f "$P"); do kill -9 $p 2>/dev/null; done')

    # ── the stack ───────────────────────────────────────────────────────────

    def battery_limits(self) -> explore.BatteryLimits:
        return self.limits

    def arm(self, halted: threading.Event) -> None:
        self._halted = halted

    # halt, hold and stop_explorer are on every stop path: none of them raises. A step that
    # fails is reported on stderr, and the steps after it still run.

    def halt(self) -> None:
        self.stop_explorer()      # the binding's own stop cancels its goal; this is the explorer
        if self._halted is not None:
            self._halted.set()

    def hold(self) -> None:
        self.stop_explorer()
        try:
            self.binding.command_pause(self.robot_id)
        except Exception as exc:  # noqa: BLE001 - a stop path: report, never raise
            print(f"explore-floor: pausing the base raised {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    def load_bound(self, bound_ref: str) -> explore.Bound:
        if not _NAME.match(bound_ref or ""):
            raise LookupError("a bound is named by a plain lower-case name")
        d = Path(self.site.bounds_dir) / bound_ref
        try:
            meta = _yaml_fields((d / "keepout.yaml").read_text())
            origin = _origin(meta)
            mask = steps.mask_from_pgm((d / meta["image"]).read_bytes(),
                                       res=float(meta["resolution"]), ox=origin[0], oy=origin[1])
            bound_doc = json.loads((d / "bound.json").read_text())
            theta0 = float(bound_doc["theta0_deg"])
            ref_map = None
            if (d / "map.yaml").is_file():
                mm = _yaml_fields((d / "map.yaml").read_text())
                mo = _origin(mm)
                ref_map = steps.grid_from_pgm((d / mm["image"]).read_bytes(),
                                              res=float(mm["resolution"]), ox=mo[0], oy=mo[1])
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise LookupError(f"not a bound this adapter can read ({exc})") from exc
        return explore.Bound(mask=mask, theta0_deg=theta0, ref_map=ref_map,
                             hand_keepouts=explore.read_hand_keepouts(d, bound_doc))

    def read_param(self, node: str, name: str) -> str | None:
        resp = self.ros.call(f"{node}/get_parameters", "rcl_interfaces/srv/GetParameters",
                             {"names": [name]})
        if resp is None or not resp.get("values"):
            return None
        return _param_text(resp["values"][0])

    def start_slam(self, graph_ref: str | None) -> bool | None:
        """slam_toolbox's own launch, detached; up when exactly one runs and map -> base_footprint
        resolves. A resume rewrites the resume parameters to the saved graph first."""
        if graph_ref is not None and not self._graph.match(graph_ref):
            return False
        if not self._slam_launched:
            params = self._paths["slam_fresh"]
            if graph_ref is not None:
                live = self._paths["slam_live"]
                r = self.shell.run(f"sed -E 's#^([[:space:]]*map_file_name:).*#\\1 {graph_ref}#' "
                                   f"{self._paths['slam_resume']} > {live}")
                if r is None:
                    return None
                check = self.shell.run(f"grep -c {graph_ref} {live}")
                if check is None or check[1].strip() in (b"", b"0"):
                    return False
                params = live
            launch = self.shell.run(self._sourced(
                f"exec ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false "
                f"slam_params_file:={params} > {self._paths['out']}/slam.log 2>&1"), detach=True)
            if launch is None:
                return None
            self._slam_launched = True
            if self._wait(SLAM_SETTLE_S):
                return None
        n = self._count("async_slam_toolbox_node")
        if n is None:
            return None
        if n > 1:
            return False
        return True if n == 1 and self.ros.transform("map", "base_footprint") else None

    def robot_pose(self):
        return self.ros.transform("map", "base_footprint")

    def load_keepout_mask(self, mask: steps.Mask, delta_deg: float) -> bool | None:
        self._mask_at = self.clock.now()     # a mask published before this is not this one
        live = self._paths["mask_live"]
        files = {"keepout.pgm": steps.mask_to_pgm(mask),
                 "keepout.yaml": steps.mask_yaml(mask, "keepout.pgm")}
        for name, data in files.items():
            r = self.shell.run(f"mkdir -p {live} && cat > {live}/{name}", stdin=data)
            if r is None or r[0] != 0:
                return None
        r = self.shell.run(f"cd {live} && md5sum keepout.pgm keepout.yaml")
        if r is None:
            return None
        remote = {parts[1]: parts[0] for parts in
                  (ln.split() for ln in r[1].decode().splitlines()) if len(parts) == 2}
        return all(remote.get(n) == hashlib.md5(d).hexdigest() for n, d in files.items())

    def _launch_keepout(self) -> bool:
        r = self.shell.run(self._sourced(
            f"exec ros2 launch {self._paths['keepout_launch']} "
            f"mask:={self._paths['mask_live']}/keepout.yaml > {self._paths['out']}/keepout.log "
            f"2>&1"), detach=True)
        return r is not None

    def _filter_up(self) -> bool | None:
        n = self.ros.publisher_count("/costmap_filter_info")
        return True if n else None

    def _lifecycle_state(self, node: str) -> int | None:
        resp = self.ros.call(f"/{node}/get_state", "lifecycle_msgs/srv/GetState", {})
        if resp is None:
            return None
        return int(resp.get("current_state", {}).get("id", 0)) or None

    def _bring_up(self, node: str) -> bool | None:
        """``node`` driven to active: True when it reports active; False when it is finalized
        and cannot be; None when it is not active after LIFECYCLE_TRIES transitions asked (a
        lost reply, a refusal while its manager is mid-transition, or no state read)."""
        tries = 0
        while True:
            state = self._lifecycle_state(node)
            if state == _ACTIVE:
                return True
            if state == _FINALIZED:
                return False
            if tries == LIFECYCLE_TRIES:
                return None
            nxt = _NEXT_TRANSITION.get(state) if state is not None else None
            if nxt is not None:
                resp = self.ros.call(f"/{node}/change_state", "lifecycle_msgs/srv/ChangeState",
                                     {"transition": {"id": nxt[0], "label": nxt[1]}})
                if resp is not None and resp.get("success"):
                    continue                    # read the state it reached, no wait
                why = "no reply" if resp is None else "refused"
                print(f"explore-floor: {node} {nxt[1]}: {why} (try {tries + 1} of "
                      f"{LIFECYCLE_TRIES}); reading its state again", file=sys.stderr)
            if self._wait(LIFECYCLE_BACKOFF_S[min(tries, len(LIFECYCLE_BACKOFF_S) - 1)]):
                return None
            tries += 1

    def reload_keepout_mask(self) -> bool | None:
        """The mask server loads the live mask file again (map_server's load_map) and publishes
        it on the mask topic, where the costmap's keep-out filter replaces the one it had.
        True on the server's success; False on its refusal; None when the reply is lost."""
        self._mask_at = self.clock.now()
        resp = self.ros.call(f"/{KEEPOUT_NODES[0]}/load_map", "nav2_msgs/srv/LoadMap",
                             {"map_url": f"{self._paths['mask_live']}/keepout.yaml"})
        if resp is None:
            return None
        return resp.get("result") == 0

    def activate_keepout(self) -> bool | None:
        """Launched once; up when both servers report active and /costmap_filter_info has a
        publisher. A configured info server already has its publisher, so the publisher alone
        is never taken for up. A server its lifecycle manager left behind (a transition whose
        reply was lost) is driven to active here, each transition bounded and its state read
        again after every try; one still not active is no answer -- and is asked again."""
        if not self._keepout_launched:
            if not self._launch_keepout():
                return None
            self._keepout_launched = True
            if self._wait(KEEPOUT_SETTLE_S):
                return None
        for node in KEEPOUT_NODES:
            up = self._bring_up(node)
            if up is not True:
                return up
        return self._filter_up()

    def restart_keepout(self) -> bool | None:
        """Every keep-out process stopped by its own name (a SIGINT to `ros2 launch` alone left
        the nodes running on the stand-in), INT then -9, and launched again."""
        self._kill("keepout_servers.launch.py|__node:=keepout_|__node:=lifecycle_manager_keepout")
        self._keepout_launched = False
        return self.activate_keepout()

    def reset_costmap(self) -> bool | None:
        t0 = self.clock.now()
        if self.ros.call("/global_costmap/clear_entirely_global_costmap",
                         "nav2_msgs/srv/ClearEntireCostmap", {}) is None:
            return None
        self._reset_at = t0
        self._wait(RESET_SETTLE_S)
        return True

    def read_costmap(self) -> steps.Grid | None:
        got = self.ros.latest("/global_costmap/costmap")
        if got is None:
            return None
        msg, t = got
        if self._reset_at is not None and t <= self._reset_at:
            return None           # from before the reset: not the costmap the check is about
        return _grid(msg)

    def read_keepout_mask(self) -> steps.Grid | None:
        got = self.ros.latest(_MASK)
        if got is None:
            return None
        msg, t = got
        if self._mask_at is not None and t <= self._mask_at:
            return None           # served before the last load: not the mask the check is about
        return _grid(msg)

    def read_map(self) -> steps.Grid | None:
        got = self.ros.latest("/map")
        return _grid(got[0]) if got is not None else None

    def start_explorer(self) -> bool | None:
        n = self._count("lib/explore_lite/explore")
        if n is None:
            return None
        if n >= 1:
            self.ros.publish("explore/resume", "std_msgs/msg/Bool", {"data": True})
            return True
        r = self.shell.run(self._sourced(
            f"exec ros2 run explore_lite explore --ros-args --params-file "
            f"{self._paths['explore_params']} > {self._paths['out']}/explore.log 2>&1",
            overlay=True), detach=True)
        if r is None or self._wait(EXPLORER_SETTLE_S):
            return None
        return True if self._count("lib/explore_lite/explore") == 1 else None

    def stop_explorer(self) -> None:
        # Once: the explorer's stop cancels every navigation goal, the return goal among them.
        try:
            self.ros.publish("explore/resume", "std_msgs/msg/Bool", {"data": False})
        except Exception as exc:  # noqa: BLE001 - a stop path: report, never raise
            print(f"explore-floor: telling the explorer to stop raised {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr)

    def explorer_status(self) -> str:
        got = self.ros.latest("explore/status")
        return str(got[0].get("status", "")) if got is not None else ""

    def recoveries(self) -> list[tuple[float, float]]:
        # Episodes, not Nav2's raw count: see explore_steps.RecoveryEpisodes.
        out, now = [], self.clock.now()
        for msg in self.ros.drain(_FEEDBACK):
            goal = tuple(msg.get("goal_id", {}).get("uuid", ()))
            fb = msg.get("feedback", {})
            pos = fb.get("current_pose", {}).get("pose", {}).get("position", {})
            began = self._episodes.feed(now, goal, int(fb.get("number_of_recoveries", 0)),
                                        float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))
            if began is not None:
                out.append(began)
        return out

    def plans(self) -> list[list[tuple[float, float]]]:
        out = []
        for msg in self.ros.drain(_PLAN):
            out.append([(float(p.get("pose", {}).get("position", {}).get("x", 0.0)),
                         float(p.get("pose", {}).get("position", {}).get("y", 0.0)))
                        for p in msg.get("poses", [])])
        return [p for p in out if p]

    def local_plans(self) -> list[list[tuple[float, float]] | None]:
        out: list[list[tuple[float, float]] | None] = []
        for msg in self.ros.drain(_LOCAL_PLAN):
            points = [(float(p.get("pose", {}).get("position", {}).get("x", 0.0)),
                       float(p.get("pose", {}).get("position", {}).get("y", 0.0)))
                      for p in msg.get("poses", [])]
            if not points:
                continue
            frame = str(msg.get("header", {}).get("frame_id", "")).lstrip("/")
            if frame != "map":
                tf = self.ros.transform("map", frame) if frame else None
                if tf is None:
                    out.append(None)
                    continue
                tx, ty, deg = tf
                c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                points = [(tx + c * x - s * y, ty + s * x + c * y) for x, y in points]
            out.append(points)
        return out

    def skip_area(self, name: str) -> None:
        # Recorded and reported. The explorer has no call to blacklist an area; it drops a
        # frontier only by its own progress timeout. So the job draws the area into the running
        # segment's keep-out mask, and later resumed segments draw it in too.
        self.skipped.append(name)
        print(f"explore-floor: exhausted, keeping it out of this segment and of later resumed "
              f"segments: {name}", file=sys.stderr)

    def go_to_mark(self, x: float, y: float, yaw_deg: float) -> None:
        self.binding.command_move(self.robot_id, x, y, math.radians(yaw_deg))

    def mark_result(self) -> str | None:
        state = self.binding.task_state(self.robot_id)
        if state == task_map.TASK_SUCCEEDED:
            return "SUCCEEDED"
        if state == task_map.TASK_FAILED:
            return self.binding.task_failure(self.robot_id) or "ABORTED"
        return None

    def save_map(self, name: str) -> explore.SavedMap | None:
        """map_saver_cli and slam_toolbox's serialize, on the robot; then every file copied back
        and compared by md5 with the robot's own. Any mismatch is no answer: copied again."""
        base_name = name.replace("/", "__")
        if not _NAME.match(base_name.replace("__", "-")):
            return None
        maps = self._paths["maps"]
        base = f"{maps}/{base_name}"
        r = self.shell.run(self._sourced(
            f"mkdir -p {maps} && cd {maps} && timeout 40 ros2 run nav2_map_server map_saver_cli "
            f"-f {base} --occ {explore.OCCUPIED_THRESH} --free {explore.FREE_THRESH} "
            f"--ros-args -p save_map_timeout:=15.0 -p "
            f"map_subscribe_transient_local:=true; echo \"map_saver rc=$?\""))
        if r is None or b"map_saver rc=0" not in r[1]:
            return None
        resp = self.ros.call("/slam_toolbox/serialize_map", "slam_toolbox/srv/SerializePoseGraph",
                             {"filename": base}, timeout=30.0)
        if resp is None or resp.get("result") != 0:
            return None
        exts = (".pgm", ".yaml", ".posegraph", ".data")
        r = self.shell.run("md5sum " + " ".join(base + e for e in exts))
        if r is None:
            return None
        remote = {Path(p[1]).name: p[0] for p in
                  (ln.split() for ln in r[1].decode().splitlines()) if len(p) == 2}
        files = {}
        for e in exts:
            got = self.shell.run(f"cat {base}{e}")
            if got is None or got[0] != 0:
                return None
            if hashlib.md5(got[1]).hexdigest() != remote.get(base_name + e):
                return None
            files["map" + e] = got[1]
        files["map.yaml"] = re.sub(rb"^image:.*$", b"image: map.pgm", files["map.yaml"],
                                   count=1, flags=re.MULTILINE)
        return explore.SavedMap(files=files, graph_ref=base)

    def end_segment(self) -> None:
        """The explorer, the keep-out servers and SLAM, each by its own name, after the save."""
        self._kill("lib/explore_lite/explore|keepout_servers.launch.py|__node:=keepout_|"
                   "__node:=lifecycle_manager_keepout|online_async_launch.py|"
                   "async_slam_toolbox_node")
        self._slam_launched = self._keepout_launched = False
        self._reset_at = None

    def battery_mv(self) -> float | None:
        v = self.binding.latest_battery(self.robot_id).voltage
        return v * 1000.0 if v is not None and math.isfinite(v) else None

    def battery_age_s(self) -> float | None:
        return self.binding.battery_age_s()

    def battery_source(self) -> str:
        return self.binding.battery_source()

    def imu_streaming(self) -> bool | None:
        samples = self.ros.recent(self.imu_topic, IMU_WINDOW_S)
        if len(samples) < IMU_MIN_SAMPLES:
            return False
        gyro = {tuple(m.get("angular_velocity", {}).get(k, 0.0) for k in "xyz")
                for _, m in samples}
        return len(gyro) > 1

    def charge_wait(self) -> None:
        """Nothing to do: on this robot a person plugs the cable in."""


def _yaml_fields(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition(":")
        if sep:
            out[k.strip()] = v.strip()
    return out


def _origin(meta: dict[str, str]) -> list[float]:
    return [float(t) for t in meta["origin"].strip("[]").split(",")]


def ros_type(name: str):
    """The Python class of ``<pkg>/<msg|srv|action>/<Class>``, where ``<Class>`` may be a dotted
    path inside the module (an action's ``NavigateToPose.Impl.FeedbackMessage``)."""
    pkg, kind, cls = name.split("/")
    obj = importlib.import_module(f"{pkg}.{kind}")
    for part in cls.split("."):
        obj = getattr(obj, part)
    return obj


class RclpyFacade:  # pragma: no cover - needs a ROS 2 runtime: exercised on the stand-in, then the robot
    """`RosFacade` on the binding's rclpy node, created on first use (the node exists once the
    binding has started). Messages are handed on as plain dicts, and every call is bounded by
    its timeout: a lost reply is None.

    A message is kept as it arrived and turned into a dict only when it is read, once. The map
    and the costmap are whole grids, published about once a second, and only the latest of each
    is ever read; converting each on arrival, and keeping hundreds of them, cost CPU and a
    steadily growing memory on the stand-in. So only the topics read as a sequence -- the
    navigation feedback (drained) and the IMU (a recent window) -- are queued."""

    def __init__(self, binding, imu_topic: str = "imu") -> None:
        self.binding = binding
        self.queued = {_FEEDBACK, _PLAN, _LOCAL_PLAN, imu_topic}
        self.topics = {
            "/map": ("nav_msgs/msg/OccupancyGrid", True),
            "/global_costmap/costmap": ("nav_msgs/msg/OccupancyGrid", False),
            _MASK: ("nav_msgs/msg/OccupancyGrid", True),
            "explore/status": ("explore_lite_msgs/msg/ExploreStatus", True),
            # An action's feedback topic carries the action's own FeedbackMessage class, which
            # rclpy nests in the action type; no module-level name for it exists.
            _FEEDBACK: ("nav2_msgs/action/NavigateToPose.Impl.FeedbackMessage", False),
            _PLAN: ("nav_msgs/msg/Path", False),
            _LOCAL_PLAN: ("nav_msgs/msg/Path", False),
            imu_topic: ("sensor_msgs/msg/Imu", False),
        }
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._queues: dict = {}
        self._dicts: dict = {}           # topic -> (arrival time, dict): the latest, converted once
        self._convert = None
        self._clients: dict = {}
        self._pubs: dict = {}
        self._node = None

    _type = staticmethod(ros_type)

    def _ensure(self):
        if self._node is not None:
            return self._node
        node = self.binding._node
        if node is None:
            raise RuntimeError("the binding has not started; there is no node to read from")
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.convert import message_to_ordereddict
        self._convert = message_to_ordereddict

        for topic, (type_name, latched) in self.topics.items():
            qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                             durability=(DurabilityPolicy.TRANSIENT_LOCAL if latched
                                         else DurabilityPolicy.VOLATILE))
            queued = topic in self.queued

            def on_msg(msg, topic=topic, queued=queued):
                t = time.monotonic()
                with self._lock:
                    self._latest[topic] = (msg, t)
                    if queued:
                        q = self._queues.setdefault(topic, [])
                        q.append((t, msg))
                        del q[:-500]
            node.create_subscription(self._type(type_name), topic, on_msg, qos)
        self._node = node
        return node

    def now(self) -> float:
        return time.monotonic()

    def call(self, service, srv_type, fields, timeout=10.0):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.set_message import set_message_fields
        node = self._ensure()
        client = self._clients.get(service)
        if client is None:
            client = self._clients[service] = node.create_client(self._type(srv_type), service)
        if not client.wait_for_service(timeout_sec=min(timeout, 5.0)):
            return None
        req = self._type(srv_type).Request()
        set_message_fields(req, fields)
        fut = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.05)          # the binding's executor thread completes the future
        if not fut.done() or fut.result() is None:
            return None
        return dict(message_to_ordereddict(fut.result()))

    def latest(self, topic):
        self._ensure()
        with self._lock:
            got = self._latest.get(topic)
            done = self._dicts.get(topic)
        if got is None:
            return None
        msg, t = got
        if done is None or done[0] != t:
            done = (t, self._convert(msg))
            with self._lock:
                self._dicts[topic] = done
        return done[1], t

    def drain(self, topic):
        self._ensure()
        with self._lock:
            q, self._queues[topic] = self._queues.get(topic, []), []
        return [self._convert(m) for _, m in q]

    def recent(self, topic, window_s):
        self._ensure()
        now = time.monotonic()
        with self._lock:
            window = [(t, m) for t, m in self._queues.get(topic, []) if now - t <= window_s]
        return [(t, self._convert(m)) for t, m in window]

    def publish(self, topic, msg_type, fields):
        from rosidl_runtime_py.set_message import set_message_fields
        node = self._ensure()
        pub = self._pubs.get(topic)
        if pub is None:
            pub = self._pubs[topic] = node.create_publisher(self._type(msg_type), topic, 10)
        msg = self._type(msg_type)()
        set_message_fields(msg, fields)
        pub.publish(msg)

    def publisher_count(self, topic):
        try:
            return self._ensure().count_publishers(topic)
        except Exception:  # noqa: BLE001 - a graph query that fails is no answer
            return None

    def transform(self, target, source):
        from rclpy.time import Time
        try:
            tf = self.binding._tf_buffer.lookup_transform(target, source, Time()).transform
        except Exception:  # noqa: BLE001 - no transform yet is no answer
            return None
        r = tf.rotation
        yaw = math.degrees(math.atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y ** 2 + r.z ** 2)))
        return tf.translation.x, tf.translation.y, yaw
