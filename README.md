# fleet-adapter-ros2

The **ROS 2 / Nav2 reference Fleet Adapter** for Swarmada. It covers the ROS 2
*class* — any robot driven through ROS 2 + the Nav2 navigation stack — with **no
vendor-specific branches** (ADR-0005). It also reaches Isaac-Sim robots via the
Isaac Sim ROS 2 bridge.

Feature-basic but **safety-complete**: the CONFORMANCE.md safety MUSTs (C3 fencing,
C4 lease self-stop, C5 confirmed estop) come from the audited `swarmada-sdk`;
optional commands are declined with `unsupported = true` (C7).

## Architecture

The adapter is transport + safety wiring around a `RobotBinding` seam
(`fleet_adapter_ros2/robot.py`):

- **`SimulatedRobot`** — the `$0` default binding (no ROS 2 install). Lets the
  skeleton dial in and pass the handshake immediately.
- **`Nav2Binding`** (`fleet_adapter_ros2/nav2_binding.py`) — the ROS 2 class
  binding: `rclpy` + Nav2. Select with `--binding ros2`. Implemented across the
  sub-items below.

`RobotBinding` declares a **lifecycle** — `start()` and `stop()` — alongside the
command and telemetry methods, and the adapter drives it in `bring_up_robot()` for
every binding alike, with no capability check and no per-binding branch. If your
binding needs no bring-up, implement both as explicit no-ops, as `SimulatedRobot`
does; `start()` MUST raise if the transport cannot be established, because returning
normally is the adapter's only evidence that the binding is live.

That is not decoration. The lifecycle was previously absent from the contract while
`Nav2Binding` had one anyway, so the adapter constructed the binding and never started
it: `--binding ros2` shipped talking to nothing, every live path short-circuited on an
absent rclpy node, and nothing raised — `pose()` reported the origin,
`battery_percent()` reported 0 and `is_stopped()` reported False forever, which failed
three conformance estop checks for a reason no output named.

## Roadmap (conformance-gated)

| Sub-item | Scope | Checks |
| :------- | :---- | :----- |
| 1 ✅ | Skeleton dials ControlStream/SafetyStream; AdapterHello/HelloAck | **C1** |
| 2 ✅ | `odom`/`battery`/`diagnostics` → telemetry (numeric-only) | C6 |
| 3 ✅ | FleetTask → `NavigateToPose` goal; task-status stream | C2, C7 |
| 4 ✅ | confirmed estop → Nav2 cancel+hold; fencing; lease self-stop | C3, C4, C5, C8 |
| 5 ✅ | pause/resume → Nav2 goal cancel-and-hold + re-dispatch | beyond C1–C8 |
| 6 ✅ | task completion streaming → `NavigateToPose` GoalStatus → `SUCCEEDED`/`FAILED` | beyond C1–C8 |
| 7 ✅ | intermediate `progressPct` → `NavigateToPose` feedback `distance_remaining` | beyond C1–C8 |

**CONFORMANT** vs the Swarmada C1–C8 harness (`--binding simulated`): 14 passed, 0
failed, 2 skipped (C4.2 lease self-stop and C6.3 RA-1 are verified separately, not
harness-observable). 26 unit tests pass (`pytest tests/`). See `../../REGISTRY.md`
for the full detail.

**Beyond the safety+task subset.** In addition to C1–C8, the adapter now maps
Swarmada `pause` / `resume` onto **Nav2 goal cancel-and-hold** (the base halts, the
`NavigateToPose` target is retained so `resume` re-dispatches it — distinct from the
estop `command_stop`, which drops the target), acking `paused=true` only once the
base is CONFIRMED at rest from odom; and it **streams task completion**, deriving
`SUCCEEDED` (goal SUCCEEDED) / `FAILED` (ABORTED) from the action `GoalStatus` and
reporting the terminal `TaskStatusUpdate` once with the executing fencing token
echoed (C6.4), so the control-plane FleetTask lifecycle sees the goal finish.

## Install

On ROS 2 Humble (Ubuntu 22.04, Python 3.10), install into a virtual environment created with
`--system-site-packages`:

```bash
sudo apt install python3-venv
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages ~/fleet-adapter-venv
~/fleet-adapter-venv/bin/python -m pip install --upgrade 'pip>=23.1'
~/fleet-adapter-venv/bin/python -m pip install <swarmada checkout>/sdk/python
~/fleet-adapter-venv/bin/python -m pip install <this checkout>
```

Why:

* **A virtual environment with `--system-site-packages`** keeps the adapter's packages apart
  from the system's, while still seeing what apt installed for the system Python: Humble's
  `rclpy` and the ROS message packages the `ros2` binding imports. A plain virtual environment
  hides them.
* **pip 23.1 or later, in that environment only.** Ubuntu 22.04's pip (22.0.2) does not keep a
  package's build apart from the system's packages, on the bare system or in a
  `--system-site-packages` environment. It builds these packages with the system's setuptools
  (59.6), which is too old to read `pyproject.toml`'s `[project]` table, and installs a package
  called `UNKNOWN 0.0.0`. pip fixed this upstream
  ([pypa/pip#6264](https://github.com/pypa/pip/issues/6264), and
  [#11740](https://github.com/pypa/pip/pull/11740) for Debian's installation layout); Ubuntu's
  report is [LP: #2018288](https://bugs.launchpad.net/ubuntu/+source/python-pip/+bug/2018288).
  The upgrade stays inside the virtual environment; the system's pip is not changed.

The Swarmada SDK is not on PyPI yet, so it is installed from a swarmada checkout. This adapter
needs swarmada 5792bb5 or later, which confirms a stop only from a state report received
after it; with an older SDK the adapter refuses to start and says so.

Then, with ROS 2 sourced:

```bash
~/fleet-adapter-venv/bin/fleet-adapter-ros2 --binding ros2 --endpoint <control-plane> ...
```

## Run

```bash
# $0 skeleton (no ROS 2) — dial the control plane / conformance harness:
python3 fleet_adapter_ros2/adapter.py --endpoint localhost:9090

# Safety-wiring tests:
pytest tests/ -v

# With a sourced ROS 2 + Nav2 workspace (later sub-items):
python3 fleet_adapter_ros2/adapter.py --binding ros2 --endpoint <control-plane> \
  --stall-window <seconds> --stall-min-progress <metres> --goal-response-bound <seconds> \
  --odometry-source <measured|command-derived>
```

`--binding ros2` refuses to start without the two `--stall-*` values. They arm the progress
watchdog: a goal whose distance to go does not shrink by `--stall-min-progress` within
`--stall-window` is cancelled and reported `ACTION_STATE_FAILED` with the reason, instead of
`RUNNING` for as long as the navigation stack never terminates it. Both are properties of the
robot, so there is no default: the window must outlast its longest turn in place or recovery,
and the minimum must exceed the noise in its pose estimate. On a robot whose odometry echoes
the velocity command, a robot that is commanded but not moving still looks like progress, so
the watchdog cannot see that case (`fleet_adapter_ros2/watchdog.py`).

`--binding ros2` also refuses to start without `--goal-response-bound`: how long a goal sent to
the navigation server may go unanswered. Past it the goal is abandoned: it is cancelled by the
id it was sent with, the robot is stopped and the goal reported `ACTION_STATE_FAILED` with the
reason, and an answer that comes later is cancelled on arrival. No later stop waits on it. How
long an answer takes is the server's and its transport's, so it is measured, not defaulted; it
should also stay well inside the 2.0 s confirmed-stop budget, so that a stop commanded while a
goal is unanswered can still be confirmed.

`--binding ros2` also refuses to start without `--odometry-source`, which says what the
odometry this binding confirms rest from actually **is**:

* `measured` — a sensor observes the robot's motion, so the reading would differ if the drive
  did not do what it was commanded (a wheel slipped, a motor stalled, someone pushed it).
* `command-derived` — the drive controller publishes the velocity command back as odometry,
  clamped, filtered or integrated, with no encoder in the path.

On a `command-derived` base the odom twist reads **zero the instant a stop is commanded**,
whatever the wheels are doing. A stop "confirmed" from it is an echo of the stop command,
which is the inference `CONFORMANCE.md` C5.3 forbids, and ADR-0071 states the precondition: a
`STOPPED` needs a motion channel independent of the command path. So when `command-derived` is
declared this adapter never reports a confirmed stop from odometry — the estop answers
`ESTOP_STATE_STOPPING` then `ESTOP_STATE_FAILED`, a lapsed lease reports
`ACTION_STATE_FAILED`, and a cancel is answered **unacknowledged** rather than
`CANCEL_DISPOSITION_STOPPED_SAFELY` (which would let a control plane requeue the action to a
second robot while this one is still moving). Each report carries the reason. **The stop
itself is unchanged and immediate**: the goal is cancelled and a zero Twist published exactly
as before. What changes is only what may be claimed about it.

Nothing on the wire or in the ROS graph distinguishes the two — same topic, same message
type, same frame — and the difference shows only when the drive disobeys, which is the one
moment a stop report matters. So there is no default and no detection: it is declared, and it
is safe only if declared honestly.

### Other senders' goals are cancelled while the adapter drives

With the ros2 binding, **while the adapter's own navigation goal runs, it cancels every other
goal on the navigation server, an operator's included.** From the moment it sends its goal (an
assignment's, a return to an explore segment's mark, a resume) until that goal succeeds, the
segment's explorer is started, or the robot is stopped, any goal the Nav2 server lists as running
that this adapter did not send is cancelled by its id. If the adapter's goal fails or is
cancelled, this goes on until its next goal or explorer start, or a stop: the server ends the
adapter's goal when a later goal preempts it. To drive the robot by hand while it is managed,
use a manual-drive or navigate-handover assignment, not a goal of your own. While the robot is
idle, after a goal that succeeded, other goals are left alone.

### The edge estop latch

An estop from an edge node (the EdgeStream) sets the robot's local latch. From then on the
adapter refuses every assignment and resume (`ROBOT_UNAVAILABLE`), and it writes the latch to a
file on the robot's own disk, `--edge-latch-file`, by default
`~/.local/state/swarmada/edge-latch-<robot-id>`. While that file is there the adapter does not
start: it exits with `REFUSED:` and the file's path. A latch file that cannot be checked counts
as set.

**Clearing it is a step at the robot, by a person.** Once the robot has been confirmed safe at
the robot, remove the file on the robot, then start the adapter:

```sh
rm ~/.local/state/swarmada/edge-latch-<robot-id>
```

The adapter offers no way to clear it over the network, and no message from the control plane
or the edge node clears it. Run under a container, mount the file's directory from the robot's
disk, or the latch goes with the container. A conformance run ends with the harness's edge
estop (C8.3), so a run without its own `--edge-latch-file` leaves the default file set: remove
it before the next run.

Conformance is run with the published Swarmada harness (`make conformance` in the
core tree, pointed at this adapter). Give each run a latch file of its own, in a temporary
directory, so a run never sets the latch at the default path on a robot:

```sh
python -m adapters.conformance --stub-path proto --adapter-name fleet-adapter-ros2 \
  --adapter-cmd "fleet-adapter-ros2 --endpoint localhost:{port} --binding simulated \
    --edge-latch-file $(mktemp -d)/edge-latch"
```

Results are tracked in the Swarmada `adapters/REGISTRY.md`.

## Map view

`--mapview-url <service>` makes the adapter emit the map-view contract (`mapview/0`, a draft)
while an explore-floor segment runs: the map, the robot's pose, the keep-out mask as the costmap
filter serves it, and the segment's reports. The feed reads and never commands, and the segment
runs and stops as it does without it. A read-only bridge does the same for a session an operator
drives. The contract, the service and the browser viewer are in
[`fleet_adapter_ros2/mapview/`](fleet_adapter_ros2/mapview/README.md), and
[`examples/map-view-hello-world`](examples/map-view-hello-world/README.md) runs the whole path on
the simulated stack in a kind cluster, with no robot.
