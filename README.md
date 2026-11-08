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

Conformance is run with the published Swarmada harness (`make conformance` in the
core tree, pointed at this adapter). Results are tracked in the Swarmada
`adapters/REGISTRY.md`.

## Map view

`--mapview-url <service>` makes the adapter emit the map-view contract (`mapview/0`, a draft)
while an explore-floor segment runs: the map, the robot's pose, the keep-out mask as the costmap
filter serves it, and the segment's reports. The feed reads and never commands, and the segment
runs and stops as it does without it. A read-only bridge does the same for a session an operator
drives. The contract, the service and the browser viewer are in
[`fleet_adapter_ros2/mapview/`](fleet_adapter_ros2/mapview/README.md), and
[`examples/map-view-hello-world`](examples/map-view-hello-world/README.md) runs the whole path on
the simulated stack in a kind cluster, with no robot.
