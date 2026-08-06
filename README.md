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
python3 fleet_adapter_ros2/adapter.py --binding ros2 --endpoint <control-plane>
```

Conformance is run with the published Swarmada harness (`make conformance` in the
core tree, pointed at this adapter). Results are tracked in the Swarmada
`adapters/REGISTRY.md`.
