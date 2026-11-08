# R1 / R2 testbed — step-by-step

Live-runtime validation for `fleet-adapter-ros2`. **No GPU, no Isaac, no robot.**
Everything here runs in Docker on an Apple Silicon Mac.

**Goal:** execute the four blocks in `nav2_binding.py` that carry
`# pragma: no cover — requires the ROS 2 runtime`, then retire
`adapters/REGISTRY.md`'s *"no live ROS 2/Nav2 runtime yet"*.

| | R1 | R2 | R3 |
|---|---|---|---|
| What's on the other end | a fake robot node | **real Nav2** | **Isaac Sim** |
| Proves | transport: `start()`, dispatch, cancel, halt, confirmed stop | semantics: rejection, ABORT, replanning, cancel latency | scale, scene, multi-robot |
| Time | ~1–2 days | ~3–5 days | after a GPU exists |
| Cost | free | free | rented GPU time |
| Hardware | your Mac | your Mac | **rented RTX box** |

**R3 is documented in [`R3.md`](R3.md)** — GPU requirements, the RT-core trap, bootstrap,
and what changes. Do not start it until R1 and R2 are green.

---

## 0. Prerequisites

On your Mac, once:

```bash
# Docker Desktop, if not already installed
brew install --cask docker
open -a Docker            # let it finish starting

docker --version && docker compose version
```

You do **not** need ROS 2, Nav2, or Python 3.12 on the host. Everything lives in
the container.

### Where these files go

```bash
cd /path/to/swarmada/adapters/external/fleet-adapter-ros2
mkdir -p testbed
# copy the delivered files into testbed/
```

They belong in the **adapter's** repo, not the core tree — they test that adapter,
and a fake ROS 2 robot is vendor-neutral Apache-2.0 test infrastructure that
passes the neutral-standard test on sight.

Final layout:

```
fleet-adapter-ros2/
├── fleet_adapter_ros2/        # the adapter under test (unchanged)
├── tests/                     # existing unit tests (unchanged)
└── testbed/
    ├── README.md              # this file
    ├── Dockerfile             # ros:jazzy + nav2 + tooling
    ├── docker-compose.yml     # services, one DDS domain
    ├── Makefile               # make r1 / make r2
    ├── fake_robot_node.py     # R1 robot; R2 plant (--passive)
    ├── r1_smoke.py            # R1 assertions
    ├── r2_sensors.py          # R2: TF tree + raycast /scan
    ├── nav2_params.yaml       # R2: Nav2 configuration
    ├── make_map.py            # R2: generates map/warehouse.{pgm,yaml}
    ├── r2_smoke.py            # R2 assertions
    ├── R3.md                  # R3: hardware, Isaac, and the demo ladder
    ├── cloud_bootstrap.sh     # R3: GPU gate + Docker + Isaac image
    ├── isaac_ros2_graph.py    # R3: the Isaac OmniGraph ROS 2 bridge
    └── vda5050_shim.py        # R3: Isaac/ROS 2 -> VDA5050 MQTT (also runs on real robots)
```

---

## 1. R1 — real ROS 2, fake robot

### 1.1 Build the image (once, ~5 minutes)

```bash
cd testbed
make build
```

### 1.2 Install the adapter and the SDK into the image

`fleet-adapter-ros2` depends on `swarmada-sdk`, which is **not on PyPI** — it lives
in the core repo at `sdk/python`. Point `SWARMADA_CORE` at your checkout:

```bash
export SWARMADA_CORE=/path/to/swarmada
make deps
```

Expected last line: `imports OK`.

> If this fails on `swarmada-sdk`, confirm `$SWARMADA_CORE/sdk/python/pyproject.toml`
> exists. That path is the only thing `make deps` assumes about the core repo.

### 1.3 Sanity-check DDS before asserting anything

```bash
make topics
```

Expected — `/odom`, `/battery_state`, `/diagnostics`, `/cmd_vel` in the topic list,
and `/navigate_to_pose` in the action list. **If the lists are empty, stop here** and
read §4 (DDS discovery). Every later failure will be a discovery failure wearing a
different mask.

### 1.4 Run R1

```bash
make r1
```

Expected output:

```
A. start() - node, subscriptions, action client, cmd_vel publisher
  PASS  start() returned without raising
  PASS  pose() is populated from live /odom
  PASS  battery_percent() is populated from live /battery_state
B. goal dispatch - a real NavigateToPose goal over DDS
  PASS  the robot moved after the goal was dispatched
  PASS  is_stopped() is False while moving
C+D. command_stop() - zero-Twist on /cmd_vel AND live goal cancel
  PASS  is_stopped() is NOT immediately True after command_stop (not inferred)
  PASS  is_stopped() became True from live odometry
  PASS  confirmation took a measurable, non-zero interval
  PASS  the robot stays at rest (the halt held)

ALL R1 ASSERTIONS PASSED
```

Exit code 0. **That is R1 complete** — all four uncovered blocks executed, and
C5.3 confirmed-stop demonstrated from live odometry rather than inference.

### 1.5 Fault injection — do this before calling R1 done

The happy path passing is necessary, not sufficient. Run each of these and record
what the adapter does:

```bash
# Nav2-side goal rejection: the adapter must map it to FAILED, not hang
docker compose run --rm fake-robot bash -lc \
  'source /opt/ros/jazzy/setup.bash && python3 /ws/testbed/fake_robot_node.py --reject-goals'

# terminal ABORTED instead of SUCCEEDED
... --goal-outcome abort

# progress stalls mid-goal without a terminal status — does the adapter notice?
... --stall-after 3.0

# slow deceleration widens the confirmed-stop window; is_stopped must stay False longer
... --decel 0.3
```

Anything that hangs is a **real defect** — open an issue against this repository.

### 1.6 Wire it into CI

`fleet-adapter-ros2/.github/workflows/conformance.yml` currently runs the
`--binding simulated` path only, so without this step the live path is uncovered in
CI the moment you stop looking. Add a job that runs `make -C testbed r1` on
`ubuntu-latest` — no GPU, no self-hosted runner needed.

---

## 2. R2 — real Nav2

Do not start R2 until R1 is green. Every R2 failure is ambiguous until the
transport is known-good.

### 2.1 What changes

`fake_robot_node.py` is **unmodified** — it runs with `--passive`, which switches off
its own action server and built-in controller so it becomes a pure plant: integrate
the commanded twist, report odometry. Nav2 owns `navigate_to_pose` and drives via
`/cmd_vel`. An R1 regression therefore stays an R1 regression.

Two things Nav2 needs that R1 does not provide, both supplied by `r2_sensors.py`:

1. a TF chain `map → odom → base_link → laser`
2. a `sensor_msgs/LaserScan` for the obstacle costmap, raycast against the map image

`map → odom` is published as **identity** — "perfect localisation" — so AMCL does not
run. That is deliberate: R2 tests Nav2's *planning and recovery* semantics, which is
what the adapter touches, not its particle filter.

**Why not Gazebo?** On Apple Silicon a headless Gazebo world is slow and fragile, and
adds a rendering dependency that proves nothing about the adapter. Nav2 cannot tell
the difference — it consumes `/scan` and TF, and both are real here. What you give up
is contact physics and wheel slip, which no Nav2 semantic under test depends on.

### 2.2 Generate the map

```bash
make r2-map
```

Writes `map/warehouse.pgm` and `map/warehouse.yaml`: a 20×20 m floor with perimeter
walls and two racking blocks, so the planner has something to plan around and the
costmap something to inflate.

### 2.3 Run R2

```bash
make r2
```

This brings up the passive robot, the sensor node, Nav2 (via `navigation_launch.py`
— planner, controller, bt_navigator, behaviors, smoother; **no map_server, no AMCL**),
then the assertions.

Watch for, in order:

1. Nav2's lifecycle nodes reaching `active` (~10–20 s)
2. `r2_smoke.py` driving a reachable goal to completion
3. the **unreachable** goal terminating rather than hanging
4. a cancel under an active controller producing a confirmed stop, with the latency printed

### 2.4 The number to write down

R2 prints the **cancel → confirmed-rest latency** under a live controller. That is the
real-world bound on the e-stop path. Quote it in `safety.md` rather than estimating it —
it is one of the few safety figures the project can state from measurement.

### 2.5 Expect to tune Nav2, not the adapter

`nav2_params.yaml` is a starting configuration. The first few runs will fail inside
Nav2 — costmap parameters, controller tolerances, TF timing. **That is normal and it is
not the adapter's fault.** Resist changing adapter code to make Nav2 happy; if a change
looks necessary, it is a finding worth filing rather than a patch worth applying.

### 2.6 When R2 is green

Update `adapters/REGISTRY.md`'s `fleet-adapter-ros2` row: replace *"no live ROS 2/Nav2
runtime yet"* with the live-runtime result and the date. Then the RFC-0001 maturity
language that draft 3 rests on can be revisited — that is the whole reason draft 3
is waiting on this.

---

## 3. What this does and does not prove

**Does:** the adapter starts against a real ROS 2 runtime; goals dispatch and cancel
over DDS; the halt reaches the robot; confirmed stop is read from live odometry;
the adapter survives Nav2 rejecting, aborting, replanning and preempting.

**Does not:** anything about physical hardware, contact dynamics, sensor noise,
multi-robot contention, or scale. Those are R3 (Isaac) and, ultimately, a real robot.
Do not let a green R2 become a claim it does not support — that overstatement is
exactly what draft 3 was written to remove.

---

## 4. Troubleshooting

**Empty topic list / nodes cannot see each other.** DDS discovery uses multicast,
which does not cross Docker's default bridge reliably, and Docker Desktop on macOS has
no true host network. Every service here shares one compose network and
`ROS_DOMAIN_ID=42`. If discovery still fails, force unicast:

```bash
export ROS_STATIC_PEERS="tb-fake-robot;tb-r1-smoke"
# or set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp and supply a CYCLONEDDS_URI
# with <Peers> entries naming the container hostnames
```

**`ModuleNotFoundError: fleet_adapter_ros2`.** `PYTHONPATH=/ws/adapter` is set in
compose, and the repo is mounted there. Confirm with
`docker compose run --rm fake-robot ls /ws/adapter/fleet_adapter_ros2`.

**`No module named 'nav2_msgs'`.** The image installs `ros-jazzy-nav2-msgs`, but the
ROS overlay must be sourced — every command in compose starts with
`source /opt/ros/jazzy/setup.bash` for this reason.

**Nav2 never activates.** Check the TF tree first:
`docker compose --profile r2 exec nav2 bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 run tf2_ros tf2_echo map base_link'`.
A missing or stale transform is the cause far more often than a Nav2 parameter.

**Costmap is empty though `/scan` publishes.** Sensor-data QoS. Nav2's observation
sources expect `BEST_EFFORT`; `r2_sensors.py` sets this deliberately. A `RELIABLE`
publisher produces an empty costmap and no error message.

---

## 5. Not yet done

- **Multi-robot.** `nav2_binding.py` uses relative topic names, so a node namespace
  would remap all five together — but `start()` calls `rclpy.init()` with no args and
  builds `Node(...)` with no `namespace=`, and `adapter.py` uses `ap.parse_args()`,
  which rejects a trailing `--ros-args`. Plumbing a namespace through is the change.
  Not needed for R1 or R2.
- **Full-loop R1.** These smokes drive `Nav2Binding` directly, with no control plane.
  Running the whole vertical — `swarmctl` → control plane on kind → adapter
  `--binding ros2` → fake robot — is worth doing once R1 and R2 are green, and is what
  would justify moving any `end-to-end` grade in RFC-0001.
