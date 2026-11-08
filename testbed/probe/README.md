# Plant probe

Two scripts that write a draft `plant.yaml`: what a running ROS 2 robot stack says about
the robot, each value with where it came from, and `unknown` with a reason wherever it
says nothing. The schema is `conformance/PLANT.md` in the Swarmada core tree.

| script | who moves the robot | time |
| :-- | :-- | :-- |
| `plant_probe.py` | nobody. Observe only | ~20 s on the desk testbed; `--window` (10 s) sets most of it |
| `plant_guided.py` | the operator, by hand, when told | ~1 min of windows + one typed number per question |

Neither script creates a publisher, sends a goal, or sets a parameter. The probe's node
has no `/rosout`, no parameter services and no `/parameter_events` publisher.
`probe.own_publishers` in the output is read from the graph on every run, and is `[]`.

## Requirements

`rclpy` and the Python standard library, Python ≥ 3.10 (ROS 2 Humble or later). PyYAML is
used only to parse parameter files on disk; without it those are `unknown`. Nothing is
installed, and nothing is written outside `--out`.

## Run

Copy this directory anywhere writable (`/tmp/probe`), source the robot's ROS 2
environment, and **do not set `PYTHONPATH`**: each script finds its siblings itself, and
replacing `PYTHONPATH` drops `rclpy`.

    python3 plant_probe.py --out /tmp/probe/out --declare-kind hardware \
        --reference-utc <epoch seconds from your own machine, taken just before>
    python3 plant_guided.py --plant /tmp/probe/out/plant.json      # needs a terminal (-t)
    python3 plant_guided.py --print-script                          # the operator's lines

`--declare-kind hardware` is your statement that the robot is physical, and it is recorded
as `source: operator`. The probe refuses it if it finds a simulator in the graph. Without
the flag and without a simulator marker, `plant.kind` is `unknown`, because no reading
proves a plant is physical.

Output in `--out`: `plant.yaml` (for people), `plant.json` (for tools; same content),
`summary.txt` (ends with **Needs a guided step:**), and after the guided run
`guided_summary.txt`.

## What it cannot tell you

- A parameter FILE is not the running configuration. With Nav2 down, every live Nav2
  field is `unknown` and the files on disk are listed as `file_candidates`.
- `max_velocity` is the velocity smoother's limit. A driver below it may clamp lower,
  and the graph does not show that.
- Host NTP state. The container shares the host's clock but cannot see whether it is
  synchronised: run `timedatectl` on the host.
- Across ROS 2 distributions (e.g. a Jazzy tool against a Humble robot) node discovery
  fails while topics still work. `ros.node_discovery: inconsistent` flags it.

## Tests

`tests/test_plant_probe.py` makes every read fail, time out, or answer `NOT_SET`, and
requires `unknown`. It also parses the captured files in `tests/fixtures/plant/`, which
come from a physical robot.
