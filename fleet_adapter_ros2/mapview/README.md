# Map view

A read-only view of what a robot maps: the occupancy grid SLAM builds, where the robot is, what is
drawn on top, and what its session reports. Writers emit the map-view contract; the map-view
service keeps the latest of each robot's session and streams it to browsers. No part of it sends
anything to a robot.

**Status: experimental.** The contract is `mapview/0`, a sidecar format of this adapter outside the Swarmada contract, with its own versioning and no compatibility promise. See
[CONTRACT.md](CONTRACT.md) for every field, unit, frame, rate, the staleness rules and what a
reader does with a version it does not know.

To see it run with no robot, follow
[examples/map-view-hello-world](../../examples/map-view-hello-world/README.md).

## Writers

| Writer | Session | Who moves the robot | Run with |
|---|---|---|---|
| The adapter's feed (`emitter.py`) | an explore-floor segment | the robot, inside its bound | `fleet-adapter-ros2 ... --mapview-url http://<service>:8080` |
| The guided-session bridge (`bridge.py`) | a guided session (`guided:<label>`) | an operator | `python -m fleet_adapter_ros2.mapview.bridge --session s1 --url http://<service>:8080` |
| The record tool (`contract_tool.py record`) | a guided session, through a shell runner | an operator | `python -m fleet_adapter_ros2.mapview.contract_tool record --runner ...` |
| Replay (`contract_tool.py convert`) | a stored guided run (`replay:<label>`) | nobody: it is played back | `python -m fleet_adapter_ros2.mapview.contract_tool convert ...` |
| The simulated source (`hello_world.py`) | explore-floor on the simulated stack | a simulated robot | `python -m fleet_adapter_ros2.mapview.hello_world --url ...` |

### The adapter's feed

With `--mapview-url URL` (or `--mapview-file FILE`), the adapter emits the contract while an
explore-floor segment runs; `--mapview-robot` sets the robot's alias, a DNS label (default:
`--robot-id`). The service's writer token, if it has one, is read from `SWARMADA_MAPVIEW_TOKEN`.

The feed reads three accessors of the exploration stack (`robot_pose`, `read_map`,
`read_keepout_mask`) from a thread of its own, and commands nothing. The segment runs and stops
exactly as it does without a feed: the control plane receives the same reports, and cancel,
estop and a lapsed lease end it the same way. The tests in `tests/test_mapview_emitter.py` hold
each of these. Without either flag, nothing changes.

### The guided-session bridge

In a guided session an operator drives the robot, and no FleetTask exists, so no adapter job
emits the map. The bridge is a ROS 2 node that subscribes to `/map`, reads `map` to
`base_footprint` through TF, and writes the contract. It creates no publisher, no service or
action client and no goal; `tests/test_mapview_bridge.py` fails if its source names one. A viewer
shows its session as operator-driven, never as autonomous.

## The service

```sh
python -m fleet_adapter_ros2.mapview.service [--host 127.0.0.1] [--port 8080]
```

| Path | What |
|---|---|
| `POST /v0/ingest` | Contract lines from a writer (`application/x-ndjson`). With `SWARMADA_MAPVIEW_TOKEN` set, a writer sends `Authorization: Bearer <token>`. The answer counts what was accepted, and what was refused by reason: `version`, `contract`, `frame`. |
| `GET /v0/robots` | Each robot's frame, phase and seconds since its last message of each kind. |
| `GET /v0/events?robot=<alias>` | Server-sent events: the robot's session as it stands, then live. |
| `GET /` | The browser viewer. |
| `GET /healthz` | `ok` |

It listens on loopback unless `--host` says otherwise, opens no connection of its own, and needs
only the Python standard library. The desktop viewer (`contract_view.py`, which needs matplotlib)
reads a contract file, for development and replay.

## Privacy

A map shows the layout of someone's home or site. Writers carry no address, place name or network
name: the contract refuses a robot alias that is not a DNS label, so an address or host name cannot
be sent as one. Reach the service through a port forward or an authenticated proxy, and keep a
real robot's viewer out of public recordings.

## Not proven on hardware

The feed, the bridge and the viewer are tested against the simulated stack and recorded fakes. The
bridge's subscription and TF reading have not run against a robot's ROS 2 graph, and the feed has
not run during a segment on a robot.
