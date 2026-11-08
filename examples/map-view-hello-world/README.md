# Map view: Hello World

A simulated robot explores a synthetic 8 m x 6 m room, and your browser draws the map it builds,
live: the occupancy grid as it grows, the robot's path and heading, a keep-out zone, and what the
segment reports. It runs in a local [kind](https://kind.sigs.k8s.io/) cluster and needs no robot
and no ROS.

What runs is this adapter's own explore-floor segment, the code an `AssignAction` runs, on its
simulated exploration stack, with the map-view feed the adapter runs on a robot. What does not
run is the Swarmada control plane: there is no FleetTask, admission or lease in this example.
Every message is marked simulated, and the viewer says so.

## Requirements

- Docker, running
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) v0.20 or later
- kubectl
- About 1 GB of free memory, and port 8080 free (set `MAPVIEW_PORT` to use another)

## Run it

```sh
git clone https://github.com/swarmada/fleet-adapter-ros2.git
cd fleet-adapter-ros2
examples/map-view-hello-world/run.sh
```

The script creates the cluster `mapview-hello`, builds and loads the image, starts the map-view
service and the simulated source, and forwards port 8080. The first run pulls the
`python:3.12-slim` base image. When the script prints `Viewer: open http://localhost:8080/`,
open that address and leave the script running.

## What you see

- **The map.** Grey is unknown, light is free, dark is occupied. The map grows as the robot
  drives a loop of the room at 0.18 m/s.
- **The robot.** A filled dot with a heading line, and its path behind it. If no pose arrives
  for 2 s, the dot turns hollow and grey and the panel gives its age; after 10 s the heading line
  goes.
- **The keep-out.** The red block is a keep-out measured by hand, as the segment's costmap filter
  serves it.
- **The panel.** Who moves the robot (here, "the robot explores inside its bound"), the
  segment's own reports, the map's sequence number and when it last changed, and how many
  messages the viewer refused, and why.

A segment takes about two and a half minutes: it proves the keep-out in force, explores, returns
to its start, and saves its map. A new segment starts 15 s later, in a new frame, and the viewer
starts the new map from empty.

## Clean up

Ctrl-C ends the port forward and leaves the cluster running. To remove everything:

```sh
kind delete cluster --name mapview-hello
```

## Without a cluster

The service and the simulated source need Python 3.10 or later and nothing else:

```sh
python3 -m fleet_adapter_ros2.mapview.service &
python3 -m fleet_adapter_ros2.mapview.hello_world --url http://127.0.0.1:8080
```

Then open http://127.0.0.1:8080/.

## Troubleshooting

| Message | What to do |
|---|---|
| `a kind cluster named mapview-hello exists` | Run with `MAPVIEW_USE_EXISTING=1` to use it, or delete it first. |
| `docker is installed but not answering` | Start Docker Desktop, or the Docker service, and run the script again. |
| `bind: address already in use` | Port 8080 is taken: run with `MAPVIEW_PORT=8081` and open that port. |
| The page says `Reconnecting` | The port forward ended. Run `kubectl --context kind-mapview-hello -n mapview-hello port-forward svc/map-view 8080:8080`. |

## How it fits together

```
 sim-explorer pod                    map-view pod                     your browser
 explore-floor segment  ── POST ──▶  map-view service  ── events ──▶  viewer
 on the simulated stack   mapview/0  latest per robot   (read-only)
```

- The data is the map-view contract, `mapview/0`, a draft:
  [`fleet_adapter_ros2/mapview/CONTRACT.md`](../../fleet_adapter_ros2/mapview/CONTRACT.md).
- The service has a ClusterIP only. Nothing outside the cluster reaches it except through your
  port forward. A writer needs the token in Secret `mapview-ingest`, which `run.sh` generates and
  never prints.
- The viewer is read-only. It sends nothing to any robot, and the contract has no command
  message.
- On a robot, the same feed runs inside the adapter with `--mapview-url`. For a session an
  operator drives, `python -m fleet_adapter_ros2.mapview.bridge` reads the map and the pose and
  sends nothing. See [`fleet_adapter_ros2/mapview/README.md`](../../fleet_adapter_ros2/mapview/README.md).

A real map shows the layout of a home or a site. Keep the viewer of a real robot out of public
recordings.
