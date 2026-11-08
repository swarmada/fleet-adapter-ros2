# Plant probe fixtures

Input for `tests/test_plant_probe.py`. Every file here is synthetic: written for this
repository by a script, from what a physical ROS 2 robot reported (a Hiwonder MentorPi,
Mecanum chassis, ROS 2 Humble in the vendor's container). None of them is a configuration
for any robot, and none is a capture. The md5 of each file is below and is pinned by
`test_fixture_bytes_are_the_pinned_ones`.

| file | md5 | what it is |
| :-- | :-- | :-- |
| `synthetic_nav2_params.yaml` | `810cffe5b9a6df72d90894c5c28daac0` | `velocity_smoother` and both costmaps, with no `footprint` key |
| `synthetic_nav2_controller_teb.yaml` | `f053863df6d544ad7bfe6e35ac21aa2a` | a `controller_server` whose `FollowPath` is TEB, with `footprint_model` |
| `synthetic_nav2_controller_dwb.yaml` | `9202a4b43ca93fc8db83ee1f2bdbb405` | a `controller_server` whose `FollowPath` is DWB, with no `footprint_model` or `min_obstacle_dist` |
| `synthetic_lidar_mount.txt` | `dd4963a88ac215bda88fd833bf3f898f` | `tf2_echo base_footprint lidar_frame`, `tf2_echo base_link lidar_frame`, and `ros2 param get /robot_state_publisher robot_description` |
| `synthetic_velocity_limits.txt` | `41cfc6370cbfddc163034d6afbc7bee7` | `ros2 param get /velocity_smoother max_velocity` and `max_decel`, read with Nav2 running |

## What they hold

Each file holds only what the probe's parsers (`testbed/probe/plant_fields.py`) and the tests
read, laid out the way the robot's files and the commands present it, so the parsers take the
same paths they take on the real thing:

- **The parameter files:** the nested `local_costmap.local_costmap` layout; a TEB and a DWB
  controller told apart by their `plugin`; and the keys a file leaves out (`footprint`,
  `footprint_padding`, and on DWB `footprint_model` and `min_obstacle_dist`), which the parser
  must leave absent rather than fill.
- **The lidar mount:** the first `Translation:` line each `tf2_echo` printed, and a URDF with
  every joint the robot's URDF declared (name, type, parent, child and origin, which is what
  `urdf_joints` reads). No links, inertials, meshes, materials or comments.
- **The velocity limits:** the two `Double values are:` lines, in the order the tests read them.

The numbers are the ones the robot reported, and so are the frame and joint names, because the
probe reports them. That keeps the cross-checks meaningful:
`test_file_cap_equals_the_live_param_dump_taken_with_nav2_up` compares the parameter file with
the live limits, and `test_urdf_dump_recovers_the_lidar_pose_and_agrees_with_tf2_echo` compares
the URDF with the live transform. Neither compares against a value chosen to fit it.

These files replace byte copies of the robot's files and of two session transcripts. The
parsers return the same result for each synthetic file as they did for the file it replaces.
