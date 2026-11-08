# A stand-in ROS 2 layer, for the probe's exit code only

`testbed/goal_response_probe.py` keeps its measuring separate from its plumbing, and every
rule it applies is tested directly in `tests/test_goal_response_probe.py`. One property is
not reachable that way: that the PROCESS exits 0 once the numbers are printed and the JSON is
written. That is a property of teardown order, and it only exists when there is a spin thread
and a node to take down in the wrong order.

These modules are the smallest thing `_ros_main` will run against: an action server that
accepts and answers at once, an odometry publisher with one pose, and a spin call.

The part that matters is `rclpy.spin_once`. On a real runtime, spinning a node after the
context is shut down is reported by rcl from C++, on a daemon thread with no Python handler
above it, and the process dies on `terminate called without an active exception` -- exit 134,
after everything was already measured and written. Python cannot raise that, so `spin_once`
here ends the process the same abrupt way when it is called after `shutdown()`. A probe that
tears its node down without joining the spinner exits 134 against this stub, and a probe that
joins first exits 0.

Not a ROS 2 simulator, and it is not on the path of anything else. The testbed's own
`fake_robot_node.py` is the stand-in robot for everything else, and it needs a real rclpy.
