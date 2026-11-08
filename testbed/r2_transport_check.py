#!/usr/bin/env python3
"""One-shot: is the binding's transport healthy against the live R2 stack, and does
is_stopped() still confirm rest? Rules the transport-liveness change in or out as the
explanation for a `NO CONFIRMED REST` sample.

AN INSTRUMENT, and the most clear-cut of them. It scores nothing: no
assertion, no exit code, no verdict -- it prints six raw readings and leaves the reading
to a human. Its whole subject is the binding's TRANSPORT, so `start()` is not a step
borrowed from the adapter, it is the apparatus: without it `transport_alive()` is False by
construction and every line below would be a tautology rather than a diagnosis.

Contrast `r2_smoke.py`, which passes or fails the adapter and therefore brings up through
`adapter.bring_up_robot()`: a harness that starts the binding on the adapter's behalf is
scoring a configuration the adapter cannot produce."""
import os
import sys
import time
from fleet_adapter_ros2.nav2_binding import Nav2Binding

b = Nav2Binding()
b.start()
b.spawn("robot-conformance-1")   # apparatus, not a verdict
time.sleep(6.0)
print("transport_alive :", b.transport_alive())
print("transport_fault :", repr(b.transport_fault()))
print("raw _speed      :", getattr(b, "_speed", None))
print("is_stopped()    :", b.is_stopped("robot-conformance-1"))
print("pose            :", b.pose("robot-conformance-1"))
print("hardware        :", [(h.component_name, h.status) for h in b.latest_hardware("robot-conformance-1")])
sys.stdout.flush()
os._exit(0)
