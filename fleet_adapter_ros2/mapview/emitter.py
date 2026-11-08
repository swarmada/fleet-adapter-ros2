# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter's map-view feed: the contract (CONTRACT.md) emitted while an explore-floor
segment runs.

A feed watches; it never acts. It runs on its own thread and calls the exploration stack's
read accessors only -- ``READS`` names all three, and a test fails if it calls anything else.
The job runs exactly as it does without a feed: the feed sees the job's reports through
:meth:`SegmentFeed.observe`, which passes each one on first and only then notes it, and it ends
itself when the adapter no longer holds the job as its current action, whatever ended it. No
stop path calls into it or waits for it, and nothing it does can raise into the job or the
adapter.

Where the lines go is a sink's: :class:`FileSink` appends JSON Lines to a file, and
:class:`HttpSink` posts them to the map-view service from its own thread, holding a bounded
queue that keeps the newest message of each kind when the service cannot be reached.
"""

from __future__ import annotations

import collections
import math
import re
import sys
import threading
import time
import urllib.request

from fleet_adapter_ros2.mapview import contract as c

#: Every call a feed makes into the exploration stack. All three read; none commands.
READS = ("robot_pose", "read_map", "read_keepout_mask")

#: The source a segment's SessionStatus names: the operation, as the contract says.
SOURCE = "explore-floor"


def _frame_for(action_id: str) -> str:
    """``map@<action>``: each segment is its own SLAM session, so its own frame."""
    return ("map@" + re.sub(r"[^A-Za-z0-9_./-]", "-", action_id))[:128]


# -- sinks -------------------------------------------------------------------------------------------

class FileSink:
    """Appends each message as one JSON line. Never raises: a write that fails is counted."""

    def __init__(self, path: str) -> None:
        self._f = open(path, "a", buffering=1)  # noqa: SIM115 - held for the adapter's life
        self._lock = threading.Lock()
        self.failed = 0

    def put(self, m: dict) -> None:
        try:
            line = c.dumps(m)
            with self._lock:
                self._f.write(line + "\n")
        except Exception:  # noqa: BLE001 - a feed never raises into the adapter
            self.failed += 1

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


class HttpSink:
    """Posts messages to the map-view service (``POST <url>/v0/ingest``, JSON Lines) from its
    own thread. :meth:`put` only queues, and never blocks or raises.

    The queue holds at most ``max_queue`` messages. Past that, the oldest message whose kind has
    a newer one queued is dropped (so the oldest poses go first), and the newest MapSnapshot,
    Overlays and SessionStatus are kept, as the contract asks of a writer that cannot reach its
    reader. A batch that fails to post goes back to the front of the queue, and the next try
    waits longer, up to ``max_backoff_s``."""

    def __init__(self, url: str, token: str | None = None, *, max_queue: int = 600,
                 batch: int = 200, timeout_s: float = 3.0, max_backoff_s: float = 10.0,
                 opener=urllib.request.urlopen) -> None:
        self.url = url.rstrip("/") + "/v0/ingest"
        self._token, self._open = token, opener
        self.max_queue, self.batch, self.timeout_s = max_queue, batch, timeout_s
        self.max_backoff_s = max_backoff_s
        self._q: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._closed = False
        self.posted = self.dropped = self.failed_posts = 0
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, daemon=True, name="mapview-post")
        self._thread.start()

    def put(self, m: dict) -> None:
        with self._cv:
            self._q.append(m)
            self._trim()
            self._cv.notify()

    def _trim(self) -> None:
        while len(self._q) > self.max_queue:
            newest = {}
            for i, m in enumerate(self._q):
                newest[m.get("kind")] = i
            for i, m in enumerate(self._q):
                if newest[m.get("kind")] != i:
                    del self._q[i]
                    break
            else:
                self._q.popleft()
            self.dropped += 1

    def _post(self, msgs: list[dict]) -> None:
        body = "".join(c.dumps(m) + "\n" for m in msgs).encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/x-ndjson"})
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with self._open(req, timeout=self.timeout_s) as resp:
            resp.read()

    def _run(self) -> None:
        backoff = 0.0
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait()
                if not self._q and self._closed:
                    return
                msgs = [self._q.popleft() for _ in range(min(self.batch, len(self._q)))]
            try:
                self._post(msgs)
                self.posted += len(msgs)
                backoff = 0.0
            except Exception as exc:  # noqa: BLE001 - the service down is not the adapter's fault
                self.failed_posts += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                with self._cv:
                    self._q.extendleft(reversed(msgs))
                    self._trim()
                    if self._closed:
                        return
                backoff = min(self.max_backoff_s, max(0.5, backoff * 2))
                time.sleep(backoff)

    def close(self, flush_s: float = 2.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
        self._thread.join(flush_s)


# -- one segment's feed ------------------------------------------------------------------------------

class SegmentFeed:
    """The contract for one explore-floor segment. See the module docstring."""

    def __init__(self, *, stack, sink, robot: str, action_id: str, simulated: bool = False,
                 clock=time.time, pose_period_s: float = c.POSE_PERIOD_S,
                 map_check_s: float = c.MAP_CHECK_S, map_repeat_s: float = c.MAP_REPEAT_S) -> None:
        self._stack, self._sink, self.robot = stack, sink, robot
        self.frame, self.simulated = _frame_for(action_id), simulated
        self._clock = clock
        self.pose_period_s, self.map_check_s, self.map_repeat_s = (pose_period_s, map_check_s,
                                                                   map_repeat_s)
        self._lock = threading.Lock()
        self._report: tuple[str, str, int | None] | None = None     # (state, text, progress)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.read_errors = 0
        self._seq = 0
        self._map_sha = None
        self._map_sent_at = None
        self._keepout_sha = None
        self._area = None

    # -- the job's side: noted, never acted on -----------------------------------------------

    def observe(self, emit):
        """``emit`` wrapped: each report is passed on first, then noted for the feed."""
        def report(state_name, message="", progress_pct=0, *rest, **kw):
            emit(state_name, message, progress_pct, *rest, **kw)
            try:
                text = str(message)[:c.MAX_TEXT]
                pct = progress_pct if isinstance(progress_pct, int) else None
                with self._lock:
                    self._report = (str(state_name), text, pct)
            except Exception:  # noqa: BLE001, S110 - noting a report may never fail the report
                pass
        return report

    def start(self, alive) -> None:
        """Run until ``alive()`` is false (the job is no longer the adapter's current action)
        or :meth:`stop` is called."""
        self._thread = threading.Thread(target=self._run, args=(alive,), daemon=True,
                                        name=f"mapview-{self.frame}"[:60])
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the feed's own thread ----------------------------------------------------------------

    def _read(self, name: str):
        try:
            return getattr(self._stack, name)()
        except Exception:  # noqa: BLE001 - a read that fails is no reading
            self.read_errors += 1
            return None

    def _put(self, kind: str, **fields) -> None:
        try:
            self._sink.put(c.message(kind, self.robot, self.frame, self._clock(), **fields))
        except Exception as exc:  # noqa: BLE001 - a feed never raises into the adapter
            print(f"mapview: {kind} not emitted: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _status(self, phase: str, text: str, pct: int | None) -> None:
        self._put("SessionStatus", source=SOURCE, phase=phase, text=text[:c.MAX_TEXT],
                  area_free_m2=self._area, simulated=self.simulated,
                  progress_pct=pct if pct is None or 0 <= pct <= 100 else None)

    def _map(self, now: float) -> None:
        g = self._read("read_map")
        if g is None:
            return
        try:
            fields = c.encode_flat(g.data, g.width, g.height, g.res, (g.ox, g.oy))
        except c.ContractError:
            self.read_errors += 1
            return
        self._area = round(sum(v == 0 for v in g.data) * g.res * g.res, 3)
        if (fields["sha256"] != self._map_sha or self._map_sent_at is None
                or now - self._map_sent_at >= self.map_repeat_s):
            self._put("MapSnapshot", seq=self._seq, **fields)
            self._seq += 1
            self._map_sha, self._map_sent_at = fields["sha256"], now

    def _keepouts(self) -> None:
        k = self._read("read_keepout_mask")
        if k is None:
            return
        try:
            data = [100 if v == 100 else -1 for v in k.data]
            fields = c.encode_flat(data, k.width, k.height, k.res, (k.ox, k.oy))
        except c.ContractError:
            self.read_errors += 1
            return
        if fields["sha256"] != self._keepout_sha:
            self._keepout_sha = fields["sha256"]
            self._put("Overlays", keepouts=[{"name": "keep-out (as served)", "grid": fields}],
                      zones=[], target=None)

    def _run(self, alive) -> None:
        try:
            self._status("started", "explore-floor segment started", None)
            self._put("Overlays", keepouts=[], zones=[], target=None)
            said, next_map, posed = None, 0.0, False
            while not self._stop.is_set() and alive():
                now = self._clock()
                p = self._read("robot_pose")
                if (isinstance(p, tuple) and len(p) == 3
                        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
                    self._put("PoseSample", x=float(p[0]), y=float(p[1]), yaw_deg=float(p[2]),
                              source="sim" if self.simulated else "slam")
                    posed = True
                # The map and the mask are read once a pose has been: by then SLAM is up and
                # the job has brought up everything they are read through.
                if posed and now >= next_map:
                    next_map = now + self.map_check_s
                    self._keepouts()
                    self._map(now)
                with self._lock:
                    report = self._report
                if report is not None and report[1] != said:
                    said = report[1]
                    self._status("running", report[1], report[2])
                self._stop.wait(self.pose_period_s)
            if posed:
                self._map(self._clock())
            with self._lock:
                report = self._report
            if report is not None and report[0] in ("ACTION_STATE_SUCCEEDED",
                                                    "ACTION_STATE_FAILED"):
                self._status("stopped", report[1], report[2])
            else:
                self._status("stopped", "the segment ended without a final report: a stop, a "
                             "cancel or a lease lapse ended it", None)
        except Exception as exc:  # noqa: BLE001 - the feed's boundary: never into the adapter
            print(f"mapview: the feed for {self.frame} ended on {type(exc).__name__}: {exc}",
                  file=sys.stderr)


class MapviewEmitter:
    """What the adapter holds: a sink and a robot alias, and a feed for each segment or
    navigate-handover.

    ``session``: the SLAM session whose frame (``map@<session>``) a navigate-handover's circle is
    drawn in -- the session the map in the viewer comes from, so the circle lands on it. Without
    one, each handover is its own frame, as each explore-floor segment is."""

    def __init__(self, sink, robot: str, *, simulated: bool = False, session: str | None = None,
                 **feed_kw) -> None:
        # Refused at start-up, not at the first segment: an alias the contract refuses would
        # make every message of every segment fail.
        c.message("PoseSample", robot, "map@check", 0.0, x=0.0, y=0.0, yaw_deg=0.0, source="sim")
        self.session = session
        if session is not None:
            c.message("PoseSample", robot, f"map@{session}", 0.0, x=0.0, y=0.0, yaw_deg=0.0,
                      source="sim")
        self.sink, self.robot, self.simulated, self._feed_kw = sink, robot, simulated, feed_kw

    def segment(self, stack, action_id: str) -> SegmentFeed:
        return SegmentFeed(stack=stack, sink=self.sink, robot=self.robot, action_id=action_id,
                           simulated=self.simulated, **self._feed_kw)

    def handover(self, *, job, gate_open, action_id: str, label: str, last: bool,
                 operation: str = "navigate-handover", gate_log=None):
        """The circle's feed for one navigate-handover or manual-drive (`handover_feed`), with
        the baton read from ``gate_log`` (the binding's gate record) when there is one."""
        from fleet_adapter_ros2.mapview.handover_feed import HandoverFeed
        frame = f"map@{self.session}" if self.session is not None else _frame_for(action_id)
        return HandoverFeed(job=job, gate_open=gate_open, sink=self.sink, robot=self.robot,
                            frame=frame, label=label, last=last, simulated=self.simulated,
                            operation=operation, action_id=action_id, gate_log=gate_log)

    def close(self) -> None:
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()

