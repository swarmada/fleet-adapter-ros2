# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view service: takes the map-view contract in from writers, keeps the latest of each
robot's session, and streams it to read-only browser viewers.

    python -m fleet_adapter_ros2.mapview.service [--host 127.0.0.1] [--port 8080]

  POST /v0/ingest           contract lines (application/x-ndjson) from a writer: the adapter's
                            feed, the guided-session bridge, a replay. With SWARMADA_MAPVIEW_TOKEN
                            set, a writer sends ``Authorization: Bearer <token>``.
  GET  /v0/robots           each robot's frame, phase and how long since its last message
  GET  /v0/events?robot=R   server-sent events: robot R's session as it stands, then live
  GET  /                    the browser viewer
  GET  /healthz             ``ok``

Standard library only. It opens no connection of its own -- it answers -- and it has no way to
send anything to a robot: the contract has no command message. It listens on loopback unless
``--host`` says otherwise, because a map shows the layout of someone's home or site.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fleet_adapter_ros2.mapview import contract as c

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
         "/viewer.css": ("viewer.css", "text/css; charset=utf-8")}
#: The largest ingest body, in bytes.
MAX_BODY = 32 * 1024 * 1024
#: Poses a session keeps for a viewer that joins late (the path drawn), one per PATH_STEP_M.
MAX_PATH = 5000
PATH_STEP_M = 0.02
#: A viewer's queue: past this it is too slow to follow, and it is disconnected.
VIEWER_QUEUE = 2000
KEEPALIVE_S = 15.0


class Session:
    """One robot's session as the service holds it: the latest of each kind, and the path."""

    def __init__(self, frame: str) -> None:
        self.frame = frame
        self.status = self.overlays = self.map = self.pose = None
        self.path: list[dict] = []
        self.arrived: dict[str, float] = {}

    def keep(self, m: dict, now: float) -> None:
        k = m["kind"]
        if k == "MapSnapshot":
            if self.map is None or m["seq"] >= self.map["seq"]:
                self.map = m
        elif k == "PoseSample":
            self.pose = m
            last = self.path[-1] if self.path else None
            if last is None or math.hypot(m["x"] - last["x"], m["y"] - last["y"]) >= PATH_STEP_M:
                self.path.append(m)
                del self.path[:-MAX_PATH]
        elif k == "Overlays":
            self.overlays = m
        elif k == "SessionStatus":
            self.status = m
        self.arrived[k] = now

    def replay(self) -> list[dict]:
        """What a viewer joining now is sent first, in an order it can draw from."""
        out = [m for m in (self.status, self.overlays, self.map) if m is not None]
        out += self.path
        if self.pose is not None and (not self.path or self.path[-1] is not self.pose):
            out.append(self.pose)
        return out


class Viewer:
    """One browser's stream: a bounded queue, and whether ingest gave up on it."""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue(VIEWER_QUEUE)
        self.dropped = False


class MapView:
    """The service's state. Thread-safe; every method returns without waiting on a viewer."""

    def __init__(self, token: str | None = None, clock=time.monotonic) -> None:
        self.token = token
        self._clock = clock
        self._lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self._viewers: dict[str, list[Viewer]] = {}
        self.refused = {"version": 0, "contract": 0, "frame": 0}
        self.versions_seen: set[str] = set()
        self.accepted = 0
        self.closed = False

    def authorised(self, header: str | None) -> bool:
        if not self.token:
            return True
        want = f"Bearer {self.token}"
        return header is not None and hmac.compare_digest(header.encode(), want.encode())

    def _check(self, m) -> str | None:
        """None if ``m`` may be kept, or why it is refused (version, contract, frame)."""
        try:
            c.check(m)
            if m["kind"] == "MapSnapshot":
                c.decode_raw(m)
            elif m["kind"] == "Overlays":
                for ko in m["keepouts"]:
                    c.decode_raw(ko["grid"], "keepout")
        except c.UnknownVersion:
            self.versions_seen.add(str(m.get("contractVersion") if isinstance(m, dict)
                                       else None)[:32])
            return "version"
        except c.ContractError:
            return "contract"
        return None

    def ingest(self, lines: list[str]) -> dict:
        """Check and keep each line; forward what was kept to the robot's viewers."""
        counts = {"accepted": 0, "version": 0, "contract": 0, "frame": 0}
        for line in lines:
            if not line.strip():
                continue
            m = c.parse(line)
            why = self._check(m) if m is not None else "contract"
            with self._lock:
                if why is None:
                    s = self.sessions.get(m["robot"])
                    started = m["kind"] == "SessionStatus" and m["phase"] == "started"
                    if s is None or (s.frame != m["frame"] and started):
                        s = self.sessions[m["robot"]] = Session(m["frame"])
                    elif s.frame != m["frame"]:
                        why = "frame"
                if why is not None:
                    counts[why] += 1
                    self.refused[why] += 1
                    continue
                s.keep(m, self._clock())
                counts["accepted"] += 1
                self.accepted += 1
                for v in list(self._viewers.get(m["robot"], [])):
                    try:
                        v.q.put_nowait(m)
                    except queue.Full:            # too slow to follow: its stream is ended
                        self._viewers[m["robot"]].remove(v)
                        v.dropped = True
        return counts

    def subscribe(self, robot: str) -> tuple[Viewer, list[dict]]:
        v = Viewer()
        with self._lock:
            self._viewers.setdefault(robot, []).append(v)
            s = self.sessions.get(robot)
            first = s.replay() if s is not None else []
        return v, first

    def close(self) -> None:
        """End every viewer's stream and refuse new ones: the service is going away."""
        with self._lock:
            self.closed = True
            for viewers in self._viewers.values():
                for v in viewers:
                    v.dropped = True
                    try:
                        v.q.put_nowait(None)          # wakes the stream to see it is ended
                    except queue.Full:
                        pass
            self._viewers.clear()

    def unsubscribe(self, robot: str, v: Viewer) -> None:
        with self._lock:
            if v in self._viewers.get(robot, []):
                self._viewers[robot].remove(v)

    def robots(self) -> dict:
        now = self._clock()
        with self._lock:
            out = []
            for name, s in sorted(self.sessions.items()):
                out.append({
                    "robot": name, "frame": s.frame,
                    "phase": s.status["phase"] if s.status else None,
                    "source": s.status["source"] if s.status else None,
                    "simulated": bool(s.status and s.status.get("simulated")),
                    "seconds_since": {k: round(now - t, 1) for k, t in s.arrived.items()},
                })
            return {"contractVersion": c.VERSION, "robots": out, "accepted": self.accepted,
                    "refused": dict(self.refused), "versionsSeen": sorted(self.versions_seen)}


def handler_for(view: MapView):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mapview"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # one line per request, no client address
            sys.stderr.write(f"mapview: {self.command} {self.path.split('?')[0]} "
                             f"{args[1] if len(args) > 1 else ''}\n")

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; connect-src 'self'; "
                             "frame-ancestors 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, (json.dumps(obj) + "\n").encode(), "application/json")

        def do_POST(self):
            if urllib.parse.urlsplit(self.path).path != "/v0/ingest":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if not view.authorised(self.headers.get("Authorization")):
                return self._json(HTTPStatus.UNAUTHORIZED, {"error": "a writer's token is "
                                                            "required"})
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length"})
            if n < 0 or n > MAX_BODY:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                  {"error": f"a body over {MAX_BODY} bytes"})
            body = self.rfile.read(n).decode("utf-8", errors="replace")
            self._json(HTTPStatus.OK, view.ingest(body.splitlines()))

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path in PAGES:
                name, ctype = PAGES[url.path]
                return self._send(HTTPStatus.OK, (STATIC / name).read_bytes(), ctype)
            if url.path == "/healthz":
                return self._send(HTTPStatus.OK, b"ok\n", "text/plain")
            if url.path == "/v0/robots":
                return self._json(HTTPStatus.OK, view.robots())
            if url.path == "/v0/events":
                robot = urllib.parse.parse_qs(url.query).get("robot", [""])[0]
                return self._events(robot)
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _events(self, robot: str) -> None:
            if not c._ROBOT.match(robot):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "robot must be an alias"})
            if view.closed:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "shutting down"})
            v, first = view.subscribe(robot)
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for m in first:
                    self.wfile.write(f"data: {json.dumps(m)}\n\n".encode())
                self.wfile.flush()
                while not v.dropped:
                    try:
                        m = v.q.get(timeout=KEEPALIVE_S)
                        if m is None:
                            return
                        self.wfile.write(f"data: {json.dumps(m)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                view.unsubscribe(robot, v)

    return Handler


def serve(host: str, port: int, view: MapView) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), handler_for(view))
    httpd.daemon_threads = True
    return httpd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on (default 127.0.0.1: this machine only)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    token = os.environ.get("SWARMADA_MAPVIEW_TOKEN") or None
    view = MapView(token)
    httpd = serve(a.host, a.port, view)

    def stop(signum, frame):              # SIGTERM, as a pod is stopped: end every stream, exit
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"map-view service on port {a.port} ({'loopback only' if a.host == '127.0.0.1' else a.host})"
          f"; writers {'need a token' if token else 'need no token'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        view.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
