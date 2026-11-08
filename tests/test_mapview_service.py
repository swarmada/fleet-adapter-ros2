# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view service: what it takes in, what it refuses, what a viewer joining late is
sent, and that it can only answer -- it opens no connection and has nothing to send a robot."""

from __future__ import annotations

import ast
import http.client
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from fleet_adapter_ros2.mapview import contract as c
from fleet_adapter_ros2.mapview import emitter as mv
from fleet_adapter_ros2.mapview import hello_world as hw
from fleet_adapter_ros2.mapview import service as svc

R = "sim-a"


def line(kind, frame="map@s1", stamp=1.0, **kw):
    defaults = {
        "MapSnapshot": lambda: {"seq": 0, **c.encode_grid([[0, 100], [-1, 0]], 0.1, (0, 0))},
        "PoseSample": lambda: {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "source": "sim"},
        "Overlays": lambda: {"keepouts": [], "zones": [], "target": None},
        "SessionStatus": lambda: {"source": "guided:s1", "phase": "started", "text": "",
                                  "area_free_m2": None},
    }[kind]()
    defaults.update(kw)
    return c.dumps(c.message(kind, R, frame, stamp, **defaults))


@pytest.fixture
def server():
    view = svc.MapView(token="t0ken")
    httpd = svc.serve("127.0.0.1", 0, view)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield view, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data, resp


def post(port, lines, token="t0ken"):
    headers = {"Content-Type": "application/x-ndjson"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, data, _ = request(port, "POST", "/v0/ingest", "\n".join(lines) + "\n", headers)
    return status, json.loads(data)


def events(port, robot, until, timeout=5.0):
    """The server-sent events of ``robot`` until ``until(messages)`` holds."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", f"/v0/events?robot={robot}")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    got, deadline = [], time.monotonic() + timeout
    while not until(got):
        assert time.monotonic() < deadline, got
        raw = resp.fp.readline().decode()
        if raw.startswith("data: "):
            got.append(json.loads(raw[6:]))
    conn.close()
    return got


# ── ingest ──────────────────────────────────────────────────────────────────

def test_a_writer_needs_the_token_when_one_is_set(server):
    _, port = server
    assert post(port, [line("SessionStatus")], token=None)[0] == 401
    assert post(port, [line("SessionStatus")], token="wrong")[0] == 401
    assert post(port, [line("SessionStatus")])[1]["accepted"] == 1


def test_each_refusal_is_counted_by_its_reason_and_nothing_refused_is_kept(server):
    view, port = server
    bad_sha = json.loads(line("MapSnapshot"))
    bad_sha["sha256"] = "0" * 64
    status, counts = post(port, [
        line("SessionStatus"), line("PoseSample"),
        json.dumps({**json.loads(line("PoseSample")), "contractVersion": "mapview/1"}),
        json.dumps(bad_sha), "not json",
        line("PoseSample", frame="map@other"),
    ])
    assert status == 200
    assert counts == {"accepted": 2, "version": 1, "contract": 2, "frame": 1}
    assert view.sessions[R].map is None
    r = json.loads(request(port, "GET", "/v0/robots")[1])
    assert r["refused"] == {"version": 1, "contract": 2, "frame": 1}
    assert r["versionsSeen"] == ["mapview/1"]


def test_a_session_started_in_a_new_frame_replaces_the_old_one(server):
    view, port = server
    post(port, [line("SessionStatus"), line("PoseSample", x=1.0)])
    post(port, [line("SessionStatus", frame="map@s2", stamp=5.0)])
    s = view.sessions[R]
    assert s.frame == "map@s2" and s.pose is None and s.path == []


def test_a_body_over_the_limit_is_refused_unread(server, monkeypatch):
    _, port = server
    monkeypatch.setattr(svc, "MAX_BODY", 10)
    assert post(port, [line("SessionStatus")])[0] == 413


# ── viewers ─────────────────────────────────────────────────────────────────

def test_a_viewer_joining_late_gets_the_session_as_it_stands_then_live(server):
    _, port = server
    post(port, [line("SessionStatus"), line("Overlays"), line("MapSnapshot"),
                line("PoseSample", x=0.0), line("PoseSample", x=0.5, stamp=2.0),
                line("PoseSample", x=0.505, stamp=3.0)])
    got = []

    def watch():
        got.extend(events(port, R, lambda ms: len(ms) >= 7))

    t = threading.Thread(target=watch)
    t.start()
    time.sleep(0.3)
    post(port, [line("PoseSample", x=1.0, stamp=4.0)])
    t.join(10)
    kinds = [m["kind"] for m in got]
    assert kinds[:3] == ["SessionStatus", "Overlays", "MapSnapshot"]
    assert [m["x"] for m in got if m["kind"] == "PoseSample"] == [0.0, 0.5, 0.505, 1.0]


def test_the_viewer_pages_are_served_with_a_policy_that_allows_only_this_service(server):
    _, port = server
    for path, ctype in (("/", "text/html"), ("/viewer.js", "text/javascript"),
                        ("/viewer.css", "text/css")):
        status, body, resp = request(port, "GET", path)
        assert status == 200 and resp.getheader("Content-Type").startswith(ctype) and body
        assert "default-src 'self'" in resp.getheader("Content-Security-Policy")
    assert request(port, "GET", "/healthz")[1] == b"ok\n"
    assert request(port, "GET", "/v0/events?robot=10.0.0.1")[0] == 400
    assert request(port, "GET", "/../service.py")[0] == 404


def test_the_service_opens_no_connection_of_its_own():
    """It answers; nothing in it dials out, and it has no message a robot could act on."""
    tree = ast.parse(Path(svc.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert not names & {"urllib.request", "http.client", "socket", "requests", "urlopen",
                        "create_connection", "connect"}
    assert "fleet_adapter_ros2.mapview" in names
    assert not [n for n in names if n.startswith("fleet_adapter_ros2.") and "mapview" not in n]


def test_the_viewer_script_sends_nothing_anywhere():
    js = (Path(svc.STATIC) / "viewer.js").read_text()
    assert "EventSource(" in js and 'fetch("v0/robots"' in js
    for word in ("XMLHttpRequest", "WebSocket", "sendBeacon", 'method: "POST"', "http://",
                 "https://"):
        assert word not in js


def test_sigterm_ends_the_streams_and_the_service(tmp_path):
    """A pod is stopped with SIGTERM: open streams end and the process exits."""
    import os
    import signal
    import socket
    import sys

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(Path(svc.__file__).resolve().parents[2])}
    proc = subprocess.Popen([sys.executable, "-m", "fleet_adapter_ros2.mapview.service",
                             "--port", str(port)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        assert "map-view service" in proc.stdout.readline()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", f"/v0/events?robot={R}")
        resp = conn.getresponse()
        assert resp.status == 200
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(5) == 0
        assert resp.read() == b""                    # the stream ended, not hung
    finally:
        proc.kill()
        proc.wait(5)


# ── the Hello World, end to end ─────────────────────────────────────────────

def test_a_simulated_segment_reaches_a_viewer_through_the_service(server):
    view, port = server
    sink = mv.HttpSink(f"http://127.0.0.1:{port}", "t0ken")
    from fleet_adapter_ros2 import explore
    from fleet_adapter_ros2.mapview import sim_source

    final = hw.run_segment(1, sink, explore.ArtifactDir(tempfile.mkdtemp(), "hello-local"),
                           explore_seconds=60.0, clock=sim_source.FastClock(30), say=lambda t: None,
                           feed_kw={"pose_period_s": 0.02, "map_check_s": 0.1, "map_repeat_s": 1.0})
    sink.close(5)
    assert "segment done" in final
    n = len(view.sessions["sim-robot"].replay())
    got = events(port, "sim-robot", lambda ms: len(ms) >= n)
    st = c.ViewState()
    for m in got:
        assert st.accept(m)
    assert st.frame == "map@hello/explore-1" and st.status["simulated"] is True
    assert st.status["text"] == final[:c.MAX_TEXT]
    assert st.map is not None and st.keepout_grids and len(st.poses) > 10
    assert view.refused == {"version": 0, "contract": 0, "frame": 0}


CHROME = shutil.which("google-chrome") or shutil.which("chromium") or next(
    (p for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",)
     if Path(p).exists()), None)


@pytest.mark.skipif(CHROME is None, reason="no Chrome to run the browser viewer in")
def test_the_browser_viewer_checks_and_draws_what_the_service_streams(server, tmp_path):
    """Headless Chrome loads the viewer from the service; the page's own counters say what it
    accepted and refused, and a version it does not know is named, not drawn."""
    _, port = server
    post(port, [line("SessionStatus", source="guided:s1"), line("Overlays", keepouts=[
        {"name": "k", "grid": c.encode_grid([[100, -1]], 0.1, (0, 0))}]),
        line("MapSnapshot"), line("PoseSample", x=0.1), line("PoseSample", x=0.2, stamp=2.0)])
    view, _ = server
    # A line in a version the viewer does not know, put straight into the stream.
    viewer_bad = {**json.loads(line("PoseSample", x=9.0)), "contractVersion": "mapview/1"}
    with view._lock:
        view.sessions[R].path.append(viewer_bad)
        expected = len(view.sessions[R].replay()) - 1          # all but the unknown version
    # An open event stream keeps a headless page loading for ever, so the service is closed
    # once the page has had its stream: the stream ends, a reconnect is refused, and the page
    # is dumped as it stands. Headless Chrome may not exit after the dump: it is ended here.
    chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                               "--use-mock-keychain", "--password-store=basic",
                               f"--user-data-dir={tmp_path / 'profile'}",
                               "--virtual-time-budget=3000", "--dump-dom",
                               f"http://127.0.0.1:{port}/?robot={R}"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    # Chrome's start-up time varies (seconds on a loaded runner), so the service is closed only
    # once the page's stream is open, never after a fixed wait: closed earlier, the page's first
    # request is refused (503) and it draws nothing. The DOM is read on a thread so that a
    # Chrome that never writes it cannot hang the test past its deadline.
    out: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [out.put(ln) for ln in chrome.stdout] + [out.put(None)],
                     daemon=True).start()
    try:
        deadline = time.monotonic() + 60
        while chrome.poll() is None and time.monotonic() < deadline:
            with view._lock:
                if view._viewers.get(R):
                    break
            time.sleep(0.05)
        time.sleep(0.5)                       # the replay is written; let the page take it
        view.close()
        dom = ""
        while "</html>" not in dom and time.monotonic() < deadline:
            try:
                chunk = out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if chunk is None:
                break
            dom += chunk
    finally:
        chrome.kill()
        chrome.wait(10)
    assert f'data-accepted="{expected}"' in dom, dom[-2000:]
    assert 'data-refused-version="1"' in dom and 'data-refused-contract="0"' in dom
    assert 'data-map-seq="0"' in dom and 'data-keepouts="1"' in dom
    assert 'data-digest="checked"' in dom and 'data-frame="map@s1"' in dom
    assert "an operator drives the robot" in dom and "mapview/1" in dom
