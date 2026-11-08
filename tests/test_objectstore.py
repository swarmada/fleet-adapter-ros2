# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Maps and keep-out sets through an ArtifactStore of kind ObjectStore.

The adapter on a robot shares no directory with the cluster. So a segment it saves is also
uploaded to ``<bucket>/<namespace>/<action>/`` in the store (each object's md5 compared with the
bytes sent; the manifest last, so a reader that finds it finds everything it lists), and only
then is the reference reported. A keep-out set is fetched by its reference: the manifest must
hash to the digest the reference names, only what it lists is read, and every file is proven
against it before the set is loaded. The credential comes from the environment and appears in
no message.

The signer is checked against AWS's own published example; the round trips run against a small
in-process store that answers as the Amazon API does (ETag = md5, 404 for a missing object).
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import urllib.parse

import pytest
from test_explore_job import DOC, FakeClock, _keepout_set
from test_stall_reporting import _main

from fleet_adapter_ros2 import explore, objectstore
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack

AK, SK = "fz-robot-test", "test-secret-never-printed-0123456789"


# ── the signature ───────────────────────────────────────────────────────────

def test_the_signature_is_awss_published_example() -> None:
    """AWS Signature Version 4, "GET Object" example in the Amazon object-storage API reference.
    The signature is AWS's own; it depends on the secret, the date, the scope and the request, not
    on the access key id, so a neutral id stands in for the example's (a key id is never written
    in this tree)."""
    empty = hashlib.sha256(b"").hexdigest()
    auth = objectstore.sign(
        "GET", "https://examplebucket.s3.amazonaws.com/test.txt",
        {"host": "examplebucket.s3.amazonaws.com", "range": "bytes=0-9",
         "x-amz-content-sha256": empty, "x-amz-date": "20130524T000000Z"}, empty,
        access_key="EXAMPLE-KEY-ID",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", amz_date="20130524T000000Z")
    assert auth == ("AWS4-HMAC-SHA256 Credential=EXAMPLE-KEY-ID/20130524/us-east-1/s3/"
                    "aws4_request, SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
                    "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")


# ── a store that answers as the Amazon API does ──────────────────────────────

class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[str] = []
        self.auth: list[str] = []
        self.refuse_put = False
        self.wrong_etag = False
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _key(self):
                return urllib.parse.unquote(self.path.split("?")[0].lstrip("/"))

            def do_PUT(self):
                fake.auth.append(self.headers.get("Authorization", ""))
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if fake.refuse_put:
                    self.send_response(403)
                    self.end_headers()
                    return
                fake.objects[self._key()] = body
                fake.puts.append(self._key())
                self.send_response(200)
                etag = "0" * 32 if fake.wrong_etag else hashlib.md5(body).hexdigest()
                self.send_header("ETag", f'"{etag}"')
                self.end_headers()

            def do_GET(self):
                fake.auth.append(self.headers.get("Authorization", ""))
                data = fake.objects.get(self._key())
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"

    def store(self) -> objectstore.ObjectStore:
        return objectstore.ObjectStore(self.endpoint, "floor-maps", AK, SK, timeout=2.0)


@pytest.fixture()
def s3():
    fake = FakeStore()
    yield fake
    fake.server.shutdown()


def test_put_compares_the_stores_md5_and_get_answers_none_for_nothing(s3) -> None:
    st = s3.store()
    st.put("lab/a/map.pgm", b"P5 bytes")
    assert s3.objects["floor-maps/lab/a/map.pgm"] == b"P5 bytes"
    assert st.get("lab/a/map.pgm") == b"P5 bytes" and st.get("lab/a/none") is None
    assert all(a.startswith(f"AWS4-HMAC-SHA256 Credential={AK}/") for a in s3.auth)
    s3.wrong_etag = True
    with pytest.raises(objectstore.StoreError, match="holds md5"):
        st.put("lab/a/map.yaml", b"x")


def test_no_message_carries_the_credential(s3) -> None:
    s3.refuse_put = True
    with pytest.raises(objectstore.StoreError) as exc:
        s3.store().put("lab/a/map.pgm", b"x")
    assert "HTTP 403" in str(exc.value) and SK not in str(exc.value) and AK not in str(exc.value)
    s3.server.shutdown()
    with pytest.raises(objectstore.StoreError) as exc:
        s3.store().get("lab/a/map.pgm")
    assert SK not in str(exc.value) and "did not answer" in str(exc.value)


@pytest.mark.parametrize("key", ["", "/abs", "a/../b"])
def test_an_object_name_that_is_no_name_is_refused(s3, key) -> None:
    with pytest.raises(objectstore.StoreError):
        s3.store().get(key)


@pytest.mark.parametrize("endpoint, bucket, ak", [("ftp://h:1", "b", AK), ("http://h:1/p", "b", AK),
                                                  ("http://h:1", "a/b", AK), ("http://h:1", "b", "")])
def test_an_unusable_store_configuration_is_refused(endpoint, bucket, ak) -> None:
    with pytest.raises(ValueError):
        objectstore.ObjectStore(endpoint, bucket, ak, SK)


# ── maps up, keep-outs down ─────────────────────────────────────────────────

def test_a_saved_segment_is_uploaded_manifest_last_and_reported_after(tmp_path, s3) -> None:
    arts = explore.ArtifactDir(str(tmp_path / "copy"), "floor-maps", remote=s3.store())
    stack = SimulatedExploreStack(clock=FakeClock())
    emits = []
    job = explore.ExploreJob(action_id="explore-first", doc=dict(DOC), stack=stack,
                             limits=stack.battery_limits(), artifacts=arts, namespace="lab",
                             clock=stack.clock, emit=lambda st, msg, *a: emits.append((st, msg)))
    job.run()
    state, message = emits[-1]
    assert state == "ACTION_STATE_SUCCEEDED"
    keys = [k.removeprefix("floor-maps/lab/explore-first/") for k in s3.puts]
    assert keys[-1] == explore.MANIFEST and set(keys) >= {"map.yaml", "map.pgm", "segment.json"}
    digest = hashlib.sha256(s3.objects["floor-maps/lab/explore-first/MANIFEST.sha256"]).hexdigest()
    assert f"map-ref=floor-maps/lab/explore-first map-sha256={digest}" in message
    assert any("uploaded to the store at lab/explore-first" in m for _, m in emits)


def test_an_upload_the_store_refuses_ends_failed_naming_it_and_reports_no_ref(
        tmp_path, s3) -> None:
    s3.refuse_put = True
    arts = explore.ArtifactDir(str(tmp_path / "copy"), "floor-maps", remote=s3.store())
    stack = SimulatedExploreStack(clock=FakeClock())
    emits = []
    job = explore.ExploreJob(action_id="explore-first", doc=dict(DOC), stack=stack,
                             limits=stack.battery_limits(), artifacts=arts, namespace="lab",
                             clock=stack.clock, emit=lambda st, msg, *a: emits.append((st, msg)))
    job.run()
    state, message = emits[-1]
    assert state == "ACTION_STATE_FAILED" and "did not reach the store" in message
    assert "HTTP 403" in message and "map-ref=" not in message
    assert (tmp_path / "copy/lab/explore-first/MANIFEST.sha256").is_file()   # kept on the robot


def _publish(s3, tmp_path, **kw):
    """A keep-out set rendered at the desk, then put into the store as the worker puts it."""
    ref = _keepout_set(tmp_path / "desk", **kw)
    _, ns, name, _ = explore.parse_keepouts_ref(ref)
    src = tmp_path / "desk" / ns / name
    for f in sorted(p.name for p in src.iterdir()):
        s3.objects[f"floor-maps/{ns}/{name}/{f}"] = (src / f).read_bytes()
    return ref, ns, name


def test_a_keep_out_set_is_fetched_by_reference_and_proven_by_its_digest(tmp_path, s3) -> None:
    ref, ns, name = _publish(s3, tmp_path)
    arts = explore.ArtifactDir(str(tmp_path / "robot"), "floor-maps", remote=s3.store())
    kset = explore.load_keepout_set(arts, ref)
    assert kset.keepouts and (tmp_path / "robot" / ns / name / "keepouts.json").is_file()
    s3.objects.clear()                                  # a proven copy is the set
    assert explore.load_keepout_set(arts, ref).ref == ref


def test_a_set_whose_manifest_is_not_the_one_referenced_is_refused_naming_the_ref(
        tmp_path, s3) -> None:
    ref, ns, name = _publish(s3, tmp_path)
    s3.objects[f"floor-maps/{ns}/{name}/MANIFEST.sha256"] += b"\n"
    arts = explore.ArtifactDir(str(tmp_path / "robot"), "floor-maps", remote=s3.store())
    with pytest.raises(LookupError) as exc:
        explore.load_keepout_set(arts, ref)
    assert ref in str(exc.value) and "not loaded" in str(exc.value)
    assert not (tmp_path / "robot" / ns / name).exists()


def test_a_listed_file_that_differs_in_the_store_is_refused(tmp_path, s3) -> None:
    ref, ns, name = _publish(s3, tmp_path)
    s3.objects[f"floor-maps/{ns}/{name}/keepouts.json"] = b'{"keepouts": []}'
    arts = explore.ArtifactDir(str(tmp_path / "robot"), "floor-maps", remote=s3.store())
    with pytest.raises(LookupError, match="keepouts.json hashes to"):
        explore.load_keepout_set(arts, ref)


def test_a_set_the_store_does_not_hold_is_refused(tmp_path, s3) -> None:
    arts = explore.ArtifactDir(str(tmp_path / "robot"), "floor-maps", remote=s3.store())
    with pytest.raises(LookupError, match="is not in the store"):
        explore.load_keepout_set(arts, "floor-maps/lab/keepouts-0123456789abcdef@sha256:"
                                 + "a" * 64)


# ── start-up ────────────────────────────────────────────────────────────────

def test_a_store_without_a_credential_refuses_to_start_naming_the_variables(
        monkeypatch, capsys) -> None:
    monkeypatch.delenv(objectstore.ACCESS_KEY_ENV, raising=False)
    monkeypatch.delenv(objectstore.SECRET_KEY_ENV, raising=False)
    code, err, _ = _main(monkeypatch, capsys, "--store-endpoint", "http://127.0.0.1:9")
    assert code == 2
    assert objectstore.ACCESS_KEY_ENV in err and objectstore.SECRET_KEY_ENV in err


def test_the_store_is_described_and_its_credential_never_printed(monkeypatch, capsys) -> None:
    monkeypatch.setenv(objectstore.ACCESS_KEY_ENV, AK)
    monkeypatch.setenv(objectstore.SECRET_KEY_ENV, SK)
    code, err, _ = _main(monkeypatch, capsys, "--store-endpoint", "http://127.0.0.1:9",
                         "--artifact-store", "floor-maps")
    assert code is None
    assert "object store, bucket floor-maps at --store-endpoint" in err
    assert "127.0.0.1" not in err                       # an address on a site's network
    assert AK not in err and SK not in err


def test_a_bucket_without_a_store_is_refused(monkeypatch, capsys) -> None:
    code, err, _ = _main(monkeypatch, capsys, "--store-bucket", "floor-maps")
    assert code == 2 and "--store-endpoint" in err

