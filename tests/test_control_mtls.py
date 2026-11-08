# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The ROS 2 adapter presents a client certificate to the control plane (C1.3).

The control plane serves ControlStream over mTLS only, and reads the adapter's name from the
client certificate's SAN (<adapter>.<namespace>.svc.cluster.local). An adapter that can only
dial plaintext cannot join it. The flags follow the reference simulation adapter's: the CA,
the certificate and the key together, or none; ``--tls-server-name`` when the endpoint is a
forward's address no server SAN covers. Each can also name ``env:VAR``, so a key handed to an
adapter inside a robot's container never has to be a file there.

The proof here is offline and real: a gRPC server that requires client certificates, signed by
a throwaway CA; the hello arrives under the certificate's SAN, and a client without one is
refused.
"""

from __future__ import annotations

import shutil
import subprocess
from concurrent import futures
from types import SimpleNamespace as NS

import grpc
import pytest
from test_stall_reporting import _main

from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter.v1 import fleet_adapter_pb2_grpc as pb_grpc
from fleet_adapter_ros2 import adapter as adapter_mod

SAN = "fleet-adapter-ros2.warehouse-a.svc.cluster.local"
SERVER_NAME = "swarmada-controlstream.swarmada-system.svc"

needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")


def _openssl(*args, cwd) -> None:
    subprocess.run(["openssl", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """A CA, a server certificate for SERVER_NAME, and a client certificate for SAN."""
    if shutil.which("openssl") is None:
        pytest.skip("needs openssl")
    d = tmp_path_factory.mktemp("pki")
    _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
             "/CN=test-ca", "-keyout", "ca.key", "-out", "ca.crt", cwd=d)
    for name, cn, san, usage in (("server", SERVER_NAME, SERVER_NAME, "serverAuth"),
                                 ("client", "fleet-adapter-ros2", SAN, "clientAuth")):
        _openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={cn}",
                 "-keyout", f"{name}.key", "-out", f"{name}.csr", cwd=d)
        (d / f"{name}.ext").write_text(f"subjectAltName=DNS:{san}\nextendedKeyUsage={usage}\n")
        _openssl("x509", "-req", "-in", f"{name}.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
                 "-CAcreateserial", "-days", "1", "-extfile", f"{name}.ext",
                 "-out", f"{name}.crt", cwd=d)
    return d


class _Plane(pb_grpc.FleetAdapterServiceServicer):
    """Answers the first ControlStream message, recording who sent it."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, list[bytes]]] = []

    def ControlStream(self, request_iterator, context):
        first = next(request_iterator)
        sans = context.auth_context().get("x509_subject_alternative_name", [])
        self.seen.append((first.WhichOneof("payload"), sans))
        yield pb.ControlPlaneMessage()


@pytest.fixture()
def plane(pki):
    servicer = _Plane()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_FleetAdapterServiceServicer_to_server(servicer, server)
    creds = grpc.ssl_server_credentials(
        [((pki / "server.key").read_bytes(), (pki / "server.crt").read_bytes())],
        root_certificates=(pki / "ca.crt").read_bytes(), require_client_auth=True)
    port = server.add_secure_port("127.0.0.1:0", creds)
    server.start()
    yield servicer, f"127.0.0.1:{port}"
    server.stop(0)


def _args(endpoint, pki=None, **over):
    base = {"endpoint": endpoint, "tls_ca": None, "tls_cert": None, "tls_key": None,
            "tls_server_name": None}
    if pki is not None:
        base.update(tls_ca=str(pki / "ca.crt"), tls_cert=str(pki / "client.crt"),
                    tls_key=str(pki / "client.key"), tls_server_name=SERVER_NAME)
    base.update(over)
    return NS(**base)


def _hello(channel, timeout=5.0):
    stub = pb_grpc.FleetAdapterServiceStub(channel)
    msg = pb.AdapterMessage(hello=pb.AdapterHello(namespace="warehouse-a"))
    return next(stub.ControlStream(iter([msg]), timeout=timeout))


@needs_openssl
def test_the_hello_arrives_under_the_client_certificates_san(plane, pki) -> None:
    servicer, endpoint = plane
    with adapter_mod.open_control_channel(_args(endpoint, pki)) as ch:
        _hello(ch)
    assert servicer.seen == [("hello", [SAN.encode()])]


@needs_openssl
def test_the_key_and_certificates_can_come_from_the_environment(plane, pki, monkeypatch) -> None:
    servicer, endpoint = plane
    for var, f in (("T_CA", "ca.crt"), ("T_CERT", "client.crt"), ("T_KEY", "client.key")):
        monkeypatch.setenv(var, (pki / f).read_text())
    args = _args(endpoint, pki, tls_ca="env:T_CA", tls_cert="env:T_CERT", tls_key="env:T_KEY")
    with adapter_mod.open_control_channel(args) as ch:
        _hello(ch)
    assert servicer.seen[-1][1] == [SAN.encode()]


@needs_openssl
def test_a_connection_without_a_certificate_is_refused(plane, pki) -> None:
    servicer, endpoint = plane
    creds = grpc.ssl_channel_credentials(root_certificates=(pki / "ca.crt").read_bytes())
    opts = (("grpc.ssl_target_name_override", SERVER_NAME),)
    with grpc.secure_channel(endpoint, creds, options=opts) as ch, \
            pytest.raises(grpc.RpcError) as exc:
        _hello(ch, timeout=3.0)
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE
    assert servicer.seen == []


@needs_openssl
def test_plaintext_cannot_reach_an_mtls_server(plane) -> None:
    servicer, endpoint = plane
    with adapter_mod.open_control_channel(_args(endpoint)) as ch, pytest.raises(grpc.RpcError):
        _hello(ch, timeout=3.0)
    assert servicer.seen == []


# ── start-up: all three or none ─────────────────────────────────────────────

@pytest.mark.parametrize("given", [("--tls-ca",), ("--tls-cert", "--tls-key"),
                                   ("--tls-ca", "--tls-key")])
def test_some_but_not_all_tls_flags_refuse_to_start(monkeypatch, capsys, tmp_path, given) -> None:
    f = tmp_path / "x.pem"
    f.write_text("x")
    argv = [a for flag in given for a in (flag, str(f))]
    code, err, robot = _main(monkeypatch, capsys, *argv)
    assert code == 2 and robot is None
    assert "--tls-ca, --tls-cert and --tls-key must be supplied together" in err


def test_a_server_name_without_tls_is_refused(monkeypatch, capsys) -> None:
    code, err, _ = _main(monkeypatch, capsys, "--tls-server-name", SERVER_NAME)
    assert code == 2 and "--tls-server-name" in err


def test_an_unreadable_key_is_refused_naming_the_flag_not_a_value(
        monkeypatch, capsys, tmp_path) -> None:
    ca = tmp_path / "ca.pem"
    ca.write_text("x")
    monkeypatch.delenv("NO_SUCH_KEY_VAR", raising=False)
    code, err, _ = _main(monkeypatch, capsys, "--tls-ca", str(ca), "--tls-cert", str(ca),
                         "--tls-key", "env:NO_SUCH_KEY_VAR")
    assert code == 2 and "--tls-key cannot be read" in err and "NO_SUCH_KEY_VAR" in err


def test_without_tls_flags_the_channel_is_plaintext_as_the_harness_serves() -> None:
    made = []
    fake = NS(insecure_channel=lambda ep: made.append(("insecure", ep)) or "ch",
              secure_channel=lambda *a, **k: pytest.fail("secure without flags"))
    assert adapter_mod.open_control_channel(_args("localhost:9"), grpc_mod=fake) == "ch"
    assert made == [("insecure", "localhost:9")]


def test_the_endpoint_is_never_printed(monkeypatch, capsys, pki) -> None:
    """On a robot run the endpoint is an address on a site's own network, and the adapter's
    start-up lines are kept as evidence."""
    made = []
    monkeypatch.setattr(adapter_mod, "open_control_channel", lambda args: made.append(1))
    monkeypatch.setattr(pb_grpc, "FleetAdapterServiceStub", lambda channel: None)
    monkeypatch.setattr(adapter_mod.fleet_adapter_ros2Adapter, "run_control_stream",
                        lambda self, stub: None)
    monkeypatch.setattr(adapter_mod.fleet_adapter_ros2Adapter, "run_safety_stream",
                        lambda self, stub: None)
    monkeypatch.setattr("sys.argv", ["adapter", "--endpoint", "10.99.88.77:9445",
                                     "--tls-ca", str(pki / "ca.crt"),
                                     "--tls-cert", str(pki / "client.crt"),
                                     "--tls-key", str(pki / "client.key"),
                                     "--tls-server-name", SERVER_NAME])
    adapter_mod.main()
    err = capsys.readouterr().err
    assert made and "mTLS to --endpoint" in err and "10.99.88.77" not in err
