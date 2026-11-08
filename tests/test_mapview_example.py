# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The map-view Hello World as shipped: the image holds what its two commands import, and the
manifests keep the service inside the cluster, unprivileged, with the writers' token from a
Secret."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "examples" / "map-view-hello-world"


def _copied_files() -> list[Path]:
    """The files the Dockerfile copies, read from its COPY lines."""
    text = (HERE / "Dockerfile").read_text().replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        if line.startswith("COPY "):
            for src in line.split()[1:-1]:
                out += sorted(p for p in (REPO / src).rglob("*") if p.is_file()) \
                    if (REPO / src).is_dir() else [REPO / src]
    return out


def test_the_image_holds_what_its_two_commands_import(tmp_path):
    """Copy only what the Dockerfile copies, and import both commands from that alone."""
    for f in _copied_files():
        if "__pycache__" in f.parts:
            continue
        dst = tmp_path / f.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    code = ("import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; "
            "import fleet_adapter_ros2.mapview.service, fleet_adapter_ros2.mapview.hello_world; "
            "print('ok')")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path),
           "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.stdout.strip() == "ok", r.stderr
    assert (tmp_path / "fleet_adapter_ros2" / "mapview" / "static" / "viewer.js").is_file()


def test_the_dockerignore_lets_through_what_the_dockerfile_copies():
    keep = [ln[1:] for ln in (HERE / "Dockerfile.dockerignore").read_text().splitlines()
            if ln.startswith("!")]
    for f in _copied_files():
        rel = str(f.relative_to(REPO))
        assert any(rel == k or rel.startswith(k) for k in keep), rel


def test_the_manifests_keep_the_service_inside_the_cluster_and_unprivileged():
    docs = list(yaml.safe_load_all((HERE / "hello-world.yaml").read_text()))
    kinds = [d["kind"] for d in docs]
    assert "Ingress" not in kinds
    for d in docs:
        if d["kind"] == "Service":
            assert d["spec"]["type"] == "ClusterIP"
        if d["kind"] == "Deployment":
            pod = d["spec"]["template"]["spec"]
            assert pod["securityContext"]["runAsNonRoot"] is True
            assert pod["automountServiceAccountToken"] is False
            for ctr in pod["containers"]:
                sc = ctr["securityContext"]
                assert sc["readOnlyRootFilesystem"] and not sc["allowPrivilegeEscalation"]
                assert sc["capabilities"]["drop"] == ["ALL"]
                assert ctr["imagePullPolicy"] == "Never" and ctr["image"] == "swarmada-mapview:dev"
                token = [e for e in ctr["env"] if e["name"] == "SWARMADA_MAPVIEW_TOKEN"]
                assert token[0]["valueFrom"]["secretKeyRef"]["name"] == "mapview-ingest"
                module = ctr["args"][0]
                assert (REPO / (module.replace(".", "/") + ".py")).is_file()


def test_run_sh_parses_and_never_prints_the_token():
    script = (HERE / "run.sh").read_text()
    assert subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=False).returncode == 0
    assert os.access(HERE / "run.sh", os.X_OK)
    token_lines = [ln for ln in script.splitlines() if "token" in ln.lower()
                   and not ln.lstrip().startswith("#")]
    assert token_lines and all("echo" not in ln and "printf" not in ln for ln in token_lines)
    assert re.search(r"kind delete cluster --name mapview-hello", script)
