#!/usr/bin/env bash
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# The map-view Hello World: a simulated robot explores a synthetic room in a kind cluster, and
# a browser shows the map it builds, live. No robot and no ROS are involved.
#
#   examples/map-view-hello-world/run.sh
#
# 1. Creates kind cluster mapview-hello (or uses an existing one; see below).
# 2. Builds image swarmada-mapview:dev and loads it into the cluster.
# 3. Creates the writers' token (Secret mapview-ingest) and applies hello-world.yaml: the
#    map-view service and the simulated explore-floor source.
# 4. Forwards localhost:8080 to the service. Open http://localhost:8080/ and leave this
#    running; Ctrl-C ends the forward and leaves the cluster up.
#
# Environment:
#   MAPVIEW_CLUSTER        kind cluster name (default mapview-hello)
#   MAPVIEW_USE_EXISTING=1 use the existing cluster MAPVIEW_CLUSTER instead of creating one
#   MAPVIEW_PORT           local port for the viewer (default 8080)
#   MAPVIEW_PORT_FORWARD=0 stop after the rollout, without forwarding the port
#
# Remove everything with: kind delete cluster --name mapview-hello

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

HERE="examples/map-view-hello-world"
CLUSTER="${MAPVIEW_CLUSTER:-mapview-hello}"
NS="mapview-hello"
IMAGE="swarmada-mapview:dev"
PORT="${MAPVIEW_PORT:-8080}"

step() { printf '\n==> %s\n' "$*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

for tool in docker kind kubectl; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH (see $HERE/README.md, Requirements)"
done
docker info >/dev/null 2>&1 || fail "docker is installed but not answering: start Docker and run this again"

KUBE=(kubectl --context "kind-$CLUSTER")

step "Cluster $CLUSTER"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  [[ "${MAPVIEW_USE_EXISTING:-0}" == "1" ]] \
    || fail "a kind cluster named $CLUSTER exists. Set MAPVIEW_USE_EXISTING=1 to use it, or MAPVIEW_CLUSTER to pick another name"
  echo "    using the existing cluster"
else
  [[ "${MAPVIEW_USE_EXISTING:-0}" == "1" ]] && fail "MAPVIEW_USE_EXISTING=1, but there is no kind cluster named $CLUSTER"
  kind create cluster --name "$CLUSTER" --wait 120s
fi

step "Image $IMAGE"
docker build -q -f "$HERE/Dockerfile" -t "$IMAGE" . >/dev/null
kind load docker-image "$IMAGE" --name "$CLUSTER"

step "Map-view service and simulated source"
"${KUBE[@]}" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Namespace
metadata: {name: $NS}
YAML
if ! "${KUBE[@]}" -n "$NS" get secret mapview-ingest >/dev/null 2>&1; then
  # The writers' token. It is generated here, kept in the cluster and never printed.
  "${KUBE[@]}" -n "$NS" create secret generic mapview-ingest \
    --from-literal=token="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')" >/dev/null
fi
"${KUBE[@]}" apply -f "$HERE/hello-world.yaml"
"${KUBE[@]}" -n "$NS" rollout restart deployment/map-view deployment/sim-explorer >/dev/null
"${KUBE[@]}" -n "$NS" rollout status deployment/map-view --timeout=180s
"${KUBE[@]}" -n "$NS" rollout status deployment/sim-explorer --timeout=180s

if [[ "${MAPVIEW_PORT_FORWARD:-1}" == "0" ]]; then
  step "Ready"
  echo "    forward the viewer with:"
  echo "    kubectl --context kind-$CLUSTER -n $NS port-forward svc/map-view $PORT:8080"
  exit 0
fi

step "Viewer: open http://localhost:$PORT/  (Ctrl-C ends the forward; the cluster stays up)"
exec "${KUBE[@]}" -n "$NS" port-forward svc/map-view "$PORT:8080"
