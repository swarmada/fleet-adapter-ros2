#!/usr/bin/env bash
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# R3 bootstrap for a rented GPU box (any provider, or a bare cloud VM).
#
# Order matters: this REFUSES TO CONTINUE on a GPU without RT cores, before it
# installs or downloads anything. A100/H100/H200/V100/T4 are common and often the
# cheapest listings on AI-training marketplaces, and Isaac Sim will not render on
# any of them. Finding that out after a 30 GB image pull is an expensive lesson.
#
#   bash cloud_bootstrap.sh              # verify + install + pull
#   bash cloud_bootstrap.sh --check-only # just tell me if this box is usable
#
set -euo pipefail

ISAAC_IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '    \033[33mWARN\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mSTOP\033[0m %s\n\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- GPU gate ----
say "Checking the GPU before anything is installed"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found. This is not a GPU box, or the driver is not installed."

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
printf '    GPU:     %s\n    VRAM:    %s MiB\n    Driver:  %s\n' \
  "$GPU_NAME" "$GPU_MEM_MB" "$DRIVER"

# Datacenter compute parts with NO RT cores. Isaac Sim's renderer requires them.
if grep -qiE '\b(A100|H100|H200|B200|V100|T4|P100|K80|MI[0-9])' <<<"$GPU_NAME"; then
  die "$GPU_NAME has NO RT (ray-tracing) cores. Isaac Sim will not render on it.
     Destroy this instance and pick one of:
       RTX 3090 / 4090 / 5090 · A10 / A10G · L4 · L40 / L40S
       RTX A5000 / A6000 · RTX PRO 6000
    "
fi

if grep -qiE '\b(RTX|A10|A10G|L4|L40|L40S|A5000|A6000|PRO 6000|Ada)\b' <<<"$GPU_NAME"; then
  ok "RT cores present"
else
  warn "Could not positively identify RT cores on '$GPU_NAME'."
  warn "Verify the model against R3.md section 1 before spending time here."
fi

if (( GPU_MEM_MB < 15000 )); then
  die "$GPU_MEM_MB MiB VRAM is below the 16 GB practical minimum for Isaac Sim 6.0."
elif (( GPU_MEM_MB < 23000 )); then
  warn "$GPU_MEM_MB MiB VRAM works but is tight. 24 GB+ is preferred for a multi-robot scene."
else
  ok "VRAM sufficient"
fi

[[ "$(uname -m)" == "x86_64" ]] || die "Isaac Sim is Linux x86_64 only (this box: $(uname -m))."
ok "Architecture x86_64"

AVAIL_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
if (( AVAIL_GB < 60 )); then
  warn "${AVAIL_GB} GB free on /. The Isaac Sim image alone is ~30 GB, plus assets."
  warn "If the pull fails on space, resize the volume rather than pruning mid-pull."
else
  ok "${AVAIL_GB} GB free on /"
fi

if (( CHECK_ONLY )); then
  say "Check-only: this box is usable for Isaac Sim."
  exit 0
fi

# ------------------------------------------------------------------ Docker ----
say "Docker"
if command -v docker >/dev/null 2>&1; then
  ok "already installed ($(docker --version))"
else
  curl -fsSL https://get.docker.com | sh
  ok "installed"
fi

# ------------------------------------------------- NVIDIA Container Toolkit ----
say "NVIDIA Container Toolkit (lets containers see the GPU)"
if docker info 2>/dev/null | grep -qi nvidia; then
  ok "runtime already registered"
else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker 2>/dev/null || service docker restart
  ok "installed and registered"
fi

say "Verifying GPU passthrough into a container"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L \
  || die "Containers cannot see the GPU. Fix this before pulling 30 GB."
ok "passthrough works"

# ------------------------------------------------------------- Isaac image ----
say "Isaac Sim image: $ISAAC_IMAGE"
if [[ -z "${NGC_API_KEY:-}" ]]; then
  warn "NGC_API_KEY is not set. nvcr.io needs a (free) NGC account."
  warn "  Get a key at ngc.nvidia.com, then:  export NGC_API_KEY=..."
  warn "Skipping the pull. Re-run this script once the key is set."
else
  echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  say "Pulling (~30 GB — this is billed wall-clock time)"
  docker pull "$ISAAC_IMAGE"
  ok "image ready"
fi

# ------------------------------------------------------------------- ROS 2 ----
say "Testbed image (ROS 2 Jazzy + Nav2)"
if [[ -f "$(dirname "$0")/Dockerfile" ]]; then
  docker build -t swarmada/ros2-testbed:jazzy "$(dirname "$0")"
  ok "built"
else
  warn "Dockerfile not found next to this script — copy testbed/ across and run 'make build'."
fi

cat <<'NEXT'

==> Bootstrap complete.

Next:
  1. make r3-params                     # Nav2 params with use_sim_time: true
  2. docker run --rm --gpus all --network=host \
       -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e ROS_DOMAIN_ID=42 \
       -v "$PWD:/testbed:ro" nvcr.io/nvidia/isaac-sim:6.0.0 \
       ./python.sh /testbed/isaac_ros2_graph.py --robot-count 1
  3. docker compose --profile r3 up nav2-r3
  4. docker compose --profile r3 up r2-smoke     # the SAME assertions, unedited

BILLING: many GPU marketplaces bill for storage while an instance is STOPPED.
Destroy rather than stop unless you are returning within hours.

NEXT
