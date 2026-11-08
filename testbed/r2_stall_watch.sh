#!/usr/bin/env bash
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# Catch r2_sensors in the act of stalling and sample its stack.
#
# WHAT IT DOES
#   Polls tb-r2-sensors' CPU. The signature of the fault is: the process publishes
#   normally (15-23% of one core), then drops to exactly 0.00% and stays there,
#   alive, publishing nothing, until teardown. When that transition is seen, this
#   takes two py-spy dumps five seconds apart. Two dumps, not one: a single stack
#   cannot distinguish "blocked here" from "passing through here". If both dumps
#   show the same frame, the process is blocked in it.
#
# WHAT IT DOES NOT DO
#   It does not touch r2_sensors.py, the image, or the compose file. py-spy runs
#   through `docker exec --privileged`, which grants CAP_SYS_PTRACE to the exec'd
#   process only -- the measured container's own capabilities are unchanged. The
#   standing rule is not to adapt the instrument to the thing it measures, and a
#   container that gains a capability is no longer quite the container that stalled.
#
# WHY IT REPORTS FAILURE LOUDLY
#   The hypothesis under test is that publish() blocks on a full RELIABLE send
#   queue. If py-spy cannot attach, this script must say so and exit non-zero. A
#   watcher that prints nothing looks exactly like a watcher that saw no stall,
#   and "no stall observed" is the one conclusion this run must never reach by
#   accident.
#
# USAGE
#   ./r2_stall_watch.sh            # run in a second terminal, before `make -f Makefile.r2 r2-smoke`
#   ./r2_stall_watch.sh --self-test  # prove py-spy can attach, then exit
#
set -uo pipefail

CONTAINER="${CONTAINER:-tb-r2-sensors}"
TARGET="${TARGET:-r2_sensors.py}"
BUSY_PCT="${BUSY_PCT:-3.0}"     # above this = publishing
IDLE_SAMPLES="${IDLE_SAMPLES:-4}" # consecutive ~0% samples that count as a stall
INTERVAL="${INTERVAL:-2}"
OUT="${OUT:-stall-watch-$(date +%Y%m%d-%H%M%S)}"

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 2; }
say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

command -v docker >/dev/null || die "docker not on PATH"

pid_of() {
  # -fn: newest full-command match. Plain -f also matches the bash wrapper, and
  # py-spy refuses two PIDs -- which is how this was first got wrong.
  docker exec "$CONTAINER" bash -lc "pgrep -fn $TARGET" 2>/dev/null | tr -d '\r'
}

cpu_of() {
  docker stats --no-stream --format '{{.CPUPerc}}' "$CONTAINER" 2>/dev/null \
    | tr -d '%\r' | head -1
}

running() {
  # `docker stats` reports 0.00% for a container that is STOPPING, not empty output.
  # Without this check the watcher reads a normal teardown as a stall, fires, and then
  # fails to attach because the process is already gone -- observed 2026-08-21, one
  # second after a passing run tore itself down. A watcher whose loudest output is
  # produced by success is worse than no watcher.
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

dump_to() {
  local file="$1" pid="$2"
  docker exec --privileged "$CONTAINER" \
    py-spy dump --pid "$pid" --locals > "$file" 2>&1
  return $?
}

# ---------------------------------------------------------------- preflight --
docker inspect "$CONTAINER" >/dev/null 2>&1 \
  || die "container '$CONTAINER' is not running. Bring the stack up first:
    make -f Makefile.r2 r2-up"

PID=$(pid_of)
[ -n "${PID:-}" ] || die "no '$TARGET' process inside $CONTAINER"
say "target pid $PID in $CONTAINER"

# Prove the instrument works BEFORE relying on it. If ptrace is refused, we find
# out now -- not at minute 2.5, with the reproduction gone.
PRE="/tmp/${OUT}-preflight.txt"
if ! dump_to "$PRE" "$PID"; then
  echo
  cat "$PRE" >&2
  die "py-spy could not attach.
    If this says 'Permission denied' or 'Operation not permitted', this Docker
    build does not honour --privileged on exec. Fall back to adding
        cap_add: [SYS_PTRACE]
    to the r2-sensors service in docker-compose.yml -- and record with the run
    that the measured container was modified, because it then is."
fi
say "py-spy attaches — $(grep -c 'Thread' "$PRE" 2>/dev/null || echo '?') thread(s) in the preflight dump"
[ "${1:-}" = "--self-test" ] && { say "self-test only; exiting"; exit 0; }

# ------------------------------------------------------------------- watch --
say "watching (busy > ${BUSY_PCT}%, stall = ${IDLE_SAMPLES} consecutive samples at ~0%)"
seen_busy=0
idle=0

while :; do
  if ! running; then
    say "container is no longer running — stack torn down before any stall was observed"
    say "(this is what a NORMAL end-of-run looks like; it is not a finding)"
    exit 1
  fi

  CPU=$(cpu_of)
  if [ -z "$CPU" ]; then
    say "no cpu reading for $CONTAINER — cannot measure, so reporting nothing"
    exit 2
  fi

  if awk -v c="$CPU" -v b="$BUSY_PCT" 'BEGIN{exit !(c>b)}'; then
    [ "$seen_busy" -eq 0 ] && say "publishing at ${CPU}% — baseline established"
    seen_busy=1
    idle=0
  elif [ "$seen_busy" -eq 1 ]; then
    idle=$((idle + 1))
    say "cpu ${CPU}%  ($idle/$IDLE_SAMPLES)"
    if [ "$idle" -ge "$IDLE_SAMPLES" ]; then
      if ! running; then
        say "container stopped during the idle window — normal teardown, not a stall"
        exit 1
      fi
      echo
      say "STALL — sampling the stack"
      A="/tmp/${OUT}-dump1.txt"; B="/tmp/${OUT}-dump2.txt"
      dump_to "$A" "$PID" || die "py-spy failed mid-run; see $A"
      sleep 5
      dump_to "$B" "$PID" || die "py-spy failed on the second dump; see $B"

      echo; echo "──────── dump 1 ────────"; cat "$A"
      echo; echo "──────── dump 2 (+5s) ────────"; cat "$B"
      echo; echo "────────────────────────"
      if diff -q "$A" "$B" >/dev/null; then
        printf '\033[33mIDENTICAL across 5s — the process is BLOCKED in the frame above, not slow.\033[0m\n'
      else
        printf '\033[33mDIFFERENT across 5s — it is still executing. A blocked publish() is REFUTED;\n'
        printf 'the 0%% CPU reading and the stack disagree, and the stack wins.\033[0m\n'
      fi
      echo "saved: $A  $B"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
