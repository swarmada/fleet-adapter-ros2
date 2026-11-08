# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# R2 measurement targets that CANNOT inherit a degraded stack.
#
# Every target here does a full `down -v --remove-orphans` and a fresh `up` before it
# measures anything. That is not tidiness: the TF publisher's output was observed decaying
# over the life of a stack (10 Hz -> 2.7 Hz over ~6 minutes), and a measurement taken on a
# stack that has been up for a while is a measurement through a degraded instrument. The
# scripts now assert TF liveness themselves (tf_liveness.py), but a clean stack per
# measurement is the cheaper half of the fix.
#
#   make -f Makefile.r2 r2-clean      tear everything down, including the r3 profile
#   make -f Makefile.r2 r2-up         clean, bring the R2 stack up, wait for READINESS
#   make -f Makefile.r2 r2-smoke      clean stack, then the R2 assertions
#   make -f Makefile.r2 r2-latency    clean stack, then the cancel -> rest distribution
#   make -f Makefile.r2 r2-tf-soak    clean stack, then TF hz at t+30s / t+2m / t+5m
#   make -f Makefile.r2 r2-costmap    clean stack, then the costmap lethality probe
#   make -f Makefile.r2 r2-stamp      identify this testbed + the image it runs in
#   make -f Makefile.r2 r2-manifest   write a sha256sum -c manifest of this testbed
#   make -f Makefile.r2 r2-verify     check this testbed against a recorded manifest
#   make -f Makefile.r2 r2-ready-timings   what readiness costs on THIS host
#
# SWARMADA_CORE must be exported: compose interpolates the whole file on EVERY command,
# including `exec` on an unrelated service.
#
# THIS FILE DOES NOT OVERRIDE THE SCRIPTS' OWN PARAMETERS.
# It used to pass `--settle 5` to r2_cancel_latency.py, whose own default was 20 s -- the
# target everyone runs quietly shortened the wait the script's author had chosen, by 4x,
# and on a cold stack the documented route to the R2 number returned NO SAMPLES. Both the
# override and the sleep are gone: `r2-wait` now gates on readiness (r2_ready.py) and the
# scripts gate on it again in-process. If a measurement needs a parameter, the parameter
# belongs in the script, where its default is visible next to the code that uses it.

COMPOSE ?= docker compose
# Recorded demos and repeat runs pass BUILD= to skip a fully-cached rebuild. Hardcoding
# `--build` here made `r2-up` export three 4.9 GB images on every invocation (~90 s each
# on this host, >20 min observed on a loaded one), which is why the 2026-09-02 R1/R2
# session ran raw `docker compose` instead of the documented targets -- the runbook's own
# commands were not the commands that produced its numbers.
BUILD ?= --build
SHELL := /bin/bash
FILES  := -f docker-compose.yml
# Every profile, so nothing is left behind. This needs the compose anchor fix to be in
# place -- before it, naming --profile r3 made the whole project invalid and `down`
# silently did nothing at all.
ALLPROF := --profile r2 --profile r2conf --profile r3 --profile conformance
R2PROF  := --profile r2

# Where a manifest is written by `r2-manifest`. Overridable so a measurement session can
# drop it beside that session's results document.
MANIFEST ?= testbed.manifest
# Deadline for the readiness gate. Not a settle -- see r2_ready.py.
READY_TIMEOUT ?= 180

.PHONY: r2-clean r2-up r2-wait r2-smoke r2-latency r2-tf-soak r2-costmap \
        r2-stamp r2-manifest r2-verify r2-ready-timings

r2-clean:         ## Tear down every profile and its volumes
	@: $${SWARMADA_CORE:?export SWARMADA_CORE=/path/to/swarmada}
	-$(COMPOSE) $(ALLPROF) down -v --remove-orphans

r2-up: r2-clean   ## Clean, bring up robot + sensors + Nav2, stamp, and wait for READINESS
	@: $${SWARMADA_CORE:?export SWARMADA_CORE=/path/to/swarmada}
	$(COMPOSE) $(FILES) $(R2PROF) up -d $(BUILD) r2-robot r2-sensors nav2
	$(MAKE) -f Makefile.r2 r2-stamp
	$(MAKE) -f Makefile.r2 r2-wait

r2-wait:          ## Block until Nav2 can actually PLAN and consume a scan (not merely `active`)
	@: $${SWARMADA_CORE:?export SWARMADA_CORE=/path/to/swarmada}
	@echo "waiting for Nav2 READINESS (planner round-trip + scan into the costmap frame) ..."
	$(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && cd /ws/testbed && \
	   python3 r2_ready.py --timeout $(READY_TIMEOUT)'

r2-ready-timings: ## Diagnostic: when each readiness signal arrives on THIS host
	@: $${SWARMADA_CORE:?export SWARMADA_CORE=/path/to/swarmada}
	$(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && cd /ws/testbed && \
	   python3 r2_ready.py --timings --timeout $(READY_TIMEOUT)'

r2-stamp:         ## Identify the apparatus: testbed digest + the image it runs in
	@echo "testbed:  $$(python3 testbed_provenance.py)"
	@echo "image:    swarmada/ros2-testbed:jazzy $$(docker image inspect \
	        --format '{{.Id}}' swarmada/ros2-testbed:jazzy 2>/dev/null || echo '<not built>')"
	@echo "host:     $$(uname -srm), docker $$(docker --version | cut -d, -f1), \
	load$$(uptime | sed 's/.*averages*://')"

# `eval echo` so a MANIFEST= given with a leading ~ resolves. make hands the recipe the
# literal string, and `test -f "~/..."` does NOT tilde-expand inside quotes -- which made
# r2-verify report "no such manifest" for a file r2-manifest had just written.
TILDE = M=$$(eval echo "$(MANIFEST)")

r2-manifest:      ## Write a sha256sum -c manifest of this testbed to $$(MANIFEST)
	@$(TILDE); python3 testbed_provenance.py --manifest > "$$M"; \
	 echo "wrote $$M  ($$(python3 testbed_provenance.py))"; \
	 echo "check with: make -f Makefile.r2 r2-verify MANIFEST=$$M"

r2-verify:        ## Check THIS testbed against a recorded manifest: MANIFEST=<path>
	@$(TILDE); test -f "$$M" || { echo "no such manifest: $$M"; exit 2; }; \
	 echo "manifest: $$(grep '^# digest' "$$M")"; \
	 echo "here:     # digest      $$(python3 testbed_provenance.py --digest)"; \
	 if command -v sha256sum >/dev/null; then CHECK="sha256sum -c -"; \
	 else CHECK="shasum -a 256 -c -"; fi; \
	 grep -v '^#' "$$M" | $$CHECK
	@# The `#` header lines are for a human reader; sha256sum has no comment syntax and
	@# reports them as "improperly formatted", which buries the real result, so they are
	@# stripped above rather than left to produce a warning next to every check.

r2-smoke: r2-up   ## Clean stack, then the R2 assertions
	$(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && cd /ws/testbed && python3 r2_smoke.py'

r2-latency: r2-up ## Clean stack, then cancel -> confirmed-rest, 6 runs
	$(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && cd /ws/testbed && \
	   python3 r2_cancel_latency.py --runs 6'

r2-costmap: r2-up ## Clean stack, then the costmap lethality probe
	$(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && python3 /ws/testbed/costmap_probe_tl.py'

r2-tf-soak: r2-up ## Clean stack, then sample /tf at t+30s, t+2m, t+5m
	@for t in 30 90 180; do \
	  sleep $$t; \
	  echo "--- /tf hz after $$t more seconds ---"; \
	  $(COMPOSE) $(R2PROF) exec -T r2-robot bash -lc \
	    'source /opt/ros/jazzy/setup.bash && timeout 8 ros2 topic hz /tf --window 50 2>&1 | tail -2'; \
	done
