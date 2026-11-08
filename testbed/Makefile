# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# R1/R2 testbed targets. Run from testbed/.
#
#   make build     build the image (once, ~5 min)
#   make deps      install the adapter + SDK inside the image's mounted tree
#   make r1        run R1 end to end and print PASS/FAIL
#   make r1-shell  interactive shell with ROS 2 sourced, for poking at topics
#   make r2-map    generate the blank warehouse map Nav2 needs
#   make r2        run R2 (Nav2) end to end
#   make r3-check  ON THE GPU BOX: is this hardware usable for Isaac Sim?
#   make r3-params R3: Nav2 params with use_sim_time=true (Isaac owns /clock)
#   make down      stop everything

COMPOSE ?= docker compose
# Recorded demos pass BUILD= to keep BuildKit's (fully cached) plan off camera, and repeat
# runs pass it to skip a 4.9 GB image export. Hardcoding `--build` in the targets below is
# what forced the 2026-09-02 R1/R2 session to invoke raw `docker compose` instead of the
# documented `make r1` / `make r2` -- so the runbook's own commands were not the commands
# that produced its numbers.
BUILD ?= --build
SHELL := /bin/bash

.PHONY: build deps r1 r1-shell r1-logs r2-map r2 r2-logs r3-params r3-check r3-vda5050 r3-vda5050-watch down clean topics

build:            ## Build the testbed image
	$(COMPOSE) build

deps:             ## Install the adapter + swarmada-sdk into the image (editable, from the mount)
	@echo "NOTE: SWARMADA_CORE must point at the swarmada checkout that holds sdk/python."
	$(COMPOSE) run --rm --no-deps \
	  -v "$${SWARMADA_CORE:?set SWARMADA_CORE=/path/to/swarmada}":/ws/core:ro \
	  fake-robot bash -lc '\
	    source /opt/ros/jazzy/setup.bash && \
	    rm -rf /tmp/sdk && cp -r /ws/core/sdk/python /tmp/sdk && \
	    pip install /tmp/sdk && \
	    pip install -e /ws/adapter && \
	    python3 -c "import fleet_adapter_ros2, rclpy, nav2_msgs.action; print(\"imports OK\")"'

r1:               ## R1: real ROS 2 + fake robot; asserts the four uncovered blocks
	$(COMPOSE) up $(BUILD) --abort-on-container-exit --exit-code-from r1-smoke fake-robot r1-smoke

r1-shell:         ## Interactive shell alongside a running fake robot
	$(COMPOSE) up -d fake-robot
	$(COMPOSE) run --rm --no-deps fake-robot bash -lc 'source /opt/ros/jazzy/setup.bash && exec bash'

r1-logs:
	$(COMPOSE) logs -f fake-robot

topics:           ## Sanity: list topics as seen from inside the DDS domain
	$(COMPOSE) run --rm --no-deps fake-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && ros2 topic list && echo "---" && ros2 action list'

r2-map:           ## Generate map/warehouse.{pgm,yaml} -- Nav2 refuses to start without a map
	$(COMPOSE) run --rm --no-deps fake-robot bash -lc \
	  'source /opt/ros/jazzy/setup.bash && python3 /ws/testbed/make_map.py --out /ws/testbed/map'

r2: r2-map        ## R2: real Nav2 against the fake robot
	# r2-sensors is NOT optional and NOT reachable through depends_on: it publishes the
	# TF chain (map->odom->base_link->laser) and the /scan the costmaps are built from.
	# Compose starts only the services named here plus their dependencies, and nothing
	# depends_on r2-sensors -- so omitting it brought Nav2 up with no TF and no sensor
	# data, which presents as r2_smoke assertion 2 failing with "check TF map->odom->
	# base_link is complete". Naming it here is the fix.
	$(COMPOSE) --profile r2 up $(BUILD) --abort-on-container-exit --exit-code-from r2-smoke \
	  r2-robot r2-sensors nav2 r2-smoke

r2-logs:
	$(COMPOSE) --profile r2 logs -f nav2

r3-params:        ## R3: nav2_params.r3.yaml with use_sim_time flipped to true (Isaac publishes /clock)
	@sed 's/use_sim_time: false/use_sim_time: true/g' nav2_params.yaml > nav2_params.r3.yaml
	@echo "wrote nav2_params.r3.yaml ($$(grep -c 'use_sim_time: true' nav2_params.r3.yaml) nodes on sim time)"

r3-check:         ## R3: is this GPU box usable for Isaac Sim? (run ON the rented instance)
	bash cloud_bootstrap.sh --check-only

r3-vda5050:       ## R3: MQTT broker + Isaac->VDA5050 shim + the real vda5050 adapter
	@echo "Isaac must already be running with --robot-count 3 (robot2 is the VDA5050 AMR)."
	$(COMPOSE) --profile r3 up mosquitto vda5050-shim vda5050-adapter

r3-vda5050-watch: ## Watch the raw VDA5050 traffic -- the first thing to check when it misbehaves
	$(COMPOSE) --profile r3 exec vda5050-shim bash -lc \
	  'mosquitto_sub -h localhost -t "uagv/v2/#" -v'

down:             ## Stop and remove everything
	-$(COMPOSE) --profile r2 --profile r3 down -v --remove-orphans

clean: down
	-docker image rm swarmada/ros2-testbed:jazzy
