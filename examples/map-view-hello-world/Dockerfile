# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0
#
# The map-view Hello World image: the map-view service and the simulated explore-floor source,
# from this repository's fleet_adapter_ros2/mapview. Standard library only: nothing is
# pip-installed. Built from the repository root:
#
#   docker build -f examples/map-view-hello-world/Dockerfile -t swarmada-mapview:dev .
#
# Dockerfile.dockerignore beside this file narrows the build context to the files copied here.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY fleet_adapter_ros2/__init__.py fleet_adapter_ros2/explore.py \
     fleet_adapter_ros2/explore_steps.py fleet_adapter_ros2/sim_explore.py fleet_adapter_ros2/
COPY fleet_adapter_ros2/mapview/ fleet_adapter_ros2/mapview/
# A user with no shell and no home; the manifests also run it with a read-only root.
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["python", "-m"]
CMD ["fleet_adapter_ros2.mapview.service", "--host", "0.0.0.0", "--port", "8080"]
