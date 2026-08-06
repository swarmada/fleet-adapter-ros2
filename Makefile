# ROS 2 / Nav2 Fleet Adapter — developer targets.
PYTHON ?= python3

.PHONY: install build lint test conformance

install:         ## Install with dev deps into the active venv (falls back to --break-system-packages)
	$(PYTHON) -m pip install -e '.[dev]' || \
	  $(PYTHON) -m pip install --break-system-packages -e '.[dev]'

build:           ## Build a wheel + sdist to dist/ (needs the `build` package: pip install build)
	$(PYTHON) -m build

lint:            ## Lint with ruff (dev dependency, declared in pyproject.toml)
	$(PYTHON) -m ruff check fleet_adapter_ros2/ tests/

test:            ## Run the safety-wiring tests
	pytest tests/ -v

conformance:     ## Drive this adapter against the Swarmada conformance harness.
	@echo "Point the published harness at:  $(PYTHON) -m fleet_adapter_ros2.adapter --endpoint localhost:{port}"
	@echo "See the Swarmada adapters/conformance/README.md for running the harness."
