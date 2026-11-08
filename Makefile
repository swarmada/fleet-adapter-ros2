# ROS 2 / Nav2 Fleet Adapter — developer targets.
PYTHON ?= python3

.PHONY: install build lint check-error-strings test conformance

install:         ## Install with dev deps into the active venv (falls back to --break-system-packages)
	$(PYTHON) -m pip install -e '.[dev]' || \
	  $(PYTHON) -m pip install --break-system-packages -e '.[dev]'

build:           ## Build a wheel + sdist to dist/ (needs the `build` package: pip install build)
	$(PYTHON) -m build

lint:            ## Lint with ruff (dev dependency, declared in pyproject.toml)
	$(PYTHON) -m ruff check fleet_adapter_ros2/ tests/ testbed/ scripts/

check-error-strings: ## Check the public text: the trees AND this branch's commit messages
	$(PYTHON) scripts/check-error-strings.py fleet_adapter_ros2 tests testbed scripts fleet_adapter examples
	@# The history is public text too, and it is the half no tree walk reaches. A message
	@# is read by everyone who runs `git log` on the published branch and cannot be
	@# corrected after a push, so it is checked here rather than by hand.
	git log --format=%B main..HEAD | $(PYTHON) scripts/check-error-strings.py -

test:            ## Run the safety-wiring tests
	pytest tests/ -v

conformance:     ## Drive this adapter against the Swarmada conformance harness.
	@echo "Point the published harness at:  $(PYTHON) -m fleet_adapter_ros2.adapter --endpoint localhost:{port}"
	@echo "See the Swarmada adapters/conformance/README.md for running the harness."
