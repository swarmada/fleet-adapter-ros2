# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The adapter's version: one string, read by the hello, whose X.Y is the contract it implements.

The hello reported a hard-coded "0.1.0" while the package said the same thing in two other
places, so a release could change one and leave the others behind. Now the hello reads
`__version__`, pyproject.toml's version is checked against it, and its first two numbers are the
contract version the hello declares (ADR-0032); the third counts this adapter's own fixes.
"""

from __future__ import annotations

import pathlib
import re

import fleet_adapter_ros2
from fleet_adapter.v1 import fleet_adapter_pb2 as pb
from fleet_adapter_ros2 import adapter as adapter_mod
from fleet_adapter_ros2.adapter import fleet_adapter_ros2Adapter as Adapter
from fleet_adapter_ros2.robot import SimulatedRobot

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def _hello():
    return next(Adapter(pb, None, SimulatedRobot(), "amr-1")._control_requests()).hello


def _parts(v: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
    assert m, f"not X.Y.Z: {v!r}"
    return int(m[1]), int(m[2]), int(m[3])


def test_the_hello_reports_the_package_version() -> None:
    assert _hello().adapter_version == fleet_adapter_ros2.__version__


def test_the_versions_first_two_numbers_are_the_declared_contracts() -> None:
    hello = _hello()
    assert _parts(hello.adapter_version)[:2] == _parts(hello.contract_version)[:2]
    assert hello.contract_version == adapter_mod.CONTRACT_VERSION


def test_pyproject_declares_the_same_version() -> None:
    m = re.search(r'(?m)^version = "([^"]+)"$', PYPROJECT.read_text(encoding="utf-8"))
    assert m, "pyproject.toml has no version line"
    assert m[1] == fleet_adapter_ros2.__version__
