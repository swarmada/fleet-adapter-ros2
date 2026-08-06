# Copyright 2026 The Swarmada Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scenario preset loader and a pure, proto-free scenario engine.

The engine computes what a simulated robot's hardware looks like at an elapsed
time ``t`` (seconds since telemetry start), applying the preset's fault timeline.
It is deliberately proto-free so it is unit-testable on its own and reusable by
every simulated backend; the adapter maps ``HardwareState`` onto the proto
``HardwareStatusUpdate`` / ``HardwareComponent`` fields.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

_PRESETS_DIR = pathlib.Path(__file__).resolve().parent / "presets"

# Hardware status tokens mirror the proto ``HardwareStatus`` enum names, kept as
# plain strings here so this module never imports the generated proto.
HW_HEALTHY = "HEALTHY"
HW_DEGRADED = "DEGRADED"
HW_FAILED = "FAILED"
_VALID_STATUS = frozenset({HW_HEALTHY, HW_DEGRADED, HW_FAILED})


@dataclass
class HardwareSpec:
    """A declared hardware component in the manifest."""

    name: str
    type: str = ""
    model: str = ""


@dataclass
class InstalledModelSpec:
    """A declared installed model in the manifest."""

    name: str
    version: str = ""


@dataclass
class FaultEvent:
    """A single transition on the fault timeline: at ``at_seconds`` the named
    component takes ``status`` (with an optional human-readable ``reason``)."""

    component: str
    at_seconds: float
    status: str
    reason: str = ""
    affects_capabilities: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    """A fully-parsed scenario preset. Covers every plan knob; a given simulated
    backend consumes the subset it supports."""

    name: str
    description: str = ""
    robot_id_prefix: str = "robot-sim"
    fleet_size: int = 1
    capabilities: list[str] = field(default_factory=list)
    hardware: list[HardwareSpec] = field(default_factory=list)
    installed_models: list[InstalledModelSpec] = field(default_factory=list)
    battery: dict = field(default_factory=dict)
    position_pattern: str = "static"
    fault_timeline: list[FaultEvent] = field(default_factory=list)
    comms: dict = field(default_factory=dict)
    estop: dict = field(default_factory=dict)


@dataclass
class HardwareFaultOverrides:
    """CLI overrides for the ``hardware-fault`` preset (``--fault-component`` /
    ``--fault-at`` / ``--recover-at``). ``None`` leaves the preset value."""

    component: str | None = None
    fault_at: float | None = None
    recover_at: float | None = None


@dataclass
class HardwareState:
    """The computed state of one component at a given time (proto-free)."""

    name: str
    type: str
    model: str
    status: str
    reason: str = ""


def _parse(doc: dict) -> Scenario:
    if not isinstance(doc, dict) or "name" not in doc:
        raise ValueError("scenario preset must be a mapping with a 'name' field")
    fleet = doc.get("fleet") or {}
    manifest = doc.get("manifest") or {}

    hardware = [
        HardwareSpec(name=h["name"], type=h.get("type", ""), model=h.get("model", ""))
        for h in (manifest.get("hardware") or [])
    ]
    models = [
        InstalledModelSpec(name=m["name"], version=m.get("version", ""))
        for m in (manifest.get("installed_models") or [])
    ]

    timeline: list[FaultEvent] = []
    for e in doc.get("fault_timeline") or []:
        status = str(e["status"]).upper()
        if status not in _VALID_STATUS:
            raise ValueError(
                f"fault_timeline status {e['status']!r} must be one of {sorted(_VALID_STATUS)}"
            )
        timeline.append(
            FaultEvent(
                component=e["component"],
                at_seconds=float(e["at_seconds"]),
                status=status,
                reason=e.get("reason", ""),
                affects_capabilities=list(e.get("affects_capabilities") or []),
            )
        )
    timeline.sort(key=lambda ev: ev.at_seconds)

    return Scenario(
        name=doc["name"],
        description=doc.get("description", ""),
        robot_id_prefix=fleet.get("robot_id_prefix", "robot-sim"),
        fleet_size=int(fleet.get("size", 1)),
        capabilities=list(manifest.get("capabilities") or []),
        hardware=hardware,
        installed_models=models,
        battery=doc.get("battery") or {},
        position_pattern=(doc.get("position") or {}).get("pattern", "static"),
        fault_timeline=timeline,
        comms=doc.get("comms") or {},
        estop=doc.get("estop") or {},
    )


def _apply_overrides(scenario: Scenario, overrides: HardwareFaultOverrides) -> None:
    """Retarget/retime a hardware-fault timeline from CLI overrides. Applies to
    the degrade→recover shape: the non-HEALTHY event is the fault, a HEALTHY event
    the recovery."""
    if not scenario.fault_timeline:
        return
    if overrides.component is not None:
        names = {h.name for h in scenario.hardware}
        if overrides.component not in names:
            raise ValueError(
                f"fault component {overrides.component!r} is not in the scenario "
                f"hardware manifest {sorted(names)}"
            )
        for e in scenario.fault_timeline:
            e.component = overrides.component
    for e in scenario.fault_timeline:
        if e.status != HW_HEALTHY and overrides.fault_at is not None:
            e.at_seconds = float(overrides.fault_at)
        if e.status == HW_HEALTHY and overrides.recover_at is not None:
            e.at_seconds = float(overrides.recover_at)
    scenario.fault_timeline.sort(key=lambda ev: ev.at_seconds)


def available_presets() -> list[str]:
    """Names of the bundled presets (files in ``presets/``)."""
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.yaml"))


def load_scenario(
    name_or_path: str, overrides: HardwareFaultOverrides | None = None
) -> Scenario:
    """Load a preset by name (from ``presets/``) or an explicit path, applying any
    hardware-fault overrides. Raises ``ValueError`` for an unknown name."""
    path = pathlib.Path(name_or_path)
    if not path.suffix:
        path = _PRESETS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise ValueError(
            f"unknown scenario {name_or_path!r}; available presets: {available_presets()}"
        )
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    scenario = _parse(doc)
    if overrides is not None:
        _apply_overrides(scenario, overrides)
    return scenario


class ScenarioEngine:
    """Pure evaluator over a :class:`Scenario`. Proto-free and deterministic."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def robot_ids(self, size: int | None = None) -> list[str]:
        """The fleet's robot IDs: ``<prefix>-1 .. <prefix>-N``."""
        n = self.scenario.fleet_size if size is None else int(size)
        n = max(1, n)
        return [f"{self.scenario.robot_id_prefix}-{i}" for i in range(1, n + 1)]

    def hardware_at(self, t: float) -> list[HardwareState]:
        """Every component's state at elapsed time ``t`` (seconds). Base is all
        HEALTHY; each timeline event with ``at_seconds <= t`` is applied in order,
        so the latest matching event per component wins."""
        state: dict[str, HardwareState] = {
            h.name: HardwareState(h.name, h.type, h.model, HW_HEALTHY, "")
            for h in self.scenario.hardware
        }
        for e in self.scenario.fault_timeline:  # ascending by at_seconds
            if e.at_seconds <= t and e.component in state:
                cur = state[e.component]
                state[e.component] = HardwareState(
                    cur.name, cur.type, cur.model, e.status, e.reason
                )
        return list(state.values())

    def hardware_delta(
        self, t: float, prev: dict[str, str] | None
    ) -> list[HardwareState]:
        """Components whose status differs from ``prev`` (name→status). With an
        empty/None ``prev`` (first payload after (re)connect) every component is
        returned — matching the proto's delta-compression contract."""
        current = self.hardware_at(t)
        if not prev:
            return current
        return [h for h in current if prev.get(h.name) != h.status]

    # ── battery-edge ────────────────────────────────────────────────────────────

    def battery_at(self, t: float) -> int | None:
        """Battery percent at elapsed ``t`` from the preset's curve
        (``start_percent - drain_per_second * t``, clamped at ``floor_percent``).
        Returns ``None`` when the preset declares no curve, so the adapter keeps the
        simulator's own battery. Drives ``TelemetryPayload.battery.percent``."""
        b = self.scenario.battery
        if not b or "start_percent" not in b:
            return None
        start = float(b.get("start_percent", 100))
        drain = float(b.get("drain_per_second", 0.0))
        floor = float(b.get("floor_percent", 0))
        return int(max(floor, start - drain * max(0.0, t)))

    # ── comms-flaky ─────────────────────────────────────────────────────────────

    def telemetry_gap_at(self, t: float) -> bool:
        """Whether elapsed ``t`` falls inside a telemetry outage: an explicit comms
        gap (``comms.gaps[].{start_seconds,end_seconds}``) or an active stream drop
        (``comms.drop`` — the stream is down, so no telemetry). Drives whether the
        adapter withholds its ``TelemetryPayload``."""
        if self.stream_down_at(t):
            return True
        for g in self.scenario.comms.get("gaps") or []:
            if float(g["start_seconds"]) <= t < float(g["end_seconds"]):
                return True
        return False

    def stream_drop_window(self) -> tuple[float, float] | None:
        """The bidirectional ControlStream drop window ``[at_seconds, reconnect_seconds)``
        from ``comms.drop``, or ``None`` when the preset declares no drop. A drop (unlike
        a gap) tears the whole stream down, so on reconnect the adapter re-runs the
        Hello/Register handshake and the control plane's lease renewals stop during the
        outage — exercising the RegisterRobot reconnect, fencing-token staleness (C3),
        and lease self-stop (C4) paths."""
        d = self.scenario.comms.get("drop")
        if not d:
            return None
        return (float(d["at_seconds"]), float(d["reconnect_seconds"]))

    def stream_down_at(self, t: float) -> bool:
        """Whether the ControlStream is dropped at elapsed ``t``."""
        win = self.stream_drop_window()
        return win is not None and win[0] <= t < win[1]

    # ── estop-drill ─────────────────────────────────────────────────────────────

    def estop_due(self, t: float) -> bool:
        """Whether a timer-driven estop drill is due at elapsed ``t``
        (``estop.trigger == at_seconds`` and ``t >= estop.at_seconds``). Command-mode
        drills (``trigger == on_command``) are never time-due. The adapter confirms the
        stop through ``safety.confirm_estop`` (ground truth, never faked)."""
        e = self.scenario.estop
        if e.get("trigger") == "at_seconds":
            return t >= float(e.get("at_seconds", 0))
        return False
