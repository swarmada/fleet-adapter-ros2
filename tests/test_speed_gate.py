# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The speed printed "at halt" must be the speed at the halt.

`wait_for_cruise` returns the PEAK over its watch when it times out, and `r1_smoke`
printed that value as "speed at halt". On the MentorPi, which caps linear speed at
0.2 m/s under a 0.35 m/s floor, that was every run: the `speed at halt 0.202 m/s`
printed by the hardware runs of 2026-09-17 and 2026-09-18 are peaks. These tests pin the two values apart and the words that go
beside each.

The same fix applies to `r2_smoke`'s "speed at cancel", and "SPIN-UP" is taken out with it
-- a motion phase the harness never measured: on the MentorPi the robot was at its own capped
cruise, not accelerating.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

_TESTBED = pathlib.Path(__file__).resolve().parents[1] / "testbed"
if str(_TESTBED) not in sys.path:
    sys.path.insert(0, str(_TESTBED))

import speed_gate  # noqa: E402

r1_smoke = pytest.importorskip("r1_smoke")


class Clock:
    """Deterministic time for wait_for_cruise: sleep() advances monotonic()."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class ScriptedBinding:
    """`_speed` follows a function of the fake clock, as the binding's odom cache would."""

    def __init__(self, clock: Clock, v_of_t) -> None:
        self._clock, self._v = clock, v_of_t

    @property
    def _speed(self):
        return (self._v(self._clock.t), 0.0)


def mentorpi(t: float) -> float:
    """Accelerate, hold at the driver's clamp (odom reads 0.202 over it), then slow for
    the goal approach from t = 8 s. Below the floor throughout."""
    if t < 1.0:
        return 0.202 * t
    if t < 8.0:
        return 0.202
    return max(0.05, 0.202 - 0.04 * (t - 8.0))


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(speed_gate.time, "monotonic", c.monotonic)
    monkeypatch.setattr(speed_gate.time, "sleep", c.sleep)
    return c


# ── what wait_for_cruise returns ─────────────────────────────────────────────

def test_on_timeout_the_second_value_is_the_peak_not_the_speed_now(clock) -> None:
    b = ScriptedBinding(clock, mentorpi)
    reached, peak = speed_gate.wait_for_cruise(b, timeout=12.0)
    assert reached is False
    assert peak == pytest.approx(0.202)
    now = speed_gate.speed(b)
    assert now == pytest.approx(0.05, abs=1e-6)   # 12 s in: well into the approach
    assert peak != pytest.approx(now)             # the reported case: they differ


def test_on_reaching_cruise_the_second_value_is_the_crossing_sample(clock) -> None:
    b = ScriptedBinding(clock, lambda t: min(0.5, 1.0 * t))  # fake_robot_node --decel 1.0
    reached, v = speed_gate.wait_for_cruise(b, timeout=12.0)
    assert reached is True
    assert speed_gate.CRUISE_FLOOR <= v < speed_gate.CRUISE_FLOOR + 0.03


# ── the words beside the number ──────────────────────────────────────────────

def test_timeout_label_prints_the_instant_as_at_halt_and_names_the_peak() -> None:
    text = speed_gate.label(0.150, False, peak=0.202)
    assert text.startswith("speed at halt 0.150 m/s")
    assert "peak 0.202 m/s while waiting for cruise, which timed out" in text
    assert "LOWER BOUND" in text
    assert "at halt 0.202" not in text


def test_the_verdict_follows_the_printed_number_not_the_watch() -> None:
    # Crossed the floor during the watch, below it at the halt: never "(>= floor)".
    text = speed_gate.label(0.30, True)
    assert "(>=" not in text and "LOWER BOUND" in text and "had reached" in text
    # Watch timed out, but the instant is at or over the floor: the number decides.
    text = speed_gate.label(0.36, False, peak=0.349)
    assert "(>= 0.35 m/s cruise floor, peak 0.349" in text and "LOWER BOUND" not in text


def test_a_capped_cruise_is_not_called_spin_up(clock) -> None:
    """The MentorPi's driver clamps at 0.2 m/s: odom reads a flat 0.2000 for
    the whole approach to the halt. The watch times out, and the label must say what the
    harness knows -- below the floor, the instant, the peak -- and name no motion phase."""
    b = ScriptedBinding(clock, lambda t: min(0.200, 0.202 * t))
    reached, peak = speed_gate.wait_for_cruise(b, timeout=12.0)
    assert reached is False
    text = speed_gate.label(speed_gate.speed(b), reached, peak=peak)
    assert text == (
        "speed at halt 0.200 m/s, peak 0.200 m/s while waiting for cruise, which timed "
        "out -- below the 0.35 m/s cruise floor: this interval is a LOWER BOUND, not the "
        "e-stop bound")


@pytest.mark.parametrize("v,reached,peak", [
    (0.200, False, 0.200), (0.150, False, 0.202), (0.100, False, None),
    (0.300, True, None), (0.412, True, None), (0.36, False, 0.349),
])
def test_no_label_names_a_motion_phase_it_did_not_measure(v, reached, peak) -> None:
    text = speed_gate.label(v, reached, peak=peak).lower()
    for phase in ("spin-up", "spin up", "accelerat", "ramp", "capped"):
        assert phase not in text


def test_r1_and_r2_print_the_same_words_for_the_same_numbers() -> None:
    """R2's cancel and R1's halt are different paths, and the label says which;
    everything else on the line -- the instant, the named peak, the verdict -- is one
    convention, not two."""
    for v, reached, peak in [(0.200, False, 0.202), (0.412, True, 0.412), (0.30, True, 0.36)]:
        r1 = speed_gate.label(v, reached, "halt", peak=peak)
        r2 = speed_gate.label(v, reached, "cancel", peak=peak)
        assert r2 == r1.replace("halt", "cancel")
    assert speed_gate.label(0.150, False, "cancel", peak=0.202) == (
        "speed at cancel 0.150 m/s, peak 0.202 m/s while waiting for cruise, which timed "
        "out -- below the 0.35 m/s cruise floor: this interval is a LOWER BOUND, not the "
        "e-stop bound")


# ── both harnesses feed label() the instant, and the watch's peak as `peak` ──

def _label_wiring(path: pathlib.Path) -> list[tuple[str, str | None, str | None, str | None]]:
    """For each label() call: (its first argument, its peak= argument, what the first
    argument was assigned from). Read from the source, because r2_smoke cannot be
    imported without rclpy and R2 is not to be run here."""
    tree = ast.parse(path.read_text())
    peak_name = None
    assigned: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = ast.unparse(node.value.func)
            tgt = node.targets[0]
            if fn == "wait_for_cruise" and isinstance(tgt, ast.Tuple):
                peak_name = tgt.elts[1].id
            names = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
            for n in names:
                if isinstance(n, ast.Name):
                    assigned[n.id] = fn
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "label"]
    out = []
    for c in calls:
        first = ast.unparse(c.args[0])
        peak = next((ast.unparse(k.value) for k in c.keywords if k.arg == "peak"), None)
        out.append((first, peak, assigned.get(first), peak_name))
    return out


@pytest.mark.parametrize("harness,reader", [
    ("r1_smoke.py", "_observe_halt"),   # the halt's own odom message
    ("r2_smoke.py", "speed"),           # the latest odom before command_stop
])
def test_each_harness_prints_the_instant_and_names_the_peak(harness, reader) -> None:
    wiring = _label_wiring(_TESTBED / harness)
    assert wiring, f"{harness} prints no speed label"
    for first, peak, source, peak_name in wiring:
        assert peak_name is not None, f"{harness} no longer calls wait_for_cruise"
        assert first != peak_name, (
            f"{harness} prints wait_for_cruise's peak `{peak_name}` as the speed at the "
            f"moment, not the speed at it")
        assert peak == peak_name, f"{harness} does not pass the watch's peak as peak="
        assert source == reader, f"{harness} reads `{first}` from {source}(), not {reader}()"


# ── r1_smoke reads the halt speed from the halt's own odom message ───────────

class OdomBinding:
    """Odom messages, each (stamp_ns, linear speed). `arrive_on_read` delivers the next
    message after that many accessor reads -- a message landing mid-observation."""

    def __init__(self, msgs, arrive_on_read=None) -> None:
        self._msgs, self._i, self._reads, self._arrive = msgs, 0, 0, arrive_on_read

    def _tick(self) -> None:
        self._reads += 1
        if self._reads == self._arrive and self._i + 1 < len(self._msgs):
            self._i += 1

    def odom_stamp(self, robot):
        self._tick()
        return self._i + 1, self._msgs[self._i][0]

    def is_stopped(self, robot):
        self._tick()
        return self._msgs[self._i][1] < 0.01

    @property
    def _speed(self):
        self._tick()
        return (self._msgs[self._i][1], 0.0)


def test_observe_halt_pairs_the_bracket_edge_with_its_own_speed() -> None:
    b = OdomBinding([(1_000, 0.198)])
    (count, stamp, stopped), v = r1_smoke._observe_halt(b)
    assert (count, stamp, stopped, v) == (1, 1_000, False, 0.198)


def test_observe_halt_retries_when_a_message_lands_between_observation_and_speed() -> None:
    # Reads: odom_stamp(1) is_stopped(2) odom_stamp(3) _speed(4) -> message 2 lands on
    # the speed read. The pair would mix message 1's stamp with message 2's speed.
    b = OdomBinding([(1_000, 0.198), (34_000, 0.150)], arrive_on_read=4)
    (count, stamp, _), v = r1_smoke._observe_halt(b)
    assert (count, stamp, v) == (2, 34_000, 0.150)
