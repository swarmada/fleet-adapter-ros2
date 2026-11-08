# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""A map this adapter saves keeps unknown unknown: it declares ``free_thresh: 0.196``.

``map_saver_cli`` writes never-observed cells grey 205, which a reader takes as occupancy
(255 - 205) / 255 = 0.19608. Its default ``free_thresh`` is 0.25: above that line, so every
cell nothing observed reads as free floor, and the zone-proposal worker refuses the map. The
historical 0.196 sits just below it; 0.1961 is already above it. So the saver is told the
thresholds, and a saved map that declares anything else is refused before it is handed on.
"""

from __future__ import annotations

import pytest
from test_explore_job import FakeClock, _job
from test_ros2_explore import _saved_files, _serve_saved, _stack

from fleet_adapter_ros2 import explore
from fleet_adapter_ros2.sim_explore import SimulatedExploreStack


def test_the_declared_free_threshold_keeps_grey_205_unknown() -> None:
    assert explore.FREE_THRESH == 0.196
    assert explore.FREE_THRESH < (255 - 205) / 255 < 0.1961


@pytest.mark.parametrize("value, ok", [("0.196", True), ("0.19", True), ("0.1961", False),
                                       ("0.25", False), ("0.19608", False)])
def test_the_boundary_is_grey_205s_own_occupancy(value, ok) -> None:
    why = explore.map_yaml_refusal(f"image: map.pgm\nfree_thresh: {value}\n".encode())
    assert (why == "") is ok, why
    if not ok:
        assert "reads unknown (grey 205" in why


def test_a_map_that_declares_no_threshold_or_a_word_is_refused() -> None:
    assert "no free_thresh" in explore.map_yaml_refusal(b"image: map.pgm\n")
    assert "not a number" in explore.map_yaml_refusal(b"free_thresh: low\n")


def test_the_maps_the_adapter_renders_declare_the_threshold() -> None:
    import fleet_adapter_ros2.explore_steps as steps
    grid = steps.Grid(width=2, height=1, res=0.1, ox=0.0, oy=0.0, data=[-1, 0])
    yaml, _ = explore.render_map(grid)
    assert b"free_thresh: 0.196\n" in yaml and explore.map_yaml_refusal(yaml) == ""


def test_the_saver_on_the_robot_is_told_the_thresholds() -> None:
    st, ros, shell, *_ = _stack()
    ros.answers["/slam_toolbox/serialize_map"] = {"result": 0}
    _serve_saved(shell, _saved_files())
    st.save_map("lab/explore-first")
    saver = next(s for s in shell.scripts() if "map_saver_cli" in s)
    assert "--occ 0.65 --free 0.196 --ros-args" in saver


def test_a_segment_whose_saved_map_reads_unknown_as_free_ends_failed(tmp_path) -> None:
    stack = SimulatedExploreStack(clock=FakeClock())
    real_save = stack.save_map

    def saver_default(name):
        saved = real_save(name)
        saved.files["map.yaml"] = saved.files["map.yaml"].replace(b"free_thresh: 0.196",
                                                                  b"free_thresh: 0.25")
        return saved
    stack.save_map = saver_default
    job, stack, emits, _ = _job(tmp_path, stack=stack)
    job.run()
    state, message, _, _ = emits.last
    assert state == "FAILED" and "free_thresh 0.25" in message
    assert "map-ref=" not in message
