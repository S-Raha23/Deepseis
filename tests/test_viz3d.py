"""
Tests for 3D placement.

A mirrored or transposed plane renders as a perfectly plausible seismic cube --
it just shows the survey backwards. That is the same failure mode as everything
else this project has caught: no exception, no visual glitch, wrong answer. So
each plane's position and orientation is asserted against the section it came
from rather than checked by eye.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis.viz3d import (Plane, crossline_plane, horizon_polylines, inline_plane,
                            scene_aspect, symmetric_limits, timeslice_plane)


def ramp_section(n_samples=40, n_traces=60):
    """A section whose value encodes its own (sample, trace) position."""
    s = np.arange(n_samples)[:, None] * 1000
    t = np.arange(n_traces)[None, :]
    return (s + t).astype(np.float32)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def test_inline_plane_sits_at_its_inline_and_spans_crosslines():
    sec = ramp_section(40, 60)
    p = inline_plane(sec, inline_index=137, decimate=1)

    assert np.all(p.y == 137), "inline plane is not at its own inline"
    assert p.x.min() == 0 and p.x.max() == 59, "does not span the crossline axis"
    assert p.z.max() == 0 and p.z.min() == -39, "does not span the full time axis"
    assert p.shape == sec.shape


def test_crossline_plane_sits_at_its_crossline_and_spans_inlines():
    sec = ramp_section(40, 25)          # (n_samples, n_inlines)
    p = crossline_plane(sec, crossline_index=402, decimate=1)

    assert np.all(p.x == 402), "crossline plane is not at its own crossline"
    assert p.y.min() == 0 and p.y.max() == 24, "does not span the inline axis"
    assert p.z.max() == 0 and p.z.min() == -39


def test_timeslice_plane_is_flat_at_its_sample():
    sl = np.zeros((30, 50), dtype=np.float32)
    p = timeslice_plane(sl, sample_index=88, decimate=1)

    assert np.all(p.z == -88), "time slice is not flat at its own sample"
    assert p.x.min() == 0 and p.x.max() == 49, "x should be crossline"
    assert p.y.min() == 0 and p.y.max() == 29, "y should be inline"


def test_time_increases_downwards():
    """z is negated, so a deeper sample must plot lower. Getting this wrong
    renders the whole survey upside down and still looks like seismic."""
    sec = ramp_section(40, 20)
    p = inline_plane(sec, 0, decimate=1)
    shallow = p.z[0, 0]
    deep = p.z[-1, 0]
    assert deep < shallow, "deeper samples are not lower in the scene"


def test_colour_is_not_transposed_relative_to_position():
    """The colour at a mesh node must be the amplitude at that node's own
    (sample, trace). A transpose here mirrors the section silently."""
    sec = ramp_section(40, 60)
    p = inline_plane(sec, 5, decimate=1)

    for r, c in [(0, 0), (7, 13), (39, 59), (20, 3)]:
        sample = int(-p.z[r, c])
        trace = int(p.x[r, c])
        assert p.color[r, c] == sec[sample, trace], \
            f"colour at mesh node {(r, c)} does not match the section it came from"


def test_crossline_colour_is_not_transposed():
    sec = ramp_section(40, 25)
    p = crossline_plane(sec, 11, decimate=1)
    for r, c in [(0, 0), (9, 17), (39, 24)]:
        sample = int(-p.z[r, c])
        inline = int(p.y[r, c])
        assert p.color[r, c] == sec[sample, inline]


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step", [1, 2, 3, 7])
def test_decimation_shrinks_the_mesh_but_keeps_the_full_extent(step):
    """Rendering 178k points per plane is what makes a browser stall, but the
    cube must not shrink each time the factor changes."""
    sec = ramp_section(64, 100)
    p = inline_plane(sec, 0, decimate=step)

    assert p.x.min() == 0 and p.x.max() == 99, "lost the end of the crossline axis"
    assert p.z.min() == -63, "lost the bottom of the section"
    if step > 1:
        assert p.color.size < sec.size


def test_all_mesh_arrays_share_one_shape():
    """Plotly silently misrenders if surfacecolor does not match the grid."""
    sec = ramp_section(55, 83)
    for p in [inline_plane(sec, 3, 2), crossline_plane(sec, 4, 2),
              timeslice_plane(np.zeros((20, 30), np.float32), 5, 2)]:
        assert p.x.shape == p.y.shape == p.z.shape == p.color.shape


# ---------------------------------------------------------------------------
# Horizons
# ---------------------------------------------------------------------------

def test_horizon_polylines_follow_their_picks():
    tracks = np.array([[10.0, 11.0, 12.0, 13.0],
                       [30.0, 30.0, 29.0, 28.0]])
    lines = horizon_polylines(tracks, inline_index=7, decimate=1)

    assert len(lines) == 2
    for (x, y, z), track in zip(lines, tracks):
        assert np.all(y == 7)
        assert np.array_equal(x, np.arange(4))
        assert np.array_equal(z, -track)


def test_lost_horizon_picks_stay_as_gaps():
    """track_horizons returns NaN where tracking was lost, typically across a
    fault. Plotly breaks a line at NaN, so dropping or interpolating them would
    draw a straight segment across a fault that no pick supports."""
    tracks = np.array([[10.0, np.nan, 12.0, 13.0]])
    (x, y, z), = horizon_polylines(tracks, 0, decimate=1)
    assert np.isnan(z[1]), "a lost pick was filled in rather than left as a gap"
    assert x.size == 4, "the trace axis was shortened to hide the gap"


def test_horizon_polylines_reject_a_wrong_shape():
    with pytest.raises(ValueError):
        horizon_polylines(np.zeros(10), 0)


# ---------------------------------------------------------------------------
# Colour scale and aspect
# ---------------------------------------------------------------------------

def test_colour_limits_are_symmetric_about_zero():
    """Seismic amplitude is signed on a diverging colour map. Asymmetric limits
    move the zero crossing off the middle colour and the polarity reads wrong."""
    rng = np.random.default_rng(0)
    lo, hi = symmetric_limits(rng.standard_normal((50, 50)) + 0.4)
    assert lo == -hi and hi > 0


def test_colour_limits_ignore_a_few_bright_samples():
    a = np.concatenate([np.full(9000, 0.5), np.full(10, 500.0)])
    lo, hi = symmetric_limits(a, percentile=99.0)
    assert hi < 10.0, "a handful of spikes flattened the whole display"


def test_colour_limits_survive_all_nan_input():
    lo, hi = symmetric_limits(np.full(10, np.nan))
    assert np.isfinite(lo) and np.isfinite(hi) and lo == -hi


def test_scene_aspect_exaggerates_the_vertical():
    """A true-scale seismic cube is a thin sheet with no visible structure."""
    a = scene_aspect(701, 401, 255, vertical_exaggeration=1.6)
    assert a["x"] == pytest.approx(1.0)
    assert a["y"] == pytest.approx(401 / 701)
    true_scale = 255 / 701
    assert a["z"] > true_scale, "no vertical exaggeration applied"


# ---------------------------------------------------------------------------
# Cuboid assembly
# ---------------------------------------------------------------------------

def test_offset_moves_position_without_touching_amplitudes():
    """Separation must be a change of position only -- otherwise pulling the
    cuboid apart would mean re-denoising every face on each slider move."""
    from deepseis.viz3d import offset_plane

    sec = ramp_section(20, 30)
    p = inline_plane(sec, 5, decimate=1)
    q = offset_plane(p, dx=3.0, dy=-7.0, dz=2.0)

    assert np.array_equal(q.color, p.color), "amplitudes changed under a translation"
    assert np.allclose(q.x, p.x + 3.0)
    assert np.allclose(q.y, p.y - 7.0)
    assert np.allclose(q.z, p.z + 2.0)


def test_notch_removes_a_corner_as_a_hole_not_a_grey_patch():
    """Plotly omits a cell whose vertex is NaN but still draws one whose colour
    is NaN, so the hole has to be cut in z."""
    from deepseis.viz3d import apply_notch

    sec = ramp_section(20, 40)
    p = inline_plane(sec, 0, decimate=1)
    n = apply_notch(p, row_from=0.5, col_from=0.5)

    assert np.isnan(n.z).any(), "no hole was cut in the geometry"
    assert np.isnan(n.z).sum() == pytest.approx(0.25 * n.z.size, rel=0.15), \
        "the notch is not roughly a quarter of the face"
    # the untouched corner must survive intact
    assert not np.isnan(n.z[0, 0])
    assert n.color[0, 0] == p.color[0, 0]


def test_notch_does_not_mutate_the_plane_it_was_given():
    from deepseis.viz3d import apply_notch

    p = inline_plane(ramp_section(16, 16), 0, decimate=1)
    before = p.z.copy()
    apply_notch(p, 0.5, 0.5)
    assert np.array_equal(p.z, before), "apply_notch modified its input in place"


def test_closed_cuboid_has_zero_separation():
    from deepseis.viz3d import shell_offsets

    off = shell_offsets(0.0, 701, 401, 255)
    for name, (dx, dy, dz) in off.items():
        assert (dx, dy, dz) == (0.0, 0.0, 0.0), f"{name} moved at separation 0"


def test_faces_move_outward_along_their_own_normals():
    """Each face slides along its own axis, or the block opens lopsidedly."""
    from deepseis.viz3d import shell_offsets

    off = shell_offsets(1.0, 701, 401, 255)
    assert off["inline_near"][1] < 0 < off["inline_far"][1], "inline faces do not oppose"
    assert off["crossline_near"][0] < 0 < off["crossline_far"][0], "crossline faces do not oppose"
    assert off["time_base"][2] < 0 < off["time_top"][2], "time faces do not oppose"
    # and each moves only along its own axis
    assert off["inline_near"][0] == 0 and off["inline_near"][2] == 0
    assert off["crossline_far"][1] == 0 and off["crossline_far"][2] == 0
    assert off["time_top"][0] == 0 and off["time_top"][1] == 0


def test_separation_scales_each_axis_by_its_own_extent():
    """A fixed offset in survey units would fling the short axis far and barely
    move the long one."""
    from deepseis.viz3d import shell_offsets

    off = shell_offsets(1.0, n_crosslines=700, n_inlines=350, n_samples=250)
    assert off["crossline_far"][0] == pytest.approx(2 * off["inline_far"][1], rel=1e-6)


def test_separation_is_monotone_and_clamped():
    from deepseis.viz3d import shell_offsets

    a = shell_offsets(0.25, 700, 400, 250)["inline_far"][1]
    b = shell_offsets(0.75, 700, 400, 250)["inline_far"][1]
    assert 0 < a < b
    # out-of-range input must not fling the faces to infinity
    assert shell_offsets(5.0, 700, 400, 250) == shell_offsets(1.0, 700, 400, 250)
    assert shell_offsets(-3.0, 700, 400, 250) == shell_offsets(0.0, 700, 400, 250)
