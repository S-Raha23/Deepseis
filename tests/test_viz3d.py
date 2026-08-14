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
