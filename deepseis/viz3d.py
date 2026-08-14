"""
Geometry for the 3D view: seismic sections placed in survey coordinates.

Kept separate from the dashboard so the arithmetic that positions a section in
space can be tested without a browser. Getting a plane's orientation wrong
produces a picture that looks plausible and is mirrored or transposed, which is
exactly the class of error this project keeps finding, so the placement is
asserted in ``tests/test_viz3d.py`` rather than eyeballed.

Coordinate convention, matching how a seismic interpreter reads a survey:

* ``x`` is the crossline number,
* ``y`` is the inline number,
* ``z`` is **negated** sample index, so two-way time increases downwards and
  the display is not upside down.

One thing this module deliberately cannot do is denoise a time slice. A time
slice is an (inline x crossline) plane at fixed two-way time -- a completely
different geometry from the (time x trace) sections the denoiser was trained
on, with different statistics. Running the model on one would return something
plausible and meaningless, so ``timeslice_plane`` takes whatever array it is
given and the caller is responsible for labelling it honestly.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Plane:
    """A section positioned in survey coordinates, ready for a 3D surface.

    ``x``, ``y``, ``z`` and ``color`` all share a shape, which is what a
    Plotly ``Surface`` with an explicit ``surfacecolor`` requires. Without
    ``surfacecolor`` the surface is coloured by height and a seismic section
    renders as a depth ramp instead of an amplitude display.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.color.shape


def _decimate(n: int, step: int) -> np.ndarray:
    """Indices into an axis of length ``n``, always including the last sample.

    Dropping the final index would shrink the rendered cube slightly each time
    the decimation factor changed, so the endpoint is kept explicitly.
    """
    step = max(int(step), 1)
    idx = np.arange(0, n, step)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def inline_plane(section: np.ndarray, inline_index: int, decimate: int = 2) -> Plane:
    """An inline section (``n_samples, n_crosslines``) placed at ``y = inline_index``."""
    n_samples, n_crosslines = section.shape
    rows = _decimate(n_samples, decimate)
    cols = _decimate(n_crosslines, decimate)

    xx, zz = np.meshgrid(cols.astype(np.float32), -rows.astype(np.float32))
    return Plane(x=xx, y=np.full_like(xx, float(inline_index)), z=zz,
                 color=section[np.ix_(rows, cols)])


def crossline_plane(section: np.ndarray, crossline_index: int, decimate: int = 2) -> Plane:
    """A crossline section (``n_samples, n_inlines``) placed at ``x = crossline_index``.

    Crossline sections are legitimate input to the denoiser: the training
    sampler draws from both orientations, so this is in distribution rather
    than an extrapolation.
    """
    n_samples, n_inlines = section.shape
    rows = _decimate(n_samples, decimate)
    cols = _decimate(n_inlines, decimate)

    yy, zz = np.meshgrid(cols.astype(np.float32), -rows.astype(np.float32))
    return Plane(x=np.full_like(yy, float(crossline_index)), y=yy, z=zz,
                 color=section[np.ix_(rows, cols)])


def timeslice_plane(slice2d: np.ndarray, sample_index: int, decimate: int = 3) -> Plane:
    """A time slice (``n_inlines, n_crosslines``) placed at ``z = -sample_index``."""
    n_inlines, n_crosslines = slice2d.shape
    rows = _decimate(n_inlines, decimate)
    cols = _decimate(n_crosslines, decimate)

    xx, yy = np.meshgrid(cols.astype(np.float32), rows.astype(np.float32))
    return Plane(x=xx, y=yy, z=np.full_like(xx, -float(sample_index)),
                 color=slice2d[np.ix_(rows, cols)])


def horizon_polylines(tracks: np.ndarray, inline_index: int,
                      decimate: int = 2) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Tracked horizons on one inline, as 3D polylines.

    ``tracks`` is ``(n_horizons, n_traces)`` of sample indices with NaN where
    tracking was lost -- see ``interpretation.horizon.track_horizons``. NaNs are
    preserved rather than dropped or interpolated: Plotly breaks a line at NaN,
    so a horizon that was genuinely lost across a fault shows as a gap instead
    of being bridged by a straight segment that no data supports.
    """
    if tracks.ndim != 2:
        raise ValueError(f"expected (n_horizons, n_traces), got shape {tracks.shape}")

    n_traces = tracks.shape[1]
    cols = _decimate(n_traces, decimate)

    out = []
    for track in tracks:
        picks = track[cols]
        out.append((cols.astype(np.float32),
                    np.full(cols.size, float(inline_index), dtype=np.float32),
                    -picks.astype(np.float32)))
    return out


def symmetric_limits(*arrays: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    """Colour limits centred on zero.

    Seismic amplitude is signed and its colour maps are diverging, so limits
    have to be symmetric or the zero crossing shifts off the middle colour and
    the polarity reads wrong. A high percentile rather than the maximum keeps a
    few bright samples from flattening everything else to grey.
    """
    # Filtered *after* masking, not before: an array that is entirely NaN
    # survives the `a.size` check but is empty once the finite mask is applied,
    # and np.percentile raises on that rather than returning NaN.
    finite = [np.abs(a[np.isfinite(a)]) for a in arrays if a is not None and a.size]
    finite = [f for f in finite if f.size]
    if not finite:
        return -1.0, 1.0
    limit = float(np.percentile(np.concatenate([f.ravel() for f in finite]), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def scene_aspect(n_crosslines: int, n_inlines: int, n_samples: int,
                 vertical_exaggeration: float = 1.6) -> dict:
    """Axis ratios for the 3D scene.

    Seismic cubes are far wider than they are deep, so a true-scale cube
    renders as a thin sheet with no visible structure. A vertical exaggeration
    is standard practice in interpretation software for exactly this reason.
    """
    longest = float(max(n_crosslines, n_inlines, 1))
    return {
        "x": n_crosslines / longest,
        "y": n_inlines / longest,
        "z": max(n_samples / longest, 0.05) * vertical_exaggeration,
    }


# ---------------------------------------------------------------------------
# Cuboid assembly
# ---------------------------------------------------------------------------

def offset_plane(plane: Plane, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Plane:
    """Translate a plane without touching its amplitudes.

    This is what makes an exploded view free: pulling the shell of the cuboid
    apart is a change of position, not of data, so no section has to be
    denoised again when the separation slider moves.
    """
    return Plane(x=plane.x + dx, y=plane.y + dy, z=plane.z + dz, color=plane.color)


def apply_notch(plane: Plane, row_from: float = 0.5, col_from: float = 0.5) -> Plane:
    """Blank out one corner of a face, so the interior can be seen through it.

    This is the "chair" display every interpretation package offers: a solid
    block with a corner removed, which shows interior slices against the outer
    faces instead of making the viewer choose between the two.

    The hole is made by writing NaN into ``z`` rather than into ``color``.
    Plotly omits a surface cell whose vertex is NaN, whereas a NaN colour is
    merely an undefined colour on a cell that still gets drawn -- the
    difference between a hole and a grey patch.
    """
    n_rows, n_cols = plane.color.shape
    rows = np.arange(n_rows)[:, None] >= row_from * n_rows
    cols = np.arange(n_cols)[None, :] >= col_from * n_cols
    cut = rows & cols

    z = plane.z.astype(np.float32).copy()
    color = plane.color.astype(np.float32).copy()
    z[cut] = np.nan
    color[cut] = np.nan
    return Plane(x=plane.x, y=plane.y, z=z, color=color)


def shell_offsets(separation: float, n_crosslines: int, n_inlines: int,
                  n_samples: int, reach: float = 0.55) -> dict:
    """How far each face of the shell slides out, for a given separation.

    ``separation`` runs 0 (closed cuboid) to 1 (fully exploded). Each face
    moves along its own outward normal and scaled by that axis's own extent,
    so the block opens evenly instead of one face flying off while another
    barely moves.
    """
    s = float(np.clip(separation, 0.0, 1.0)) * reach
    return {
        "inline_near": (0.0, -s * n_inlines, 0.0),
        "inline_far": (0.0, +s * n_inlines, 0.0),
        "crossline_near": (-s * n_crosslines, 0.0, 0.0),
        "crossline_far": (+s * n_crosslines, 0.0, 0.0),
        "time_top": (0.0, 0.0, +s * n_samples),
        "time_base": (0.0, 0.0, -s * n_samples),
    }
