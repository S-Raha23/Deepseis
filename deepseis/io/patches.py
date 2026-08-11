"""Patch extraction / stitching.

Blind-spot training and 3D-style fault segmentation both work far better on
small patches than on a whole (potentially huge) seismic volume: it keeps
memory bounded, gives many training samples out of one section, and is what
lets everything here run on a single consumer GPU (or, for the demo, a CPU)
as the spec's Sec. 12 ("compute cost") promises.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def extract_patches(volume: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Extract overlapping square patches from a 2D array.

    Returns an array of shape ``(n_patches, size, size)``. Edge regions that
    don't divide evenly are covered by shifting the last patch back inside
    the bounds (no padding / no data loss).
    """
    h, w = volume.shape
    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    if not ys or ys[-1] != h - size:
        ys.append(max(h - size, 0))
    if not xs or xs[-1] != w - size:
        xs.append(max(w - size, 0))

    patches = []
    for y in ys:
        for x in xs:
            patches.append(volume[y:y + size, x:x + size])
    return np.stack(patches, axis=0)


def patch_coords(h: int, w: int, size: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    if not ys or ys[-1] != h - size:
        ys.append(max(h - size, 0))
    if not xs or xs[-1] != w - size:
        xs.append(max(w - size, 0))
    return [(y, x) for y in ys for x in xs]


def stitch_patches(patches: np.ndarray, h: int, w: int, size: int, stride: int) -> np.ndarray:
    """Inverse of :func:`extract_patches` — averages overlapping regions back together."""
    coords = patch_coords(h, w, size, stride)
    assert len(coords) == patches.shape[0], "patch count mismatch during stitching"

    out = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    # a mild cosine window reduces seam artefacts at patch boundaries
    win_1d = np.hanning(size) if size > 1 else np.ones(1)
    win_1d = np.clip(win_1d, 0.2, 1.0)
    window = np.outer(win_1d, win_1d)

    for (y, x), patch in zip(coords, patches):
        out[y:y + size, x:x + size] += patch * window
        weight[y:y + size, x:x + size] += window

    weight[weight == 0] = 1.0
    return (out / weight).astype(np.float32)


def batches(patches: np.ndarray, batch_size: int, shuffle: bool = True,
            rng: np.random.Generator | None = None) -> Iterator[np.ndarray]:
    """Yield mini-batches from an array of patches."""
    if rng is None:
        rng = np.random.default_rng(0)
    idx = np.arange(patches.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield patches[idx[i:i + batch_size]]
