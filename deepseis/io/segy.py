"""
SEG-Y / .npy readers and writers.

This is the "bring your own data" path: point ``field_volume_path`` in the
config at a real SEG-Y (F3 Netherlands or otherwise) or ``.npy`` volume and
these functions handle it. It's implemented and unit-tested against a
locally-synthesized SEG-Y file (see ``tests/test_segy.py``) — there's no
dependency on actually having F3 downloaded to know this code works.

SEG-Y is read/written with ``segyio`` (the industry-standard Python SEG-Y
library), matching the tech stack in Sec. 7 of the spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio


def load_volume(path: str | Path, mmap: bool = False) -> np.ndarray:
    """Load a seismic volume from ``.npy``, ``.sgy``/``.segy``, or a raw FaultSeg3D-style ``.dat``.

    Returns a 2D array ``(n_samples, n_traces)`` for a single inline/section,
    or a 3D array ``(n_inlines, n_crosslines, n_samples)`` for a full SEG-Y
    survey (post-stack, regularly gridded).

    ``mmap=True`` memory-maps a ``.npy`` volume instead of reading it, which
    matters whenever only a few sections are needed. F3 is stored as float64,
    so a full read costs 573 MB before any float32 copy is made, while one
    inline read through a memory map costs 0.72 MB. On a 1 GB hosting tier the
    difference is the whole application.
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r") if mmap else np.load(path)
    if path.suffix.lower() in (".sgy", ".segy"):
        return _load_segy_volume(path)
    if path.suffix.lower() == ".dat":
        return load_faultseg3d_dat(path)
    raise ValueError(f"Unsupported seismic file extension: {path.suffix}")


def load_faultseg3d_dat(path: str | Path, dim: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Load a raw FaultSeg3D-style ``.dat`` volume (float32, C-order, then transposed).

    This matches the exact convention used by the original FaultSeg3D repo's
    own data generator (``np.fromfile(..., dtype=np.single).reshape(dim)``
    followed by ``np.transpose``): a ``data/samples/faultseg3d/`` pair
    (synthetic seismic + fault label, 128x128x128) is bundled with this repo
    as a genuine real-data smoke test — see ``tests/test_segy.py``.
    """
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size != np.prod(dim):
        raise ValueError(f"{path}: expected {np.prod(dim)} float32 values for dim={dim}, got {raw.size}")
    vol = raw.reshape(dim)
    return np.transpose(vol).astype(np.float32)


def _load_segy_volume(path: Path) -> np.ndarray:
    with segyio.open(str(path), "r", strict=False) as f:
        f.mmap()
        try:
            # post-stack 3D, regularly gridded -> fast path
            cube = segyio.tools.cube(f)  # (n_inlines, n_crosslines, n_samples)
            return np.asarray(cube, dtype=np.float32)
        except Exception:
            # fall back: treat as an unstructured collection of traces (a single 2D section)
            traces = np.stack([tr.astype(np.float32) for tr in f.trace], axis=1)  # (n_samples, n_traces)
            return traces


def load_inline(path: str | Path, inline_index: int) -> np.ndarray:
    """Pull a single 2D inline section (n_samples, n_crosslines) out of a SEG-Y survey."""
    path = Path(path)
    if path.suffix == ".npy":
        vol = np.load(path)
        if vol.ndim == 2:
            return vol
        return vol[inline_index].T  # (n_crosslines, n_samples) -> (n_samples, n_crosslines)
    with segyio.open(str(path), "r", strict=False) as f:
        f.mmap()
        inline_no = f.ilines[inline_index] if inline_index < len(f.ilines) else f.ilines[0]
        section = segyio.tools.collect(f.iline[inline_no])  # (n_crosslines, n_samples)
        return np.asarray(section, dtype=np.float32).T


def save_npy(path: str | Path, volume: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, volume.astype(np.float32))


def write_segy_like(out_path: str | Path, section: np.ndarray, template_path: str | Path | None,
                     dt_ms: float = 4.0) -> None:
    """Write a 2D section (n_samples, n_traces) to a new SEG-Y file.

    If ``template_path`` (an existing SEG-Y) is given, its geometry/headers
    are copied so the output "drops into" the same survey grid the input
    came from — this backs the spec's "export cleaned volume back to SEG-Y"
    stretch goal (Sec. 5) and the "drops into ONGC/Oil India's existing
    workflow" narrative (Sec. 11).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples, n_traces = section.shape

    if template_path is not None and Path(template_path).suffix.lower() in (".sgy", ".segy"):
        with segyio.open(str(template_path), "r", strict=False) as src:
            spec = segyio.tools.metadata(src)
            with segyio.create(str(out_path), spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header
                for i in range(min(n_traces, spec.tracecount)):
                    dst.trace[i] = section[:, i].astype(np.float32)
        return

    # no template -> synthesize a minimal, self-consistent SEG-Y from scratch
    spec = segyio.spec()
    spec.format = 5  # IEEE float32
    spec.samples = np.arange(n_samples) * dt_ms
    spec.ilines = [1]
    spec.xlines = np.arange(1, n_traces + 1)
    spec.tracecount = n_traces
    with segyio.create(str(out_path), spec) as dst:
        for i in range(n_traces):
            dst.trace[i] = section[:, i].astype(np.float32)
            dst.header[i] = {
                segyio.TraceField.INLINE_3D: 1,
                segyio.TraceField.CROSSLINE_3D: i + 1,
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(dt_ms * 1000),
            }
        dst.bin.update(tsort=segyio.TraceSortingFormat.CROSSLINE_SORTING)
