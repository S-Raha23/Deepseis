"""
Volume-level data access with honest train/validation/test separation.

The previous pipeline trained on a single section -- ``volume[0].T``, one
inline out of 401 -- and then served that model across the whole survey, with
the facies head additionally fitted and scored on the same line. There was no
held-out data anywhere in the project, so no number it reported could
distinguish fitting from generalising.

Two things are fixed here.

**Many sections, not one.** Patches are drawn from inlines *and* crosslines
across the training block, which is roughly two orders of magnitude more
independent structure than a single line offers, and is what lets a model
with real capacity be fitted without simply memorising one image.

**Splits that are actually split.** Adjacent inlines in a 3D survey are very
nearly the same picture -- measured on F3, inline 200 correlates with inline
201 at 0.93. A random per-section split would therefore put near-duplicates of
the training data into the validation set and report a generalisation number
that is nothing of the sort. Splits are contiguous blocks separated by
discarded buffer zones wide enough for the correlation to decay (0.16 at 20
lines, 0.01 at 60; a 30-line buffer is used). Crossline sections cut across
*every* inline, so the ones used for training are truncated to the training
inline range -- otherwise each of them would carry a stripe of the test block.

Normalisation is a single global scalar computed on the training block only
and stored with the checkpoint, applied identically at train and serve time.
The old code normalised each served section by its own standard deviation
while training on an unnormalised volume, a 4.4x mismatch. Per-section
normalisation is a bad idea even when it is applied consistently: it makes the
denoised amplitude of a section depend on what else happens to be in it,
which breaks the amplitude comparisons interpreters make between lines.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Splits:
    """Contiguous inline blocks, separated by discarded buffers."""
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    buffer: int

    def as_dict(self) -> dict:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test),
                "buffer": self.buffer}

    def contains(self, split: str, index: int) -> bool:
        lo, hi = getattr(self, split)
        return lo <= index < hi


def make_splits(n_inlines: int, buffer: int = 30, val_frac: float = 0.09,
                test_frac: float = 0.07) -> Splits:
    """Lay out train / buffer / val / buffer / test along the inline axis."""
    n_val = max(int(round(n_inlines * val_frac)), 1)
    n_test = max(int(round(n_inlines * test_frac)), 1)
    n_train = n_inlines - n_val - n_test - 2 * buffer
    if n_train < 1:
        raise ValueError(f"volume has too few inlines ({n_inlines}) for buffer={buffer}")

    train = (0, n_train)
    val_start = n_train + buffer
    val = (val_start, val_start + n_val)
    test_start = val[1] + buffer
    test = (test_start, min(test_start + n_test, n_inlines))
    return Splits(train=train, val=val, test=test, buffer=buffer)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

class SeismicVolume:
    """Memory-mapped access to a 3D volume, in ``(n_samples, n_traces)`` sections.

    The on-disk convention is ``(n_inlines, n_crosslines, n_samples)``; every
    section this class hands out is transposed to ``(n_samples, n_traces)``,
    i.e. time/depth downwards, which is what the models and the dashboard use.
    """

    def __init__(self, path: str | Path, buffer: int = 30, norm_sections: int = 40) -> None:
        self.path = Path(path)
        self.volume = np.load(self.path, mmap_mode="r")
        if self.volume.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {self.volume.shape}")
        self.n_inlines, self.n_crosslines, self.n_samples = self.volume.shape
        self.splits = make_splits(self.n_inlines, buffer=buffer)
        self.scale, self.offset = self._compute_normalization(norm_sections)

    # -- normalisation ------------------------------------------------------

    def _compute_normalization(self, n_sections: int) -> tuple[float, float]:
        """One global (scale, offset) from the training block only."""
        lo, hi = self.splits.train
        idx = np.linspace(lo, hi - 1, min(n_sections, hi - lo)).astype(int)
        sample = np.asarray(self.volume[idx], dtype=np.float64)
        offset = float(sample.mean())
        scale = float(sample.std())
        if scale < 1e-12:
            raise ValueError("training block has near-zero variance -- check the input volume")
        return scale, offset

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.offset) / self.scale).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x * self.scale + self.offset).astype(np.float32)

    # -- section access -----------------------------------------------------

    def inline(self, index: int, normalized: bool = True) -> np.ndarray:
        """Inline section, ``(n_samples, n_crosslines)``."""
        sec = np.asarray(self.volume[index], dtype=np.float32).T
        return self.normalize(sec) if normalized else sec

    def crossline(self, index: int, inline_range: tuple[int, int] | None = None,
                  normalized: bool = True) -> np.ndarray:
        """Crossline section, ``(n_samples, n_inlines_in_range)``."""
        lo, hi = inline_range if inline_range is not None else (0, self.n_inlines)
        sec = np.asarray(self.volume[lo:hi, index], dtype=np.float32).T
        return self.normalize(sec) if normalized else sec

    def inline_stack(self, index: int, n_neighbors: int, normalized: bool = True) -> np.ndarray:
        """``(2*n_neighbors+1, n_samples, n_crosslines)`` -- the line and its neighbours.

        Neighbours are clamped at the volume edge. Whether feeding them helps
        depends on whether the noise is independent between adjacent lines,
        which is an empirical question for any given survey (migration
        correlates noise laterally); ``evaluate.py`` measures it rather than
        assuming it.
        """
        if n_neighbors == 0:
            return self.inline(index, normalized)[None]
        idx = np.clip(np.arange(index - n_neighbors, index + n_neighbors + 1), 0, self.n_inlines - 1)
        sec = np.asarray(self.volume[idx], dtype=np.float32).transpose(0, 2, 1)
        return self.normalize(sec) if normalized else sec

    def split_indices(self, split: str) -> np.ndarray:
        lo, hi = getattr(self.splits, split)
        return np.arange(lo, hi)

    def metadata(self) -> dict:
        return {"path": str(self.path), "shape": list(self.volume.shape),
                "scale": self.scale, "offset": self.offset, "splits": self.splits.as_dict()}


# ---------------------------------------------------------------------------
# Controlled noise injection (for validation with a known reference)
# ---------------------------------------------------------------------------

def inject_noise(section: np.ndarray, noise_cfg: dict, seed: int) -> np.ndarray:
    """Add reproducible synthetic noise to a real section.

    This exists to solve the project's central evaluation problem. Fully
    synthetic data has ground truth but not real geology; raw field data has
    real geology but no ground truth, which is why the old pipeline could only
    report a leakage number that a do-nothing filter scores perfectly on.
    Adding *known* noise to a real section gives both at once: the model is
    trained self-supervised on the contaminated version alone and scored
    against the original, so PSNR/SSIM/SNR become available on real geology.

    The seed is derived from the section's identity rather than drawn fresh,
    so a given section always carries the same noise. That is not a
    reproducibility nicety -- resampling the noise every epoch would show the
    network the same geology under independent noise realisations, which is
    the Noise2Noise setting and a strictly easier problem than the field one
    this is meant to stand in for.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(section, dtype=np.float64).copy()
    ref_std = float(out.std()) + 1e-30

    coherent_amp = float(noise_cfg.get("coherent_amp", 0.0) or 0.0)
    if coherent_amp > 0:
        nt, nx = out.shape
        t = np.arange(nt)[:, None]
        x = np.arange(nx)[None, :]
        for _ in range(int(noise_cfg.get("n_coherent_events", 3))):
            # Steeply dipping, low-frequency events: the ground-roll-like
            # contaminant that sits outside the geological dip fan.
            dip = rng.uniform(2.0, 6.0) * rng.choice([-1.0, 1.0])
            intercept = rng.uniform(-nt, 2 * nt)
            freq = rng.uniform(0.02, 0.06)
            width = rng.uniform(6, 18)
            dist = t - (intercept + dip * x)
            out += (coherent_amp * ref_std * np.exp(-(dist ** 2) / (2 * width ** 2))
                    * np.sin(2 * np.pi * freq * dist))

    random_std = float(noise_cfg.get("random_std", 0.0) or 0.0)
    if random_std > 0:
        out += rng.normal(0.0, random_std * ref_std, size=out.shape)

    return out.astype(np.float32)


def section_seed(base_seed: int, orientation: str, index: int) -> int:
    """Stable per-section seed."""
    return int(base_seed) * 1_000_003 + (0 if orientation == "inline" else 7919) + int(index)


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

class PatchSampler:
    """Random crops from a preloaded pool of training sections.

    Sections are preloaded once rather than read per batch, because the
    mmap-plus-transpose read dominates step time otherwise. Crops are random
    rather than taken from a fixed grid: a grid at stride 32 hands the model
    the same 147 windows every epoch, while random offsets make the effective
    training set continuous in position.
    """

    def __init__(self, volume: SeismicVolume, n_inline_sections: int = 120,
                 n_crossline_sections: int = 60, n_neighbors: int = 0,
                 rng: np.random.Generator | None = None,
                 noise_cfg: dict | None = None, noise_seed: int = 1234) -> None:
        self.volume = volume
        self.n_neighbors = n_neighbors
        rng = rng or np.random.default_rng(0)

        lo, hi = volume.splits.train
        self.sections: list[np.ndarray] = []

        def contaminate(sec: np.ndarray, orientation: str, index: int) -> np.ndarray:
            if not noise_cfg:
                return sec
            seed = section_seed(noise_seed, orientation, index)
            # Each channel is a different physical line, so each gets its own
            # noise realisation -- sharing one across the stack would make the
            # neighbour channels a free look at the centre line's noise.
            return np.stack([inject_noise(sec[c], noise_cfg, seed + 104729 * c)
                             for c in range(sec.shape[0])])

        if n_inline_sections > 0:
            for i in np.linspace(lo, hi - 1, min(n_inline_sections, hi - lo)).astype(int):
                self.sections.append(contaminate(volume.inline_stack(int(i), n_neighbors),
                                                 "inline", int(i)))

        if n_crossline_sections > 0 and n_neighbors == 0:
            # Truncated to the training inline range: a full-length crossline
            # would include samples from the val and test blocks. Crossline
            # neighbours are a different physical spacing from inline ones, so
            # the stacked-channel mode uses inlines only.
            for j in np.linspace(0, volume.n_crosslines - 1,
                                 min(n_crossline_sections, volume.n_crosslines)).astype(int):
                sec = volume.crossline(int(j), inline_range=(lo, hi))[None]
                self.sections.append(contaminate(sec, "crossline", int(j)))

        if not self.sections:
            raise ValueError("no training sections were loaded")

        self.rng = rng

    def __len__(self) -> int:
        return len(self.sections)

    def sample_batch(self, batch_size: int, patch_size: int,
                     augment_trace_flip: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(patches, depth_offsets)``.

        ``patches`` is ``(B, C, patch_size, patch_size)``; ``depth_offsets`` is
        the absolute starting sample index of each crop, which the
        depth-varying noise model needs in order to index the right bins.
        """
        out = np.empty((batch_size, self.sections[0].shape[0], patch_size, patch_size),
                       dtype=np.float32)
        offsets = np.empty(batch_size, dtype=np.int64)

        for b in range(batch_size):
            sec = self.sections[self.rng.integers(len(self.sections))]
            _, h, w = sec.shape
            y0 = int(self.rng.integers(0, max(h - patch_size, 0) + 1))
            x0 = int(self.rng.integers(0, max(w - patch_size, 0) + 1))
            crop = sec[:, y0:y0 + patch_size, x0:x0 + patch_size]

            # Trace-order flip only. Flipping the time axis reverses the
            # seismic wavelet's phase, and post-stack wavelets are not
            # time-symmetric, so vertical flips teach the model a wavelet that
            # does not occur in the data. The old pipeline did both.
            if augment_trace_flip and self.rng.random() < 0.5:
                crop = crop[:, :, ::-1]

            out[b] = crop
            offsets[b] = y0

        return out, offsets


def save_norm_metadata(path: str | Path, volume: SeismicVolume) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(volume.metadata(), f, indent=2)
