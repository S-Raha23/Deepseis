"""
Behavioural tests for the structure-preservation terms.

The loss these replace had its sign effectively inverted -- it rewarded
*raising* gradient magnitude, so the model it supervised produced an output
with more energy than its input -- and that survived review because nothing
ever checked which way the number moved when structure was actually
destroyed. Every term here is therefore checked against a known-degraded
input: if a penalty claims to detect lost geology, it has to go up when
geology is lost, and stay at zero when nothing changes.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from deepseis.losses.geology import (dip_fan_loss, dip_fan_mask, fault_contrast_loss,
                                     local_semblance)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def faulted_section(h: int = 96, w: int = 96, throw: int = 6) -> torch.Tensor:
    """Flat reflectors cut by a vertical fault with a real throw."""
    t = torch.arange(h, dtype=torch.float32).view(h, 1)
    x = torch.arange(w, dtype=torch.float32).view(1, w)
    left = torch.sin(2 * math.pi * t / 9.0).expand(h, w).clone()
    shifted = torch.sin(2 * math.pi * (t + throw) / 9.0).expand(h, w)
    sec = torch.where(x >= w // 2, shifted, left)
    return sec.view(1, 1, h, w)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(int(3 * sigma), 1)
    g = torch.exp(-torch.arange(-radius, radius + 1, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1, 1)
    xp = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(xp, g)
    xp = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    return F.conv2d(xp, g.view(1, 1, 1, -1))


# ---------------------------------------------------------------------------
# Semblance
# ---------------------------------------------------------------------------

def test_semblance_is_bounded_and_drops_at_a_fault():
    sec = faulted_section()
    c = local_semblance(sec)
    assert float(c.min()) >= -1e-5 and float(c.max()) <= 1.0 + 1e-4, "semblance must lie in [0, 1]"

    w = sec.shape[-1]
    at_fault = c[0, 0, :, w // 2 - 2: w // 2 + 2].mean()
    away = c[0, 0, :, 5:20].mean()
    assert float(at_fault) < float(away), "semblance should fall across a fault"


def test_semblance_rises_when_incoherent_noise_is_removed():
    """Guards the premise of fault_contrast_loss: denoising legitimately raises
    the coherence *level*, which is why the loss compares contrast instead."""
    torch.manual_seed(0)
    clean = faulted_section()
    noisy = clean + 0.5 * torch.randn_like(clean)
    assert float(local_semblance(clean).mean()) > float(local_semblance(noisy).mean())


# ---------------------------------------------------------------------------
# Fault contrast
# ---------------------------------------------------------------------------

def test_fault_contrast_loss_is_zero_for_an_untouched_section():
    sec = faulted_section()
    assert float(fault_contrast_loss(sec, sec)) == pytest.approx(0.0, abs=1e-9)


def test_fault_contrast_loss_grows_as_the_fault_is_smeared():
    sec = faulted_section()
    losses = [float(fault_contrast_loss(gaussian_blur(sec, s), sec)) for s in (0.0, 0.8, 1.6, 3.0)]
    assert losses[0] == pytest.approx(0.0, abs=1e-9)
    assert all(b > a for a, b in zip(losses, losses[1:])), f"not monotone in blur: {losses}"


def test_fault_contrast_loss_does_not_punish_removing_incoherent_noise():
    """Recovering the clean section from a noisy one must not be penalised
    more than smearing the fault out of it."""
    torch.manual_seed(0)
    clean = faulted_section()
    noisy = clean + 0.5 * torch.randn_like(clean)
    recovering = float(fault_contrast_loss(clean, noisy))
    smearing = float(fault_contrast_loss(gaussian_blur(noisy, 3.0), noisy))
    assert recovering < smearing, f"denoising ({recovering}) penalised as much as blurring ({smearing})"


# ---------------------------------------------------------------------------
# Dip fan
# ---------------------------------------------------------------------------

def test_dip_fan_mask_admits_geological_dip_and_excludes_steeper_events():
    h = w = 64
    mask = dip_fan_mask(h, w, max_dip=1.0, device=torch.device("cpu"), dtype=torch.float32)

    def energy_fraction_inside(dip: float) -> float:
        t = torch.arange(h, dtype=torch.float32).view(h, 1)
        x = torch.arange(w, dtype=torch.float32).view(1, w)
        event = torch.sin(2 * math.pi * (t - dip * x) / 8.0).view(1, 1, h, w)
        spec = torch.abs(torch.fft.fft2(event)) ** 2
        return float((spec * mask).sum() / spec.sum())

    assert energy_fraction_inside(0.25) > 0.9, "a gently dipping reflector must sit inside the fan"
    assert energy_fraction_inside(4.0) < 0.3, "a steeply dipping (low-velocity) event must fall outside"


def test_dip_fan_loss_is_zero_for_an_untouched_section_and_grows_with_blur():
    sec = faulted_section()
    assert float(dip_fan_loss(sec, sec)) == pytest.approx(0.0, abs=1e-9)
    losses = [float(dip_fan_loss(gaussian_blur(sec, s), sec)) for s in (0.0, 1.0, 2.0, 4.0)]
    assert all(b > a for a, b in zip(losses, losses[1:])), f"not monotone in blur: {losses}"


def test_dip_fan_loss_ignores_energy_outside_the_fan():
    """Removing a steep, low-velocity coherent event is what a denoiser is for,
    so the term that protects signal must not charge for it."""
    torch.manual_seed(0)
    h = w = 64
    t = torch.arange(h, dtype=torch.float32).view(h, 1)
    x = torch.arange(w, dtype=torch.float32).view(1, w)
    geology = torch.sin(2 * math.pi * (t - 0.2 * x) / 9.0).view(1, 1, h, w)
    ground_roll = 1.5 * torch.sin(2 * math.pi * (t - 5.0 * x) / 14.0).view(1, 1, h, w)

    contaminated = geology + ground_roll
    removed_the_noise = float(dip_fan_loss(geology, contaminated))
    removed_the_signal = float(dip_fan_loss(ground_roll, contaminated))
    assert removed_the_noise < 0.05, f"charged {removed_the_noise} for removing out-of-fan noise"
    assert removed_the_signal > removed_the_noise, "removing the geology must cost more"


def test_taper_prevents_patch_edge_energy_from_dominating():
    """The old F-K term took an unwindowed FFT of a patch; the resulting edge
    step is broadband and can exceed the signal it is meant to measure."""
    torch.manual_seed(0)
    sec = faulted_section(64, 64)
    ramp = torch.linspace(0, 6, 64).view(1, 1, 1, 64).expand_as(sec)
    with_step = sec + ramp   # a large edge-to-edge discontinuity under FFT wraparound

    # The tapered loss should be near-zero for an unchanged section regardless
    # of how big the wraparound step is.
    assert float(dip_fan_loss(with_step, with_step)) == pytest.approx(0.0, abs=1e-9)
    blurred = float(dip_fan_loss(gaussian_blur(with_step, 2.0), with_step))
    assert blurred > 1e-6, "the term stopped responding to blur once a step was present"
