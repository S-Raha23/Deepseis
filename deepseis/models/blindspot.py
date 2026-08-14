"""
Blind-spot denoiser with an explicit noise model.

This replaces the masking-based Noise2Void denoiser in ``models/unet.py``.
The problem with masking is not that it is wrong in principle, it is that it
buys its blind spot at a very high price:

* the reconstruction loss only sees ``mask_fraction`` of the pixels (2% in
  the shipped config), so 98% of every forward pass produces no gradient;
* training inputs are corrupted and inference inputs are not, so the model is
  served a distribution it never trained on;
* whether the blind spot is *actually* blind depends on the correctness of
  the masking code, which is exactly where this project's previous denoiser
  was silently broken (donor pixels resolving back onto the source).

Here the blind spot is a property of the **architecture**, not of the input.
The receptive field is restricted so that the prediction at pixel ``(y, x)``
provably cannot see ``input[:, y, x]``, for any input. Nothing is masked, the
loss is evaluated at every pixel, and train and inference see identical
inputs. ``tests/test_blindspot.py`` verifies the blind spot by autograd rather
than trusting this docstring.

Construction (Laine et al. 2019, *High-Quality Self-Supervised Deep Image
Denoising*): a backbone whose receptive field is restricted to the half-plane
strictly above the centre row is run over the four 90-degree rotations of the
input. Rotating each branch's output back gives four strict half-planes --
above, below, left, right -- whose union is every pixel except the centre one.
A 1x1 head mixes the four branches.

Two further departures from ``unet.py``:

**Bias-free.** Every convolution has ``bias=False`` and every activation is
positively homogeneous (LeakyReLU), so the network is exactly scale
equivariant: ``f(a*x) == a*f(x)`` for ``a > 0``. A denoiser that is not scale
equivariant has to be served at exactly the amplitude scale it was fitted at,
and this project has already been bitten by that once (a 4.4x train/serve
mismatch). It also generalises across noise levels for free (Mohan et al.
2020). Normalisation layers are removed for the same reason: ``GroupNorm``
rescales per sample over the spatial extent, which both destroys relative
amplitude -- the quantity bright-spot and AVO interpretation depend on -- and
makes the output depend on the patch the pixel happened to land in.

**Predicts a distribution, not a point.** The network outputs a per-pixel
Gaussian ``(mu, sigma_p)`` over the clean value *given the surroundings*. On
its own a blind-spot network can only ever produce ``E[x | context excluding
the centre pixel]``, which is intrinsically blurrier than the best estimate
available, because it throws away the single most informative measurement of
``x``: the observed value at that pixel. Combining the network's prior with
that observation through the noise model (see ``losses/nll.py``) recovers
``E[x | context AND centre]``, which is sharper -- and sharper in exactly the
geologically right way. Where context predicts the pixel well (conformable
reflectors) the prior is confident and the observation is smoothed away;
where context fails (fault cuts, pinch-outs, channel edges) the prior is
uncertain and the observation is preserved. That behaviour is what "denoise
without removing geology" actually means, and it falls out of the model
rather than having to be bolted on by a penalty term.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Receptive-field bookkeeping
# ---------------------------------------------------------------------------

def shift_down(x: torch.Tensor, n: int) -> torch.Tensor:
    """Translate the feature map down by ``n`` rows, zero-filling the top.

    This is the whole mechanism behind the half-plane restriction. Every
    operation that lets information reach *downwards* is followed (or, for
    pooling, preceded) by a compensating shift, which keeps the invariant

        feature[y] depends only on input rows <= y - 1

    true from the input all the way to the output.
    """
    if n <= 0:
        return x
    return F.pad(x, (0, 0, n, 0))[:, :, : x.shape[-2], :]


class HalfPlaneConv(nn.Module):
    """Bias-free conv whose output stays strictly above the centre row.

    A ``k x k`` convolution reaches ``k // 2`` rows downwards, so it is
    followed by a shift of ``k // 2`` to undo exactly that reach.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.reach = kernel // 2
        self.act = nn.LeakyReLU(0.1) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = shift_down(self.conv(x), self.reach)
        return self.act(h) if self.act is not None else h


class HalfPlaneUNet(nn.Module):
    """U-Net backbone restricted to the half-plane strictly above each pixel."""

    def __init__(self, in_ch: int, base_ch: int = 32, depth: int = 3, max_mult: int = 4) -> None:
        super().__init__()
        chs = [base_ch * min(2 ** i, max_mult) for i in range(depth + 1)]
        self.out_channels = chs[0]

        self.encoders = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.encoders.append(nn.Sequential(HalfPlaneConv(prev, c), HalfPlaneConv(c, c)))
            prev = c

        self.bottleneck = nn.Sequential(HalfPlaneConv(chs[-2], chs[-1]), HalfPlaneConv(chs[-1], chs[-1]))

        self.decoders = nn.ModuleList(
            [nn.Sequential(HalfPlaneConv(chs[i + 1] + chs[i], chs[i]), HalfPlaneConv(chs[i], chs[i]))
             for i in reversed(range(depth))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The one shift that creates the blind spot: after this, row y of the
        # feature map holds row y-1 of the input, so "everything the network
        # can still reach" already excludes the centre row.
        h = shift_down(x, 1)

        skips = []
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            # Pooling aggregates rows 2y and 2y+1, which reaches one row too
            # far once the result is upsampled back; shifting first pays for it.
            h = F.max_pool2d(shift_down(h, 1), 2)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips)):
            # Nearest-neighbour upsampling to the skip's exact size handles odd
            # extents (F3 sections are 255 samples deep) without breaking the
            # invariant: the sampled low-res index is never more than floor(y/2).
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))

        return h


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

class NoiseModel(nn.Module):
    """Learned observation-noise standard deviation.

    The marginal likelihood of the observed value only ever constrains the
    *total* variance ``sigma_p^2 + sigma_n^2``, so the split between "the
    context could not predict this" and "this is noise" is not identifiable if
    both terms are free per pixel -- the model could call everything noise, or
    nothing. What breaks the degeneracy is that ``sigma_n`` is forced to be
    shared across pixels while ``sigma_p`` varies freely.

    ``mode="scalar"`` shares it across the whole section. ``mode="depth"``
    shares it across traces but lets it vary with two-way time, which is the
    physically right shape for seismic: signal amplitude decays with depth
    through spherical divergence and absorption while much of the noise does
    not, so the noise-to-signal ratio grows downwards. That is 255 parameters
    against ~180k supervised pixels per section, still heavily constrained.
    """

    def __init__(self, mode: str = "depth", n_depth: int = 512, init_sigma: float = 0.1,
                 init_profile=None, freeze: bool = False) -> None:
        super().__init__()
        if mode not in ("scalar", "depth"):
            raise ValueError(f"unknown noise model mode: {mode}")
        self.mode = mode
        self.frozen = freeze

        n = 1 if mode == "scalar" else n_depth
        if init_profile is not None:
            sigma = torch.as_tensor(np.asarray(init_profile), dtype=torch.float32).reshape(-1)
            if sigma.numel() != n:
                idx = torch.linspace(0, sigma.numel() - 1, n)
                sigma = torch.from_numpy(np.interp(idx.numpy(),
                                                   np.arange(sigma.numel()),
                                                   sigma.numpy())).float()
            raw = torch.log(torch.expm1(sigma.clamp_min(1e-6)))
        else:
            # softplus^-1(init) so the initial sigma is what the caller asked for
            raw = torch.full((n,), float(torch.log(torch.expm1(torch.tensor(init_sigma)))))

        self.raw = nn.Parameter(raw, requires_grad=not freeze)

    def sigma(self, height: int, depth_offset: int = 0, device=None) -> torch.Tensor:
        """Noise sigma as a (1, 1, height, 1) tensor, broadcastable over traces."""
        if self.mode == "scalar":
            s = F.softplus(self.raw).view(1, 1, 1, 1)
            return s.expand(1, 1, height, 1)
        raw = self.raw[depth_offset: depth_offset + height]
        if raw.shape[0] < height:  # section deeper than the table: hold the last value
            raw = torch.cat([raw, self.raw[-1:].expand(height - raw.shape[0])])
        return F.softplus(raw).view(1, 1, height, 1)

    def profile(self) -> torch.Tensor:
        return F.softplus(self.raw).detach()


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class BlindSpotDenoiser(nn.Module):
    """Blind-spot denoiser predicting a per-pixel Gaussian over the clean value.

    ``forward`` returns ``(mu, sigma_p)``, the mean and standard deviation of
    the prior implied by everything *except* the pixel itself. Turning that
    into a denoised image requires the noise model too -- see
    ``losses.nll.posterior_mean`` -- because the point of the construction is
    to fold the observed pixel back in.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, depth: int = 3,
                 max_mult: int = 4) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.backbone = HalfPlaneUNet(in_channels, base_channels, depth, max_mult)

        c = self.backbone.out_channels
        # 1x1 only: mixes the four half-planes without letting any of them
        # reach sideways into the centre pixel they were built to exclude.
        self.head = nn.Sequential(
            nn.Conv2d(4 * c, 4 * c, 1, bias=False), nn.LeakyReLU(0.1),
            nn.Conv2d(4 * c, 2 * c, 1, bias=False), nn.LeakyReLU(0.1),
            nn.Conv2d(2 * c, 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if h == w:
            # Square input: the four rotations keep the same shape, so they can
            # go through the backbone as one batch of 4B instead of four
            # separate passes. Identical arithmetic, materially better BLAS
            # utilisation -- and training patches are always square.
            stacked = torch.cat([torch.rot90(x, k, dims=(2, 3)) for k in range(4)], dim=0)
            hs = self.backbone(stacked)
            branches = [torch.rot90(hs[k * b:(k + 1) * b], -k, dims=(2, 3)) for k in range(4)]
        else:
            branches = [torch.rot90(self.backbone(torch.rot90(x, k, dims=(2, 3))), -k, dims=(2, 3))
                        for k in range(4)]

        out = self.head(torch.cat(branches, dim=1))
        mu = out[:, 0:1]

        # abs() rather than softplus keeps the network exactly scale
        # equivariant -- softplus(a*z) != a*softplus(z). No epsilon floor
        # either, for the same reason: an additive constant is a bias by
        # another name. sigma_p is allowed to reach exactly zero because
        # every consumer divides by sigma_p^2 + sigma_n^2, and sigma_n is a
        # softplus and therefore strictly positive.
        sigma_p = out[:, 1:2].abs()
        return mu, sigma_p


def receptive_field_halfwidth(model: BlindSpotDenoiser, probe: int = 241) -> int:
    """Measure how far the prediction at a pixel actually reaches, in pixels.

    Measured rather than derived, because the shift bookkeeping that creates
    the blind spot also perturbs the extent in ways that are easy to get wrong
    on paper. Tiled inference needs this to choose a crop margin: below it the
    tiled result stops matching the single-pass one, which is the whole reason
    the tiled path is allowed to exist.

    Depth 2 gives 48 and depth 3 gives 100 for the default widths. The extent
    is symmetric in all four directions, which is itself a check that every
    rotated branch is contributing.
    """
    was_training = model.training
    model.eval()
    # Callers are typically inference paths running under torch.no_grad(), so
    # grad has to be turned back on explicitly for the probe.
    with torch.enable_grad():
        x = torch.zeros(1, model.in_channels, probe, probe, requires_grad=True)
        mu, _ = model(x)
        (grad,) = torch.autograd.grad(mu[0, 0, probe // 2, probe // 2], x)
    nonzero = (grad[0].abs().sum(0) > 0).nonzero()
    if was_training:
        model.train()
    if nonzero.numel() == 0:
        return 0
    centre = probe // 2
    return int((nonzero - centre).abs().max())
