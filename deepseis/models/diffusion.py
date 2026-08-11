"""
Physics-guided diffusion refiner (spec Sec. 3.3 — the "wow" stretch goal).

A small DDPM-style model that takes the *already denoised* output of stage 1
and refines it further. It is trained self-supervised (same blind-spot
target: the observed data itself, not a separate clean label) and steered
by a lightweight "physics prior" — a wave-equation-inspired smoothness
penalty (discrete Laplacian) that discourages the sampler from hallucinating
high-frequency detail that doesn't correspond to a physically plausible
propagating wavefield. This keeps generated detail geologically plausible
rather than invented, which is the whole point of adding physics guidance
instead of running an unconstrained generative model (spec: "keeps the
generated detail geologically plausible rather than hallucinated").

This is deliberately small (few timesteps, few channels) so it can train in
a couple of minutes on CPU for the demo; the architecture scales up
trivially (more channels/timesteps) once run on a real GPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class _TimeConditionedBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class DiffusionUNet(nn.Module):
    """Small time-conditioned U-Net predicting the noise epsilon added at step t."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 2) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        chs = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.stem = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList([_TimeConditionedBlock(chs[i], time_dim) for i in range(depth)])
        self.downsamplers = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1) for i in range(depth)])

        self.bottleneck = _TimeConditionedBlock(chs[-1], time_dim)

        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1)
                                          for i in reversed(range(depth))])
        self.merge_convs = nn.ModuleList([nn.Conv2d(chs[i] * 2, chs[i], 1) for i in reversed(range(depth))])
        self.up_blocks = nn.ModuleList([_TimeConditionedBlock(chs[i], time_dim) for i in reversed(range(depth))])

        self.head = nn.Conv2d(chs[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.stem(x)
        skips = []
        for res, down in zip(self.down_blocks, self.downsamplers):
            h = res(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h, t_emb)

        for up, merge, res, skip in zip(self.upsamplers, self.merge_convs, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = merge(torch.cat([h, skip], dim=1))
            h = res(h, t_emb)

        return self.head(h)


class GaussianDiffusion:
    """Linear-beta-schedule DDPM wrapper: training loss + a DDIM-style fast sampler."""

    def __init__(self, model: DiffusionUNet, timesteps: int = 60, physics_prior_weight: float = 0.05,
                 device: str = "cpu") -> None:
        self.model = model
        self.timesteps = timesteps
        self.physics_prior_weight = physics_prior_weight
        self.device = device

        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + b * noise

    @staticmethod
    def _laplacian(x: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)

    def physics_prior_loss(self, x0_pred: torch.Tensor, x0_ref: torch.Tensor) -> torch.Tensor:
        """Penalize Laplacian-energy that grew beyond what's in the stage-1 reference.

        A propagating wavefield doesn't spontaneously acquire new curvature
        out of nowhere; if the refiner's output has meaningfully *more*
        second-derivative energy than the input it started from, that's a
        sign it invented detail rather than sharpening real structure.
        """
        lap_pred = self._laplacian(x0_pred)
        lap_ref = self._laplacian(x0_ref)
        excess = F.relu(lap_pred.abs() - lap_ref.abs())
        return excess.pow(2).mean()

    def training_loss(self, x0: torch.Tensor, x0_ref: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t)
        recon_loss = F.mse_loss(eps_pred, noise)

        # reconstruct an x0 estimate from the predicted noise for the physics term
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x0_pred = (x_t - b * eps_pred) / (a + 1e-6)
        prior_loss = self.physics_prior_loss(x0_pred, x0_ref)

        return recon_loss + self.physics_prior_weight * prior_loss

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], x_init: torch.Tensor | None = None,
               sample_steps: int = 20) -> torch.Tensor:
        """DDIM-style sampler subsampled to ``sample_steps`` for fast demo-time inference."""
        device = self.device
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()

        step_indices = torch.linspace(self.timesteps - 1, 0, sample_steps, device=device).long()
        for i, t_val in enumerate(step_indices):
            t = torch.full((shape[0],), int(t_val), device=device, dtype=torch.long)
            eps_pred = self.model(x, t)
            a = self.sqrt_alphas_cumprod[t][:, None, None, None]
            b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x0_pred = (x - b * eps_pred) / (a + 1e-6)

            if i == len(step_indices) - 1:
                x = x0_pred
            else:
                t_next = step_indices[i + 1]
                a_next = self.sqrt_alphas_cumprod[t_next]
                b_next = self.sqrt_one_minus_alphas_cumprod[t_next]
                x = a_next * x0_pred + b_next * eps_pred
        return x
