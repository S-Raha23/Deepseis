"""
Verification of the blind-spot denoiser's two structural guarantees.

These are not smoke tests. The previous denoiser in this project shipped a
blind spot that was not blind -- masked pixels could be replaced by
themselves, so the reconstruction loss rewarded copying the input -- and the
defect survived because nothing checked the property directly, only the
training curve, which looked fine. A blind-spot architecture is only worth
anything if the blindness is actually true, so it is asserted here by
autograd rather than by reading the code.
"""
from __future__ import annotations

import torch

from deepseis.losses.nll import posterior_mean
from deepseis.models.blindspot import BlindSpotDenoiser


def _grad_of_output_pixel(model, x, y, xi, out_index=0):
    """d output[0, out_index, y, xi] / d input, as a tensor shaped like x."""
    x = x.clone().requires_grad_(True)
    mu, sigma_p = model(x)
    target = (mu if out_index == 0 else sigma_p)[0, 0, y, xi]
    (grad,) = torch.autograd.grad(target, x, retain_graph=False)
    return grad


def test_blind_spot_is_exhaustively_blind():
    """For EVERY pixel, the prediction there has zero gradient w.r.t. the input there.

    Small net and small input so the exhaustive loop is cheap; the property is
    architectural and does not depend on width or depth.
    """
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=4, depth=2).eval()
    h = w = 20
    x = torch.randn(1, 1, h, w)

    worst = 0.0
    for y in range(h):
        for xi in range(w):
            g = _grad_of_output_pixel(model, x, y, xi)
            worst = max(worst, float(g[0, :, y, xi].abs().max()))
    assert worst == 0.0, f"blind spot leaks: max |d out(y,x) / d in(y,x)| = {worst}"


def test_blind_spot_holds_for_the_sigma_head_too():
    """The uncertainty head shares the backbone, so it must be blind as well."""
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=4, depth=2).eval()
    x = torch.randn(1, 1, 16, 16)
    for y, xi in [(0, 0), (3, 11), (8, 8), (15, 15), (15, 0), (0, 15)]:
        g = _grad_of_output_pixel(model, x, y, xi, out_index=1)
        assert float(g[0, :, y, xi].abs().max()) == 0.0, f"sigma head sees its own pixel at {(y, xi)}"


def test_blind_spot_holds_across_all_input_channels():
    """With neighbouring inlines as extra channels, none of them may leak either."""
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=3, base_channels=4, depth=2).eval()
    x = torch.randn(1, 3, 16, 16)
    for y, xi in [(1, 1), (7, 9), (14, 4)]:
        g = _grad_of_output_pixel(model, x, y, xi)
        assert float(g[0, :, y, xi].abs().max()) == 0.0, f"channel leak at {(y, xi)}"


def test_the_network_actually_uses_its_context():
    """Guards against the trivial way to pass the test above: ignoring the input.

    A network whose output is constant is perfectly blind and perfectly
    useless, so the neighbourhood gradient must be substantially non-zero.
    """
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=8, depth=2).eval()
    x = torch.randn(1, 1, 24, 24)
    g = _grad_of_output_pixel(model, x, 12, 12)
    neighbourhood = g[0, 0, 9:16, 9:16].abs().sum() - g[0, 0, 12, 12].abs()
    assert float(neighbourhood) > 1e-6, "output does not depend on its context at all"

    # and it must see all four half-planes, not just one
    for name, patch in [("above", g[0, 0, :12, :]), ("below", g[0, 0, 13:, :]),
                        ("left", g[0, 0, :, :12]), ("right", g[0, 0, :, 13:])]:
        assert float(patch.abs().sum()) > 1e-8, f"the {name} half-plane is never reached"


def test_bias_free_network_is_scale_equivariant():
    """f(a*x) == a*f(x): the model cannot be sensitive to amplitude scaling.

    This is the property that makes a train/serve amplitude mismatch
    structurally impossible rather than a bug waiting to be reintroduced.
    """
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=8, depth=2).eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        mu1, sig1 = model(x)
        for a in (0.1, 3.0, 47.0):
            mu2, sig2 = model(a * x)
            assert torch.allclose(mu2, a * mu1, atol=1e-4, rtol=1e-4), f"mu not scale equivariant at a={a}"
            assert torch.allclose(sig2, a * sig1, atol=1e-4, rtol=1e-4), f"sigma not scale equivariant at a={a}"


def test_no_bias_parameters_anywhere():
    model = BlindSpotDenoiser(in_channels=1, base_channels=8, depth=2)
    offenders = [n for n, _ in model.named_parameters() if n.endswith("bias")]
    assert not offenders, f"bias parameters break scale equivariance: {offenders}"


def test_posterior_mean_interpolates_between_prior_and_observation():
    """The estimator must fall back to the observation where the prior is weak."""
    y = torch.full((1, 1, 4, 4), 2.0)
    mu = torch.zeros(1, 1, 4, 4)
    sigma_n = torch.full((1, 1, 4, 4), 1.0)

    confident = posterior_mean(mu, torch.full_like(y, 1e-3), sigma_n, y)
    assert torch.allclose(confident, mu, atol=1e-4), "a confident prior should override the observation"

    uncertain = posterior_mean(mu, torch.full_like(y, 1e3), sigma_n, y)
    assert torch.allclose(uncertain, y, atol=1e-4), "an uncertain prior should defer to the observation"


def test_handles_odd_sized_sections():
    """F3 inlines are 255 samples deep; nothing may assume powers of two."""
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=4, depth=3).eval()
    with torch.no_grad():
        mu, sigma_p = model(torch.randn(1, 1, 255, 101))
    assert mu.shape == (1, 1, 255, 101)
    assert sigma_p.shape == (1, 1, 255, 101)
    assert torch.isfinite(mu).all() and (sigma_p > 0).all()


def test_odd_sized_sections_are_still_blind():
    """The odd-extent interpolation path must not quietly reintroduce the centre pixel."""
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=4, depth=3).eval()
    x = torch.randn(1, 1, 31, 23)
    for y, xi in [(0, 0), (15, 11), (30, 22), (7, 19)]:
        g = _grad_of_output_pixel(model, x, y, xi)
        assert float(g[0, :, y, xi].abs().max()) == 0.0, f"odd-size leak at {(y, xi)}"
