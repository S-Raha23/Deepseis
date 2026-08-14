"""
Training objective and output rule for the blind-spot denoiser.

The network predicts a Gaussian prior ``N(mu, sigma_p^2)`` over the clean
value at a pixel, from everything except that pixel. The observation model is
``y = x + n`` with ``n ~ N(0, sigma_n^2)``. Two things follow directly.

**What to train on.** Marginalising the unknown clean value out of the joint
gives the likelihood of the one quantity actually observed::

    y ~ N(mu, sigma_p^2 + sigma_n^2)

so maximum likelihood on the *noisy* data is a well-posed objective with no
clean reference anywhere -- and, unlike masked MSE, it is evaluated at every
pixel of every patch rather than at the 2% that happened to be masked.

It is also better behaved than MSE for this problem. Plain MSE against a
noisy target treats a pixel the context cannot possibly predict (a fault cut)
the same as one it can, and the only way to reduce the loss at the former is
to regress towards the local mean -- i.e. to blur the discontinuity. Here the
model can instead raise ``sigma_p`` there and pay ``0.5*log(var)`` rather than
the full squared error, which is a much cheaper way to be honest than
smoothing is to be wrong.

**What to output.** The denoised value is *not* ``mu``. ``mu`` is
``E[x | context excluding y]`` -- it has thrown away the most informative
single measurement of ``x``, which is ``y`` itself, and is therefore
systematically over-smoothed. Multiplying the prior by the likelihood and
taking the mean of the posterior gives ``E[x | context AND y]``::

    x_hat = (mu * sigma_n^2 + y * sigma_p^2) / (sigma_p^2 + sigma_n^2)

This is a per-pixel adaptive blend, and its behaviour is exactly the one the
geology requires. Where the context predicts the pixel well, ``sigma_p`` is
small, the weight lands on ``mu``, and the noise is removed. Where the context
fails -- fault planes, pinch-outs, channel margins, unconformities, precisely
the features an interpreter is looking for -- ``sigma_p`` is large, the weight
lands on the observation, and the structure survives untouched. Fault
preservation is a consequence of the estimator rather than a penalty term
that has to be tuned against the reconstruction loss.
"""
from __future__ import annotations

import torch


def blindspot_nll(mu: torch.Tensor, sigma_p: torch.Tensor, sigma_n: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood of the observed noisy image under the model.

    Shapes: ``mu``, ``sigma_p``, ``y`` are ``(B, 1, H, W)``; ``sigma_n``
    broadcasts against them (``(1, 1, H, 1)`` for the depth-varying model).
    Constant terms are dropped -- they do not affect the gradient.
    """
    var = sigma_p ** 2 + sigma_n ** 2
    return (0.5 * (y - mu) ** 2 / var + 0.5 * torch.log(var)).mean()


def posterior_mean(mu: torch.Tensor, sigma_p: torch.Tensor, sigma_n: torch.Tensor,
                   y: torch.Tensor) -> torch.Tensor:
    """``E[x | context, y]`` -- the denoised estimate."""
    var_p = sigma_p ** 2
    var_n = sigma_n ** 2
    return (mu * var_n + y * var_p) / (var_p + var_n)


def posterior_std(sigma_p: torch.Tensor, sigma_n: torch.Tensor) -> torch.Tensor:
    """Posterior standard deviation -- a per-pixel uncertainty map.

    Useful as a QC display: it highlights where the denoiser is guessing,
    which on a seismic section is where the structure is complicated.
    """
    var_p = sigma_p ** 2
    var_n = sigma_n ** 2
    return torch.sqrt(var_p * var_n / (var_p + var_n))
