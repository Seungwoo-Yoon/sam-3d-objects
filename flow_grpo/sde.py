"""SDE utilities for Flow-GRPO-Fast.

Convention: t=0 is pure noise, t=1 is data.
"""

from typing import Dict, Tuple

import torch


def sde_sigma(t: float, a: float = 0.7, t_eps: float = 1e-3) -> float:
    """
    SDE noise level: σ_t = a · √((1−t)/t).

    High near t=0 (noise endpoint), zero at t=1 (data endpoint).
    Clamped away from singularities.
    """
    t_safe = max(min(t, 1.0 - t_eps), t_eps)
    return a * ((1.0 - t_safe) / t_safe) ** 0.5


def sde_mu_dict(
    x_t: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    t: float,
    dt: float,
    sigma: float,
) -> Dict[str, torch.Tensor]:
    """
    Drift mean for one SDE Euler–Maruyama step (all modalities).

      μ_θ(x_t, t) = x_t + [v − σ²/(2(1−t)) · (x_t + t·v)] · dt
    """
    t_safe = max(t, 1e-6)
    one_minus_t = max(1.0 - t_safe, 1e-6)
    correction_scale = -(sigma ** 2) / (2.0 * one_minus_t)

    mu = {}
    for k in x_t:
        corr = correction_scale * (x_t[k] + t_safe * v[k])
        mu[k] = x_t[k] + (v[k] + corr) * dt
    return mu


def sde_step_dict(
    x_t: Dict[str, torch.Tensor],
    v: Dict[str, torch.Tensor],
    t: float,
    dt: float,
    sigma: float,
    epsilon: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    One SDE step returning (mu, x_{t+dt}).

      x_{t+dt} = μ_θ(x_t, t) + σ · √dt · ε
    """
    mu = sde_mu_dict(x_t, v, t, dt, sigma)
    x_new = {k: mu[k] + sigma * (dt ** 0.5) * epsilon[k] for k in mu}
    x_new["shape"] = x_t["shape"] + v["shape"] * dt  # shape modality has no noise

    return mu, x_new
