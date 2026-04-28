"""GRPO loss computation with per-step backward for memory efficiency."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .sde import sde_mu_dict

logger = logging.getLogger(__name__)


def compute_grpo_loss_single(
    model: nn.Module,
    ref_model: nn.Module,
    traj: Dict,
    adv_g: torch.Tensor,
    cond_embed: torch.Tensor,
    device: torch.device,
    n_terms: int,               # G * T_sde  — used to normalise each step's loss
    kl_coeff: float = 0.04,
    clip_epsilon: float = 0.2,
    time_scale: float = 1000.0,
) -> Tuple[float, float]:
    """
    Compute and immediately back-propagate the GRPO loss for a SINGLE trajectory.

    x_steps uses the interleaved format produced by generate_sde_group:
      [x_in_0, x_out_0, x_in_1, x_out_1, …, x_final]
    For SDE step i: x_t = x_steps[2*i], x_next = x_steps[2*i+1].

    Each SDE step's loss is divided by n_terms (= G*T_sde) and back-propagated
    immediately, so at most ONE forward-pass activation graph lives in memory
    at any given time.

    Returns (pg_sum_scalar, kl_sum_scalar) for logging only.
    """
    T     = len(traj["t_steps"])
    adv_t = adv_g.to(device=device, dtype=torch.float32)  # (N_obj,)
    # print(f"adv_t: {adv_t.cpu().numpy()}")

    total_pg_val = 0.0
    total_kl_val = 0.0

    excluded = {'shape', 'translation_scale'}
    noisy_key = traj.get("noisy_key", None)

    for step_idx in range(T):
        t     = traj["t_steps"][step_idx]
        dt    = traj["dt_steps"][step_idx]
        sigma = traj["sigma_steps"][step_idx]

        # Interleaved storage: x_in at even index, x_out at odd index
        x_t    = {k: v.detach() for k, v in traj["x_steps"][2 * step_idx].items()}
        x_next = {k: v.detach() for k, v in traj["x_steps"][2 * step_idx + 1].items()}
        mu_old = {k: v.detach() for k, v in traj["mu_steps"][step_idx].items()}

        t_tensor = torch.tensor([t * time_scale], device=device, dtype=torch.float32)
        d_tensor = torch.tensor([dt * time_scale], device=device, dtype=torch.float32)

        # ── forward (with gradient) ──
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_new = model.reverse_fn.backbone(x_t, t_tensor, cond_embed, d=d_tensor)

        # pg_loss — free mu_new immediately after sq_new is computed
        denom  = 2.0 * (sigma ** 2) * dt + 1e-12
        sq_old = sum(((x_next[k] - mu_old[k]) ** 2).sum(dim=(1, 2)) if k == noisy_key else 0 for k in x_next)
        mu_new = sde_mu_dict(x_t, v_new, t, dt, sigma)
        sq_new = sum(((x_next[k] - mu_new[k]) ** 2).sum(dim=(1, 2)) if k == noisy_key else 0 for k in x_next)
        del mu_new

        ratio        = torch.exp(((sq_old - sq_new) / denom).clamp(-20.0, 20.0))
        pg_unclipped = ratio * adv_t
        pg_clipped   = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_t
        pg_loss      = -torch.min(pg_unclipped, pg_clipped).sum()  # sum over objects

        # kl gradient via register_hook — lets us free v_ref before backward
        # d(kl)/d(v_new[k]) = 2*(v_new[k] - v_ref[k]) / numel, injected as a hook
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_ref = ref_model(x_t, t_tensor, cond_embed, d=d_tensor)

        if kl_coeff > 0.0:
            kl_val = 0.0
            for k, v in v_new.items():
                if k in excluded:
                    continue
                diff = (v.detach() - v_ref[k]).float()
                kl_val += (diff ** 2).mean().item()
                if v.requires_grad:
                    kl_grad = ((2.0 * kl_coeff / (n_terms * v.numel())) * diff).to(dtype=v.dtype)
                    v.register_hook(lambda g, kg=kl_grad: g + kg.to(g.dtype))
            del v_ref  # free before backward — not referenced by any live graph
        else:
            kl_val = 0.0

        pg_val = pg_loss.item()
        (pg_loss / n_terms).backward()  # pg grad + injected kl grad in one pass
        del v_new, pg_loss

        total_pg_val += pg_val
        total_kl_val += kl_val

    return total_pg_val, total_kl_val
