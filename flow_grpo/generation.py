"""Flow-GRPO-Fast generation phase (ODE pre-branch + SDE window + ODE post-branch)."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .sde import sde_sigma, sde_mu_dict, sde_step_dict

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_sde_group(
    model: nn.Module,           # ShortCut (trainable, but called in eval)
    cond_embed: torch.Tensor,   # pre-computed condition embedding  (N, L, C)
    latent_shape_dict: Dict,    # {latent_name: (N, seq_len, channels)}
    device: torch.device,
    G: int = 8,
    T_train: int = 10,
    T_sde: int = 2,
    sde_a: float = 0.7,
    cfg_strength: float = 7.0,
    time_scale: float = 1000.0,
) -> Tuple[List[Dict], List[int]]:
    """
    Flow-GRPO-Fast: T_sde steps randomly sampled (without replacement) from
    [0, T_train − 1].  At those steps SDE noise is injected; all other steps
    use a deterministic ODE update.  G trajectories run in parallel.

    Storage format per trajectory dict:
      x_steps     : interleaved [x_in_0, x_out_0, x_in_1, x_out_1, …, x_final]
                    length = 2*T_sde + 1
                    x_final is x_1 (after all T_train steps), used for reward.
      mu_steps    : list of T_sde mean dicts   (SDE steps only)
      sigma_steps : list of T_sde floats
      t_steps     : list of T_sde floats
      dt_steps    : list of T_sde floats

    compute_grpo_loss_single uses x_steps[2*i] and x_steps[2*i+1] for SDE step i.

    Returns:
      trajectories   : list of G trajectory dicts (described above)
      sde_step_indices: sorted list of T_sde chosen step indices (for logging)
    """
    # Switch model to eval (CFG active)
    model.eval()
    model.rescale_t = 3.0

    # Apply rescale_t the same way as ShortCut._prepare_t_and_d
    t_seq = np.linspace(0.0, 1.0, T_train + 1)
    if model.rescale_t and model.rescale_t != 1.0:
        t_seq = t_seq / (1.0 + (model.rescale_t - 1.0) * (1.0 - t_seq))

    model.reverse_fn.strength = cfg_strength

    # Randomly select T_sde step indices (without replacement, sorted)
    sde_indices_sorted = sorted(
        np.random.choice(T_train, size=T_sde, replace=False).tolist()
    )
    sde_index_set = set(sde_indices_sorted)

    # Pick one component to perturb with noise; all G share the same choice
    noisy_key = str(np.random.choice(["6drotation_normalized", "translation", "scale"]))

    trajectories = []
    for g in range(G):
        x_t = model._generate_noise(latent_shape_dict, device)

        # x_steps stores interleaved (x_in, x_out) for each SDE step + x_final
        x_steps: List[Dict] = []
        mu_steps, sigma_steps, t_steps, dt_steps = [], [], [], []

        for step_idx in range(T_train):
            t  = float(t_seq[step_idx])
            dt = float(t_seq[step_idx + 1]) - t

            t_tensor = torch.tensor([t * time_scale], device=device, dtype=torch.float32)
            # d_tensor = torch.zeros(1, device=device, dtype=torch.float32)
            d_tensor = torch.tensor([dt * time_scale], device=device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = model.reverse_fn.backbone(x_t, t_tensor, cond_embed, d=d_tensor)

            if step_idx in sde_index_set:
                # SDE step: inject noise only on noisy_key; others get zeros → ODE
                # g==0 is fixed as ODE (sigma=0 → no noise injected)
                sigma = 0.0 if g < G // 4 or g == 0 else sde_sigma(t, a=sde_a)
                eps   = {
                    k: (torch.randn_like(x_t[k]) if k == noisy_key else torch.zeros_like(x_t[k]))
                    for k in x_t
                }
                mu, x_new = sde_step_dict(x_t, v, t, dt, sigma, eps)

                x_steps.append({k: x_t[k].detach() for k in x_t})       # x_in
                x_steps.append({k: x_new[k].detach() for k in x_new})   # x_out

                mu_steps.append({k: mu[k].detach() for k in mu})
                sigma_steps.append(sigma)
                t_steps.append(t)
                dt_steps.append(dt)
            else:
                # ODE step: deterministic update, not stored
                x_new = {k: (x_t[k] + v[k] * dt).detach() for k in x_t}

            x_t = x_new

        # Append x_final (= x_1) for reward decoding
        x_steps.append({k: v.detach() for k, v in x_t.items()})

        trajectories.append(
            dict(
                x_steps=x_steps,
                mu_steps=mu_steps,
                sigma_steps=sigma_steps,
                t_steps=t_steps,
                dt_steps=dt_steps,
                noisy_key=noisy_key,
            )
        )

    return trajectories, sde_indices_sorted
