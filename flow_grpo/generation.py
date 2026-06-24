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
    generation_batch_size: int = 0,
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

    generation_batch_size controls how many group trajectories are concatenated
    into one backbone forward. 0 means all G groups; 1 is the old serial layout.

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

    num_objects = next(iter(latent_shape_dict.values()))[0]
    group_batch_size = G if generation_batch_size <= 0 else min(generation_batch_size, G)

    def _cat_dicts(dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.cat([d[k] for d in dicts], dim=0) for k in dicts[0]}

    def _slice_group(d: Dict[str, torch.Tensor], local_g: int) -> Dict[str, torch.Tensor]:
        start = local_g * num_objects
        end = start + num_objects
        return {k: v[start:end].detach() for k, v in d.items()}

    trajectories = []
    for group_start in range(0, G, group_batch_size):
        group_end = min(group_start + group_batch_size, G)
        group_indices = list(range(group_start, group_end))
        chunk_size = len(group_indices)

        # Preserve the original per-group RNG order: initial noise first, then
        # the SDE-window noise tensors for that group.
        x0_list = []
        eps_noisy_by_group = []
        for _g in group_indices:
            x0 = model._generate_noise(latent_shape_dict, device)
            x0_list.append(x0)
            eps_noisy_by_group.append([
                torch.randn_like(x0[noisy_key])
                for _ in range(T_sde)
            ])

        x_t = _cat_dicts(x0_list)
        cond_embed_batch = cond_embed.repeat(chunk_size, 1, 1)

        chunk_x_steps: List[List[Dict]] = [[] for _ in group_indices]
        chunk_mu_steps: List[List[Dict]] = [[] for _ in group_indices]
        chunk_sigma_steps: List[List[float]] = [[] for _ in group_indices]
        chunk_t_steps: List[List[float]] = [[] for _ in group_indices]
        chunk_dt_steps: List[List[float]] = [[] for _ in group_indices]

        sde_pos = 0
        for step_idx in range(T_train):
            t = float(t_seq[step_idx])
            dt = float(t_seq[step_idx + 1]) - t

            # Keep the original singleton timestep/delta semantics. The model
            # broadcasts these across all objects/groups, matching the old serial
            # generation path more closely than per-row repeated tensors.
            t_tensor = torch.tensor([t * time_scale], device=device, dtype=torch.float32)
            d_tensor = torch.tensor([dt * time_scale], device=device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = model.reverse_fn.backbone(x_t, t_tensor, cond_embed_batch, d=d_tensor)

            if step_idx in sde_index_set:
                # SDE step: inject noise only on noisy_key; others get zeros -> ODE.
                # Keep the same per-group sigma rule as the serial implementation.
                sigma_values = [
                    0.0 if g < G // 4 or g == 0 else sde_sigma(t, a=sde_a)
                    for g in group_indices
                ]

                # Keep backbone evaluation batched, but apply the SDE update one
                # group at a time. This matches the old serial path where each
                # sde_step_dict call sees all objects for one trajectory.
                mu_groups = []
                x_new_groups = []
                for local_g, sigma in enumerate(sigma_values):
                    x_group = _slice_group(x_t, local_g)
                    v_group = _slice_group(v, local_g)
                    eps_group = {k: torch.zeros_like(val) for k, val in x_group.items()}
                    eps_group[noisy_key] = eps_noisy_by_group[local_g][sde_pos]

                    mu_group, x_new_group = sde_step_dict(
                        x_group, v_group, t, dt, sigma, eps_group
                    )
                    mu_groups.append(mu_group)
                    x_new_groups.append(x_new_group)

                mu = _cat_dicts(mu_groups)
                x_new = _cat_dicts(x_new_groups)

                for local_g, sigma in enumerate(sigma_values):
                    chunk_x_steps[local_g].append(_slice_group(x_t, local_g))       # x_in
                    chunk_x_steps[local_g].append(_slice_group(x_new, local_g))     # x_out
                    chunk_mu_steps[local_g].append(_slice_group(mu, local_g))
                    chunk_sigma_steps[local_g].append(sigma)
                    chunk_t_steps[local_g].append(t)
                    chunk_dt_steps[local_g].append(dt)

                sde_pos += 1
            else:
                # ODE step: deterministic update, not stored
                x_new = {k: (x_t[k] + v[k] * dt).detach() for k in x_t}

            x_t = x_new

        for local_g in range(chunk_size):
            chunk_x_steps[local_g].append(_slice_group(x_t, local_g))  # x_final
            trajectories.append(
                dict(
                    x_steps=chunk_x_steps[local_g],
                    mu_steps=chunk_mu_steps[local_g],
                    sigma_steps=chunk_sigma_steps[local_g],
                    t_steps=chunk_t_steps[local_g],
                    dt_steps=chunk_dt_steps[local_g],
                    noisy_key=noisy_key,
                )
            )

    return trajectories, sde_indices_sorted
