"""GRPO training epoch loop."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import trimesh
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


class RunningStats:
    """Welford's online algorithm for running mean and variance of advantages."""

    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, values: torch.Tensor):
        """Incorporate a new batch of scalar values."""
        for x in values.tolist():
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return (self.M2 / (self.count - 1)) ** 0.5

    def state_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "M2": self.M2}

    def load_state_dict(self, d: Dict[str, Any]):
        self.count = d["count"]
        self.mean = d["mean"]
        self.M2 = d["M2"]


from sam3d_objects.data.utils import to_device
from sam3d_objects.utils.dist_utils import unwrap_dist
from sam3d_objects.pipeline.inference_utils import get_pose_decoder

from train_dual_backbone_foundationpose import (
    log_metrics,
    prepare_conditioning_for_scene,
    get_condition_input,
)
from train_midi_lora_foundationpose import save_checkpoint

from .generation import generate_sde_group
from .decoding import decode_shape_to_sdf
from .reward import compute_reward
from .loss import compute_grpo_loss_single

logger = logging.getLogger(__name__)


def train_epoch_grpo(
    model: nn.Module,
    ref_backbone: nn.Module,        # frozen reference backbone
    ss_decoder: nn.Module,
    slat_generator: nn.Module,
    slat_decoder_mesh: nn.Module,
    slat_preprocessor: Any,
    slat_condition_embedder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    ss_preprocessor: Any,
    G: int = 8,
    T_train: int = 10,
    T_sde: int = 2,
    sde_a: float = 0.7,
    clip_epsilon: float = 0.2,
    kl_coeff: float = 0.04,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    scheduler: Optional[Any] = None,
    exp_logger: Optional[Any] = None,
    global_step: int = 0,
    save_interval_steps: int = 0,
    output_dir: Optional[Path] = None,
    is_distributed: bool = False,
    best_reward: float = -float("inf"),
    cfg_strength: float = 7.0,
    adv_stats: Optional[RunningStats] = None,
) -> Tuple[Dict[str, float], float]:
    """GRPO training epoch."""
    model.train()
    pose_decoder = get_pose_decoder("ScaleShiftInvariant")

    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    total_loss_sum = 0.0
    total_pg_sum = 0.0
    total_kl_sum = 0.0
    total_reward_sum = 0.0
    detailed_reward_sum = {}
    num_scenes = 0
    current_best_reward = best_reward

    pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch}", disable=(rank != 0))

    for step, batch in enumerate(pbar):
        try:
            conditionals = to_device(batch["conditionals"], device)
            scene_num_objects = batch["num_objects"]

            # GT pose (from latents) and GT mesh data — present only when load_meshes=True
            gt_latents   = to_device(batch.get("latents",   {}), device)
            gt_mesh_data = to_device(batch["mesh_data"], device) if "mesh_data" in batch else None

            # -----------------------------------------------------------------
            # Prepare SS conditioning (same as LoRA training)
            # -----------------------------------------------------------------
            ss_input_dicts = prepare_conditioning_for_scene(
                image=conditionals["image"],
                pointmap=conditionals["pointmap"],
                object_masks=conditionals["object_masks"],
                preprocessor=ss_preprocessor,
                device=device,
            )

            embedder = unwrap_dist(model).condition_embedder
            ss_cond_list = [
                get_condition_input(embedder, d, [])[0][0]
                for d in ss_input_dicts
            ]
            ss_cond_embed = torch.cat(ss_cond_list, dim=0)  # (N_obj, L, C)

            raw_model = unwrap_dist(model)
            latent_shape_dict = {
                k: (scene_num_objects,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in raw_model.reverse_fn.backbone.latent_mapping.items()
            }

            # -------------------------------------------------------------
            # SLAT conditioning for this object
            # Mirrors prepare_conditioning_for_scene, but uses slat_preprocessor
            # and slat_condition_embedder (same get_condition_input pattern).
            # -------------------------------------------------------------
            slat_input_dicts = prepare_conditioning_for_scene(
                image=conditionals["image"],
                pointmap=None,
                object_masks=conditionals["object_masks"],
                preprocessor=slat_preprocessor,
                device=device,
            )
            slat_cond_list = [
                get_condition_input(slat_condition_embedder, d, [])[0][0]
                for d in slat_input_dicts
            ]
            slat_cond_embed = torch.cat(slat_cond_list, dim=0)

            # -------------------------------------------------------------
            # Generation phase: Flow-GRPO-Fast (ODE + random SDE window + ODE)
            # -------------------------------------------------------------
            trajectories, sde_indices = generate_sde_group(
                model=raw_model,
                cond_embed=ss_cond_embed,
                latent_shape_dict=latent_shape_dict,
                device=device,
                G=G,
                T_train=T_train,
                T_sde=T_sde,
                sde_a=sde_a,
                cfg_strength=cfg_strength,
                time_scale=raw_model.time_scale,
            )

            # -------------------------------------------------------------
            # Decode final samples and compute rewards
            # -------------------------------------------------------------
            all_reward_dicts = []   # one per trajectory, None for failures
            for g_idx, traj in enumerate(trajectories):
                x_1 = traj["x_steps"][-1]   # final latent dict
                shape_latent = x_1["shape"]   # (N_obj, L, 8)

                scale = torch.zeros((scene_num_objects, 1, 3), device=device)
                rotation = torch.zeros((scene_num_objects, 1, 4), device=device)
                translation = torch.zeros((scene_num_objects, 1, 3), device=device)

                for i, ss_input_dict in enumerate(ss_input_dicts):
                    pose = pose_decoder({
                        "shape": shape_latent[i:i+1],
                        "scale": x_1["scale"][i:i+1],
                        "6drotation_normalized": x_1["6drotation_normalized"][i:i+1],
                        "translation": x_1["translation"][i:i+1],
                        "translation_scale": x_1["translation_scale"],
                    }, scene_scale=ss_input_dict['pointmap_scale'], scene_shift=ss_input_dict['pointmap_shift'])

                    scale[i:i+1] = pose["scale"]
                    rotation[i:i+1] = pose["rotation"]
                    translation[i:i+1] = pose["translation"]

                meshes = decode_shape_to_sdf(
                    shape_latent=shape_latent,
                    ss_decoder=ss_decoder,
                    slat_generator=slat_generator,
                    slat_decoder_mesh=slat_decoder_mesh,
                    cond_embed=slat_cond_embed,
                    device=device,
                )

                # [DEBUG] Output combined mesh (Only for the first trajectory)
                # Also save the input image for reference
                # if g_idx == 0 and rank == 0 and output_dir is not None and len(meshes) > 0:
                #     try:
                #         debug_dir = output_dir / "debug"
                #         debug_dir.mkdir(parents=True, exist_ok=True)
                #         tag = f"step{global_step + step:06d}"

                #         # Build per-object trimesh and apply predicted pose transform
                #         combined_meshes = []
                #         for obj_idx, m in enumerate(meshes):
                #             if m is None or not m.success:
                #                 continue
                #             verts = m.vertices.detach().cpu().float().numpy()
                #             faces = m.faces.detach().cpu().numpy()
                #             tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

                #             s = scale[obj_idx, 0].detach().cpu().float().numpy()       # (3,)
                #             q = rotation[obj_idx, 0].detach().cpu().float()             # (4,) [w,x,y,z]
                #             t = translation[obj_idx, 0].detach().cpu().float().numpy()  # (3,)

                #             R = quaternion_to_matrix(q).numpy()  # (3, 3)
                #             M = np.eye(4)
                #             M[:3, :3] = R @ np.diag([1, 1, 1]) @ np.diag(s)
                #             M[:3, 3] = t
                #             tri.apply_transform(M)
                #             combined_meshes.append(tri)

                #         if combined_meshes:
                #             combined = trimesh.util.concatenate(combined_meshes)
                #             combined.export(str(debug_dir / f"{tag}_combined.glb"))

                #         # Save GT meshes in camera space as GLB (one per object + combined)
                #         if (
                #             gt_mesh_data is not None
                #             and gt_mesh_data.get("mesh_verts") is not None
                #             and gt_latents.get("scale") is not None
                #             and gt_latents.get("6drotation_normalized") is not None
                #             and gt_latents.get("translation") is not None
                #         ):
                #             gt_verts_list = gt_mesh_data["mesh_verts"]   # list[Tensor(V,3)|None]
                #             gt_faces_list = gt_mesh_data["mesh_faces"]   # list[Tensor(F,3)|None]
                #             gt_avail      = gt_mesh_data["mesh_available"].cpu().numpy()
                #             gt_s   = gt_latents["scale"].float().cpu().numpy()                  # (N, 3)
                #             gt_r6d = gt_latents["6drotation_normalized"].float()                # (N, 6)
                #             gt_t   = gt_latents["translation"].float().cpu().numpy()            # (N, 3)

                #             # 6D rotation → matrix  (Zhou et al., 2019)
                #             a1 = torch.nn.functional.normalize(gt_r6d[:, :3], dim=-1)
                #             a2 = gt_r6d[:, 3:6]
                #             a2 = torch.nn.functional.normalize(
                #                 a2 - (a2 * a1).sum(-1, keepdim=True) * a1, dim=-1
                #             )
                #             a3 = torch.linalg.cross(a1, a2)
                #             R_gt = torch.stack([a1, a2, a3], dim=-1).cpu().numpy()  # (N, 3, 3)

                #             gt_tris = []
                #             for obj_idx, (verts_t, faces_t) in enumerate(
                #                 zip(gt_verts_list, gt_faces_list)
                #             ):
                #                 if not gt_avail[obj_idx] or verts_t is None or faces_t is None:
                #                     continue
                #                 verts = verts_t.cpu().float().numpy()  # (V, 3) in [-0.5, 0.5]
                #                 faces = faces_t.cpu().numpy()
                #                 # Apply GT pose: pts_cam = (verts * s) @ R^T + t
                #                 M = np.eye(4)
                #                 M[:3, :3] = R_gt[obj_idx] @ np.diag(gt_s[obj_idx])
                #                 M[:3, 3]  = gt_t[obj_idx]
                #                 tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                #                 tri.apply_transform(M)
                #                 # tri.export(str(debug_dir / f"{tag}_gt_obj{obj_idx}.glb"))
                #                 gt_tris.append(tri)

                #             if gt_tris:
                #                 trimesh.util.concatenate(gt_tris).export(
                #                     str(debug_dir / f"{tag}_gt_combined.glb")
                #                 )

                #         # Save input image with per-object mask overlay
                #         img_tensor = conditionals["image"]  # [H, W, 3]
                #         img_np = img_tensor.detach().cpu().numpy()
                #         if img_np.dtype != np.uint8:
                #             img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                #         vis = img_np.astype(np.float32)
                #         if "object_masks" in conditionals:
                #             masks_np = conditionals["object_masks"].detach().cpu().numpy()  # (N, H, W)
                #             _COLORS = [
                #                 (255,  60,  60),
                #                 ( 60, 200,  60),
                #                 ( 60, 120, 255),
                #                 (255, 200,  40),
                #                 (200,  60, 255),
                #                 ( 40, 220, 220),
                #             ]
                #             for obj_idx, mask in enumerate(masks_np):
                #                 color = _COLORS[obj_idx % len(_COLORS)]
                #                 alpha = 0.45
                #                 for c in range(3):
                #                     vis[:, :, c] = np.where(
                #                         mask > 0.5,
                #                         vis[:, :, c] * (1 - alpha) + color[c] * alpha,
                #                         vis[:, :, c],
                #                     )
                #         Image.fromarray(vis.clip(0, 255).astype(np.uint8)).save(
                #             str(debug_dir / f"{tag}_input.png")
                #         )
                #     except Exception as _dbg_e:
                #         logger.warning(f"[DEBUG] mesh/image save failed: {_dbg_e}")

                if len(meshes) == 0:
                    all_reward_dicts.append(None)   # penalty for empty structure
                else:
                    try:
                        reward_dict = compute_reward(
                            meshes, scale, rotation, translation,
                            conditionals["camera_view_transform"],
                            conditionals["pointmap"], conditionals["object_masks"],
                            gt_mesh_points=gt_mesh_data["mesh_points"]    if gt_mesh_data is not None else None,
                            gt_scale=gt_latents.get("scale"),
                            gt_rotation_6d=gt_latents.get("6drotation_normalized"),
                            gt_translation=gt_latents.get("translation"),
                            gt_mesh_available=gt_mesh_data["mesh_available"] if gt_mesh_data is not None else None,
                        )
                    except NotImplementedError:
                        raise
                    except Exception as e:
                        logger.warning(f"Reward computation failed: {e}")
                        reward_dict = None
                    all_reward_dicts.append(reward_dict)

                # Free SDF grids — only needed for reward, not for GRPO loss
                for m in meshes:
                    if m is not None:
                        m.sdf_d = None
                del meshes
                torch.cuda.empty_cache()

            n_valid = sum(1 for rd in all_reward_dicts if rd is not None)
            if n_valid < 2:
                logger.warning("Not enough valid samples to compute advantages. Skipping.")
                continue

            # -------------------------------------------------------------
            # Per-component advantage: normalize each reward component
            # separately, then sum to get total advantage per object per trajectory.
            # Each rd[k] is a tensor of shape (n_obj_g,); object count may vary
            # across samples but is consistent across G trajectories for the same scene.
            # -------------------------------------------------------------
            reward_keys = next(rd for rd in all_reward_dicts if rd is not None).keys()

            # advantages[g] = None or tensor of shape (n_obj_g,)
            advantages = [None] * len(all_reward_dicts)
            for k in reward_keys:
                k_vals = [(g_idx, rd[k]) for g_idx, rd in enumerate(all_reward_dicts) if rd is not None]

                # Stack valid trajectories → (n_valid, n_obj); compute per-object mean/std across trajectories
                stacked = torch.stack([v for _, v in k_vals], dim=0)  # (n_valid, n_obj)
                mean_k = stacked.mean(dim=0)                           # (n_obj,)
                std_k  = stacked.std(dim=0).clamp(min=1e-8)           # (n_obj,)

                if k == "collision":
                    coeff = 3.0
                else:
                    coeff = 1.0

                for g_idx, v in k_vals:
                    norm_v = (v - mean_k) / std_k  # (n_obj,)
                    if advantages[g_idx] is None:
                        advantages[g_idx] = norm_v * coeff
                    else:
                        advantages[g_idx] = advantages[g_idx] + norm_v * coeff

            # Re-normalize summed advantages using global running stats.
            # g_mean / g_std are shared across all objects.
            all_adv_vals = torch.cat([a for a in advantages if a is not None])  # all objects, all valid trajs
            if adv_stats is None:
                adv_stats = RunningStats()
            adv_stats.update(all_adv_vals)
            if adv_stats.count >= G * 2:
                # Global normalization: stable across training
                g_mean = all_adv_vals.new_tensor(adv_stats.mean)
                g_std  = all_adv_vals.new_tensor(adv_stats.std).clamp(min=1e-8)
                advantages = [((a - g_mean) / g_std) if a is not None else None for a in advantages]
            else:
                # Fall back to local batch normalization until enough samples
                adv_mean = all_adv_vals.mean()
                adv_std  = all_adv_vals.std().clamp(min=1e-8)
                advantages = [((a - adv_mean) / adv_std) if a is not None else None for a in advantages]

            # Total rewards for logging
            rewards = [sum(v.sum().item() for v in rd.values()) if rd is not None else -1.0 for rd in all_reward_dicts]
            reward_dicts = [rd for rd in all_reward_dicts if rd is not None]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            r_mean    = rewards_t.mean()
            r_std     = rewards_t.std().clamp(min=1e-8)
            G_valid     = len(trajectories)
            T           = len(trajectories[0]["t_steps"])   # = T_sde
            n_terms     = G_valid * T
            logger.debug("SDE step indices this batch: %s", sde_indices)

            optimizer.zero_grad(set_to_none=True)
            # Keep eval() mode from generate_sde_group so that ClassifierFreeGuidance
            # applies CFG in the same way as during generation.
            # LoRA gradients flow regardless of train/eval mode.

            details_pg, details_kl = 0.0, 0.0
            for traj, adv_g in zip(trajectories, advantages):
                if adv_g is None:
                    continue
                # backward is called per-step inside; returns scalars for logging
                pg_val, kl_val = compute_grpo_loss_single(
                    model=raw_model,
                    ref_model=ref_backbone,
                    traj=traj,
                    adv_g=adv_g,
                    cond_embed=ss_cond_embed,
                    device=device,
                    n_terms=n_terms,
                    kl_coeff=kl_coeff,
                    clip_epsilon=clip_epsilon,
                    time_scale=raw_model.time_scale,
                )
                details_pg += pg_val
                details_kl += kl_val

            # Free trajectory tensors now that backward is done
            del trajectories
            torch.cuda.empty_cache()

            trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
            grad_has_nan = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable_params
            )
            if grad_has_nan:
                logger.warning(f"NaN/Inf gradient at step {step}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            # print("[DEBUG] Gradient norms:", {name: p.grad.norm().item() for name, p in raw_model.named_parameters() if p.grad is not None})
            # print("[DEBUG] Parameter norms:", {name: p.norm().item() for name, p in unwrap_dist(model).named_parameters() if p.requires_grad and p.grad is not None})
            if r_mean > 0:
                optimizer.step()

            details = {
                "pg_loss":       details_pg / n_terms,
                "kl_loss":       details_kl / n_terms,
                "total_loss":    (details_pg + kl_coeff * details_kl) / n_terms,
                "reward_mean":   r_mean.item(),
                "reward_std":    r_std.item(),
                "adv_global_mean": adv_stats.mean,
                "adv_global_std":  adv_stats.std,
            }

            if scheduler is not None:
                scheduler.step()

            total_loss_sum += details.get("total_loss", 0.0)
            total_pg_sum += details.get("pg_loss", 0.0)
            total_kl_sum += details.get("kl_loss", 0.0)
            if r_mean > 0:
                total_reward_sum += details.get("reward_mean", 0.0)
            else:
                logger.warning(f"Non-positive reward mean ({r_mean.item():.4f}) at step {step}, not updating model and not counting this scene for logging/ckpt.")


            if r_mean > 0:
                reward_key_count = {}
                reward_value_sum = {}
                for rd in reward_dicts:
                    if rd is not None:
                        for k, v in rd.items():
                            reward_key_count[k] = reward_key_count.get(k, 0) + 1
                            reward_value_sum[k] = reward_value_sum.get(k, 0.0) + v.mean().item()
                for k in reward_value_sum:
                    detailed_reward_sum[k] = detailed_reward_sum.get(k, 0.0) + reward_value_sum[k] / reward_key_count[k]

                num_scenes += 1

        except NotImplementedError:
            raise   # Propagate unimplemented reward
        except Exception as e:
            logger.error(f"Error at step {step} (epoch {epoch}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

        # Logging
        if num_scenes > 0 and (step + 1) % log_interval == 0 and rank == 0:
            avg_loss = total_loss_sum / num_scenes
            avg_pg = total_pg_sum / num_scenes
            avg_kl = total_kl_sum / num_scenes
            avg_r = total_reward_sum / num_scenes
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "pg": f"{avg_pg:.4f}",
                "kl": f"{avg_kl:.4f}",
                "r": f"{avg_r:.4f}",
                "lr": f"{lr:.2e}",
            })
            if exp_logger is not None:
                basic_info = {
                    "train/loss": avg_loss,
                    "train/pg_loss": avg_pg,
                    "train/kl_loss": avg_kl,
                    "train/reward_mean": avg_r,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/adv_global_mean": adv_stats.mean,
                    "train/adv_global_std": adv_stats.std,
                    "train/adv_global_count": adv_stats.count,
                }
                detailed_avg_rewards = {f"train/{k}_reward": v / num_scenes for k, v in detailed_reward_sum.items()}
                log_metrics(exp_logger, basic_info | detailed_avg_rewards, global_step + step)

        # Step-based checkpoint
        current_gs = global_step + step + 1
        if (
            save_interval_steps > 0
            and current_gs % save_interval_steps == 0
            and rank == 0
            and output_dir is not None
            and num_scenes > 0
        ):
            step_metrics = {
                "loss": total_loss_sum / num_scenes,
                "reward_mean": total_reward_sum / num_scenes,
            }
            _adv_extra = {"adv_stats": adv_stats.state_dict()}
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, current_gs, step_metrics,
                str(output_dir / f"step_{current_gs:08d}.pt"),
                is_distributed,
                extra_state=_adv_extra,
            )
            if step_metrics["reward_mean"] > current_best_reward:
                current_best_reward = step_metrics["reward_mean"]
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, current_gs, step_metrics,
                    str(output_dir / "best.pt"),
                    is_distributed,
                    extra_state=_adv_extra,
                )
                logger.info(f"New best reward: {current_best_reward:.4f}")

    metrics = {
        "loss": total_loss_sum / max(num_scenes, 1),
        "pg_loss": total_pg_sum / max(num_scenes, 1),
        "kl_loss": total_kl_sum / max(num_scenes, 1),
        "reward_mean": total_reward_sum / max(num_scenes, 1),
    }

    if is_dist:
        for key in metrics:
            t = torch.tensor(metrics[key], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            metrics[key] = t.item()

    return metrics, current_best_reward
