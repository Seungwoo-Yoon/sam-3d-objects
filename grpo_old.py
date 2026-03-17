"""
Flow-GRPO-Fast RL finetuning of MidiSparseStructureFlowTdfyWrapper (LoRA).

Algorithm: Flow-GRPO-Fast (arxiv:2505.05470 + yifan123/flow_grpo), adapted for
3D sparse structure generation.

Flow-GRPO-Fast vs Flow-GRPO:
  Regular Flow-GRPO applies SDE sampling and GRPO training across ALL T_train steps.
  Flow-GRPO-Fast confines stochasticity to only T_sde steps (T_sde < T_train),
  randomly placed within the full T_train-step trajectory.
  The remaining (T_train − T_sde) steps use deterministic ODE sampling.
  This yields comparable reward with significantly less memory and compute.

Training overview:
  For each batch (one scene / one object conditioning):
    1. [Generation phase — ODE pre-branch, no_grad, single trajectory]
       Randomly sample branch_idx ∈ [0, T_train − T_sde].
       Run branch_idx deterministic ODE steps to reach x_branch:
         x_{t+dt} = x_t + v_θ(x_t, t) · dt       ← pure Euler ODE (no noise)

    2. [Generation phase — SDE window, no_grad, G trajectories]
       From x_branch, branch G diverse samples via T_sde SDE steps:
         x_{t+dt} = x_t
                  + [v_θ - σ_t²/(2(1-t)) · (x_t + t·v_θ)] · dt   ← drift
                  + σ_t · √dt · ε                                   ← diffusion
       where σ_t = a · √((1-t)/t)  (high at t≈0, zero at t=1)
       Store trajectories for the SDE window only.

    3. [Generation phase — ODE post-branch, no_grad, G trajectories]
       Continue each of the G trajectories with ODE for the remaining
       (T_train − branch_idx − T_sde) steps (not stored, only final x_1 used).

    4. [Reward phase, no_grad]
       Decode each x_1 through the frozen pipeline:
         shape_latent → ss_decoder → coords → slat_generator → slat_decoder → SDF
       Compute scalar reward r_g for each sample g.

    5. [Update phase, with grad for LoRA params]
       Group-relative advantage: A_g = (r_g − mean(r)) / (std(r) + ε)
       GRPO loss per SDE-window step t, per sample g:
         log_ratio   = [‖x_{t+dt} − μ_θ_old‖² − ‖x_{t+dt} − μ_θ_new‖²]
                       / (2 · σ_t² · dt)
         ratio       = exp(log_ratio)
         pg_term     = min(ratio·A, clip(ratio, 1−ε_clip, 1+ε_clip)·A)
         kl_term     = ‖v_θ_new − v_ref‖²  (simplified KL, closed-form proxy)
         L           = −pg_term + β·kl_term
       Backprop L, update LoRA adapters.

Only the LoRA adapters inside MidiSparseStructureFlowTdfyWrapper are trained.
ss_decoder, slat_generator, slat_decoder are frozen (eval mode, no grad).

Usage:
    python train_flow_grpo_foundationpose.py \\
        --config                      checkpoints/hf/midi_ss_generator.yaml \\
        --ss_generator_checkpoint     checkpoints/hf/ss_generator.ckpt \\
        --lora_checkpoint             ./outputs/midi_lora_fp/latest_peft \\
        --ss_decoder_config           checkpoints/hf/ss_decoder.yaml \\
        --ss_decoder_checkpoint       checkpoints/hf/ss_decoder.ckpt \\
        --slat_generator_config       checkpoints/hf/slat_generator.yaml \\
        --slat_generator_checkpoint   checkpoints/hf/slat_generator.ckpt \\
        --slat_decoder_mesh_config    checkpoints/hf/slat_decoder_mesh.yaml \\
        --slat_decoder_mesh_checkpoint checkpoints/hf/slat_decoder_mesh.ckpt \\
        --data_root /path/to/foundationpose_data \\
        --output_dir ./outputs/flow_grpo_fp

Model loading mirrors inference_pipeline.py exactly:
  - ss_decoder:              direct config + ckpt  (state_dict_key=None)
  - slat_generator:          config["module"]["generator"]["backbone"]
                             ckpt prefix "_base_models.generator."
  - slat_decoder_mesh:       direct config + ckpt  (state_dict_key=None)
  - slat_condition_embedder: config["module"]["condition_embedder"]["backbone"]
                             ckpt prefix "_base_models.condition_embedder."
                             (from slat_generator_checkpoint)
  - slat_preprocessor:       config["tdfy"]["val_preprocessor"]
                             (from slat_generator_config)
"""

import os

from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm

from omegaconf import OmegaConf
from hydra.utils import instantiate
from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel

from foundation_pose_dataset import FoundationPoseDataset, collate_fn
from sam3d_objects.data.utils import to_device, tree_tensor_map
from sam3d_objects.utils.dist_utils import setup_dist, unwrap_dist
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

# Reuse helpers from LoRA training script
from train_midi_lora_foundationpose import (
    build_model_from_config,
    load_ss_generator_checkpoint,
    apply_lora,
    freeze_condition_embedder,
    freeze_non_lora_params,
    get_parameter_groups,
    save_checkpoint,
    load_checkpoint,
)
from train_dual_backbone_foundationpose import (
    setup_logger,
    log_metrics,
    prepare_conditioning_for_scene,
    get_condition_input,
)
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    prune_sparse_structure,
    downsample_sparse_structure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Frozen Downstream Model Loading
# (mirrors inference_pipeline.py's instantiate_and_load_from_pretrained pattern)
# ============================================================================

from sam3d_objects.model.io import filter_and_remove_prefix_state_dict_fn


def _instantiate_and_load(
    cfg,
    ckpt_path: str,
    device: torch.device,
    state_dict_fn=None,
    state_dict_key: Optional[str] = "state_dict",
    strict: bool = True,
) -> nn.Module:
    """
    Instantiate model from OmegaConf cfg and load weights from ckpt_path.

    state_dict_key=None  →  treat the whole checkpoint as the state dict
                           (used for ss_decoder, slat_decoder_mesh)
    state_dict_fn        →  prefix-stripping function applied before load
                           (used for generator / condition-embedder sub-dicts)
    """
    model = instantiate(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if state_dict_key is None:
        sd = ckpt
    else:
        sd = ckpt.get(state_dict_key, ckpt)

    if state_dict_fn is not None:
        sd = state_dict_fn(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ss_decoder(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """Load ss_decoder (direct checkpoint, no prefix stripping)."""
    cfg = OmegaConf.load(config_path)
    if "pretrained_ckpt_path" in cfg:
        del cfg["pretrained_ckpt_path"]
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_key=None)
    logger.info(f"Loaded ss_decoder from {ckpt_path}")
    return model


def load_slat_generator(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load slat_generator (ShortCut + CFG + Backbone).

    Config section: config["module"]["generator"]["backbone"]
    Checkpoint prefix: _base_models.generator.
    """
    cfg = OmegaConf.load(config_path)["module"]["generator"]["backbone"]
    prefix_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_fn=prefix_fn, strict=False)
    logger.info(f"Loaded slat_generator from {ckpt_path}")
    return model


def load_slat_decoder_mesh(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """Load slat_decoder_mesh (direct checkpoint, no prefix stripping)."""
    cfg = OmegaConf.load(config_path)
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_key=None, strict=False)
    logger.info(f"Loaded slat_decoder_mesh from {ckpt_path}")
    return model


def load_condition_embedder(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load a condition embedder from a generator config + checkpoint.

    Config section: config["module"]["condition_embedder"]["backbone"]
    Checkpoint prefix: _base_models.condition_embedder.
    """
    cfg = OmegaConf.load(config_path)["module"]["condition_embedder"]["backbone"]
    prefix_fn = filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder.")
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_fn=prefix_fn)
    logger.info(f"Loaded condition embedder from {ckpt_path}")
    return model


def load_slat_preprocessor(config_path: str) -> Any:
    """Load SLAT preprocessor from a generator config's val_preprocessor section."""
    cfg = OmegaConf.load(config_path)
    preprocessor = instantiate(cfg["slat_preprocessor"])
    logger.info(f"Loaded SLAT preprocessor from {config_path}")
    return preprocessor


# ============================================================================
# QLoRA helpers
# ============================================================================

def _make_linear4bit(linear: nn.Linear, quant_type: str, compute_dtype, double_quant: bool):
    """Return a bitsandbytes Linear4bit copy of an nn.Linear layer."""
    import bitsandbytes as bnb

    q = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        compress_statistics=double_quant,
    )
    # Params4bit stores the fp16 weight; actual quantization triggers on first forward
    q.weight = bnb.nn.Params4bit(
        linear.weight.data.cpu(),
        requires_grad=False,
        quant_type=quant_type,
    )
    if linear.bias is not None:
        q.bias = nn.Parameter(linear.bias.data)
    return q


def quantize_backbone_4bit(
    backbone: nn.Module,
    quant_type: str = "nf4",
    compute_dtype=None,
    double_quant: bool = True,
) -> nn.Module:
    """
    Quantize all nn.Linear weights inside *backbone* to 4-bit NF4 in-place.

    Works on both plain backbones and PEFT PeftModel wrappers:
      - plain nn.Linear            →  bnb.nn.Linear4bit
      - peft lora.Linear.base_layer →  bnb.nn.Linear4bit  (LoRA adapters stay fp16)

    Call this AFTER apply_lora() + loading LoRA checkpoint, but BEFORE
    moving the model back to GPU (or call .to(device) after to trigger quant).
    Then call peft.prepare_model_for_kbit_training() on the backbone.
    """
    import bitsandbytes as bnb
    from peft.tuners.lora.layer import Linear as LoraLinear

    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    def _recurse(module: nn.Module):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoraLinear):
                # Quantize only the frozen base weight; LoRA A/B stay fp16
                base = child.base_layer
                if isinstance(base, nn.Linear) and not isinstance(base, bnb.nn.Linear4bit):
                    q = _make_linear4bit(base, quant_type, compute_dtype, double_quant)
                    child.base_layer = q
                    child._modules["base_layer"] = q
                # Do NOT recurse further — lora_A/B are nn.Linear but must stay fp16
            elif isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
                setattr(module, child_name,
                        _make_linear4bit(child, quant_type, compute_dtype, double_quant))
            else:
                _recurse(child)

    _recurse(backbone)
    logger.info(
        "Quantized backbone: nn.Linear → bnb.nn.Linear4bit "
        "(quant_type=%s, double_quant=%s, compute_dtype=%s)",
        quant_type, double_quant, compute_dtype,
    )
    return backbone


# ============================================================================
# Gradient Checkpointing
# ============================================================================

def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing on all transformer blocks inside the backbone.

    MidiSparseStructureFlowModel passes use_checkpoint down to every
    InteractionModule (and other blocks) that use it.  Setting use_checkpoint=True
    causes torch.utils.checkpoint.checkpoint to be called inside each block's
    forward(), so intermediate activations are NOT retained — they are recomputed
    during backward instead.  This trades ~2× more compute for a large reduction
    in peak activation memory.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "use_checkpoint") and not module.use_checkpoint:
            module.use_checkpoint = True
            count += 1
    logger.info("Gradient checkpointing enabled on %d module(s).", count)


# ============================================================================
# SDE Utilities  (convention: t=0 is pure noise, t=1 is data)
# ============================================================================

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


# ============================================================================
# Generation Phase
# ============================================================================

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

    model.rescale_t = 3

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

    trajectories = []
    for _ in range(G):
        x_t = model._generate_noise(latent_shape_dict, device)

        # x_steps stores interleaved (x_in, x_out) for each SDE step + x_final
        x_steps: List[Dict] = []
        mu_steps, sigma_steps, t_steps, dt_steps = [], [], [], []

        for step_idx in range(T_train):
            t  = float(t_seq[step_idx])
            dt = float(t_seq[step_idx + 1]) - t

            t_tensor = torch.tensor([t * time_scale], device=device, dtype=torch.float32)
            d_tensor = torch.zeros(1, device=device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = model.reverse_fn.backbone(x_t, t_tensor, cond_embed, d=d_tensor)

            if step_idx in sde_index_set:
                # SDE step: inject noise, store (x_in, x_out) for GRPO training
                sigma = sde_sigma(t, a=sde_a)
                eps   = {k: torch.randn_like(x_t[k]) for k in x_t}
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
                x_new["shape"] = (x_t["shape"] + v["shape"] * dt).detach()

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
            )
        )

    return trajectories, sde_indices_sorted


# ============================================================================
# Frozen Downstream Decoding
# ============================================================================

@torch.no_grad()
def decode_shape_to_sdf(
    shape_latent: torch.Tensor,     # (N, L, 8) – the 'shape' entry of x_1
    ss_decoder: nn.Module,
    slat_generator: nn.Module,
    slat_decoder_mesh: nn.Module,
    cond_embed: torch.Tensor,     # (N, L_slat, C_slat) – from slat_condition_embedder
    device: torch.device,
) -> Optional[Dict]:
    """
    Decode a shape latent through the frozen pipeline to produce SDF output.

    Steps:
      1. ss_decoder:         shape_latent  →  voxel occupancy  →  coords
      2. slat_condition_embedder + slat_input_dict  →  slat_cond_embed
      3. slat_generator:     coords + slat_cond_embed  →  SLAT
      4. slat_decoder_mesh:  SLAT  →  mesh (contains SDF)

    Returns:
      dict with 'mesh', 'coords', 'slat' keys; or None if voxel is empty.
    """
    # 1. ss_decoder → voxel → coords
    ss_vol = ss_decoder(
        shape_latent.permute(0, 2, 1)
        .contiguous()
        .view(shape_latent.shape[0], 8, 16, 16, 16)
    )
    coords = torch.argwhere(ss_vol > 0)[:, [0, 2, 3, 4]].int()

    if coords.shape[0] == 0:
        logger.debug("Empty sparse structure – skipping this sample.")
        return []   # return empty list, not None — callers use len()

    meshes = []

    for i, object_embed in enumerate(cond_embed):
        # 3. slat_generator  (coords passed as numpy, same as inference_pipeline)
        obj_coords = coords[coords[:, 0] == i]
        if obj_coords.shape[0] == 0:
            # This object has no voxels — append None as placeholder so that
            # len(meshes) == len(cond_embed) and pose indices stay aligned.
            logger.debug(f"Object {i} has no voxels, skipping mesh decode.")
            meshes.append(None)
            continue

        # Reset batch index to 0 — each slat_generator call is single-object (batch=1)
        obj_coords_local = obj_coords.clone()
        obj_coords_local[:, 0] = 0
        # Mirror inference_pipeline: prune interior voxels then cap at 42k for mesh decoding
        obj_coords_local = prune_sparse_structure(obj_coords_local)
        obj_coords_local, _ = downsample_sparse_structure(obj_coords_local)
        if obj_coords_local.shape[0] == 0:
            meshes.append(None)
            continue

        latent_shape = (1, obj_coords_local.shape[0], 8)
        coords_np = obj_coords_local.cpu().numpy()

        slat = slat_generator(latent_shape, device, object_embed[None, ...], coords_np)

        slat = sp.SparseTensor(
            coords=obj_coords_local,
            feats=slat[0],
        ).to(device)
        slat = slat * SLAT_STD.to(device) + SLAT_MEAN.to(device)

        # 4. slat_decoder_mesh  →  mesh / SDF
        mesh_out = slat_decoder_mesh(slat)[0]
        meshes.append(mesh_out)

    return meshes


# ============================================================================
# Reward Function 
# ============================================================================

def compute_reward(meshes: List[MeshExtractResult], scale, rotation, translation) -> float:
    """
    Compute scalar reward that penalizes inter-object intersection.

    For each ordered pair (i, j), the interior points of object i (SDF_i < 0)
    are transformed into object j's local coordinate frame.  The penetration
    depth at each point is max(0, -SDF_j), and its sum is the intersection
    penalty for that pair.

    Both SDF grids live in [-0.5, 0.5]^3 local space.
    Pose convention: world = (local * scale) @ R^T + t   (row-vector, R from quaternion_to_matrix)

    Args:
        meshes:      list of MeshExtractResult (one per object); each has .sdf_d (res+1)^3
        scale:       (N, 1, 3) per-object scale
        rotation:    (N, 1, 4) per-object quaternion  [w, x, y, z]
        translation: (N, 1, 3) per-object translation

    Returns:
        float: reward (≤ 0; less negative = less intersection = better).
    """
    from pytorch3d.transforms import quaternion_to_matrix
    import torch.nn.functional as F

    # Build list of (original_index, mesh) for meshes that are valid.
    # sdf_d is stored as a flat ((res+1)^3,) tensor; we reshape to
    # (res+1, res+1, res+1) when we access it below.
    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if len(valid) < 2:
        return 0.0

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    # Precompute rotation matrices and cast poses to sdf dtype/device.
    # Keyed by original index so pose arrays (scale/rotation/translation)
    # are indexed correctly even when some objects are skipped.
    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    total_penalty = 0.0
    MAX_INTERIOR_PTS = 500_000   # cap to limit memory; sample randomly if exceeded

    for vi, mesh_i in valid:
        res_i  = mesh_i.res            # e.g. 64
        sdf_i  = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)

        # ── surface points via zero-crossing interpolation along each axis ──
        # For each axis, find adjacent voxel pairs with opposite SDF signs,
        # then linearly interpolate to get sub-voxel-precise surface points.
        surface_pts = []
        for axis in range(3):
            s0 = sdf_i.narrow(axis, 0, res_i)   # all but last slice
            s1 = sdf_i.narrow(axis, 1, res_i)   # all but first slice
            cross = s0 * s1 < 0                  # sign change → zero crossing
            gi, gj, gk = torch.where(cross)      # integer grid coords of s0
            if gi.numel() == 0:
                continue
            v0 = s0[gi, gj, gk]
            v1 = s1[gi, gj, gk]
            t  = v0 / (v0 - v1)                 # interpolation factor in [0,1]
            coords = torch.stack([gi, gj, gk], dim=-1).to(dtype)
            coords[:, axis] += t                 # sub-voxel position along axis
            surface_pts.append(coords)

        if not surface_pts:
            continue
        pts_surface = torch.cat(surface_pts, dim=0)  # (N, 3)

        # Subsample to cap memory usage
        if pts_surface.shape[0] > MAX_INTERIOR_PTS:
            perm = torch.randperm(pts_surface.shape[0], device=device)[:MAX_INTERIOR_PTS]
            pts_surface = pts_surface[perm]

        # Map grid indices → local coordinates in [-0.5, 0.5]
        pts_interior = pts_surface / res_i - 0.5  # (M, 3)

        # ── forward transform: local_i  →  world ──
        # world = (local * s_i) @ R_i^T + t_i
        pts_world = (pts_interior * s_map[vi]) @ R_map[vi].T + t_map[vi]  # (M, 3)

        for vj, mesh_j in valid:
            if vi == vj:
                continue

            res_j = mesh_j.res
            sdf_j = mesh_j.sdf_d.reshape(res_j + 1, res_j + 1, res_j + 1)

            # ── inverse transform: world  →  local_j ──
            # local_j = (world - t_j) @ R_j / s_j
            pts_j = (pts_world - t_map[vj]) @ R_map[vj] / s_map[vj]   # (M, 3)

            # Discard points that fall outside j's grid  (cannot intersect)
            in_bounds = ((pts_j >= -0.5) & (pts_j <= 0.5)).all(dim=-1)  # (M,)
            if not in_bounds.any():
                continue
            pts_j_bounded = pts_j[in_bounds]   # (K, 3)

            # ── trilinear interpolation of sdf_j at pts_j_bounded ──
            # sdf_d axis layout: [dim0, dim1, dim2]  →  grid_sample axes [D, H, W]
            # grid_sample (x,y,z) maps to (W, H, D) = (dim2, dim1, dim0)
            # normalised coords: local ∈ [-0.5, 0.5]  →  grid_sample ∈ [-1, 1]  →  *2
            grid_xyz = pts_j_bounded[:, [2, 1, 0]] * 2.0          # (K, 3)  in (x,y,z) order
            grid     = grid_xyz[None, None, None, :, :]            # (1,1,1,K,3)

            sdf_j_vals = F.grid_sample(
                sdf_j[None, None].float(),   # (1, 1, D, H, W)
                grid.float(),
                mode='bilinear',             # trilinear for 5-D input
                padding_mode='border',
                align_corners=True,
            ).squeeze()                      # (K,)

            # Intersection penalty: penetration depth of i's interior into j
            penalty = F.relu(-sdf_j_vals).max().item()
            total_penalty += penalty

    # Higher reward = less intersection; negate so optimiser maximises it
    return -total_penalty
    


# ============================================================================
# GRPO Loss
# ============================================================================

def compute_grpo_loss_single(
    model: nn.Module,
    ref_model: nn.Module,
    traj: Dict,
    adv_g: float,
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
    adv_t = torch.tensor(adv_g, device=device, dtype=torch.float32)

    total_pg_val = 0.0
    total_kl_val = 0.0

    excluded = {'shape', 'translation_scale'}

    for step_idx in range(T):
        t     = traj["t_steps"][step_idx]
        dt    = traj["dt_steps"][step_idx]
        sigma = traj["sigma_steps"][step_idx]

        # Interleaved storage: x_in at even index, x_out at odd index
        x_t    = {k: v.detach() for k, v in traj["x_steps"][2 * step_idx].items()}
        x_next = {k: v.detach() for k, v in traj["x_steps"][2 * step_idx + 1].items()}
        mu_old = {k: v.detach() for k, v in traj["mu_steps"][step_idx].items()}

        t_tensor = torch.tensor([t * time_scale], device=device, dtype=torch.float32)
        d_tensor = torch.zeros(1, device=device, dtype=torch.float32)

        # ── forward (with gradient) ──
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_new = model.reverse_fn.backbone(x_t, t_tensor, cond_embed, d=d_tensor)

        # pg_loss — free mu_new immediately after sq_new is computed
        denom  = 2.0 * (sigma ** 2) * dt + 1e-12
        sq_old = sum(((x_next[k] - mu_old[k]) ** 2).sum() if k not in excluded else 0 for k in x_next)
        mu_new = sde_mu_dict(x_t, v_new, t, dt, sigma)
        sq_new = sum(((x_next[k] - mu_new[k]) ** 2).sum() if k not in excluded else 0 for k in x_next)
        del mu_new

        ratio        = torch.exp(((sq_old - sq_new) / denom).clamp(-20.0, 20.0))
        pg_unclipped = ratio * adv_t
        pg_clipped   = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_t
        pg_loss      = -torch.min(pg_unclipped, pg_clipped)

        # kl gradient via register_hook — lets us free v_ref before backward
        # d(kl)/d(v_new[k]) = 2*(v_new[k] - v_ref[k]) / numel, injected as a hook
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_ref = ref_model(x_t, t_tensor, cond_embed, d=d_tensor)

        kl_val = 0.0
        for k, v in v_new.items():
            if k in excluded:
                continue
            diff = (v.detach() - v_ref[k]).float()
            kl_val += (diff ** 2).mean().item()
            kl_grad = (2.0 * kl_coeff / (n_terms * v.numel())) * diff
            v.register_hook(lambda g, kg=kl_grad: g + kg)
        del v_ref  # free before backward — not referenced by any live graph

        pg_val = pg_loss.item()
        (pg_loss / n_terms).backward()  # pg grad + injected kl grad in one pass
        del v_new, pg_loss

        total_pg_val += pg_val
        total_kl_val += kl_val

    return total_pg_val, total_kl_val


# ============================================================================
# Training Loop (GRPO)
# ============================================================================

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
    num_scenes = 0
    current_best_reward = best_reward

    pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch}", disable=(rank != 0))

    for step, batch in enumerate(pbar):
        try:
            conditionals = to_device(batch["conditionals"], device)
            scene_num_objects = batch["num_objects"]

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

            # --- DEBUG: save input image + per-object masked images ---
            # from PIL import Image as _PILImage
            # _img = conditionals["image"]
            # if _img.dim() == 4:
            #     _img = _img[0]
            # _img_np = (_img.detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            # _masks = conditionals["object_masks"]
            # if _masks.dim() == 4:
            #     _masks = _masks[0]
            # _tiles = [_img_np]
            # for _m in _masks:
            #     _m_np = _m.detach().cpu().float().numpy()[..., None]  # (H, W, 1)
            #     _tiles.append((_img_np * _m_np).astype(np.uint8))
            # _PILImage.fromarray(np.concatenate(_tiles, axis=1)).save("./debug_image.png")
            # logger.info("[DEBUG] image saved → ./debug_image.png")
            # --- END DEBUG ---

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
            rewards = []
            valid_indices = []
            for g_idx, traj in enumerate(trajectories):
                x_1 = traj["x_steps"][-1]   # final latent dict
                shape_latent = x_1["shape"]   # (N_obj, L, 8)

                scale = torch.zeros((scene_num_objects, 1, 3), device=device)
                rotation = torch.zeros((scene_num_objects, 1, 4), device=device) # quaternion output from pose decoder
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

                # --- DEBUG: export all meshes transformed by pose as a single GLB ---
                # if len(meshes) > 0 and any(m is not None for m in meshes):
                #     import trimesh
                #     from pytorch3d.transforms import quaternion_to_matrix
                #     debug_scene = trimesh.Scene()
                #     for obj_i, mesh_result in enumerate(meshes):
                #         if mesh_result is None or not mesh_result.success:
                #             continue
                #         verts = mesh_result.vertices  # (V, 3)
                #         faces = mesh_result.faces      # (F, 3)
                #         R = quaternion_to_matrix(rotation[obj_i, 0:1])[0]  # (3, 3)
                #         s = scale[obj_i, 0]            # (3,)
                #         t = translation[obj_i, 0]      # (3,)
                #         # scale → rotate → translate  (matches compose_transform order)
                #         v = verts * s.unsqueeze(0)
                #         v = (R @ v.T).T + t.unsqueeze(0)
                #         v_np = v.detach().cpu().float().numpy()
                #         f_np = faces.detach().cpu().numpy()
                #         debug_scene.add_geometry(
                #             trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
                #         )
                #     debug_path = "./debug_mesh.glb"
                #     debug_scene.export(debug_path)
                #     logger.info(f"[DEBUG] mesh saved → {debug_path}")
                # --- END DEBUG ---

                if len(meshes) == 0:
                    rewards.append(-1.0)   # penalty for empty structure
                else:
                    try:
                        r = float(compute_reward(meshes, scale, rotation, translation))
                    except NotImplementedError:
                        raise
                    except Exception as e:
                        logger.warning(f"Reward computation failed: {e}")
                        r = -1.0
                    rewards.append(r)
                    valid_indices.append(g_idx)

                # Free SDF grids — only needed for reward, not for GRPO loss
                for m in meshes:
                    if m is not None:
                        m.sdf_d = None
                del meshes
                torch.cuda.empty_cache()

            if len(rewards) < 2:
                logger.warning("Not enough valid samples to compute advantages. Skipping.")
                continue

            # -------------------------------------------------------------
            # Update phase: per-trajectory GRPO loss with gradient accumulation
            # Keeps at most ONE trajectory's computation graph in memory.
            # -------------------------------------------------------------
            rewards_t   = torch.tensor(rewards, dtype=torch.float32, device=device)
            r_mean      = rewards_t.mean()
            r_std       = rewards_t.std().clamp(min=1e-8)
            advantages  = ((rewards_t - r_mean) / r_std).tolist()
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
            optimizer.step()

            details = {
                "pg_loss":     details_pg / n_terms,
                "kl_loss":     details_kl / n_terms,
                "total_loss":  (details_pg + kl_coeff * details_kl) / n_terms,
                "reward_mean": r_mean.item(),
                "reward_std":  r_std.item(),
            }

            if scheduler is not None:
                scheduler.step()

            total_loss_sum += details.get("total_loss", 0.0)
            total_pg_sum += details.get("pg_loss", 0.0)
            total_kl_sum += details.get("kl_loss", 0.0)
            total_reward_sum += details.get("reward_mean", 0.0)
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
                log_metrics(exp_logger, {
                    "train/loss": avg_loss,
                    "train/pg_loss": avg_pg,
                    "train/kl_loss": avg_kl,
                    "train/reward_mean": avg_r,
                    "train/lr": lr,
                    "train/epoch": epoch,
                }, global_step + step)

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
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, current_gs, step_metrics,
                str(output_dir / f"step_{current_gs:08d}.pt"),
                is_distributed,
            )
            if step_metrics["reward_mean"] > current_best_reward:
                current_best_reward = step_metrics["reward_mean"]
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, current_gs, step_metrics,
                    str(output_dir / "best.pt"),
                    is_distributed,
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


# ============================================================================
# Main
# ============================================================================

def main(args):
    # -------------------------------------------------------------------------
    # Distributed setup
    # -------------------------------------------------------------------------
    is_distributed = "RANK" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_dist(
            rank=rank,
            local_rank=local_rank,
            world_size=int(os.environ["WORLD_SIZE"]),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=os.environ.get("MASTER_PORT", "12355"),
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(output_dir / "grpo_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        shutil.copy2(args.config, output_dir / "model_config.yaml")

    exp_logger = setup_logger(
        logger_type=args.logger,
        output_dir=output_dir,
        config=vars(args),
        rank=rank,
    )

    # -------------------------------------------------------------------------
    # Build trainable SS generator model + LoRA  (optionally QLoRA)
    # -------------------------------------------------------------------------
    logger.info("Building trainable ss_generator with %s...",
                "QLoRA" if args.qlora else "LoRA")
    model, ss_preprocessor = build_model_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)

    if args.ss_generator_checkpoint:
        load_ss_generator_checkpoint(
            checkpoint_path=args.ss_generator_checkpoint,
            model=model,
            device=device,
        )

    if args.freeze_embedder:
        freeze_condition_embedder(model)

    if args.lora_checkpoint:
        # Load LoRA config (rank/alpha/target_modules) AND weights from checkpoint.
        # PeftModel.from_pretrained reads adapter_config.json automatically,
        # so --lora_rank / --lora_alpha are ignored when a checkpoint is given.
        logger.info(f"Loading pretrained LoRA from {args.lora_checkpoint}")
        raw_model = unwrap_dist(model)
        backbone = raw_model.reverse_fn.backbone
        peft_backbone = PeftModel.from_pretrained(backbone, args.lora_checkpoint)
        if not args.resume:
            raw_model.reverse_fn.backbone = peft_backbone
        logger.info("Loaded LoRA adapter (config + weights) via PeftModel.from_pretrained")
    else:
        apply_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # -------------------------------------------------------------------------
    # Reference backbone — deepcopy NOW (fp16, before any quantization).
    # This captures the exact starting-point policy for KL regularisation.
    # -------------------------------------------------------------------------
    logger.info("Creating frozen reference backbone...")
    raw_backbone = peft_backbone if args.lora_checkpoint else unwrap_dist(model).reverse_fn.backbone
    ref_backbone = copy.deepcopy(raw_backbone)
    ref_backbone.eval()
    for p in ref_backbone.parameters():
        p.requires_grad_(False)
    logger.info("Reference backbone created and frozen (fp16).")

    # -------------------------------------------------------------------------
    # QLoRA: quantize training backbone base weights to 4-bit AFTER ref copy.
    # LoRA adapters (lora_A / lora_B) remain in fp16/bf16.
    # -------------------------------------------------------------------------
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        logger.info("Applying QLoRA %d-bit quantization to training backbone...", args.qlora_bits)
        train_backbone = unwrap_dist(model).reverse_fn.backbone
        quantize_backbone_4bit(
            train_backbone,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            double_quant=True,
        )
        # Move back to GPU to materialise the Params4bit buffers, then prepare
        train_backbone.to(device)
        prepare_model_for_kbit_training(train_backbone, use_gradient_checkpointing=False)
        logger.info("QLoRA quantization complete.")

        model.reverse_fn.backbone = train_backbone

    freeze_non_lora_params(model)

    # -------------------------------------------------------------------------
    # Gradient checkpointing (reduces peak activation memory during backward)
    # -------------------------------------------------------------------------
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(unwrap_dist(model))

    # -------------------------------------------------------------------------
    # Load frozen downstream models
    # Mirrors inference_pipeline.py's init_* methods exactly.
    # -------------------------------------------------------------------------
    logger.info("Loading frozen downstream models...")

    # ss_decoder: direct checkpoint, no prefix stripping (state_dict_key=None)
    ss_decoder = load_ss_decoder(
        args.ss_decoder_config, args.ss_decoder_checkpoint, device
    )

    # slat_generator: config["module"]["generator"]["backbone"] + prefix "_base_models.generator."
    slat_generator = load_slat_generator(
        args.slat_generator_config, args.slat_generator_checkpoint, device
    )

    # slat_decoder_mesh: direct checkpoint, no prefix stripping
    slat_decoder_mesh = load_slat_decoder_mesh(
        args.slat_decoder_mesh_config, args.slat_decoder_mesh_checkpoint, device
    )

    # slat_condition_embedder: config["module"]["condition_embedder"]["backbone"]
    #   + prefix "_base_models.condition_embedder."  (from slat generator ckpt)
    logger.info("Loading SLAT condition embedder...")
    slat_condition_embedder = load_condition_embedder(
        args.slat_generator_config, args.slat_generator_checkpoint, device
    )

    # slat_preprocessor: config["tdfy"]["val_preprocessor"]  (from slat generator config)
    logger.info("Loading SLAT preprocessor...")
    slat_preprocessor = load_slat_preprocessor(args.pipeline_config)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    logger.info("Creating dataset...")
    train_dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=args.max_objects_per_scene,
        load_images=True,
        load_depth=True,
        load_masks=True,
        image_size=(args.image_height, args.image_width) if args.image_width > 0 else None,
        precomputed_latents=False,   # GRPO needs raw images for decoding
        num_renders_per_scene=args.num_renders_per_scene,
        gso_root=args.gso_root or None,
        load_meshes=args.load_meshes,
    )
    logger.info(f"Dataset size: {len(train_dataset)} scenes")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------------------------------------------
    # DDP
    # -------------------------------------------------------------------------
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_params,
        )

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    param_groups = get_parameter_groups(model, args.learning_rate, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_reward = -float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_reward = ckpt.get("metrics", {}).get("reward_mean", -float("inf"))

        raw_model = unwrap_dist(model)
        backbone = raw_model.reverse_fn.backbone
        peft_backbone = PeftModel.from_pretrained(backbone, args.resume.replace(".pt", "_peft"), device=device)
        raw_model.reverse_fn.backbone = peft_backbone

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    assert args.t_sde_steps < args.t_train_steps, (
        f"--t_sde_steps ({args.t_sde_steps}) must be strictly less than "
        f"--t_train_steps ({args.t_train_steps})"
    )

    logger.info(
        f"Starting Flow-GRPO-Fast: {args.num_epochs} epochs, "
        f"G={args.group_size}, T_train={args.t_train_steps}, T_sde={args.t_sde_steps}, "
        f"sde_a={args.sde_a}, kl_coeff={args.kl_coeff}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_metrics, best_reward = train_epoch_grpo(
            model=model,
            ref_backbone=ref_backbone,
            ss_decoder=ss_decoder,
            slat_generator=slat_generator,
            slat_decoder_mesh=slat_decoder_mesh,
            slat_preprocessor=slat_preprocessor,
            slat_condition_embedder=slat_condition_embedder,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            ss_preprocessor=ss_preprocessor,
            G=args.group_size,
            T_train=args.t_train_steps,
            T_sde=args.t_sde_steps,
            sde_a=args.sde_a,
            clip_epsilon=args.clip_epsilon,
            kl_coeff=args.kl_coeff,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            scheduler=scheduler,
            exp_logger=exp_logger if rank == 0 else None,
            global_step=global_step,
            save_interval_steps=args.save_interval_steps,
            output_dir=output_dir,
            is_distributed=is_distributed,
            best_reward=best_reward,
            cfg_strength=args.cfg_strength,
        )
        global_step += len(train_loader)

        if rank == 0:
            logger.info(
                f"Epoch {epoch}: loss={epoch_metrics['loss']:.4f}, "
                f"pg={epoch_metrics['pg_loss']:.4f}, "
                f"kl={epoch_metrics['kl_loss']:.4f}, "
                f"reward={epoch_metrics['reward_mean']:.4f}"
            )
            if exp_logger is not None:
                log_metrics(exp_logger, {
                    "epoch/loss": epoch_metrics["loss"],
                    "epoch/reward_mean": epoch_metrics["reward_mean"],
                }, global_step)

            save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step, epoch_metrics,
                str(output_dir / "latest.pt"),
                is_distributed,
            )

        if args.save_interval_epochs > 0 and (epoch + 1) % args.save_interval_epochs == 0:
            if rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, epoch_metrics,
                    str(output_dir / f"epoch_{epoch:04d}.pt"),
                    is_distributed,
                )

    logger.info("Flow-GRPO training completed.")
    if rank == 0 and exp_logger is not None:
        if hasattr(exp_logger, "finish"):
            exp_logger.finish()
        elif hasattr(exp_logger, "close"):
            exp_logger.close()

    if is_distributed:
        dist.destroy_process_group()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flow-GRPO RL finetuning of MidiSparseStructureFlowTdfyWrapper (LoRA)"
    )

    # ---- SS generator (trainable) ----
    parser.add_argument("--config", type=str, required=True,
                        help="SS generator config YAML")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--ss_generator_checkpoint", type=str, default=None,
                        help="Pretrained ss_generator.ckpt (base weights)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to a PEFT adapter directory from train_midi_lora "
                             "(e.g. ./outputs/midi_lora/latest_peft)")

    # ---- LoRA / QLoRA ----
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--qlora", action="store_true", default=False,
                        help="Enable QLoRA: quantize base model weights to 4-bit (NF4) "
                             "via bitsandbytes. LoRA adapters remain in fp16. "
                             "Requires: pip install bitsandbytes>=0.41")
    parser.add_argument("--qlora_bits", type=int, default=4, choices=[4],
                        help="Quantization bit-width for QLoRA (currently only 4 supported)")

    # ---- Frozen downstream models ----
    parser.add_argument("--ss_decoder_config", type=str, required=True)
    parser.add_argument("--ss_decoder_checkpoint", type=str, required=True)
    parser.add_argument("--slat_generator_config", type=str, required=True)
    parser.add_argument("--slat_generator_checkpoint", type=str, required=True)
    parser.add_argument("--slat_decoder_mesh_config", type=str, required=True)
    parser.add_argument("--slat_decoder_mesh_checkpoint", type=str, required=True)
    parser.add_argument("--pipeline_config", type=str, required=True)
    # NOTE: slat_preprocessor and slat_condition_embedder are loaded directly
    # from slat_generator_config/checkpoint – no separate pipeline_yaml needed.

    # ---- Data ----
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--gso_root", type=str, default=None)
    parser.add_argument("--max_objects_per_scene", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=0)
    parser.add_argument("--num_renders_per_scene", type=int, default=1)
    parser.add_argument("--load_meshes", action="store_true")

    # ---- Flow-GRPO-Fast hyperparameters ----
    parser.add_argument("--group_size", type=int, default=8,
                        help="Number of trajectories per input (G)")
    parser.add_argument("--t_train_steps", type=int, default=10,
                        help="Total denoising steps for the full trajectory (T_train)")
    parser.add_argument("--t_sde_steps", type=int, default=2,
                        help="Number of SDE steps randomly sampled for training (T_sde). "
                             "Must be strictly less than --t_train_steps. "
                             "Remaining steps use deterministic ODE sampling.")
    parser.add_argument("--sde_a", type=float, default=0.2,
                        help="SDE noise amplitude: σ_t = a·√((1−t)/t)")
    parser.add_argument("--kl_coeff", type=float, default=0.04,
                        help="KL divergence penalty coefficient (β)")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--cfg_strength", type=float, default=7.0,
                        help="CFG strength during SDE generation")

    # ---- Memory optimizations ----
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing on transformer blocks. "
                             "Reduces peak activation memory during backward at the cost "
                             "of ~2x more compute (activations are recomputed on backward).")

    # ---- Optimizer ----
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--freeze_embedder", action="store_true", default=True)
    parser.add_argument("--no_freeze_embedder", dest="freeze_embedder", action="store_false")
    parser.add_argument("--find_unused_params", action="store_true")

    # ---- Logging & checkpoints ----
    parser.add_argument("--logger", type=str, default="wandb",
                        choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval_steps", type=int, default=50)
    parser.add_argument("--save_interval_epochs", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./outputs/flow_grpo_fp")

    # ---- Resume ----
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)