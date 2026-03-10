"""
Training script for MidiSparseStructureFlowTdfyWrapper with LoRA finetuning.

Uses PEFT (HuggingFace) for LoRA adaptation. All attention projection layers
(self-attn and cross-attn) are targeted. The model is initialized from
ss_generator.ckpt (structurally compatible with the Midi backbone).

Usage:
    # Single GPU
    python train_midi_lora_foundationpose.py \
        --config checkpoints/hf/midi_ss_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
        --output_dir ./outputs/midi_lora_fp

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 train_midi_lora_foundationpose.py \
        --config checkpoints/hf/midi_ss_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
        --output_dir ./outputs/midi_lora_fp

    # Resume from checkpoint
    python train_midi_lora_foundationpose.py \
        --config checkpoints/hf/midi_ss_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
        --output_dir ./outputs/midi_lora_fp \
        --resume ./outputs/midi_lora_fp/latest.pt
"""

import os
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf
from hydra.utils import instantiate
from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel

from foundation_pose_dataset import FoundationPoseDataset, collate_fn
from sam3d_objects.data.utils import to_device
from sam3d_objects.utils.dist_utils import setup_dist, unwrap_dist

from train_dual_backbone_foundationpose import (
    setup_logger,
    log_metrics,
    prepare_conditioning_for_scene,
    get_condition_input,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Building
# ============================================================================

def _instantiate_latent_mapping(latent_mapping_cfg: dict) -> dict:
    """Instantiate latent_mapping entries that contain _target_ (Hydra-style)."""
    result = {}
    for key, val in latent_mapping_cfg.items():
        if isinstance(val, dict) and "_target_" in val:
            result[key] = instantiate(val)
        else:
            result[key] = val
    return result


def build_model_from_config(
    config_path: str,
    checkpoint_dir: Optional[str] = None,
) -> tuple[nn.Module, Any]:
    """
    Build the ShortCut model with MidiSparseStructureFlowTdfyWrapper backbone.

    The YAML config structure:
        module:
          condition_embedder: ...
          generator:
            backbone:  # ShortCut
              reverse_fn:  # ClassifierFreeGuidance
                backbone:  # MidiSparseStructureFlowTdfyWrapper

    Returns:
        (model, preprocessor)
    """
    from sam3d_objects.config.utils import locate
    from functools import partial

    config = OmegaConf.load(config_path)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(config_path)

    # --- 1. Condition embedder ---
    logger.info("Building condition embedder...")
    condition_embedder = instantiate(config.module.condition_embedder.backbone)

    # --- 2. MidiSparseStructureFlowTdfyWrapper ---
    reverse_fn_cfg = config.module.generator.backbone.reverse_fn
    backbone_plain = OmegaConf.to_container(reverse_fn_cfg.backbone, resolve=True)
    backbone_target = backbone_plain.pop("_target_")

    if "latent_mapping" in backbone_plain:
        backbone_plain["latent_mapping"] = _instantiate_latent_mapping(
            backbone_plain["latent_mapping"]
        )

    logger.info("Building MidiSparseStructureFlowTdfyWrapper...")
    BackboneClass = locate(backbone_target)
    backbone = BackboneClass(**backbone_plain)

    # --- 3. ClassifierFreeGuidance wrapper ---
    logger.info("Building ClassifierFreeGuidance wrapper...")
    cfg_plain = OmegaConf.to_container(reverse_fn_cfg, resolve=True)
    cfg_target = cfg_plain.pop("_target_")
    cfg_plain.pop("backbone")
    CFGClass = locate(cfg_target)
    reverse_fn = CFGClass(backbone=backbone, **cfg_plain)

    # --- 4. ShortCut ---
    logger.info("Building ShortCut model...")
    shortcut_plain = OmegaConf.to_container(config.module.generator.backbone, resolve=True)
    shortcut_target = shortcut_plain.pop("_target_")
    shortcut_plain.pop("reverse_fn")

    loss_weights = shortcut_plain.get("loss_weights", {})
    if isinstance(loss_weights, dict) and "_target_" in loss_weights:
        lw_target = loss_weights.pop("_target_")
        shortcut_plain["loss_weights"] = locate(lw_target)(**loss_weights)

    sampler_cfg = shortcut_plain.pop("training_time_sampler_fn", None)
    if sampler_cfg is not None and isinstance(sampler_cfg, dict):
        sampler_target = sampler_cfg.pop("_target_", None)
        sampler_cfg.pop("_partial_", None)
        if sampler_target:
            shortcut_plain["training_time_sampler_fn"] = partial(
                locate(sampler_target), **sampler_cfg
            )

    ShortCutClass = locate(shortcut_target)
    model = ShortCutClass(reverse_fn=reverse_fn, **shortcut_plain)
    model.condition_embedder = condition_embedder

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # --- 5. Preprocessor ---
    logger.info("Loading preprocessor...")
    try:
        preprocessor_path = Path(checkpoint_dir) / "pipeline.yaml"
        if not preprocessor_path.exists():
            preprocessor_path = Path("checkpoints/hf/pipeline.yaml")
        preprocessor = instantiate(OmegaConf.load(preprocessor_path)["ss_preprocessor"])
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
    except Exception as e:
        logger.warning(f"Failed to load preprocessor: {e}")
        preprocessor = None

    return model, preprocessor


# ============================================================================
# Pretrained Weight Loading
# ============================================================================

def load_ss_generator_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """
    Load ss_generator.ckpt weights into the model (before LoRA is applied).

    SparseStructureFlowTdfyWrapper and MidiSparseStructureFlowTdfyWrapper share
    the same parameter names, so weights map directly (strict=False handles
    the structural difference in global blocks).

    Args:
        checkpoint_path: Path to ss_generator.ckpt
        model: ShortCut model (must be called BEFORE apply_lora)
        device: Device to load onto
    """
    logger.info(f"Loading ss_generator checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    from sam3d_objects.model.io import filter_and_remove_prefix_state_dict_fn

    # Load backbone (reverse_fn.backbone.*)
    backbone_sd = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")(state_dict)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    logger.info(
        f"Backbone weights: {len(missing)} missing, {len(unexpected)} unexpected keys"
    )
    if missing[:3]:
        logger.info(f"  Example missing: {missing[:3]}")

    # Load condition embedder
    embedder_sd = filter_and_remove_prefix_state_dict_fn(
        "_base_models.condition_embedder."
    )(state_dict)
    if embedder_sd:
        model.condition_embedder.load_state_dict(embedder_sd, strict=True)
        logger.info("Loaded condition embedder weights")


# ============================================================================
# LoRA Setup via PEFT
# ============================================================================

def apply_lora(
    model: nn.Module,
    lora_rank: int,
    lora_alpha: float,
) -> None:
    """
    Apply LoRA adapters to the MidiSparseStructureFlowTdfyWrapper backbone
    using PEFT. Must be called AFTER loading pretrained weights.

    Target modules (regex fullmatch on full module path):
      - self_attn.to_qkv.{latent}  (self-attention QKV projections)
      - self_attn.to_out.{latent}  (self-attention output projections)
      - cross_attn.{latent}.to_q   (cross-attention Q projections)
      - cross_attn.{latent}.to_kv  (cross-attention KV projections)
      - cross_attn.{latent}.to_out (cross-attention output projections)

    Using a regex avoids accidentally matching norm layers whose ModuleDict
    keys share the same latent names.

    After this call, model.reverse_fn.backbone is a PeftModel whose forward
    is transparent to the caller (passes through all args/kwargs unchanged).
    """
    backbone = model.reverse_fn.backbone  # MidiSparseStructureFlowTdfyWrapper

    # Regex fullmatch on the full module path from backbone.named_modules().
    # Matches only the nn.Linear projection layers inside attention blocks,
    # excluding norm layers that share the same latent-name ModuleDict keys.
    target_modules = (
        r"blocks\.\d+\."
        r"(self_attn\.(to_qkv|to_out)\.[^.]+|"
        r"cross_attn\.[^.]+\.(to_q|to_kv|to_out))"
    )
    logger.info(f"LoRA target_modules regex: {target_modules}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )

    peft_backbone = get_peft_model(backbone, lora_config)
    peft_backbone.print_trainable_parameters()

    # Replace backbone in the CFG wrapper
    model.reverse_fn.backbone = peft_backbone


def freeze_condition_embedder(model: nn.Module):
    """Freeze condition embedder parameters."""
    count = 0
    for param in model.condition_embedder.parameters():
        param.requires_grad = False
        count += param.numel()
    logger.info(f"Frozen condition embedder: {count:,} parameters")


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all parameters outside the LoRA backbone.
    (ShortCut-level params, condition embedder, etc.)
    PEFT already freezes backbone params that are not LoRA adapters.
    """
    for name, param in model.named_parameters():
        # LoRA adapter params are named with "lora_A" or "lora_B" in PEFT
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif param.requires_grad:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable: {trainable:,} / {total:,} total parameters")


def get_parameter_groups(model: nn.Module, lr: float, weight_decay: float):
    """Create optimizer parameter groups (trainable params only)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    logger.info(f"Param groups: {len(decay)} decay, {len(no_decay)} no-decay")
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


# ============================================================================
# Checkpoint Saving / Loading
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    save_path: str,
    is_distributed: bool = False,
    scaler: Optional[GradScaler] = None,
):
    """Save LoRA training checkpoint (trainable params + optimizer state)."""
    if is_distributed and dist.get_rank() != 0:
        return

    raw_model = unwrap_dist(model)

    # Save only trainable parameters (LoRA adapters)
    # trainable_state = {
    #     name: param.detach().cpu()
    #     for name, param in raw_model.named_parameters()
    #     if param.requires_grad
    # }

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        # "lora_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(
        f"Saved LoRA checkpoint to {save_path}"
    )

    raw_model.reverse_fn.backbone.save_pretrained(save_path.replace(".pt", "_peft"))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load LoRA training checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_model = unwrap_dist(model)

    # lora_sd = ckpt.get("lora_state_dict", ckpt.get("model_state_dict", {}))
    # missing, unexpected = raw_model.load_state_dict(lora_sd, strict=True)
    # logger.info(
    #     f"Loaded LoRA state: {len(missing)} missing, {len(unexpected)} unexpected"
    # )

    # TODO: load peft model from checkpoint_path.replace(".pt", "_peft")
    peft_dir = checkpoint_path.replace(".pt", "_peft")
    peft_dir_path = Path(peft_dir)

    if peft_dir_path.exists() and peft_dir_path.is_dir():
        backbone = raw_model.reverse_fn.backbone

        if not isinstance(backbone, PeftModel):
            raise RuntimeError(
                "Expected raw_model.reverse_fn.backbone to be a PeftModel. "
                "Make sure apply_lora() is called BEFORE resume."
            )

        # --- load adapter state dict saved by PEFT ---
        # PEFT may save either:
        #   - adapter_model.safetensors
        #   - adapter_model.bin
        #   - pytorch_model.bin (older)
        adapter_files = [
            peft_dir_path / "adapter_model.safetensors",
            peft_dir_path / "adapter_model.bin",
            peft_dir_path / "pytorch_model.bin",
        ]

        state_dict = None
        for f in adapter_files:
            if f.exists():
                if f.suffix == ".safetensors":
                    try:
                        from safetensors.torch import load_file as safe_load_file
                    except Exception as e:
                        raise RuntimeError(
                            f"Found {f.name} but safetensors is not available: {e}"
                        )
                    state_dict = safe_load_file(str(f), device=str(device))
                else:
                    state_dict = torch.load(str(f), map_location=device)
                logger.info(f"Loading PEFT adapter weights from {f}")
                break

        if state_dict is None:
            raise FileNotFoundError(
                f"PEFT directory exists but no adapter weight file found in {peft_dir_path}"
            )

        # Preferred: use PEFT helper to correctly map keys into the adapter
        loaded = False
        try:
            # PEFT versions differ in import locations / signatures
            from peft.utils.save_and_load import set_peft_model_state_dict

            # Some versions accept (model, state_dict, adapter_name=...)
            try:
                set_peft_model_state_dict(backbone, state_dict, adapter_name="default")
            except TypeError:
                # Older versions: (model, state_dict)
                set_peft_model_state_dict(backbone, state_dict)
            loaded = True
        except Exception as e:
            logger.warning(f"set_peft_model_state_dict failed ({e}); falling back to load_state_dict")

        if not loaded:
            # Fallback: direct load into PeftModel (usually works, but may be less robust across versions)
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            logger.info(
                f"PEFT fallback load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected"
            )

        # Make sure on correct device (typically already is, but safe)
        backbone.to(device)
        logger.info(f"Loaded PEFT adapters from {peft_dir_path}")
    else:
        logger.warning(
            f"PEFT adapter directory not found ({peft_dir}). "
            "Skipping LoRA adapter weight loading."
        )

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    logger.info(
        f"Resumed from epoch {ckpt.get('epoch', '?')}, "
        f"step {ckpt.get('global_step', '?')}"
    )
    return ckpt

# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    preprocessor: Any,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    exp_logger: Optional[Any] = None,
    global_step: int = 0,
    save_interval_steps: int = 0,
    output_dir: Optional[Path] = None,
    is_distributed: bool = False,
    best_loss: float = float("inf"),
) -> tuple[Dict[str, float], float]:
    """Train for one epoch."""
    model.train()
    embedder = model.condition_embedder

    total_loss = 0.0
    total_fm_loss = 0.0
    total_sc_loss = 0.0
    num_scenes = 0
    num_objects = 0
    current_best_loss = best_loss

    is_distributed_runtime = dist.is_initialized()
    rank = dist.get_rank() if is_distributed_runtime else 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))

    for step, batch in enumerate(pbar):
        try:
            latents = to_device(batch["latents"], device)
            conditionals = to_device(batch["conditionals"], device)
            scene_num_objects = batch["num_objects"]

            optimizer.zero_grad(set_to_none=True)

            # Prepare latents
            latents["scale"] = latents["scale"].unsqueeze(1)
            latents["6drotation_normalized"] = latents["6drotation_normalized"].unsqueeze(1)
            latents["translation"] = latents["translation"].unsqueeze(1)
            latents["translation_scale"] = torch.ones(
                latents["scale"].shape[0], 1, 1, device=device
            )
            latents["shape"] = (
                latents["shape"].reshape(-1, 8, 16 ** 3).permute(0, 2, 1).contiguous()
            )

            # Prepare conditioning
            input_dicts = prepare_conditioning_for_scene(
                image=conditionals["image"],
                pointmap=conditionals["pointmap"],
                object_masks=conditionals["object_masks"],
                preprocessor=preprocessor,
                device=device,
            )

            pointmap_scales = torch.stack(
                [d["pointmap_scale"] for d in input_dicts], dim=0
            ).to(device)
            pointmap_shifts = torch.stack(
                [d["pointmap_shift"] for d in input_dicts], dim=0
            ).to(device)

            latents["scale"] = torch.log(latents["scale"] / pointmap_scales)
            latents["translation"] = (
                latents["translation"] - pointmap_shifts
            ) / pointmap_scales

            cond = torch.cat(
                [get_condition_input(embedder, d, [])[0][0] for d in input_dicts],
                dim=0,
            )

            if use_amp:
                with autocast(dtype=torch.float16):
                    loss, detail_losses = model.loss(latents, cond)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, detail_losses = model.loss(latents, cond)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss at step {step} (epoch {epoch}), skipping"
                    )
                    if scheduler is not None:
                        scheduler.step()
                    raise ValueError("NaN or Inf loss encountered")

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_fm_loss += detail_losses["flow_matching_loss"].item()
            total_sc_loss += detail_losses["self_consistency_loss"].item()
            num_scenes += 1
            num_objects += scene_num_objects

        except Exception as e:
            logger.error(f"Error at step {step} (epoch {epoch}): {e}")
            import traceback
            logger.error(traceback.format_exc())

        try:
            if num_scenes > 0 and (step + 1) % log_interval == 0:
                avg_loss = total_loss / num_scenes
                avg_fm = total_fm_loss / num_scenes
                avg_sc = total_sc_loss / num_scenes
                avg_objs = num_objects / num_scenes
                current_lr = optimizer.param_groups[0]["lr"]

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "fm": f"{avg_fm:.4f}",
                    "sc": f"{avg_sc:.4f}",
                    "objs": f"{avg_objs:.1f}",
                    "lr": f"{current_lr:.2e}",
                })

                if exp_logger is not None and rank == 0:
                    log_metrics(exp_logger, {
                        "train/loss": avg_loss,
                        "train/flow_matching_loss": avg_fm,
                        "train/self_consistency_loss": avg_sc,
                        "train/avg_objects_per_scene": avg_objs,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                    }, global_step + step)

            # Step-based checkpoint
            current_global_step = global_step + step + 1
            if (
                save_interval_steps > 0
                and current_global_step % save_interval_steps == 0
                and rank == 0
                and output_dir is not None
                and num_scenes > 0
            ):
                step_metrics = {
                    "loss": total_loss / num_scenes,
                    "flow_matching_loss": total_fm_loss / num_scenes,
                    "self_consistency_loss": total_sc_loss / num_scenes,
                }
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, current_global_step, step_metrics,
                    str(output_dir / f"step_{current_global_step:08d}.pt"),
                    is_distributed, scaler,
                )
                if step_metrics["loss"] < current_best_loss:
                    current_best_loss = step_metrics["loss"]
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, current_global_step, step_metrics,
                        str(output_dir / "best.pt"),
                        is_distributed, scaler,
                    )
                    logger.info(f"  New best loss: {current_best_loss:.4f}")

        except Exception as e:
            logger.error(f"Error during logging/checkpointing at step {step}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    metrics = {
        "loss": total_loss / max(num_scenes, 1),
        "flow_matching_loss": total_fm_loss / max(num_scenes, 1),
        "self_consistency_loss": total_sc_loss / max(num_scenes, 1),
        "avg_objects_per_scene": num_objects / max(num_scenes, 1),
    }

    if is_distributed_runtime:
        for key in metrics:
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()

    return metrics, current_best_loss


# ============================================================================
# Main
# ============================================================================

def main(args):
    # --- Distributed setup ---
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

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        shutil.copy2(args.config, output_dir / "model_config.yaml")
        logger.info(f"Output directory: {output_dir}")

    # --- Experiment logger ---
    exp_logger = setup_logger(
        logger_type=args.logger,
        output_dir=output_dir,
        config=vars(args),
        rank=rank,
    )

    # --- Build model ---
    logger.info("Building model...")
    model, preprocessor = build_model_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)

    # --- Load pretrained weights (BEFORE applying LoRA) ---
    if args.ss_generator_checkpoint:
        load_ss_generator_checkpoint(
            checkpoint_path=args.ss_generator_checkpoint,
            model=model,
            device=device,
        )

    # --- Freeze condition embedder ---
    if args.freeze_embedder:
        freeze_condition_embedder(model)

    # --- Apply PEFT LoRA to backbone ---
    # Must be done AFTER loading pretrained weights so we load into the
    # original nn.Linear layers (before PEFT replaces them with LoRA layers).
    apply_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # Freeze all remaining non-LoRA params (ShortCut-level, etc.)
    freeze_non_lora_params(model)

    # --- Dataset ---
    logger.info("Creating dataset...")
    train_dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=args.max_objects_per_scene,
        load_images=True,
        load_depth=True,
        load_masks=True,
        image_size=(args.image_height, args.image_width) if args.image_width > 0 else None,
        precomputed_latents=args.precomputed_latents,
        num_renders_per_scene=args.num_renders_per_scene,
        gso_root=args.gso_root or None,
        load_meshes=args.load_meshes,
    )
    logger.info(f"Train dataset: {len(train_dataset)} scenes")

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

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

    # --- DDP ---
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_params,
        )

    # --- Optimizer & Scheduler ---
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
    scaler = GradScaler() if args.use_amp else None

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))

    # --- Training ---
    logger.info(
        f"Starting training: {args.num_epochs} epochs, "
        f"{len(train_loader)} steps/epoch, warmup={warmup_steps} steps"
    )

    for epoch in range(start_epoch, args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_metrics, best_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            preprocessor=preprocessor,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            use_amp=args.use_amp,
            scaler=scaler,
            scheduler=scheduler,
            exp_logger=exp_logger,
            global_step=global_step,
            save_interval_steps=args.save_interval_steps,
            output_dir=output_dir,
            is_distributed=is_distributed,
            best_loss=best_loss,
        )
        global_step += len(train_loader)

        if rank == 0:
            logger.info(
                f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                f"fm={train_metrics['flow_matching_loss']:.4f}, "
                f"sc={train_metrics['self_consistency_loss']:.4f}, "
                f"objs={train_metrics['avg_objects_per_scene']:.1f}"
            )
            if exp_logger is not None:
                log_metrics(exp_logger, {
                    "epoch/loss": train_metrics["loss"],
                    "epoch/flow_matching_loss": train_metrics["flow_matching_loss"],
                    "epoch/self_consistency_loss": train_metrics["self_consistency_loss"],
                }, global_step)

            # Save latest
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step, train_metrics,
                str(output_dir / "latest.pt"),
                is_distributed, scaler,
            )

        # Epoch-based periodic checkpoint
        if args.save_interval_epochs > 0 and (epoch + 1) % args.save_interval_epochs == 0:
            if rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, train_metrics,
                    str(output_dir / f"epoch_{epoch:04d}.pt"),
                    is_distributed, scaler,
                )

    logger.info("Training completed!")

    if rank == 0 and exp_logger is not None:
        if hasattr(exp_logger, "finish"):
            exp_logger.finish()
        elif hasattr(exp_logger, "close"):
            exp_logger.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA finetuning of MidiSparseStructureFlowTdfyWrapper via PEFT"
    )

    # Config / Checkpoints
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML (e.g., checkpoints/hf/midi_ss_generator.yaml)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Base directory for pretrained checkpoints. "
                             "Defaults to the config file's directory.")
    parser.add_argument("--ss_generator_checkpoint", type=str, default=None,
                        help="Path to ss_generator.ckpt for backbone initialization")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha; effective scaling = alpha / rank (default: 16.0)")

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--gso_root", type=str, default=None)
    parser.add_argument("--max_objects_per_scene", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=0)
    parser.add_argument("--precomputed_latents", action="store_true")
    parser.add_argument("--num_renders_per_scene", type=int, default=1)
    parser.add_argument("--load_meshes", action="store_true")

    # Training hyperparams
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--freeze_embedder", action="store_true", default=True)
    parser.add_argument("--no_freeze_embedder", dest="freeze_embedder", action="store_false")
    parser.add_argument("--find_unused_params", action="store_true")

    # Logging & checkpoints
    parser.add_argument("--logger", type=str, default="wandb",
                        choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval_steps", type=int, default=100)
    parser.add_argument("--save_interval_epochs", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./outputs/midi_lora_foundationpose")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
