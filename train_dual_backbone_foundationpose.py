"""
Training script for DualBackbone ShortCut Flow Model with FoundationPose Dataset.

Trains the ShortCut model using DualBackboneSparseStructureFlowTdfyWrapper
as the denoising backbone on FoundationPose synthetic scenes.

Usage:
    # Single GPU
    python train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --output_dir ./outputs/dual_backbone_fp

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --output_dir ./outputs/dual_backbone_fp

    # Resume from checkpoint
    python train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root /path/to/foundationpose_data \
        --output_dir ./outputs/dual_backbone_fp \
        --resume ./outputs/dual_backbone_fp/latest.pt
"""

import os
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

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

from foundation_pose_dataset import FoundationPoseDataset, collate_fn
from sam3d_objects.data.utils import to_device
from sam3d_objects.utils.dist_utils import setup_dist, unwrap_dist


def setup_logger(logger_type: str, output_dir: Path, config: dict, rank: int = 0):
    """
    Setup experiment tracking logger (wandb or tensorboard).

    Args:
        logger_type: 'wandb', 'tensorboard', or 'none'
        output_dir: Output directory for logs
        config: Configuration dict to log
        rank: Process rank (only rank 0 should log)

    Returns:
        Logger instance or None
    """
    if rank != 0 or logger_type == 'none':
        return None

    if logger_type == 'wandb':
        try:
            import wandb
            wandb.init(
                project="sam3d-dual-backbone",
                name=output_dir.name,
                config=config,
                dir=str(output_dir),
                resume='allow',
            )
            logger.info("Initialized Weights & Biases logging")
            return wandb
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return None

    elif logger_type == 'tensorboard':
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = output_dir / 'tensorboard'
            log_dir.mkdir(exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"Initialized TensorBoard logging at {log_dir}")
            return writer
        except ImportError:
            logger.warning("tensorboard not installed. Install with: pip install tensorboard")
            return None

    return None


def log_metrics(exp_logger, metrics: Dict[str, float], step: int, prefix: str = ""):
    """
    Log metrics to the experiment logger.

    Args:
        exp_logger: wandb or tensorboard writer instance
        metrics: Dictionary of metrics to log
        step: Global step number
        prefix: Optional prefix for metric names
    """
    if exp_logger is None:
        return

    # Wandb
    if hasattr(exp_logger, 'log'):
        log_dict = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        log_dict['step'] = step
        exp_logger.log(log_dict, step=step)

    # TensorBoard
    elif hasattr(exp_logger, 'add_scalar'):
        for key, value in metrics.items():
            tag = f"{prefix}{key}" if prefix else key
            exp_logger.add_scalar(tag, value, step)


def merge_image_and_mask(image: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Merge RGB image with mask to create RGBA image.

    Args:
        image: [H, W, 3] torch tensor
        mask: [H, W] torch tensor

    Returns:
        RGBA image as numpy array [H, W, 4] in uint8 format
    """
    image_np = (image.cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # [H, W]
    rgba = np.concatenate([image_np, mask_np[..., None]], axis=-1)  # [H, W, 4]
    return rgba


def prepare_conditioning_for_scene(
    image: torch.Tensor,
    pointmap: torch.Tensor,
    object_masks: torch.Tensor,
    preprocessor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare conditioning inputs for all objects in a scene.

    This follows the pattern from Sam3DConditionedMixin.preprocess_image() and
    inference_pipeline_pointmap.preprocess_image().

    Args:
        image: Scene RGB image [H, W, 3], torch tensor, values in [0,1]
        pointmap: Scene pointmap [H, W, 3], torch tensor
        object_masks: Per-object masks [num_objects, H, W], torch tensor, values in [0,1]
        preprocessor: The preprocessor instance (from the model config)
        device: Target device

    Returns:
        Dictionary with preprocessed conditioning ready for the condition embedder
    """
    num_objects = object_masks.shape[0]

    # For scene-level batching, all objects share the same image/pointmap
    # but each has its own mask

    # Convert image to correct format for preprocessor
    # Preprocessor expects numpy RGBA in uint8 [0, 255]
    rgba_images = []
    for i in range(num_objects):
        rgba = merge_image_and_mask(image, object_masks[i])
        rgba_images.append(rgba)

    # Convert to tensors and preprocess
    # Following the pattern from preprocess_image in Sam3DConditionedMixin

    # Convert pointmap to expected format [3, H, W]
    pointmap_chw = pointmap.permute(2, 0, 1).contiguous()  # [3, H, W]

    # Process each object's RGBA image through the preprocessor
    batch_items = []
    for rgba in rgba_images:
        # Convert RGBA to torch tensor
        rgba_tensor = torch.from_numpy(rgba.astype(np.float32) / 255.0)  # [H, W, 4]
        rgba_tensor = rgba_tensor.permute(2, 0, 1).contiguous()  # [4, H, W]

        rgb_image = rgba_tensor[:3]  # [3, H, W]
        rgb_image_mask = (rgba_tensor[3:4] > 0).float()  # [1, H, W]

        # Call preprocessor's internal method
        preprocessor_return = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap_chw
        )

        # Build item dict following inference_pipeline pattern
        item = {
            "mask": preprocessor_return["mask"].unsqueeze(0).to(device),  # [1, ...]
            "image": preprocessor_return["image"].unsqueeze(0).to(device),  # [1, ...]
            "rgb_image": preprocessor_return["rgb_image"].unsqueeze(0).to(device),  # [1, ...]
            "rgb_image_mask": preprocessor_return["rgb_image_mask"].unsqueeze(0).to(device),  # [1, ...]
        }

        # Add pointmap-related items if preprocessor processes pointmaps
        if pointmap_chw is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = preprocessor_return["pointmap"].unsqueeze(0).to(device)
            item["rgb_pointmap"] = preprocessor_return["rgb_pointmap"].unsqueeze(0).to(device)
            item["pointmap_scale"] = preprocessor_return.get("pointmap_scale", torch.tensor([1.0])).unsqueeze(0).to(device)
            item["pointmap_shift"] = preprocessor_return.get("pointmap_shift", torch.tensor([0.0, 0.0, 0.0])).unsqueeze(0).to(device)
            item["rgb_pointmap_scale"] = preprocessor_return.get("rgb_pointmap_scale", torch.tensor([1.0])).unsqueeze(0).to(device)
            item["rgb_pointmap_shift"] = preprocessor_return.get("rgb_pointmap_shift", torch.tensor([0.0, 0.0, 0.0])).unsqueeze(0).to(device)

        batch_items.append(item)

    return batch_items

def embed_condition(condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs

def map_input_keys(item, condition_input_mapping):
    output = [item[k] for k in condition_input_mapping]

    return output

def get_condition_input(condition_embedder, input_dict, input_mapping):
    condition_args = map_input_keys(input_dict, input_mapping)
    condition_kwargs = {
        k: v for k, v in input_dict.items() if k not in input_mapping
    }
    embedded_cond, condition_args, condition_kwargs = embed_condition(
        condition_embedder, *condition_args, **condition_kwargs
    )
    if embedded_cond is not None:
        condition_args = (embedded_cond,)
        condition_kwargs = {}

    return condition_args, condition_kwargs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _instantiate_latent_mapping(latent_mapping_cfg):
    """
    Instantiate latent_mapping entries that contain _target_ (Hydra-style).
    Each entry like {_target_: ..., in_channels: ..., ...} becomes an nn.Module.
    """
    from hydra.utils import instantiate as hydra_instantiate
    result = {}
    for key, val in latent_mapping_cfg.items():
        if isinstance(val, dict) and "_target_" in val:
            result[key] = hydra_instantiate(val)
        else:
            result[key] = val
    return result


def _prepare_backbone_config(backbone_cfg_omegaconf):
    """
    Convert a backbone OmegaConf config to a plain dict,
    instantiating nested _target_ entries (latent_mapping items).
    """
    cfg = OmegaConf.to_container(backbone_cfg_omegaconf, resolve=True)

    # Instantiate latent_mapping entries (they contain _target_)
    if "latent_mapping" in cfg:
        cfg["latent_mapping"] = _instantiate_latent_mapping(cfg["latent_mapping"])

    return cfg


def build_model_from_config(config_path: str, checkpoint_dir: Optional[str] = None) -> tuple[nn.Module, Any]:
    """
    Build the full ShortCut model with DualBackbone from a YAML config using Hydra instantiate.

    The YAML config has the structure:
        module:
          condition_embedder: ...
          generator:
            backbone:  # ShortCut
              reverse_fn:  # ClassifierFreeGuidanceWithExternalUnconditionalProbability
                backbone:  # DualBackboneSparseStructureFlowTdfyWrapper

    Returns the ShortCut model with condition_embedder wired into the backbone.
    """
    config = OmegaConf.load(config_path)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(config_path)

    from sam3d_objects.config.utils import locate

    # --- 1. Instantiate the condition embedder ---
    logger.info("Building condition embedder...")
    condition_embedder = instantiate(config.module.condition_embedder.backbone)

    # --- 2. Build the DualBackbone (denoising network) ---
    reverse_fn_cfg = config.module.generator.backbone.reverse_fn
    dual_backbone_omegaconf = reverse_fn_cfg.backbone

    # Prepare sparse_flow_config with latent_mapping instantiated
    sparse_flow_config = _prepare_backbone_config(dual_backbone_omegaconf.sparse_flow_config)

    # Prepare global_sparse_flow_config with latent_mapping instantiated
    global_sparse_flow_config = _prepare_backbone_config(dual_backbone_omegaconf.global_sparse_flow_config)

    # Get remaining DualBackbone params
    dual_backbone_plain = OmegaConf.to_container(dual_backbone_omegaconf, resolve=True)
    dual_backbone_target = dual_backbone_plain.pop("_target_")
    dual_backbone_plain.pop("sparse_flow_config")
    dual_backbone_plain.pop("global_sparse_flow_config")

    logger.info("Building DualBackboneSparseStructureFlowTdfyWrapper...")
    DualBackboneClass = locate(dual_backbone_target)
    dual_backbone = DualBackboneClass(
        sparse_flow_config=sparse_flow_config,
        global_sparse_flow_config=global_sparse_flow_config,
        checkpoint_dir=checkpoint_dir,
        **dual_backbone_plain,
    )

    # --- 3. Wrap with ClassifierFreeGuidance ---
    logger.info("Building ClassifierFreeGuidance wrapper...")
    cfg_wrapper_cfg = OmegaConf.to_container(reverse_fn_cfg, resolve=True)
    cfg_wrapper_target = cfg_wrapper_cfg.pop("_target_")
    cfg_wrapper_cfg.pop("backbone")  # We pass the actual module instead

    CFGClass = locate(cfg_wrapper_target)
    reverse_fn = CFGClass(backbone=dual_backbone, **cfg_wrapper_cfg)

    # --- 4. Build the ShortCut model ---
    logger.info("Building ShortCut model...")
    shortcut_cfg = OmegaConf.to_container(config.module.generator.backbone, resolve=True)
    shortcut_target = shortcut_cfg.pop("_target_")
    shortcut_cfg.pop("reverse_fn")  # We pass the actual module instead

    # Handle loss_weights with _target_: make_dict
    loss_weights = shortcut_cfg.get("loss_weights", {})
    if isinstance(loss_weights, dict) and "_target_" in loss_weights:
        loss_weights_target = loss_weights.pop("_target_")
        LossWeightsFactory = locate(loss_weights_target)
        shortcut_cfg["loss_weights"] = LossWeightsFactory(**loss_weights)

    # Handle training_time_sampler_fn with _partial_/_target_
    sampler_cfg = shortcut_cfg.pop("training_time_sampler_fn", None)
    if sampler_cfg is not None and isinstance(sampler_cfg, dict):
        sampler_target = sampler_cfg.pop("_target_", None)
        sampler_cfg.pop("_partial_", None)
        if sampler_target:
            SamplerFn = locate(sampler_target)
            from functools import partial
            shortcut_cfg["training_time_sampler_fn"] = partial(SamplerFn, **sampler_cfg)

    ShortCutClass = locate(shortcut_target)
    model = ShortCutClass(reverse_fn=reverse_fn, **shortcut_cfg)
    model.condition_embedder = condition_embedder

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- 5. Load the preprocessor ---
    # The preprocessor is needed to prepare conditioning inputs during training
    logger.info("Loading preprocessor...")
    try:
        preprocessor_config_path = Path(checkpoint_dir) / "pipeline.yaml"
        if not preprocessor_config_path.exists():
            preprocessor_config_path = Path("checkpoints/hf/pipeline.yaml")

        pipeline_config = OmegaConf.load(preprocessor_config_path)
        preprocessor = instantiate(pipeline_config["ss_preprocessor"])
        logger.info(f"Loaded preprocessor from {preprocessor_config_path}")
    except Exception as e:
        logger.warning(f"Failed to load preprocessor: {e}")
        logger.warning("Preprocessor will not be available - conditioning may fail")
        preprocessor = None

    return model, preprocessor


def freeze_condition_embedder(model: nn.Module):
    """Freeze the condition embedder parameters (typically pretrained DINOv2)."""
    embedder = model.condition_embedder
    if hasattr(embedder, 'parameters'):
            count = 0
            for param in embedder.parameters():
                param.requires_grad = False
                count += param.numel()
            logger.info(f"Frozen condition embedder: {count:,} parameters")


def freeze_sparse_flow(model: nn.Module):
    """
    Freeze the primary sparse_flow backbone entirely.
    Only global_sparse_flow will remain trainable.
    """
    backbone = model.reverse_fn.backbone  # DualBackbone
    count = 0
    for param in backbone.sparse_flow.parameters():
        param.requires_grad = False
        count += param.numel()
    logger.info(f"Frozen sparse_flow backbone: {count:,} parameters")


def get_parameter_groups(model: nn.Module, lr: float, weight_decay: float):
    """
    Create parameter groups with optional separate learning rates.
    Excludes frozen parameters.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr},
    ]

    logger.info(f"Parameter groups: {len(decay_params)} decay, {len(no_decay_params)} no-decay")
    return param_groups


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
    best_loss: float = float('inf'),
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
            latents = to_device(batch['latents'], device)
            conditionals = to_device(batch['conditionals'], device)
            scene_num_objects = batch['num_objects']

            optimizer.zero_grad(set_to_none=True)

            # Prepare latents: add dummy dimension for token_len=1 modalities
            latents['scale'] = latents['scale'].unsqueeze(1)
            latents['6drotation_normalized'] = latents['6drotation_normalized'].unsqueeze(1)
            latents['translation'] = latents['translation'].unsqueeze(1)
            latents['translation_scale'] = torch.ones(latents['scale'].shape[0], 1, 1).to(device)
            latents['shape'] = latents['shape'].reshape(-1, 8, 16**3).permute(0, 2, 1).contiguous()

            # Prepare conditioning following Sam3DConditionedMixin pattern
            image = conditionals['image']  # [H, W, 3]
            pointmap = conditionals['pointmap']  # [H, W, 3]
            object_masks = conditionals['object_masks']  # [num_objects, H, W]

            # Preprocess conditioning for all objects in the scene
            input_dicts = prepare_conditioning_for_scene(
                image=image,
                pointmap=pointmap,
                object_masks=object_masks,
                preprocessor=preprocessor,
                device=device,
            )

            cond = [get_condition_input(embedder, input_dict, [])[0][0]
                    for input_dict in input_dicts]

            cond = torch.concat(cond, dim=0)

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
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_fm_loss += detail_losses['flow_matching_loss'].item()
            total_sc_loss += detail_losses['self_consistency_loss'].item()
            num_scenes += 1
            num_objects += scene_num_objects

            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / num_scenes
                avg_fm_loss = total_fm_loss / num_scenes
                avg_sc_loss = total_sc_loss / num_scenes
                avg_objs = num_objects / num_scenes
                current_lr = optimizer.param_groups[0]["lr"]

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'fm': f'{avg_fm_loss:.4f}',
                    'sc': f'{avg_sc_loss:.4f}',
                    'objs': f'{avg_objs:.1f}',
                    'lr': f'{current_lr:.2e}',
                })

                # Log to experiment tracker
                if exp_logger is not None and rank == 0:
                    step_metrics = {
                        'train/loss': avg_loss,
                        'train/flow_matching_loss': avg_fm_loss,
                        'train/self_consistency_loss': avg_sc_loss,
                        'train/avg_objects_per_scene': avg_objs,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch,
                    }
                    log_metrics(exp_logger, step_metrics, global_step + step)

            # Save checkpoint at step intervals
            current_global_step = global_step + step + 1
            if save_interval_steps > 0 and current_global_step % save_interval_steps == 0:
                if rank == 0 and output_dir is not None:
                    # TODO: UnboundLocalError: cannot access local variable 'avg_loss' where it is not associated with a value
                    # step_metrics = {
                    #     'loss': avg_loss,
                    #     'flow_matching_loss': avg_fm_loss,
                    #     'self_consistency_loss': avg_sc_loss,
                    #     'avg_objects_per_scene': avg_objs,
                    # }
                    step_metrics = {}

                    # Save periodic checkpoint
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=current_global_step,
                        metrics=step_metrics,
                        save_path=str(output_dir / f'step_{current_global_step:08d}.pt'),
                        is_distributed=is_distributed,
                        scaler=scaler,
                    )
                    logger.info(f"Saved checkpoint at step {current_global_step}")

                    # Save best checkpoint
                    # if avg_loss < current_best_loss:
                    #     current_best_loss = avg_loss
                    #     save_checkpoint(
                    #         model=model,
                    #         optimizer=optimizer,
                    #         scheduler=scheduler,
                    #         epoch=epoch,
                    #         global_step=current_global_step,
                    #         metrics=step_metrics,
                    #         save_path=str(output_dir / 'best.pt'),
                    #         is_distributed=is_distributed,
                    #         scaler=scaler,
                    #     )
                    #     logger.info(f"  New best loss: {current_best_loss:.4f}")

        except Exception as e:
            logger.error(f"Error in training step {step} (epoch {epoch}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    metrics = {
        'loss': total_loss / max(num_scenes, 1),
        'flow_matching_loss': total_fm_loss / max(num_scenes, 1),
        'self_consistency_loss': total_sc_loss / max(num_scenes, 1),
        'avg_objects_per_scene': num_objects / max(num_scenes, 1),
    }

    if is_distributed_runtime:
        for key in metrics:
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()

    return metrics, current_best_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    preprocessor: Any,
    use_amp: bool = False,
    exp_logger: Optional[Any] = None,
    global_step: int = 0,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    embedder = model.condition_embedder

    total_loss = 0.0
    total_fm_loss = 0.0
    total_sc_loss = 0.0
    num_scenes = 0
    num_objects = 0

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0))

    for step, batch in enumerate(pbar):
        try:
            latents = to_device(batch['latents'], device)
            conditionals = to_device(batch['conditionals'], device)
            scene_num_objects = batch['num_objects']

            # Prepare latents
            latents['scale'] = latents['scale'].unsqueeze(1)
            latents['6drotation_normalized'] = latents['6drotation_normalized'].unsqueeze(1)
            latents['translation'] = latents['translation'].unsqueeze(1)
            latents['translation_scale'] = torch.ones(latents['scale'].shape[0], 1, 1).to(device)
            latents['shape'] = latents['shape'].reshape(-1, 8, 16**3).permute(0, 2, 1).contiguous()

            # Prepare conditioning
            image = conditionals['image']
            pointmap = conditionals['pointmap']
            object_masks = conditionals['object_masks']

            # Preprocess conditioning for all objects in the scene
            input_dicts = prepare_conditioning_for_scene(
                image=image,
                pointmap=pointmap,
                object_masks=object_masks,
                preprocessor=preprocessor,
                device=device,
            )

            cond = [get_condition_input(embedder, input_dict, [])[0][0]
                    for input_dict in input_dicts]

            cond = torch.concat(cond, dim=0)

            if use_amp:
                with autocast(dtype=torch.float16):
                    loss, detail_losses = model.loss(latents, cond)
            else:
                loss, detail_losses = model.loss(latents, cond)

            total_loss += loss.item()
            total_fm_loss += detail_losses['flow_matching_loss'].item()
            total_sc_loss += detail_losses['self_consistency_loss'].item()
            num_scenes += 1
            num_objects += scene_num_objects

            # Log validation metrics periodically
            if (step + 1) % log_interval == 0 and exp_logger is not None and rank == 0:
                avg_loss = total_loss / num_scenes
                avg_fm_loss = total_fm_loss / num_scenes
                avg_sc_loss = total_sc_loss / num_scenes
                avg_objs = num_objects / num_scenes

                val_step_metrics = {
                    'val/loss': avg_loss,
                    'val/flow_matching_loss': avg_fm_loss,
                    'val/self_consistency_loss': avg_sc_loss,
                    'val/avg_objects_per_scene': avg_objs,
                }
                log_metrics(exp_logger, val_step_metrics, global_step + step)

        except Exception as e:
            logger.error(f"Error in validation step {step}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    metrics = {
        'val_loss': total_loss / max(num_scenes, 1),
        'val_flow_matching_loss': total_fm_loss / max(num_scenes, 1),
        'val_self_consistency_loss': total_sc_loss / max(num_scenes, 1),
        'val_avg_objects_per_scene': num_objects / max(num_scenes, 1),
    }

    if is_distributed:
        for key in metrics:
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()

    return metrics


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
    """Save training checkpoint."""
    if is_distributed and dist.get_rank() != 0:
        return

    raw_model = unwrap_dist(model)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_model = unwrap_dist(model)
    raw_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
                f"step {checkpoint.get('global_step', '?')}")
    return checkpoint


def main(args):
    """Main training function."""

    # --- Setup distributed ---
    is_distributed = 'RANK' in os.environ
    if is_distributed:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup_dist(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            master_addr=os.environ.get('MASTER_ADDR', 'localhost'),
            master_port=os.environ.get('MASTER_PORT', '12355'),
        )
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        config_save_path = output_dir / 'training_config.json'
        with open(config_save_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        # Also copy the model config
        shutil.copy2(args.config, output_dir / 'model_config.yaml')
        logger.info(f"Output directory: {output_dir}")

    # --- Setup experiment logger ---
    exp_logger = setup_logger(
        logger_type=args.logger,
        output_dir=output_dir,
        config=vars(args),
        rank=rank,
    )

    # --- Build model and preprocessor ---
    logger.info("Building model from config...")
    model, preprocessor = build_model_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)

    # Freeze condition embedder (typically pretrained, should not be trained)
    if args.freeze_embedder:
        freeze_condition_embedder(model)

    # Freeze sparse_flow backbone — only train global_sparse_flow
    freeze_sparse_flow(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"After freeze: {trainable_params:,} trainable / {total_params:,} total")

    # --- Create datasets ---
    logger.info("Creating datasets...")
    train_dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=args.max_objects_per_scene,
        load_images=True,
        load_depth=True,
        load_masks=True,
        image_size=(args.image_height, args.image_width) if args.image_width > 0 else None,
        precomputed_latents=args.precomputed_latents,
        num_renders_per_scene=args.num_renders_per_scene,
        gso_root=args.gso_root if args.gso_root else None,
        load_meshes=args.load_meshes,
    )
    logger.info(f"Train dataset size: {len(train_dataset)} scenes")

    # Validation split: use a fraction of scenes
    val_dataset = None
    if args.val_data_root:
        val_dataset = FoundationPoseDataset(
            data_root=args.val_data_root,
            max_objects_per_scene=args.max_objects_per_scene,
            load_images=True,
            load_depth=True,
            load_masks=True,
            image_size=(args.image_height, args.image_width) if args.image_width > 0 else None,
            precomputed_latents=args.precomputed_latents,
            num_renders_per_scene=args.num_renders_per_scene,
        )
        logger.info(f"Val dataset size: {len(val_dataset)} scenes")

    # --- Dataloaders ---
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # One scene at a time (scene = batch of objects)
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # --- DDP ---
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_params,
        )

    # --- Optimizer ---
    param_groups = get_parameter_groups(model, args.learning_rate, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # --- Scheduler ---
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- AMP scaler ---
    scaler = GradScaler() if args.use_amp else None

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('metrics', {}).get('val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    # --- Training loop ---
    logger.info(f"Starting training: {args.num_epochs} epochs, "
                f"{len(train_loader)} steps/epoch, "
                f"warmup={warmup_steps} steps")

    for epoch in range(start_epoch, args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics, best_val_loss = train_epoch(
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
            best_loss=best_val_loss,
        )

        global_step += len(train_loader)

        # Validate
        val_metrics = {}
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                preprocessor=preprocessor,
                use_amp=args.use_amp,
                exp_logger=exp_logger,
                global_step=global_step,
                log_interval=args.log_interval,
            )

        # Log
        if rank == 0:
            logger.info(f"Epoch {epoch}:")
            logger.info(f"  Train: loss={train_metrics['loss']:.4f}, "
                        f"fm={train_metrics['flow_matching_loss']:.4f}, "
                        f"sc={train_metrics['self_consistency_loss']:.4f}, "
                        f"objs={train_metrics['avg_objects_per_scene']:.1f}")
            if val_metrics:
                logger.info(f"  Val:   loss={val_metrics['val_loss']:.4f}, "
                            f"fm={val_metrics['val_flow_matching_loss']:.4f}, "
                            f"sc={val_metrics['val_self_consistency_loss']:.4f}")

        # Save latest checkpoint at end of each epoch
        if rank == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metrics={**train_metrics, **val_metrics},
                save_path=str(output_dir / 'latest.pt'),
                is_distributed=is_distributed,
                scaler=scaler,
            )

        # Periodic epoch-based checkpoint (if enabled)
        if args.save_interval_epochs > 0 and (epoch + 1) % args.save_interval_epochs == 0:
            if rank == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics={**train_metrics, **val_metrics},
                    save_path=str(output_dir / f'epoch_{epoch:04d}.pt'),
                    is_distributed=is_distributed,
                    scaler=scaler,
                )

    logger.info("Training completed!")

    # Finish experiment logging
    if rank == 0 and exp_logger is not None:
        if hasattr(exp_logger, 'finish'):
            exp_logger.finish()
        elif hasattr(exp_logger, 'close'):
            exp_logger.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DualBackbone ShortCut model with FoundationPose dataset'
    )

    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML (e.g., checkpoints/hf/dual_backbone_generator.yaml)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Base directory for loading pretrained backbone checkpoints. '
                             'Defaults to the directory containing the config file.')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing FoundationPose data')
    parser.add_argument('--val_data_root', type=str, default=None,
                        help='Root directory for validation data (optional)')
    parser.add_argument('--gso_root', type=str, default=None,
                        help='Path to GSO dataset root for mesh loading')
    parser.add_argument('--max_objects_per_scene', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_width', type=int, default=0,
                        help='Target image width (0 for original)')
    parser.add_argument('--image_height', type=int, default=0,
                        help='Target image height (0 for original)')
    parser.add_argument('--precomputed_latents', action='store_true')
    parser.add_argument('--num_renders_per_scene', type=int, default=1)
    parser.add_argument('--load_meshes', action='store_true')

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Fraction of total steps for LR warmup')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--freeze_embedder', action='store_true', default=True,
                        help='Freeze condition embedder (default: True)')
    parser.add_argument('--no_freeze_embedder', dest='freeze_embedder', action='store_false')
    parser.add_argument('--find_unused_params', action='store_true',
                        help='Enable find_unused_parameters for DDP')

    # Logging & checkpoints
    parser.add_argument('--logger', type=str, default='wandb', choices=['wandb', 'tensorboard', 'none'],
                        help='Experiment logger to use (default: wandb)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Step interval for logging metrics')
    parser.add_argument('--save_interval_steps', type=int, default=1,
                        help='Checkpoint save interval in steps (0 to disable step-based saving)')
    parser.add_argument('--save_interval_epochs', type=int, default=0,
                        help='Checkpoint save interval in epochs (0 to disable epoch-based saving)')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval (epochs)')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/dual_backbone_foundationpose')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)
