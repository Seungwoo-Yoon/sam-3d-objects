"""
Training script for ShortCut Flow Model with Scene-Level Batching

This script trains the ShortCut (SS generator) model on entire scenes at once.
For example, if a scene has 9 objects, all 9 objects are processed together
and their losses are calculated simultaneously.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
from sam3d_objects.data.utils import tree_tensor_map, to_device
from sam3d_objects.utils.dist_utils import setup_dist, unwrap_dist, master_first
from sam3d_objects.utils.general_utils import dict_flatten


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SceneDataset(Dataset):
    """
    Dataset that returns all objects from a scene as a single batch.

    Each sample contains:
    - latents: Dict of tensors with shape [num_objects_in_scene, ...]
    - conditionals: Conditioning information (e.g., images, text, etc.)
    - scene_id: Identifier for the scene
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_objects_per_scene: int = 32,
    ):
        """
        Args:
            data_root: Root directory containing scene data
            split: Data split ('train', 'val', 'test')
            max_objects_per_scene: Maximum number of objects per scene (for filtering)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_objects_per_scene = max_objects_per_scene

        # Load scene metadata (you'll need to implement this based on your data format)
        self.scene_list = self._load_scene_list()

        logger.info(f"Loaded {len(self.scene_list)} scenes for split '{split}'")

    def _load_scene_list(self) -> List[str]:
        """
        Load list of scene IDs from disk.
        Implement this based on your data structure.
        """
        scene_list_file = self.data_root / self.split / 'scenes.txt'
        if scene_list_file.exists():
            with open(scene_list_file, 'r') as f:
                scenes = [line.strip() for line in f]
            return scenes
        else:
            # If no scene list file, find all scene directories
            scene_dirs = sorted([d.name for d in (self.data_root / self.split).iterdir() if d.is_dir()])
            return scene_dirs

    def __len__(self) -> int:
        return len(self.scene_list)

    def _load_scene_latents(self, scene_id: str) -> Dict[str, torch.Tensor]:
        """
        Load latent representations for all objects in a scene.

        Returns a dictionary with keys like:
        - 'shape': [num_objects, latent_dim]
        - 'translation': [num_objects, 3]
        - 'scale': [num_objects, 3]
        - '6drotation_normalized': [num_objects, 6]
        - etc.
        """
        scene_path = self.data_root / self.split / scene_id

        # Load pre-computed latents (implement based on your data format)
        # This is a placeholder - adjust to your actual data structure
        latents_file = scene_path / 'latents.pt'
        if latents_file.exists():
            latents = torch.load(latents_file)
        else:
            # If latents don't exist, you might need to encode them using a VAE encoder
            raise NotImplementedError(
                f"Latent file not found for scene {scene_id}. "
                "You need to pre-compute latents or implement online encoding."
            )

        return latents

    def _load_scene_conditionals(self, scene_id: str) -> Dict[str, Any]:
        """
        Load conditioning information for the scene.

        This could include:
        - Images
        - Text descriptions
        - Point clouds
        - etc.
        """
        scene_path = self.data_root / self.split / scene_id

        # Load conditionals (implement based on your conditioning type)
        # This is a placeholder - adjust to your actual conditioning
        conditionals = {}

        # Example: load scene image
        # image_file = scene_path / 'image.jpg'
        # if image_file.exists():
        #     conditionals['image'] = load_image(image_file)

        return conditionals

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_id = self.scene_list[idx]

        # Load latents for all objects in the scene
        latents = self._load_scene_latents(scene_id)

        # Load conditioning information
        conditionals = self._load_scene_conditionals(scene_id)

        # Filter scenes with too many objects
        num_objects = next(iter(latents.values())).shape[0]
        if num_objects > self.max_objects_per_scene:
            # Either skip or sample subset of objects
            logger.warning(
                f"Scene {scene_id} has {num_objects} objects, "
                f"exceeding max {self.max_objects_per_scene}. Sampling subset."
            )
            indices = torch.randperm(num_objects)[:self.max_objects_per_scene]
            latents = tree_tensor_map(lambda x: x[indices], latents)
            num_objects = self.max_objects_per_scene

        return {
            'latents': latents,
            'conditionals': conditionals,
            'scene_id': scene_id,
            'num_objects': num_objects,
        }


def scene_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for scene-level batching.

    Since each scene can have a different number of objects, we process
    one scene at a time (batch_size=1 at the scene level).
    """
    assert len(batch) == 1, "Scene-level training requires batch_size=1 for scenes"

    return batch[0]


def create_shortcut_model(
    reverse_fn: nn.Module,
    config: Dict,
) -> ShortCut:
    """
    Create ShortCut model with scene-level training configuration.

    Args:
        reverse_fn: The denoising network (e.g., DiT model)
        config: Model configuration dictionary

    Returns:
        ShortCut model instance
    """
    model = ShortCut(
        reverse_fn=reverse_fn,
        batch_mode=True,  # CRITICAL: Enable scene-level processing
        self_consistency_prob=config.get('self_consistency_prob', 0.25),
        shortcut_loss_weight=config.get('shortcut_loss_weight', 1.0),
        self_consistency_cfg_strength=config.get('self_consistency_cfg_strength', 3.0),
        ratio_cfg_samples_in_self_consistency_target=config.get(
            'ratio_cfg_samples_in_self_consistency_target', 0.5
        ),
        fm_in_shortcut_target_prob=config.get('fm_in_shortcut_target_prob', 0.0),
        fm_eps_max=config.get('fm_eps_max', 0),
        cfg_modalities=config.get('cfg_modalities', ['shape']),
        # Flow matching base parameters
        sigma_min=config.get('sigma_min', 0.0),
        inference_steps=config.get('inference_steps', 100),
        time_scale=config.get('time_scale', 1000.0),
        solver_method=config.get('solver_method', 'euler'),
    )

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: ShortCut model
        dataloader: Scene dataloader
        optimizer: Optimizer
        epoch: Current epoch number
        device: Training device
        grad_clip: Gradient clipping value
        log_interval: Logging frequency

    Returns:
        Dictionary of average losses
    """
    model.train()

    total_loss = 0.0
    total_fm_loss = 0.0
    total_sc_loss = 0.0
    num_scenes = 0
    num_objects = 0

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0),
    )

    for step, batch in enumerate(pbar):
        # Move to device
        latents = to_device(batch['latents'], device)
        conditionals = to_device(batch['conditionals'], device)

        # Get number of objects in this scene
        scene_num_objects = batch['num_objects']

        # Forward pass - compute loss for all objects in the scene
        # The loss function will handle all objects simultaneously
        loss, detail_losses = model.loss(
            latents,  # x1: Dict of [num_objects, ...]
            **conditionals,  # Conditioning shared across all objects
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_fm_loss += detail_losses['flow_matching_loss'].item()
        total_sc_loss += detail_losses['self_consistency_loss'].item()
        num_scenes += 1
        num_objects += scene_num_objects

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / num_scenes
            avg_fm_loss = total_fm_loss / num_scenes
            avg_sc_loss = total_sc_loss / num_scenes
            avg_objects_per_scene = num_objects / num_scenes

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'fm': f'{avg_fm_loss:.4f}',
                'sc': f'{avg_sc_loss:.4f}',
                'objs': f'{avg_objects_per_scene:.1f}',
            })

    # Compute epoch averages
    metrics = {
        'loss': total_loss / num_scenes,
        'flow_matching_loss': total_fm_loss / num_scenes,
        'self_consistency_loss': total_sc_loss / num_scenes,
        'avg_objects_per_scene': num_objects / num_scenes,
    }

    # Aggregate across ranks if distributed
    if is_distributed:
        for key in metrics:
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: ShortCut model
        dataloader: Validation dataloader
        device: Device

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_fm_loss = 0.0
    total_sc_loss = 0.0
    num_scenes = 0
    num_objects = 0

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    pbar = tqdm(
        dataloader,
        desc="Validation",
        disable=(rank != 0),
    )

    for batch in pbar:
        # Move to device
        latents = to_device(batch['latents'], device)
        conditionals = to_device(batch['conditionals'], device)

        # Get number of objects in this scene
        scene_num_objects = batch['num_objects']

        # Forward pass
        loss, detail_losses = model.loss(
            latents,
            **conditionals,
        )

        # Accumulate
        total_loss += loss.item()
        total_fm_loss += detail_losses['flow_matching_loss'].item()
        total_sc_loss += detail_losses['self_consistency_loss'].item()
        num_scenes += 1
        num_objects += scene_num_objects

    # Compute averages
    metrics = {
        'val_loss': total_loss / num_scenes,
        'val_flow_matching_loss': total_fm_loss / num_scenes,
        'val_self_consistency_loss': total_sc_loss / num_scenes,
        'val_avg_objects_per_scene': num_objects / num_scenes,
    }

    # Aggregate across ranks if distributed
    if is_distributed:
        for key in metrics:
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_distributed: bool = False,
):
    """Save training checkpoint."""
    if is_distributed and dist.get_rank() != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': unwrap_dist(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def main(args):
    """Main training function."""

    # Setup distributed training if needed
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
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    if rank == 0:
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")

    # Create datasets
    train_dataset = SceneDataset(
        data_root=args.data_root,
        split='train',
        max_objects_per_scene=args.max_objects_per_scene,
    )

    val_dataset = SceneDataset(
        data_root=args.data_root,
        split='val',
        max_objects_per_scene=args.max_objects_per_scene,
    )

    # Create dataloaders
    # NOTE: batch_size=1 because each scene is a "batch" of objects
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # One scene at a time
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=scene_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # One scene at a time
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scene_collate_fn,
        pin_memory=True,
    )

    # Create model
    # NOTE: You need to create your reverse_fn (denoising network) here
    # This is a placeholder - replace with your actual model
    from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
        SparseStructureFlowTdfyWrapper
    )

    # Example model config - adjust based on your needs
    model_config = {
        'self_consistency_prob': args.self_consistency_prob,
        'shortcut_loss_weight': args.shortcut_loss_weight,
        'self_consistency_cfg_strength': args.cfg_strength,
        'ratio_cfg_samples_in_self_consistency_target': args.ratio_cfg_samples,
        'fm_in_shortcut_target_prob': args.fm_in_shortcut_target_prob,
        'fm_eps_max': args.fm_eps_max,
        'cfg_modalities': ['shape'],  # Adjust based on your modalities
        'time_scale': 1000.0,
        'inference_steps': 100,
    }

    # Load or create reverse_fn (denoising network)
    # This is where you'd instantiate your DiT or other denoising model
    # reverse_fn = YourDenoisingNetwork(...)

    # For now, this is a placeholder
    logger.warning(
        "You need to implement the reverse_fn (denoising network) creation. "
        "See the placeholder in the code."
    )
    # Uncomment and modify when you have your model:
    # reverse_fn = create_your_model(...)
    # model = create_shortcut_model(reverse_fn, model_config)
    # model = model.to(device)

    # Wrap with DDP if distributed
    # if is_distributed:
    #     model = DDP(model, device_ids=[local_rank])

    # Create optimizer
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay,
    # )

    # Create learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.num_epochs,
    # )

    # Training loop
    # best_val_loss = float('inf')
    #
    # for epoch in range(args.num_epochs):
    #     if is_distributed:
    #         train_sampler.set_epoch(epoch)
    #
    #     # Train
    #     train_metrics = train_epoch(
    #         model=model,
    #         dataloader=train_loader,
    #         optimizer=optimizer,
    #         epoch=epoch,
    #         device=device,
    #         grad_clip=args.grad_clip,
    #         log_interval=args.log_interval,
    #     )
    #
    #     # Validate
    #     val_metrics = validate(
    #         model=model,
    #         dataloader=val_loader,
    #         device=device,
    #     )
    #
    #     # Log
    #     if rank == 0:
    #         logger.info(f"Epoch {epoch}:")
    #         logger.info(f"  Train: {train_metrics}")
    #         logger.info(f"  Val: {val_metrics}")
    #
    #     # Save checkpoint
    #     if val_metrics['val_loss'] < best_val_loss:
    #         best_val_loss = val_metrics['val_loss']
    #         save_checkpoint(
    #             model=model,
    #             optimizer=optimizer,
    #             epoch=epoch,
    #             metrics={**train_metrics, **val_metrics},
    #             save_path=output_dir / 'best_model.pt',
    #             is_distributed=is_distributed,
    #         )
    #
    #     # Regular checkpoint
    #     if (epoch + 1) % args.save_interval == 0:
    #         save_checkpoint(
    #             model=model,
    #             optimizer=optimizer,
    #             epoch=epoch,
    #             metrics={**train_metrics, **val_metrics},
    #             save_path=output_dir / f'checkpoint_epoch_{epoch}.pt',
    #             is_distributed=is_distributed,
    #         )
    #
    #     # Step scheduler
    #     scheduler.step()

    logger.info("Training completed!")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ShortCut model with scene-level batching')

    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing scene data')
    parser.add_argument('--max_objects_per_scene', type=int, default=32,
                        help='Maximum number of objects per scene')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Model arguments
    parser.add_argument('--self_consistency_prob', type=float, default=0.25,
                        help='Probability of using self-consistency loss')
    parser.add_argument('--shortcut_loss_weight', type=float, default=1.0,
                        help='Weight for shortcut loss')
    parser.add_argument('--cfg_strength', type=float, default=3.0,
                        help='CFG strength for self-consistency target')
    parser.add_argument('--ratio_cfg_samples', type=float, default=0.5,
                        help='Ratio of CFG samples in self-consistency target')
    parser.add_argument('--fm_in_shortcut_target_prob', type=float, default=0.0,
                        help='Probability of using FM in shortcut target')
    parser.add_argument('--fm_eps_max', type=float, default=0.0,
                        help='Maximum epsilon for flow matching')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/shortcut_scene',
                        help='Output directory for checkpoints and logs')

    args = parser.parse_args()

    main(args)
