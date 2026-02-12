"""
Example training script with concrete model instantiation.

This demonstrates how to set up the ShortCut model with a proper denoising network
for scene-level training.
"""

import torch
import torch.nn as nn
from functools import partial

from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
    SparseStructureFlowTdfyWrapper
)
from sam3d_objects.model.backbone.dit.embedder import DinoEmbedder  # Example embedder


def create_denoising_network(config: dict) -> nn.Module:
    """
    Create the denoising network (reverse_fn) for the ShortCut model.

    This example uses the SparseStructureFlowTdfyWrapper which is a
    multi-object transformer that processes multiple objects simultaneously.

    Args:
        config: Configuration dictionary

    Returns:
        Denoising network module
    """

    # Example latent mapping configuration
    # This defines how different modalities are processed
    from sam3d_objects.model.backbone.tdfy_dit.models.mm_latent import LatentMapping

    latent_mapping = {
        'shape': LatentMapping(
            in_channels=config.get('shape_channels', 256),
            out_channels=config.get('model_channels', 512),
            num_tokens=config.get('shape_tokens', 256),
        ),
        'translation': LatentMapping(
            in_channels=3,
            out_channels=config.get('model_channels', 512),
            num_tokens=1,
        ),
        'scale': LatentMapping(
            in_channels=3,
            out_channels=config.get('model_channels', 512),
            num_tokens=1,
        ),
        '6drotation_normalized': LatentMapping(
            in_channels=6,
            out_channels=config.get('model_channels', 512),
            num_tokens=1,
        ),
    }

    # Example: share transformer for pose-related attributes
    latent_share_transformer = {
        'pose': ['translation', 'scale', '6drotation_normalized']
    }

    # Create condition embedder (e.g., for images, text, etc.)
    # This is optional - set to None if you don't have conditioning
    condition_embedder = None
    # Example with image conditioning:
    # condition_embedder = DinoEmbedder(
    #     model_name='dinov2_vitb14',
    #     output_dim=config.get('cond_channels', 768),
    # )

    # Create the denoising network
    reverse_fn = SparseStructureFlowTdfyWrapper(
        latent_mapping=latent_mapping,
        latent_share_transformer=latent_share_transformer,
        condition_embedder=condition_embedder,
        # Model architecture parameters
        in_channels=config.get('model_channels', 512),
        model_channels=config.get('model_channels', 512),
        cond_channels=config.get('cond_channels', 768),
        out_channels=config.get('model_channels', 512),
        num_blocks=config.get('num_blocks', 12),
        num_head_channels=config.get('num_head_channels', 64),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        pe_mode=config.get('pe_mode', 'rope'),  # 'rope' or 'ape'
        use_fp16=config.get('use_fp16', False),
        use_checkpoint=config.get('use_checkpoint', False),
        share_mod=config.get('share_mod', False),
        qk_rms_norm=config.get('qk_rms_norm', True),
        qk_rms_norm_cross=config.get('qk_rms_norm_cross', True),
        is_shortcut_model=True,  # IMPORTANT: Enable shortcut-specific features
    )

    return reverse_fn


def create_shortcut_model_example():
    """
    Example of creating a complete ShortCut model for scene-level training.
    """

    # Model configuration
    model_config = {
        # Denoising network config
        'model_channels': 512,
        'cond_channels': 768,
        'num_blocks': 12,
        'num_head_channels': 64,
        'mlp_ratio': 4.0,
        'pe_mode': 'rope',
        'use_fp16': False,
        'use_checkpoint': True,
        'qk_rms_norm': True,
        'qk_rms_norm_cross': True,

        # Shape latent config
        'shape_channels': 256,
        'shape_tokens': 256,

        # ShortCut-specific config
        'self_consistency_prob': 0.25,  # 25% of samples use self-consistency
        'shortcut_loss_weight': 1.0,
        'self_consistency_cfg_strength': 3.0,
        'ratio_cfg_samples_in_self_consistency_target': 0.5,
        'fm_in_shortcut_target_prob': 0.0,
        'fm_eps_max': 0.0,
        'cfg_modalities': ['shape'],

        # Flow matching config
        'sigma_min': 0.0,
        'time_scale': 1000.0,
        'inference_steps': 100,
    }

    # Create denoising network
    reverse_fn = create_denoising_network(model_config)

    # Create ShortCut model with batch_mode=True for scene-level training
    model = ShortCut(
        reverse_fn=reverse_fn,
        batch_mode=True,  # CRITICAL: This enables scene-level processing
        self_consistency_prob=model_config['self_consistency_prob'],
        shortcut_loss_weight=model_config['shortcut_loss_weight'],
        self_consistency_cfg_strength=model_config['self_consistency_cfg_strength'],
        ratio_cfg_samples_in_self_consistency_target=model_config[
            'ratio_cfg_samples_in_self_consistency_target'
        ],
        fm_in_shortcut_target_prob=model_config['fm_in_shortcut_target_prob'],
        fm_eps_max=model_config['fm_eps_max'],
        cfg_modalities=model_config['cfg_modalities'],
        sigma_min=model_config['sigma_min'],
        time_scale=model_config['time_scale'],
        inference_steps=model_config['inference_steps'],
    )

    return model


def example_training_step():
    """
    Example of a single training step with scene-level data.
    """

    # Create model
    model = create_shortcut_model_example()
    model = model.cuda()

    # Example scene with 9 objects
    num_objects = 9
    batch_data = {
        'latents': {
            'shape': torch.randn(num_objects, 256, 256).cuda(),  # [9, C, tokens]
            'translation': torch.randn(num_objects, 1, 3).cuda(),  # [9, 1, 3]
            'scale': torch.randn(num_objects, 1, 3).cuda(),  # [9, 1, 3]
            '6drotation_normalized': torch.randn(num_objects, 1, 6).cuda(),  # [9, 1, 6]
        },
        'conditionals': {
            # Add conditioning here if you have it
            # e.g., 'image': scene_image,
        }
    }

    # Forward pass - compute loss for ALL 9 objects simultaneously
    loss, detail_losses = model.loss(
        batch_data['latents'],
        **batch_data['conditionals']
    )

    print(f"Total loss: {loss.item():.4f}")
    print(f"Flow matching loss: {detail_losses['flow_matching_loss'].item():.4f}")
    print(f"Self-consistency loss: {detail_losses['self_consistency_loss'].item():.4f}")

    # Backward pass
    loss.backward()

    print("Training step completed successfully!")
    print(f"Processed {num_objects} objects in the scene simultaneously.")


if __name__ == '__main__':
    # Run example
    print("=" * 80)
    print("ShortCut Scene-Level Training Example")
    print("=" * 80)
    print()

    example_training_step()

    print()
    print("=" * 80)
    print("Key Concepts:")
    print("=" * 80)
    print("1. batch_mode=True: Treats all objects in a batch as belonging to the same scene")
    print("2. All objects in a scene share the same conditioning (e.g., scene image)")
    print("3. Loss is computed for all objects simultaneously")
    print("4. Each scene can have a variable number of objects")
    print("5. DataLoader batch_size=1 means one scene at a time")
