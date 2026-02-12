# Training ShortCut Flow Model with Scene-Level Batching

This guide explains how to train the ShortCut (SS generator) model on entire scenes at once, where all objects in a scene are processed simultaneously.

## Overview

The ShortCut model is a flow matching model with self-consistency regularization. When training with `batch_mode=True`, it processes all objects in a scene together, which is critical for multi-object scene generation.

### Key Concept: Scene-Level vs Object-Level Training

**Object-Level Training (batch_mode=False)**:
- Each object is processed independently
- Batch contains N objects from potentially different scenes
- Conditioning can be different for each object

**Scene-Level Training (batch_mode=True)** ⭐ **What you want**:
- All objects in a batch belong to the same scene
- All objects share the same scene-level conditioning (e.g., scene image)
- Loss is computed for all objects in the scene simultaneously
- DataLoader batch_size=1 (one scene at a time)

## Files

1. **`train_shortcut_scene.py`**: Main training script with full training loop
2. **`train_shortcut_scene_example.py`**: Example showing model instantiation and usage
3. **`TRAINING_SHORTCUT_SCENE.md`**: This documentation

## Quick Start

### 1. Prepare Your Data

Your dataset should be organized by scenes. Each scene contains multiple objects.

```
data_root/
├── train/
│   ├── scene_0001/
│   │   ├── latents.pt          # Latent codes for all objects
│   │   ├── conditionals.pt     # Scene-level conditioning (optional)
│   │   └── metadata.json       # Scene metadata
│   ├── scene_0002/
│   └── ...
└── val/
    ├── scene_1001/
    └── ...
```

#### Latent Format

The `latents.pt` file should contain a dictionary with keys for each modality:

```python
{
    'shape': torch.Tensor,           # [num_objects, latent_dim, num_tokens]
    'translation': torch.Tensor,     # [num_objects, 1, 3]
    'scale': torch.Tensor,           # [num_objects, 1, 3]
    '6drotation_normalized': torch.Tensor,  # [num_objects, 1, 6]
}
```

### 2. Run Training

```bash
# Single GPU
python train_shortcut_scene.py \
    --data_root /path/to/data \
    --output_dir ./outputs/shortcut_scene \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --self_consistency_prob 0.25 \
    --shortcut_loss_weight 1.0 \
    --cfg_strength 3.0

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 train_shortcut_scene.py \
    --data_root /path/to/data \
    --output_dir ./outputs/shortcut_scene \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### 3. Run Example

To understand how the model works:

```bash
python train_shortcut_scene_example.py
```

## Important Parameters

### ShortCut Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_mode` | `True` | **CRITICAL**: Must be `True` for scene-level training |
| `self_consistency_prob` | `0.25` | Probability of using self-consistency loss (25% of time) |
| `shortcut_loss_weight` | `1.0` | Weight for self-consistency loss component |
| `self_consistency_cfg_strength` | `3.0` | CFG strength when computing self-consistency target |
| `ratio_cfg_samples_in_self_consistency_target` | `0.5` | Ratio of samples using CFG vs conditional-only |
| `fm_in_shortcut_target_prob` | `0.0` | Probability of using flow matching in shortcut target |
| `fm_eps_max` | `0.0` | Maximum epsilon for flow matching samples |
| `cfg_modalities` | `['shape']` | Which modalities to apply CFG to |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_objects_per_scene` | `32` | Maximum number of objects per scene |
| `num_epochs` | `100` | Number of training epochs |
| `learning_rate` | `1e-4` | Learning rate |
| `weight_decay` | `0.01` | Weight decay |
| `grad_clip` | `1.0` | Gradient clipping value |

## How It Works

### Training Loop

For each scene:

1. **Load Scene Data**: Load all objects in the scene as a single batch
   ```python
   latents = {
       'shape': [9, 256, 256],      # 9 objects in this scene
       'translation': [9, 1, 3],
       'scale': [9, 1, 3],
       '6drotation_normalized': [9, 1, 6],
   }
   ```

2. **Compute Loss**: All 9 objects are processed together
   ```python
   loss, detail_losses = model.loss(
       latents,
       **conditionals  # Shared across all 9 objects
   )
   ```

3. **Loss Components**:
   - **Flow Matching Loss**: Traditional flow matching objective (75% of samples)
   - **Self-Consistency Loss**: Shortcut consistency objective (25% of samples)

   Total loss = `flow_matching_loss + shortcut_loss_weight * self_consistency_loss`

### batch_mode Behavior

When `batch_mode=True`, the model:

1. **Sampling d values**:
   - Either all objects get `d > 0` (self-consistency) or all get `d = 0` (flow matching)
   - This ensures consistency across all objects in the scene

2. **CFG sampling**:
   - Either all objects use CFG or all use conditional-only
   - Maintains scene-level coherence

3. **Conditional handling**:
   - All objects share the same conditioning tensors
   - No per-object indexing of conditionals

## Example: Processing a Scene with 9 Objects

```python
from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut

# Create model with batch_mode=True
model = ShortCut(
    reverse_fn=your_denoising_network,
    batch_mode=True,  # CRITICAL
    self_consistency_prob=0.25,
    # ... other params
)

# Scene with 9 objects
scene_data = {
    'shape': torch.randn(9, 256, 256),
    'translation': torch.randn(9, 1, 3),
    'scale': torch.randn(9, 1, 3),
    '6drotation_normalized': torch.randn(9, 1, 6),
}

# Scene-level conditioning (shared by all 9 objects)
scene_conditioning = {
    'image': scene_image,  # Same image for all objects
}

# Compute loss for all 9 objects simultaneously
loss, detail_losses = model.loss(
    scene_data,
    **scene_conditioning
)

print(f"Loss for {len(scene_data['shape'])} objects: {loss.item()}")
# Output: Loss for 9 objects: 0.1234
```

## Data Preparation

### Pre-computing Latents

You'll need to pre-encode your 3D objects into latent codes using a VAE encoder:

```python
import torch
from your_vae import VAE

# Load VAE encoder
vae = VAE.load_pretrained('path/to/vae')
vae.eval()

# For each scene
scene_objects = load_scene_objects('scene_0001')  # List of 3D objects

latents = {
    'shape': [],
    'translation': [],
    'scale': [],
    '6drotation_normalized': [],
}

for obj in scene_objects:
    # Encode object to latent
    latent = vae.encode(obj)

    latents['shape'].append(latent['shape'])
    latents['translation'].append(obj.translation)
    latents['scale'].append(obj.scale)
    latents['6drotation_normalized'].append(obj.rotation_6d)

# Stack to create [num_objects, ...] tensors
latents = {k: torch.stack(v) for k, v in latents.items()}

# Save
torch.save(latents, 'data_root/train/scene_0001/latents.pt')
```

## Customization

### Custom Dataset

Implement your own dataset by subclassing `SceneDataset`:

```python
class MySceneDataset(SceneDataset):
    def _load_scene_latents(self, scene_id: str):
        # Your custom latent loading logic
        latents = load_my_latents(scene_id)
        return latents

    def _load_scene_conditionals(self, scene_id: str):
        # Your custom conditioning loading logic
        image = load_scene_image(scene_id)
        return {'image': image}
```

### Custom Denoising Network

Create your own denoising network (reverse_fn):

```python
class MyDenoisingNetwork(nn.Module):
    def forward(self, x_t, t, cond, d=None):
        """
        Args:
            x_t: Dict of noisy latents [num_objects, ...]
            t: Timestep [num_objects] or [1] if batch_mode=True
            cond: Conditioning tensor (shared if batch_mode=True)
            d: Shortcut step size [num_objects] or [1] if batch_mode=True

        Returns:
            velocity: Dict of velocity predictions [num_objects, ...]
        """
        # Your denoising logic
        ...
        return velocity
```

## Debugging Tips

### Check batch_mode is enabled

```python
assert model.batch_mode == True, "batch_mode must be True for scene-level training"
```

### Verify scene-level processing

```python
# All objects should share the same t and d values
print(f"t values: {t}")  # Should be same for all objects
print(f"d values: {d}")  # Should be same for all objects
```

### Monitor object counts

```python
# Log average objects per scene
avg_objects = total_objects / num_scenes
print(f"Average objects per scene: {avg_objects:.1f}")
```

## Troubleshooting

### Issue: "batch_mode must be True"

**Solution**: Make sure you initialize ShortCut with `batch_mode=True`

```python
model = ShortCut(..., batch_mode=True)
```

### Issue: Different t/d values for objects in same scene

**Solution**: This shouldn't happen with `batch_mode=True`. Check that:
1. You're not manually overriding `batch_mode` somewhere
2. You're using the latest version of the code

### Issue: Out of memory with large scenes

**Solutions**:
1. Reduce `max_objects_per_scene`
2. Use gradient checkpointing: `use_checkpoint=True`
3. Use mixed precision: `use_fp16=True`
4. Use gradient accumulation

## Performance Tips

1. **Use gradient checkpointing** for large models:
   ```python
   reverse_fn = create_model(..., use_checkpoint=True)
   ```

2. **Enable mixed precision** training:
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       loss, _ = model.loss(latents, **conditionals)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Distribute across multiple GPUs**:
   ```bash
   torchrun --nproc_per_node=8 train_shortcut_scene.py ...
   ```

## Citation

If you use this code, please cite the ShortCut paper:

```bibtex
@article{shortcut2024,
  title={ShortCut: Accelerating Diffusion Models with Self-Consistency},
  author={...},
  journal={arXiv preprint arXiv:2410.12557},
  year={2024}
}
```

## Questions?

If you have questions or issues:
1. Check this documentation
2. Run `train_shortcut_scene_example.py` to see a working example
3. Check the ShortCut model implementation in `sam3d_objects/model/backbone/generator/shortcut/model.py`
