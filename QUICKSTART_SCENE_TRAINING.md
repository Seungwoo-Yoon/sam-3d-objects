# QuickStart: Scene-Level Training for ShortCut Model

This guide gets you started with training the ShortCut flow model on entire scenes at once.

## 🎯 What This Does

Trains a ShortCut (SS generator) model where:
- **Each training step processes ALL objects in a scene together** (e.g., 9 objects at once)
- All objects share the same scene-level conditioning
- Loss is calculated for the entire scene simultaneously

This is critical for multi-object scene generation!

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `train_shortcut_scene.py` | Main training script with full pipeline |
| `train_shortcut_scene_example.py` | Example showing how to use the model |
| `utils_prepare_scene_data.py` | Helper to prepare your data |
| `run_training_shortcut.sh` | Convenient bash script for training |
| `TRAINING_SHORTCUT_SCENE.md` | Detailed documentation |
| `QUICKSTART_SCENE_TRAINING.md` | This file |

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Your Data

Create example data to test:

```bash
python utils_prepare_scene_data.py example
```

This creates `./example_data/train/scene_example/` with the correct format.

For your real data, organize it like:
```
data_root/
├── train/
│   ├── scene_0001/
│   │   └── latents.pt      # Dict with keys: shape, translation, scale, 6drotation_normalized
│   ├── scene_0002/
│   └── ...
└── val/
    └── ...
```

### Step 2: Test the Model

Run the example to verify everything works:

```bash
python train_shortcut_scene_example.py
```

Expected output:
```
Total loss: 0.1234
Flow matching loss: 0.0567
Self-consistency loss: 0.0667
Processed 9 objects in the scene simultaneously.
```

### Step 3: Start Training

Using the shell script (easiest):

```bash
# Single GPU
./run_training_shortcut.sh --data_root /path/to/data --output_dir ./outputs/exp1

# Multiple GPUs (e.g., 4 GPUs)
./run_training_shortcut.sh --data_root /path/to/data --num_gpus 4 --preset large
```

Or directly with Python:

```bash
# Single GPU
python train_shortcut_scene.py \
    --data_root /path/to/data \
    --output_dir ./outputs/exp1 \
    --num_epochs 100 \
    --learning_rate 1e-4

# Multiple GPUs with torchrun
torchrun --nproc_per_node=4 train_shortcut_scene.py \
    --data_root /path/to/data \
    --output_dir ./outputs/exp1
```

## ⚙️ Key Configuration

### Critical Parameter: `batch_mode=True`

This MUST be set for scene-level training:

```python
model = ShortCut(
    reverse_fn=your_network,
    batch_mode=True,  # ⭐ CRITICAL!
    # ... other params
)
```

### Important Hyperparameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `self_consistency_prob` | 0.25 | How often to use self-consistency loss (25% of batches) |
| `shortcut_loss_weight` | 1.0 | Weight for self-consistency component |
| `cfg_strength` | 3.0 | Classifier-free guidance strength |
| `max_objects_per_scene` | 32 | Max objects per scene (filter larger scenes) |

## 📊 Understanding the Training

### What Happens During Training

For a scene with 9 objects:

```python
# Input: All 9 objects in one batch
latents = {
    'shape': [9, 256, 256],           # 9 objects
    'translation': [9, 1, 3],
    'scale': [9, 1, 3],
    '6drotation_normalized': [9, 1, 6],
}

# All objects share scene-level conditioning
conditionals = {
    'image': scene_image,  # Same for all 9 objects
}

# Compute loss for all 9 objects together
loss, details = model.loss(latents, **conditionals)

# Total loss = flow_matching_loss + self_consistency_loss
```

### Loss Components

1. **Flow Matching Loss** (75% of time): Standard flow matching objective
2. **Self-Consistency Loss** (25% of time): Shortcut consistency objective

The model randomly chooses which loss to use, but when `batch_mode=True`, all objects in the scene use the same loss type.

## 🔍 Verify Your Setup

### Check 1: Data Format

```bash
python utils_prepare_scene_data.py verify --scene_dir /path/to/scene_0001
```

Should output:
```
✓ Found latents.pt
  ✓ shape: [N, 256, 256]
  ✓ translation: [N, 1, 3]
  ✓ scale: [N, 1, 3]
  ✓ 6drotation_normalized: [N, 1, 6]
✓ All N objects have consistent shapes
✅ Scene verification passed!
```

### Check 2: Model Configuration

```python
# In your code, verify:
assert model.batch_mode == True, "batch_mode must be True!"
```

### Check 3: DataLoader

```python
# DataLoader batch_size must be 1 (one scene at a time)
dataloader = DataLoader(
    dataset,
    batch_size=1,  # ⭐ Must be 1
    collate_fn=scene_collate_fn,
)
```

## 💡 Tips & Tricks

### Start Small

Test with a small subset first:
```bash
./run_training_shortcut.sh \
    --data_root ./example_data \
    --num_epochs 5 \
    --preset small
```

### Monitor Training

Key metrics to watch:
- `flow_matching_loss`: Should decrease steadily
- `self_consistency_loss`: Should decrease (may be noisy)
- `avg_objects_per_scene`: Should match your data

### Handle Large Scenes

If you have scenes with many objects (>32):

**Option 1**: Filter them
```python
# In dataset
max_objects_per_scene = 32
```

**Option 2**: Sample subset
```python
# Randomly sample 32 objects from larger scenes
if num_objects > max_objects_per_scene:
    indices = torch.randperm(num_objects)[:max_objects_per_scene]
    latents = tree_tensor_map(lambda x: x[indices], latents)
```

**Option 3**: Use gradient checkpointing
```python
model = create_model(..., use_checkpoint=True)
```

## 🐛 Troubleshooting

### "batch_mode must be True"
✅ **Fix**: Set `batch_mode=True` when creating ShortCut model

### Out of Memory
✅ **Fixes**:
1. Reduce `max_objects_per_scene`
2. Enable `use_checkpoint=True`
3. Enable `use_fp16=True`
4. Use gradient accumulation

### Different t/d values for objects
✅ **Fix**: This shouldn't happen with `batch_mode=True`. Check your ShortCut version.

### Loss is NaN
✅ **Fixes**:
1. Reduce learning rate
2. Enable gradient clipping: `--grad_clip 1.0`
3. Check data for NaN/Inf values

## 📚 Next Steps

1. **Read the detailed docs**: [TRAINING_SHORTCUT_SCENE.md](TRAINING_SHORTCUT_SCENE.md)
2. **Customize the model**: Modify `train_shortcut_scene_example.py`
3. **Prepare your data**: Use `utils_prepare_scene_data.py`
4. **Run experiments**: Try different hyperparameters

## 🎓 Key Concepts

### Scene-Level vs Object-Level

| Aspect | Object-Level | Scene-Level (What you want) |
|--------|--------------|----------------------------|
| `batch_mode` | `False` | `True` ⭐ |
| Batch contains | N independent objects | All objects from 1 scene |
| Conditioning | Per-object | Per-scene (shared) |
| DataLoader batch_size | Any (e.g., 32) | 1 (one scene) |
| Use case | Independent generation | Scene generation |

### Why Scene-Level Training?

When generating multi-object scenes, you want:
- **Consistency**: All objects should fit together
- **Shared context**: Objects share scene-level information (e.g., lighting, style)
- **Joint optimization**: Objects influence each other during training

Scene-level training (with `batch_mode=True`) achieves this!

## 📞 Need Help?

1. Run the example: `python train_shortcut_scene_example.py`
2. Check detailed docs: [TRAINING_SHORTCUT_SCENE.md](TRAINING_SHORTCUT_SCENE.md)
3. Verify your data: `python utils_prepare_scene_data.py verify --scene_dir /path/to/scene`
4. Check the model code: [sam3d_objects/model/backbone/generator/shortcut/model.py](sam3d_objects/model/backbone/generator/shortcut/model.py)

## ✨ Summary

```bash
# 1. Create example data
python utils_prepare_scene_data.py example

# 2. Test the model
python train_shortcut_scene_example.py

# 3. Train!
./run_training_shortcut.sh --data_root /path/to/data --output_dir ./outputs/exp1
```

That's it! You're now training a ShortCut model with scene-level batching. 🎉

---

**Remember**: The key is `batch_mode=True` and `batch_size=1` (one scene at a time)!