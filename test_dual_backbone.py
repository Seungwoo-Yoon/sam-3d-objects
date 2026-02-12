#!/usr/bin/env python3
"""
Test script for DualBackboneSparseStructureFlowTdfyWrapper

This script demonstrates how to:
1. Create a dual backbone model
2. Load checkpoints (if available)
3. Run a forward pass
4. Save checkpoints
"""

import torch
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sam3d_objects.model.backbone.tdfy_dit.models import (
    DualBackboneSparseStructureFlowTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.dual_backbone_checkpoint_utils import (
    load_dual_backbone_from_checkpoints,
    save_dual_backbone_checkpoint,
    load_pretrained_checkpoint
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_config_from_yaml(yaml_path):
    """Load configuration from dual_backbone_generator.yaml."""
    print(f"Loading configuration from: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract dual backbone configuration
    dual_backbone_config = config["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]

    # Convert to OmegaConf for hydra instantiate
    cfg = OmegaConf.create(dual_backbone_config)

    # Instantiate latent_mapping for sparse_flow_config
    sparse_flow_config = dict(cfg["sparse_flow_config"])
    sparse_flow_config["latent_mapping"] = {
        key: instantiate(value)
        for key, value in cfg["sparse_flow_config"]["latent_mapping"].items()
    }

    # Instantiate latent_mapping for global_sparse_flow_config
    global_sparse_flow_config = dict(cfg["global_sparse_flow_config"])
    global_sparse_flow_config["latent_mapping"] = {
        key: instantiate(value)
        for key, value in cfg["global_sparse_flow_config"]["latent_mapping"].items()
    }

    # Extract other parameters
    combine_mode = cfg.get("combine_mode", "add")
    sparse_flow_checkpoint = cfg.get("sparse_flow_checkpoint", None)
    global_sparse_flow_checkpoint = cfg.get("global_sparse_flow_checkpoint", None)

    print(f"✓ Configuration loaded successfully")
    print(f"  - Combine mode: {combine_mode}")
    print(f"  - Sparse flow model_channels: {sparse_flow_config['model_channels']}")
    print(f"  - Sparse flow num_blocks: {sparse_flow_config['num_blocks']}")
    print(f"  - Global flow num_blocks: {global_sparse_flow_config['num_blocks']}")

    return sparse_flow_config, global_sparse_flow_config, combine_mode, sparse_flow_checkpoint, global_sparse_flow_checkpoint


def create_dummy_latent_mapping():
    """Create a dummy latent mapping configuration for testing."""
    from sam3d_objects.model.backbone.tdfy_dit.models.mm_latent import (
        Latent,
        LearntPositionEmbedder,
        ShapePositionEmbedder,
    )

    latent_mapping = {
        "6drotation_normalized": Latent(
            in_channels=6,
            model_channels=1024,
            pos_embedder=LearntPositionEmbedder(
                model_channels=1024,
                token_len=1,
            ),
        ),
        "scale": Latent(
            in_channels=3,
            model_channels=1024,
            pos_embedder=LearntPositionEmbedder(
                model_channels=1024,
                token_len=1,
            ),
        ),
        "shape": Latent(
            in_channels=8,
            model_channels=1024,
            pos_embedder=ShapePositionEmbedder(
                model_channels=1024,
                patch_size=1,
                resolution=16,
            ),
        ),
        "translation": Latent(
            in_channels=3,
            model_channels=1024,
            pos_embedder=LearntPositionEmbedder(
                model_channels=1024,
                token_len=1,
            ),
        ),
        "translation_scale": Latent(
            in_channels=1,
            model_channels=1024,
            pos_embedder=LearntPositionEmbedder(
                model_channels=1024,
                token_len=1,
            ),
        ),
    }

    return latent_mapping


def test_model_creation(use_yaml=True, yaml_path=None):
    """Test creating a dual backbone model.

    Args:
        use_yaml: If True, load configuration from yaml file. Otherwise use dummy config.
        yaml_path: Path to dual_backbone_generator.yaml file.
    """
    print("\n" + "=" * 80)
    print("Test 1: Model Creation")
    print("=" * 80)

    if use_yaml:
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "checkpoints/hf/dual_backbone_generator.yaml"

        # Load configuration from yaml
        sparse_flow_config, global_sparse_flow_config, combine_mode, sparse_flow_checkpoint, global_sparse_flow_checkpoint = load_config_from_yaml(yaml_path)
    else:
        # Create dummy latent mapping
        latent_mapping = create_dummy_latent_mapping()

        # Configuration for both backbones
        sparse_flow_config = {
            "latent_mapping": latent_mapping,
            "latent_share_transformer": {
                "6drotation_normalized": [
                    "6drotation_normalized",
                    "translation",
                    "scale",
                    "translation_scale",
                ]
            },
            "cond_channels": 1024,
            "condition_embedder": None,
            "force_zeros_cond": False,
            "in_channels": 8,
            "model_channels": 1024,
            "num_blocks": 4,  # Small for testing
            "num_heads": 8,
            "out_channels": 8,
            "mlp_ratio": 4,
            "pe_mode": "ape",
            "use_fp16": False,
            "use_checkpoint": False,
            "is_shortcut_model": True,
            "freeze_shared_parameters": False,
        }

        # Same config for global flow (can be different in practice)
        global_sparse_flow_config = sparse_flow_config.copy()
        global_sparse_flow_config["latent_mapping"] = create_dummy_latent_mapping()
        combine_mode = "add"

    # Create model
    print("\nCreating DualBackboneSparseStructureFlowTdfyWrapper...")
    model = DualBackboneSparseStructureFlowTdfyWrapper(
        sparse_flow_config=sparse_flow_config,
        global_sparse_flow_config=global_sparse_flow_config,
        combine_mode=combine_mode,
        checkpoint_dir="./checkpoints/hf/",
        sparse_flow_checkpoint=sparse_flow_checkpoint,
        global_sparse_flow_checkpoint=global_sparse_flow_checkpoint,
    )
    model = model.to("cuda")

    print(f"✓ Model created successfully")
    print(f"  - SparseStructureFlow parameters: {sum(p.numel() for p in model.sparse_flow.parameters()):,}")
    print(f"  - GlobalSparseStructureFlow parameters: {sum(p.numel() for p in model.global_sparse_flow.parameters()):,}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, sparse_flow_config, global_sparse_flow_config


def test_forward_pass(model):
    """Test forward pass through the model."""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)

    batch_size = 32
    resolution = 16
    num_cond_tokens = 256  # Typical number of tokens from image embedder

    # Create dummy input
    latents_dict = {
        "6drotation_normalized": torch.randn(batch_size, 1, 6).cuda(),
        "scale": torch.randn(batch_size, 1, 3).cuda(),
        "shape": torch.randn(batch_size, resolution**3, 8).cuda(),
        "translation": torch.randn(batch_size, 1, 3).cuda(),
        "translation_scale": torch.randn(batch_size, 1, 1).cuda(),
    }

    t = torch.rand(1).cuda()
    d = torch.rand(1).cuda()
    # Conditioning should be 3D: (batch_size, num_tokens, embed_dim)
    cond = torch.randn(batch_size, num_cond_tokens, 1024).cuda()

    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(
            latents_dict,
            t,
            cond,
            d=d,
            cfg=False,
        )

    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(output.keys())}")
    for key, value in output.items():
        print(f"  - {key}: shape={value.shape}")

    return output


def test_checkpoint_save_load(model, sparse_flow_config, global_sparse_flow_config):
    """Test saving and loading checkpoints."""
    print("\n" + "=" * 80)
    print("Test 3: Checkpoint Save/Load")
    print("=" * 80)

    # Create temporary directory for checkpoints
    checkpoint_dir = Path("tmp_dual_backbone_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        # Save checkpoints
        print("Saving checkpoints...")
        save_dual_backbone_checkpoint(
            model,
            save_path=checkpoint_dir,
            additional_state={"epoch": 0, "step": 100},
        )
        print(f"✓ Checkpoints saved to {checkpoint_dir}/")

        # Create new model
        print("\nCreating new model...")
        new_model = DualBackboneSparseStructureFlowTdfyWrapper(
            sparse_flow_config=sparse_flow_config,
            global_sparse_flow_config=global_sparse_flow_config,
            combine_mode="add",
        ).cuda()

        # Load checkpoints
        print("Loading checkpoints...")
        load_dual_backbone_from_checkpoints(
            new_model,
            sparse_flow_checkpoint=checkpoint_dir / "sparse_flow.ckpt",
            global_sparse_flow_checkpoint=checkpoint_dir / "global_sparse_flow.ckpt",
        )
        print("✓ Checkpoints loaded successfully")

        # Verify parameters match
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        loaded_params = new_model.state_dict()

        all_match = True
        for key in original_params.keys():
            if not torch.allclose(original_params[key], loaded_params[key]):
                print(f"  ✗ Parameter mismatch: {key}")
                all_match = False

        if all_match:
            print("✓ All parameters match!")
        else:
            print("✗ Some parameters don't match")

    finally:
        # Cleanup
        import shutil
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"\nCleaned up temporary directory: {checkpoint_dir}")


def test_separate_backbone_access(model):
    """Test accessing individual backbones."""
    print("\n" + "=" * 80)
    print("Test 4: Separate Backbone Access")
    print("=" * 80)

    batch_size = 32
    resolution = 16
    num_cond_tokens = 256  # Typical number of tokens from image embedder

    latents_dict = {
        "6drotation_normalized": torch.randn(batch_size, 1, 6),
        "scale": torch.randn(batch_size, 1, 3),
        "shape": torch.randn(batch_size, resolution**3, 8),
        "translation": torch.randn(batch_size, 1, 3),
        "translation_scale": torch.randn(batch_size, 1, 1),
    }

    t = torch.rand(1)
    d = torch.rand(1)
    # Conditioning should be 3D: (batch_size, num_tokens, embed_dim)
    cond = torch.randn(batch_size, num_cond_tokens, 1024)

    model.eval()
    with torch.no_grad():
        # Run through sparse flow only
        print("Running SparseStructureFlow only...")
        sparse_output = model.sparse_flow(latents_dict, t, cond, d=d)
        print(f"  ✓ SparseStructureFlow output keys: {list(sparse_output.keys())}")

        # Run through global sparse flow only
        print("Running GlobalSparseStructureFlow only...")
        global_output = model.global_sparse_flow(latents_dict, t, cond, d=d)
        print(f"  ✓ GlobalSparseStructureFlow output keys: {list(global_output.keys())}")

        # Run through combined model
        print("Running combined model...")
        combined_output = model(latents_dict, t, cond, d=d)

        # Verify combination
        print("\nVerifying output combination...")
        all_match = True
        for key in combined_output.keys():
            expected = sparse_output[key] + global_output[key]
            if not torch.allclose(combined_output[key], expected, rtol=1e-4):
                print(f"  ✗ Combination mismatch for {key}")
                all_match = False

        if all_match:
            print("  ✓ Combined output matches (sparse + global)")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing DualBackboneSparseStructureFlowTdfyWrapper")
    print("=" * 80)

    # Check if dual_backbone_generator.yaml exists
    yaml_path = Path(__file__).parent / "checkpoints/hf/dual_backbone_generator.yaml"
    use_yaml = yaml_path.exists()

    if use_yaml:
        print(f"Using configuration from: {yaml_path}")
    else:
        print("dual_backbone_generator.yaml not found, using dummy configuration")

    try:
        # Test 1: Model creation
        model, sparse_config, global_config = test_model_creation(use_yaml=use_yaml, yaml_path=yaml_path)

        # Test 2: Forward pass
        test_forward_pass(model)

        # Test 3: Checkpoint save/load
        test_checkpoint_save_load(model, sparse_config, global_config)

        # Test 4: Separate backbone access
        test_separate_backbone_access(model)

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 80 + "\n")
        raise


if __name__ == "__main__":
    main()
