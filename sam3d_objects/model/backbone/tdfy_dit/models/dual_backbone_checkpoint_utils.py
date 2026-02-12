# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Utility functions for loading checkpoints into DualBackboneSparseStructureFlowTdfyWrapper.

This module provides helper functions to:
1. Load existing SparseStructureFlow checkpoint into the dual backbone model
2. Load GlobalSparseStructureFlow checkpoint (when available)
3. Initialize GlobalSparseStructureFlow from scratch while loading SparseStructureFlow from checkpoint
"""

from typing import Optional, Dict, Any
import torch
from loguru import logger
from pathlib import Path

def load_pretrained_checkpoint(
    model,
    sparse_flow_checkpoint: Optional[str] = None,
    state_dict_key: str = "state_dict",
):
    """
    Load a pretrained SparseStructureFlow checkpoint into the dual backbone model.

    Args:
        model: DualBackboneSparseStructureFlowTdfyWrapper instance
        sparse_flow_checkpoint: Path to SparseStructureFlow checkpoint
        state_dict_key: Key to extract state_dict from checkpoint (default: "state_dict")

    Returns:
        The model with loaded SparseStructureFlow checkpoint
    """
    if sparse_flow_checkpoint is not None:
        logger.info(f"Loading SparseStructureFlow checkpoint from {sparse_flow_checkpoint}")
        checkpoint = torch.load(sparse_flow_checkpoint, map_location="cpu", weights_only=False)

        # Extract state dict
        if state_dict_key in checkpoint:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint

        # Remove prefix if it exists (e.g., "backbone.", "reverse_fn.backbone.")
        state_dict = _remove_common_prefix(state_dict)

        # Load into sparse_flow
        missing_keys, unexpected_keys = model.sparse_flow.load_state_dict(
            state_dict, strict=True
        )

        if missing_keys:
            logger.warning(f"Missing keys in SparseStructureFlow: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in SparseStructureFlow: {unexpected_keys}")

        logger.info("SparseStructureFlow checkpoint loaded successfully")
    else:
        logger.info("No SparseStructureFlow checkpoint provided, using initialized weights")

    return model

def load_dual_backbone_from_checkpoints(
    model,
    sparse_flow_checkpoint: Optional[str] = None,
    global_sparse_flow_checkpoint: Optional[str] = None,
    sparse_flow_strict: bool = True,
    global_sparse_flow_strict: bool = True,
    state_dict_key: str = "state_dict",
):
    """
    Load checkpoints into a DualBackboneSparseStructureFlowTdfyWrapper model.

    Args:
        model: DualBackboneSparseStructureFlowTdfyWrapper instance
        sparse_flow_checkpoint: Path to SparseStructureFlow checkpoint (existing model)
        global_sparse_flow_checkpoint: Path to GlobalSparseStructureFlow checkpoint (optional)
        sparse_flow_strict: Whether to strictly enforce checkpoint matching for sparse_flow
        global_sparse_flow_strict: Whether to strictly enforce checkpoint matching for global_sparse_flow
        state_dict_key: Key to extract state_dict from checkpoint (default: "state_dict")

    Returns:
        The model with loaded checkpoints
    """
    if sparse_flow_checkpoint is not None:
        logger.info(f"Loading SparseStructureFlow checkpoint from {sparse_flow_checkpoint}")
        checkpoint = torch.load(sparse_flow_checkpoint, map_location="cpu", weights_only=False)

        # Extract state dict
        if state_dict_key in checkpoint:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint

        # Remove prefix if it exists (e.g., "backbone.", "reverse_fn.backbone.")
        state_dict = _remove_common_prefix(state_dict)

        # Load into sparse_flow
        missing_keys, unexpected_keys = model.sparse_flow.load_state_dict(
            state_dict, strict=sparse_flow_strict
        )

        if missing_keys:
            logger.warning(f"Missing keys in SparseStructureFlow: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in SparseStructureFlow: {unexpected_keys}")

        logger.info("SparseStructureFlow checkpoint loaded successfully")
    else:
        logger.info("No SparseStructureFlow checkpoint provided, using initialized weights")

    if global_sparse_flow_checkpoint is not None:
        logger.info(f"Loading GlobalSparseStructureFlow checkpoint from {global_sparse_flow_checkpoint}")
        checkpoint = torch.load(global_sparse_flow_checkpoint, map_location="cpu", weights_only=False)

        # Extract state dict
        if state_dict_key in checkpoint:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint

        # Remove prefix if it exists
        state_dict = _remove_common_prefix(state_dict)

        # Load into global_sparse_flow
        missing_keys, unexpected_keys = model.global_sparse_flow.load_state_dict(
            state_dict, strict=global_sparse_flow_strict
        )

        if missing_keys:
            logger.warning(f"Missing keys in GlobalSparseStructureFlow: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in GlobalSparseStructureFlow: {unexpected_keys}")

        logger.info("GlobalSparseStructureFlow checkpoint loaded successfully")
    else:
        logger.info("No GlobalSparseStructureFlow checkpoint provided, using initialized weights")

    return model


def _remove_common_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove common prefixes from state dict keys.

    Common prefixes include:
    - "backbone."
    - "reverse_fn.backbone."
    - "module."
    - "generator.backbone.reverse_fn.backbone."
    """
    # Find common prefix
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    prefixes_to_try = [
        "generator.backbone.reverse_fn.backbone.",
        "reverse_fn.backbone.",
        "backbone.",
        "module.",
    ]

    for prefix in prefixes_to_try:
        if all(k.startswith(prefix) for k in keys):
            logger.info(f"Removing prefix '{prefix}' from checkpoint keys")
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def save_dual_backbone_checkpoint(
    model,
    save_path: str,
    save_sparse_flow_only: bool = False,
    save_global_flow_only: bool = False,
    additional_state: Optional[Dict[str, Any]] = None,
):
    """
    Save checkpoints for DualBackboneSparseStructureFlowTdfyWrapper.

    Args:
        model: DualBackboneSparseStructureFlowTdfyWrapper instance
        save_path: Path to save checkpoint (can be a directory for separate files)
        save_sparse_flow_only: If True, save only SparseStructureFlow
        save_global_flow_only: If True, save only GlobalSparseStructureFlow
        additional_state: Additional state to save (e.g., optimizer, epoch)
    """
    save_path = Path(save_path)

    if save_sparse_flow_only:
        # Save only sparse flow
        checkpoint = {
            "state_dict": model.sparse_flow.state_dict(),
        }
        if additional_state:
            checkpoint.update(additional_state)

        save_file = save_path / "sparse_flow.ckpt" if save_path.is_dir() else save_path
        torch.save(checkpoint, save_file)
        logger.info(f"SparseStructureFlow checkpoint saved to {save_file}")

    elif save_global_flow_only:
        # Save only global flow
        checkpoint = {
            "state_dict": model.global_sparse_flow.state_dict(),
        }
        if additional_state:
            checkpoint.update(additional_state)

        save_file = save_path / "global_sparse_flow.ckpt" if save_path.is_dir() else save_path
        torch.save(checkpoint, save_file)
        logger.info(f"GlobalSparseStructureFlow checkpoint saved to {save_file}")

    else:
        # Save both as separate files
        if save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

            # Save sparse flow
            sparse_flow_checkpoint = {
                "state_dict": model.sparse_flow.state_dict(),
            }
            if additional_state:
                sparse_flow_checkpoint.update(additional_state)
            torch.save(sparse_flow_checkpoint, save_path / "sparse_flow.ckpt")

            # Save global flow
            global_flow_checkpoint = {
                "state_dict": model.global_sparse_flow.state_dict(),
            }
            if additional_state:
                global_flow_checkpoint.update(additional_state)
            torch.save(global_flow_checkpoint, save_path / "global_sparse_flow.ckpt")

            # Save combined model state dict
            combined_checkpoint = {
                "state_dict": model.state_dict(),
            }
            if additional_state:
                combined_checkpoint.update(additional_state)
            torch.save(combined_checkpoint, save_path / "combined_model.ckpt")

            logger.info(f"All checkpoints saved to {save_path}/")
        else:
            # Save as single combined file
            checkpoint = {
                "state_dict": model.state_dict(),
            }
            if additional_state:
                checkpoint.update(additional_state)
            torch.save(checkpoint, save_path)
            logger.info(f"Combined checkpoint saved to {save_path}")


def create_dual_backbone_from_single_checkpoint(
    checkpoint_path: str,
    dual_backbone_config: Dict[str, Any],
    state_dict_key: str = "state_dict",
):
    """
    Create a DualBackboneSparseStructureFlowTdfyWrapper from a single SparseStructureFlow checkpoint.

    This is useful for migrating from a single backbone to dual backbone setup.

    Args:
        checkpoint_path: Path to existing SparseStructureFlow checkpoint
        dual_backbone_config: Configuration dict for DualBackboneSparseStructureFlowTdfyWrapper
        state_dict_key: Key to extract state_dict from checkpoint

    Returns:
        DualBackboneSparseStructureFlowTdfyWrapper with SparseStructureFlow loaded and
        GlobalSparseStructureFlow initialized from scratch
    """
    from .dual_backbone_sparse_structure_flow import DualBackboneSparseStructureFlowTdfyWrapper

    # Create model
    model = DualBackboneSparseStructureFlowTdfyWrapper(**dual_backbone_config)

    # Load sparse flow checkpoint
    load_dual_backbone_from_checkpoints(
        model,
        sparse_flow_checkpoint=checkpoint_path,
        global_sparse_flow_checkpoint=None,
        state_dict_key=state_dict_key,
    )

    return model
