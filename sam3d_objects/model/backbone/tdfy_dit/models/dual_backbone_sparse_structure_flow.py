# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import *
import torch
import torch.nn as nn
from loguru import logger

from .mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
from .global_sparse_structure_flow import GlobalSparseStructureFlowTdfyWrapper


class DualBackboneSparseStructureFlowTdfyWrapper(nn.Module):
    """
    Dual backbone wrapper that combines SparseStructureFlowTdfyWrapper and GlobalSparseStructureFlowTdfyWrapper.

    This class runs both models in parallel and combines their outputs using element-wise addition.
    Each backbone can be loaded from separate checkpoints.

    Args:
        sparse_flow_config: Configuration dict for SparseStructureFlowTdfyWrapper
        global_sparse_flow_config: Configuration dict for GlobalSparseStructureFlowTdfyWrapper
        sparse_flow_checkpoint: Optional path to checkpoint for SparseStructureFlowTdfyWrapper
        global_sparse_flow_checkpoint: Optional path to checkpoint for GlobalSparseStructureFlowTdfyWrapper
        combine_mode: How to combine outputs ('add', 'weighted'). Default: 'add'
        global_flow_weight: Weight for global flow output when combine_mode='weighted'. Default: 1.0
    """

    def __init__(
        self,
        sparse_flow_config: dict,
        global_sparse_flow_config: dict,
        checkpoint_dir: Optional[str] = "",
        sparse_flow_checkpoint: Optional[str] = None,
        global_sparse_flow_checkpoint: Optional[str] = None,
        combine_mode: Literal["add", "weighted"] = "add",
        global_flow_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Initialize both backbones
        logger.info("Initializing SparseStructureFlowTdfyWrapper (primary backbone)...")
        self.sparse_flow = SparseStructureFlowTdfyWrapper(**sparse_flow_config)

        logger.info("Initializing GlobalSparseStructureFlowTdfyWrapper (global backbone)...")
        self.global_sparse_flow = GlobalSparseStructureFlowTdfyWrapper(**global_sparse_flow_config)

        self.combine_mode = combine_mode

        # For weighted combination
        if combine_mode == "weighted":
            self.global_flow_weight = nn.Parameter(torch.tensor(global_flow_weight))
        else:
            self.register_buffer("global_flow_weight", torch.tensor(1.0))


        # Load checkpoints if provided
        if sparse_flow_checkpoint is not None:
            self.load_sparse_flow_checkpoint(os.path.join(checkpoint_dir, sparse_flow_checkpoint))

        if global_sparse_flow_checkpoint is not None:
            self.load_global_sparse_flow_checkpoint(os.path.join(checkpoint_dir, global_sparse_flow_checkpoint))
        else:
            logger.info("GlobalSparseStructureFlowTdfyWrapper initialized with default weights (no checkpoint loaded)")

        self.latent_mapping = self.sparse_flow.latent_mapping

    def load_sparse_flow_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load checkpoint for SparseStructureFlowTdfyWrapper.

        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in checkpoint match the model
        """
        logger.info(f"Loading SparseStructureFlowTdfyWrapper checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load with prefix handling if needed
        self.sparse_flow.load_state_dict(state_dict, strict=strict)
        logger.info("SparseStructureFlowTdfyWrapper checkpoint loaded successfully")

    def load_global_sparse_flow_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load checkpoint for GlobalSparseStructureFlowTdfyWrapper.

        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in checkpoint match the model
        """
        logger.info(f"Loading GlobalSparseStructureFlowTdfyWrapper checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load with prefix handling if needed
        self.global_sparse_flow.load_state_dict(state_dict, strict=strict)
        logger.info("GlobalSparseStructureFlowTdfyWrapper checkpoint loaded successfully")

    def forward(
        self,
        latents_dict: dict,
        t: torch.Tensor,
        *condition_args,
        **condition_kwargs,
    ) -> dict:
        """
        Forward pass through both backbones in parallel and combine outputs.

        Args:
            latents_dict: Dictionary of latent representations
            t: Timestep tensor
            *condition_args: Positional arguments for condition embedder
            **condition_kwargs: Keyword arguments for condition embedder

        Returns:
            Combined output dictionary
        """

        # Run both models in parallel
        sparse_flow_output = self.sparse_flow(
            latents_dict, t, *condition_args, **condition_kwargs
        )

        d = condition_kwargs.pop("d", [None])

        global_sparse_flow_output = self.global_sparse_flow(
            latents_dict, t[0], d=d[0], *condition_args, **condition_kwargs
        )

        # Combine outputs
        combined_output = {}

        # Ensure both outputs have the same keys
        assert set(sparse_flow_output.keys()) == set(global_sparse_flow_output.keys()), \
            f"Output keys mismatch: {sparse_flow_output.keys()} vs {global_sparse_flow_output.keys()}"

        for key in sparse_flow_output.keys():
            if self.combine_mode == "add":
                # Simple element-wise addition
                combined_output[key] = sparse_flow_output[key] + global_sparse_flow_output[key]
            elif self.combine_mode == "weighted":
                # Weighted addition
                combined_output[key] = (
                    sparse_flow_output[key] +
                    self.global_flow_weight * global_sparse_flow_output[key]
                )

        return combined_output

    def state_dict(self, *args, **kwargs):
        """
        Return state dict with prefixes for each backbone.
        """
        state_dict = {}

        # Add sparse flow state dict with prefix
        for k, v in self.sparse_flow.state_dict(*args, **kwargs).items():
            state_dict[f"sparse_flow.{k}"] = v

        # Add global sparse flow state dict with prefix
        for k, v in self.global_sparse_flow.state_dict(*args, **kwargs).items():
            state_dict[f"global_sparse_flow.{k}"] = v

        # Add combine parameters
        if self.combine_mode == "weighted":
            state_dict["global_flow_weight"] = self.global_flow_weight

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with prefix handling.
        """
        sparse_flow_state = {}
        global_sparse_flow_state = {}
        other_state = {}

        for k, v in state_dict.items():
            if k.startswith("sparse_flow."):
                sparse_flow_state[k[len("sparse_flow."):]] = v
            elif k.startswith("global_sparse_flow."):
                global_sparse_flow_state[k[len("global_sparse_flow."):]] = v
            else:
                other_state[k] = v

        print(len(sparse_flow_state), len(global_sparse_flow_state), len(other_state))
        print(global_sparse_flow_state)

        # Load each backbone
        # if sparse_flow_state:
        #     self.sparse_flow.load_state_dict(sparse_flow_state, strict=strict)

        if global_sparse_flow_state:
            self.global_sparse_flow.load_state_dict(global_sparse_flow_state, strict=strict)

        # Load other parameters
        if "global_flow_weight" in other_state and self.combine_mode == "weighted":
            self.global_flow_weight.data = other_state["global_flow_weight"]

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    @property
    def condition_embedder(self):
        """
        Return the condition embedder from sparse_flow backbone.
        This property provides compatibility with ClassifierFreeGuidance wrappers
        that expect to access backbone.condition_embedder.

        Both backbones should have the same condition_embedder, so we return
        the one from sparse_flow.
        """
        return self.sparse_flow.condition_embedder
