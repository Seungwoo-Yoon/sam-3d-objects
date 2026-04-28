"""Flow-GRPO-Fast package for RL finetuning of 3D sparse structure generation."""

from .model_loaders import (
    load_ss_decoder,
    load_slat_generator,
    load_slat_decoder_mesh,
    load_condition_embedder,
    load_slat_preprocessor,
)
from .qlora import quantize_backbone_4bit, enable_gradient_checkpointing
from .sde import sde_sigma, sde_mu_dict, sde_step_dict
from .generation import generate_sde_group
from .decoding import decode_shape_to_sdf
from .reward import compute_reward
from .loss import compute_grpo_loss_single
from .trainer import train_epoch_grpo, RunningStats

__all__ = [
    "load_ss_decoder",
    "load_slat_generator",
    "load_slat_decoder_mesh",
    "load_condition_embedder",
    "load_slat_preprocessor",
    "quantize_backbone_4bit",
    "enable_gradient_checkpointing",
    "sde_sigma",
    "sde_mu_dict",
    "sde_step_dict",
    "generate_sde_group",
    "decode_shape_to_sdf",
    "compute_reward",
    "compute_grpo_loss_single",
    "train_epoch_grpo",
    "RunningStats",
]
