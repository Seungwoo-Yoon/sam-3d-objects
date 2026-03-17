"""Frozen downstream model loading utilities.

Mirrors inference_pipeline.py's instantiate_and_load_from_pretrained pattern.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sam3d_objects.model.io import filter_and_remove_prefix_state_dict_fn

logger = logging.getLogger(__name__)


def _instantiate_and_load(
    cfg,
    ckpt_path: str,
    device: torch.device,
    state_dict_fn=None,
    state_dict_key: Optional[str] = "state_dict",
    strict: bool = True,
) -> nn.Module:
    """
    Instantiate model from OmegaConf cfg and load weights from ckpt_path.

    state_dict_key=None  →  treat the whole checkpoint as the state dict
                           (used for ss_decoder, slat_decoder_mesh)
    state_dict_fn        →  prefix-stripping function applied before load
                           (used for generator / condition-embedder sub-dicts)
    """
    model = instantiate(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if state_dict_key is None:
        sd = ckpt
    else:
        sd = ckpt.get(state_dict_key, ckpt)

    if state_dict_fn is not None:
        sd = state_dict_fn(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ss_decoder(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """Load ss_decoder (direct checkpoint, no prefix stripping)."""
    cfg = OmegaConf.load(config_path)
    if "pretrained_ckpt_path" in cfg:
        del cfg["pretrained_ckpt_path"]
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_key=None)
    logger.info(f"Loaded ss_decoder from {ckpt_path}")
    return model


def load_slat_generator(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load slat_generator (ShortCut + CFG + Backbone).

    Config section: config["module"]["generator"]["backbone"]
    Checkpoint prefix: _base_models.generator.
    """
    cfg = OmegaConf.load(config_path)["module"]["generator"]["backbone"]
    prefix_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_fn=prefix_fn, strict=False)
    logger.info(f"Loaded slat_generator from {ckpt_path}")
    return model


def load_slat_decoder_mesh(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """Load slat_decoder_mesh (direct checkpoint, no prefix stripping)."""
    cfg = OmegaConf.load(config_path)
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_key=None, strict=False)
    logger.info(f"Loaded slat_decoder_mesh from {ckpt_path}")
    return model


def load_condition_embedder(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load a condition embedder from a generator config + checkpoint.

    Config section: config["module"]["condition_embedder"]["backbone"]
    Checkpoint prefix: _base_models.condition_embedder.
    """
    cfg = OmegaConf.load(config_path)["module"]["condition_embedder"]["backbone"]
    prefix_fn = filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder.")
    model = _instantiate_and_load(cfg, ckpt_path, device, state_dict_fn=prefix_fn)
    logger.info(f"Loaded condition embedder from {ckpt_path}")
    return model


def load_slat_preprocessor(config_path: str) -> Any:
    """Load SLAT preprocessor from a generator config's val_preprocessor section."""
    cfg = OmegaConf.load(config_path)
    preprocessor = instantiate(cfg["slat_preprocessor"])
    logger.info(f"Loaded SLAT preprocessor from {config_path}")
    return preprocessor
