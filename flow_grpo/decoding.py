"""Frozen downstream pipeline decoding: shape latent → mesh/SDF."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.pipeline.inference_utils import (
    SLAT_MEAN,
    SLAT_STD,
    prune_sparse_structure,
    downsample_sparse_structure,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def decode_shape_to_sdf(
    shape_latent: torch.Tensor,     # (N, L, 8) – the 'shape' entry of x_1
    ss_decoder: nn.Module,
    slat_generator: nn.Module,
    slat_decoder_mesh: nn.Module,
    cond_embed: torch.Tensor,     # (N, L_slat, C_slat) – from slat_condition_embedder
    device: torch.device,
) -> Optional[List]:
    """
    Decode a shape latent through the frozen pipeline to produce SDF output.

    Steps:
      1. ss_decoder:         shape_latent  →  voxel occupancy  →  coords
      2. slat_condition_embedder + slat_input_dict  →  slat_cond_embed
      3. slat_generator:     coords + slat_cond_embed  →  SLAT
      4. slat_decoder_mesh:  SLAT  →  mesh (contains SDF)

    Returns:
      list of MeshExtractResult (one per object); or empty list if voxel is empty.
    """
    # 1. ss_decoder → voxel → coords
    ss_vol = ss_decoder(
        shape_latent.permute(0, 2, 1)
        .contiguous()
        .view(shape_latent.shape[0], 8, 16, 16, 16)
    )
    coords = torch.argwhere(ss_vol > 0)[:, [0, 2, 3, 4]].int()

    if coords.shape[0] == 0:
        logger.debug("Empty sparse structure – skipping this sample.")
        return []   # return empty list, not None — callers use len()

    meshes = []

    # for i, object_embed in enumerate(cond_embed):
    #     # 3. slat_generator  (coords passed as numpy, same as inference_pipeline)
    #     obj_coords = coords[coords[:, 0] == i]
    #     if obj_coords.shape[0] == 0:
    #         # This object has no voxels — append None as placeholder so that
    #         # len(meshes) == len(cond_embed) and pose indices stay aligned.
    #         logger.debug(f"Object {i} has no voxels, skipping mesh decode.")
    #         meshes.append(None)
    #         continue

    #     # Reset batch index to 0 — each slat_generator call is single-object (batch=1)
    #     obj_coords_local = obj_coords.clone()
    #     obj_coords_local[:, 0] = 0
    #     # Mirror inference_pipeline: prune interior voxels then cap at 42k for mesh decoding
    #     obj_coords_local = prune_sparse_structure(obj_coords_local)
    #     obj_coords_local, _ = downsample_sparse_structure(obj_coords_local)
    #     if obj_coords_local.shape[0] == 0:
    #         meshes.append(None)
    #         continue

    #     latent_shape = (1, obj_coords_local.shape[0], 8)
    #     coords_np = obj_coords_local.cpu().numpy()

    #     slat = slat_generator(latent_shape, device, object_embed[None, ...], coords_np)

    #     slat = sp.SparseTensor(
    #         coords=obj_coords_local,
    #         feats=slat[0],
    #     ).to(device)
    #     slat = slat * SLAT_STD.to(device) + SLAT_MEAN.to(device)

    #     # 4. slat_decoder_mesh  →  mesh / SDF
    #     mesh_out = slat_decoder_mesh(slat)[0]
    #     meshes.append(mesh_out)

    for i in range(0, cond_embed.shape[0], 8):
        # 3. slat_generator  (coords passed as numpy, same as inference_pipeline)
        start = i
        end = min(i + 8, cond_embed.shape[0])
        batch_coords = coords[(coords[:, 0] >= start) & (coords[:, 0] < end)]
        if batch_coords.shape[0] == 0:
            logger.debug(f"Objects {start} to {end-1} have no voxels, skipping mesh decode.")
            meshes.extend([None] * (end - start))
            continue
        batch_coords_local = batch_coords.clone()
        batch_coords_local[:, 0] = batch_coords_local[:, 0] - start

        batch_coords_local = prune_sparse_structure(batch_coords_local)
        batch_coords_local, _ = downsample_sparse_structure(batch_coords_local)
        if batch_coords_local.shape[0] == 0:
            meshes.extend([None] * (end - start))
            continue

        latent_shape = (1, batch_coords_local.shape[0], 8)
        coords_np = batch_coords_local.cpu().numpy()
        slat = slat_generator(latent_shape, device, cond_embed[start:end], coords_np)
        slat = sp.SparseTensor(
            coords=batch_coords_local,
            feats=slat[0],
        ).to(device)
        slat = slat * SLAT_STD.to(device) + SLAT_MEAN.to(device)
        mesh_outs = slat_decoder_mesh(slat)
        for j in range(end - start):
            meshes.append(mesh_outs[j])

    return meshes
