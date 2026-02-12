from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae import SLatGaussianDecoder, SLatEncoder
from omegaconf import OmegaConf
import os
from hydra.utils import instantiate
from safetensors.torch import load_file
from sam3d_objects.model.io import load_model_from_checkpoint
import numpy as np
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
import torch

def instantiate_and_load_from_pretrained(
    config,
    ckpt_path,
    state_dict_fn=None,
    state_dict_key="state_dict",
    device="cuda", 
):
    model = instantiate(config)

    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device="cuda")
        if state_dict_fn is not None:
            state_dict = state_dict_fn(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        model = load_model_from_checkpoint(
            model,
            ckpt_path,
            strict=True,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key=state_dict_key,
            state_dict_fn=state_dict_fn,
        )
    model = model.to(device)

    return model

encoder: SLatEncoder = instantiate_and_load_from_pretrained()

decoder: SLatGaussianDecoder = instantiate_and_load_from_pretrained(
    OmegaConf.load("../checkpoints/hf/slat_decoder_gs.yaml"),
    "../checkpoints/hf/slat_decoder_gs.ckpt",
    device='cuda',
    state_dict_key=None,
)

feature = np.load('../debug/features/dinov2_features.npy')  # (N, 1024)
indices = np.load('../debug/features/voxel_indices.npy')    # (N, 3)

sp = SparseTensor(
    feats=torch.from_numpy(feature).float().cuda(),
    coords=torch.from_numpy(indices).int().cuda()
)

gs = decoder(sp)