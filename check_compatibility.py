from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3d_objects.model.io import load_model_from_checkpoint
from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import SparseStructureEncoderTdfyWrapper
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np
import torch
from sam3d_objects.custom.utils import *

with open('./checkpoints/hf/ss_decoder.yaml'):
    config = OmegaConf.load('./checkpoints/hf/ss_decoder.yaml')
    decoder = instantiate(config)

# decoder = load_model_from_checkpoint(
#     decoder,
#     './checkpoints/hf/ss_decoder.ckpt',
#     strict=True,
#     device='cpu',
#     freeze=True,
#     eval=True,
#     state_dict_key=None
# )

model_path = hf_hub_download(repo_id="JeffreyXiang/TRELLIS-image-large",
                                     filename="ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
state_dict = load_file(model_path)
decoder.load_state_dict(state_dict, strict=True)

decoder = decoder.eval().cuda()

encoder = SparseStructureEncoderTdfyWrapper(
    in_channels=1,
    latent_channels=8,
    num_res_blocks=2,
    num_res_blocks_middle=2,
    channels=[32, 128, 512],
    use_fp16=True,
    sample_posterior=True
)

model_path = hf_hub_download(repo_id="microsoft/TRELLIS-image-large",
                                     filename="ckpts/ss_enc_conv3d_16l8_fp16.safetensors")
state_dict = load_file(model_path)
encoder.load_state_dict(state_dict, strict=True)
encoder = encoder.eval().cuda()
print("Encoder loaded successfully")

voxel = np.load("./debug/occupancy_grids/foundationpose_4_occupancy_grid.npy")
voxel = voxel.astype(np.float32)
voxel = np.expand_dims(voxel, axis=0) # add batch dimension
voxel = np.expand_dims(voxel, axis=0) # add channel dimension
voxel = torch.from_numpy(voxel) # convert to torch tensor
voxel = voxel.to(torch.float16) # convert to float16
voxel = voxel.cuda() # move to GPU

print(voxel.shape)

with torch.no_grad():
    latent = encoder(voxel)['z']
    print(latent.shape)

    decoded = decoder(latent)
    print(decoded.shape)

# save decoded output as point cloud (ply)
decoded = decoded.cpu().numpy()
decoded = np.squeeze(decoded) # remove batch and channel dimensions
decoded = np.clip(decoded, 0, 1) # clip to [0, 1]
decoded = decoded > 0.5 # threshold to binary occupancy
indices = np.argwhere(decoded) # get occupied voxel indices
points = indices / 64.0 - 0.5 # convert to [0, 1] and center at 0

save_pc(points, "./debug/decoded_pointclouds/foundationpose_4_decoded.ply", color=(255, 0, 0, 255))