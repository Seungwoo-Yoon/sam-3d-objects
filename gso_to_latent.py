from glob import glob
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)
import torch
import trimesh
import numpy as np
from tqdm import tqdm, trange
import os

def voxelize_surface(vertices, faces, resolution=64, num_samples=200_000):
    """
    Surface voxelization via sampling.
    Mesh coordinates assumed in [-1,1].
    """

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 1. sample surface
    pts, _ = trimesh.sample.sample_surface(mesh, num_samples)

    # 2. normalize [-1,1] -> [0,1]
    pts = (pts + 1.0) / 2.0

    # 3. map to voxel index
    ijk = (pts * resolution).astype(np.int32)

    # clamp (edge case when coord==1.0)
    ijk = np.clip(ijk, 0, resolution - 1)

    vox = np.zeros((resolution, resolution, resolution), dtype=bool)
    vox[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    return vox

config = OmegaConf.load('./checkpoints/hf/ss_decoder.yaml')
decoder = instantiate(config)

OPTIMIZATION_STEPS = 200
POINT_CLOUD = True

decoder = load_model_from_checkpoint(
    decoder,
    './checkpoints/hf/ss_decoder.ckpt',
    strict=True,
    device='cpu',
    freeze=True,
    eval=True,
    state_dict_key=None
)
decoder = decoder.eval().cuda()

os.makedirs('./gso/google_scanned_objects/latent_codes', exist_ok=True)
os.makedirs('./gso/google_scanned_objects/point_clouds', exist_ok=True)

# iterate over all objects in gso (./gso/google_scanned_objects/models_normalized/*/meshes/model.obj)
for obj_path in tqdm(glob('./gso/google_scanned_objects/models_normalized/*/meshes/model.obj'), position=0, desc="Processing GSO objects"):
    latent = torch.randn(1, 8, 16, 16, 16).cuda()

    # optimizer on the latent
    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OPTIMIZATION_STEPS)

    # read mesh from gso (./gso/google_scanned_objects/models_normalized/3D_Dollhouse_Happy_Brother/model.obj)
    mesh = trimesh.load_mesh(obj_path) # in [-1, 1] box

    # voxelize the mesh to 64x64x64
    voxel_grid = voxelize_surface(mesh.vertices, mesh.faces, resolution=64, num_samples=200_000)
    voxel_grid = torch.from_numpy(voxel_grid).float().unsqueeze(0).unsqueeze(0).cuda() # (1, 1, 64, 64, 64)

    with trange(OPTIMIZATION_STEPS, position=1, desc="Optimizing latent code") as steps:
        for i in steps:
            optimizer.zero_grad()
            decoded = decoder(latent)
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded, voxel_grid)
            loss += 0.5 * torch.mean(latent**2) # regularization
            loss.backward()
            optimizer.step()
            scheduler.step()

            steps.set_postfix(loss=loss.item())

    decoded_voxel = (decoder(latent).detach().cpu().numpy() > 0).astype(np.uint8) # (1, 1, 64, 64, 64)

    obj_name = obj_path.split('/')[-3]

    # save the latent as .npy
    np.save(f'./gso/google_scanned_objects/latent_codes/{obj_name}.npy', latent.detach().cpu().numpy())

    if POINT_CLOUD:
        # convert decoded voxel to point cloud and save as .ply
        decoded_voxel = decoded_voxel.squeeze() # (64, 64, 64)
        points = np.argwhere(decoded_voxel) # (N, 3)
        points = points / 63.0 * 2.0 - 1.0 # map back to [-1, 1]

        with open(f'./gso/google_scanned_objects/point_clouds/{obj_name}.ply', 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property uchar alpha\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]} 255 0 0 255\n")