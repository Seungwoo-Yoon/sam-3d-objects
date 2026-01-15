from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import utils3d
import os
import torch.nn.functional as F

image_folder = "../debug/rendered_frames"
extrinsics = np.load(f"{image_folder}/extrinsics.npy")
intrinsics = np.load(f"{image_folder}/intrinsics.npy")
voxel_path = '../debug/occupancy_grids/segment1_6_occupancy_grid.npy'

n_images = 150
batch_size = 16

images = []

for i in range(n_images):
    img = Image.open(f"{image_folder}/frame_{i:03d}.png")
    images.append(img)

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval().cuda()
n_patch = 518 // 14

patchtokens_list = []
uv_list = []

voxel = np.load(voxel_path)
indices = np.array(np.nonzero(voxel)).T.astype(np.float32)
positions = indices / np.array(voxel.shape) - 0.5
positions = torch.from_numpy(positions).float().unsqueeze(0).cuda()

for i in range(0, n_images, batch_size):
    batch_images = images[i:i+batch_size]
    batch_extrinsics = torch.from_numpy(extrinsics[i:i+batch_size]).float().cuda()
    batch_intrinsics = torch.from_numpy(intrinsics[i:i+batch_size]).float().cuda()

    for j, img in enumerate(batch_images):
        image = img.resize((518, 518), Image.Resampling.LANCZOS)

        image = np.array(image).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        batch_images[j] = image

    batch_images = torch.stack(batch_images, dim=0).cuda()
    bs = batch_images.shape[0]


    with torch.no_grad():
        features = dinov2_model(batch_images, is_training=True)

    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1

    patchtokens_list.append(patchtokens)
    uv_list.append(uv)

patchtokens_all = torch.cat(patchtokens_list, dim=0)
uv_all = torch.cat(uv_list, dim=0)

for i, uv in enumerate(uv_all):
    # visualize uv
    uv_img = (uv + 1) / 2 * 518
    uv_img = uv_img.cpu().numpy().astype(np.int32)
    vis_img = np.zeros((518, 518, 3), dtype=np.uint8)
    vis_img[uv_img[:, 1], uv_img[:, 0]] = 255
    img = Image.fromarray(vis_img)
    os.makedirs("../debug/uv_test", exist_ok=True)
    img.save(f"../debug/uv_test/uv_{i:03d}.png")


tokens = F.grid_sample(patchtokens_all, uv_all.unsqueeze(1), mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1).cpu().numpy()
tokens = np.mean(tokens, axis=0)

os.makedirs("../debug/features", exist_ok=True)

np.save("../debug/features/dinov2_features.npy", tokens)
np.save("../debug/features/voxel_indices.npy", indices)