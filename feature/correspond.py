import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

indices = np.load("../debug/features/voxel_indices.npy")
features = np.load("../debug/features/dinov2_features.npy")

image = Image.open("../notebook/images/segment1/image.jpg")
mask = Image.open("../notebook/images/segment1/6.png").convert("L")

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval().cuda()
n_patch = 518 // 14

image = image.resize((512, 512), Image.Resampling.LANCZOS)
mask = mask.resize((512, 512), Image.Resampling.NEAREST)

image_size = image.size
image = image.resize((518, 518), Image.Resampling.LANCZOS)

image = np.array(image).astype(np.float32) / 255
image = torch.from_numpy(image).permute(2, 0, 1).float()

with torch.no_grad():
    img_features = dinov2_model(image.unsqueeze(0).cuda(), is_training=True)

image_features = img_features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:] \
    .permute(0, 2, 1).reshape(1, 1024, n_patch, n_patch).squeeze(0).cpu().numpy()

# upsample to image size
image_features_upsampled = np.zeros((1024, image_size[1], image_size[0]), dtype=np.float32)
for i in range(1024):
    feat = Image.fromarray(image_features[i])
    feat = feat.resize(image_size, Image.Resampling.NEAREST)
    image_features_upsampled[i] = np.array(feat)

# image에서 mask=1인 부분과 features 사이의 대응점 찾기
mask = np.array(mask.resize(image_size, Image.Resampling.NEAREST))
mask_indices = np.array(np.nonzero(mask)).T  # (N, 2)
correspondences = []
for y, x in tqdm(mask_indices):
    img_feat = image_features_upsampled[:, y, x]  # (1024,)

    # features에서 가장 유사한 feature 찾기 (cosine similarity after normalization)
    # img_feat_norm = img_feat / np.linalg.norm(img_feat)
    # feats_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    # dists = 1 - np.dot(feats_norm, img_feat_norm)  # cosine distance

    # feature에서 가장 유사한 feature 찾기 (L2 distance)
    dists = np.linalg.norm(features - img_feat, axis=1)  # L

    min_idx = np.argmin(dists)
    correspondences.append((indices[min_idx][0], indices[min_idx][1], indices[min_idx][2]))

# correspondences (N, 3)을 point cloud (ply)로 저장
# (-0.5, -0.5, -0.5) ~ (0.5, 0.5, 0.5) 범위로 정규화해서
# 파란색으로

with open("../debug/features/correspondences.ply", "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(correspondences)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for x, y, z in correspondences:
        nx = x / 64 - 0.5
        ny = y / 64 - 0.5
        nz = z / 64 - 0.5
        f.write(f"{nx} {ny} {nz} 0 0 255\n")