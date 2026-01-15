from PIL import Image
import torch
import numpy as np
import os
import torch.nn.functional as F

# ----------------------------
# Settings
# ----------------------------
image_folder = "../debug/rendered_frames"
extrinsics = np.load(f"{image_folder}/extrinsics.npy")   # (V,4,4)
intrinsics = np.load(f"{image_folder}/intrinsics.npy")   # (V,3,3)
voxel_path = '../debug/occupancy_grids/segment1_6_occupancy_grid.npy'

n_images = 150
batch_size = 16

H = W = 518
eps = 1e-4

# ----------------------------
# Load images
# ----------------------------
images = []
for i in range(n_images):
    img = Image.open(f"{image_folder}/frame_{i:03d}.png")
    images.append(img)

# ----------------------------
# DINOv2
# ----------------------------
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval().cuda()
n_patch = 518 // 14  # 37

# ----------------------------
# Load voxel points
# ----------------------------
voxel = np.load(voxel_path)
indices = np.array(np.nonzero(voxel)).T.astype(np.float32)   # (N,3)
positions = indices / np.array(voxel.shape) - 0.5            # (N,3) in world-ish normalized
positions = torch.from_numpy(positions).float().cuda()       # (N,3)

N = positions.shape[0]

# ----------------------------
# Projection + occlusion (z-buffer)
# ----------------------------
def project_world_to_image(pts_world, extr, intr):
    """
    pts_world: (B,N,3)
    extr:      (B,4,4) world->cam
    intr:      (B,3,3)
    return:
      u_px, v_px: (B,N) float pixel coords (0..W-1 / 0..H-1)
      uv_grid:    (B,N,2) normalized to [-1,1] for grid_sample
      depth:      (B,N) camera z
      valid:      (B,N) in-front & inside image bounds
    """
    B, Np, _ = pts_world.shape

    ones = torch.ones((B, Np, 1), device=pts_world.device, dtype=pts_world.dtype)
    pts_h = torch.cat([pts_world, ones], dim=-1)          # (B,N,4)

    # world -> cam
    cam_h = torch.bmm(pts_h, extr.transpose(1, 2))        # (B,N,4)
    X = cam_h[..., 0]
    Y = cam_h[..., 1]
    Z = cam_h[..., 2].clamp(min=1e-8)                     # prevent div0, (B,N)

    fx = intr[:, 0, 0].unsqueeze(1)
    fy = intr[:, 1, 1].unsqueeze(1)
    cx = intr[:, 0, 2].unsqueeze(1)
    cy = intr[:, 1, 2].unsqueeze(1)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    # valid if Z>0 and inside image
    valid = (cam_h[..., 2] > 0) & (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))

    # grid_sample coords in [-1,1]
    # align_corners=False 기준으로는 (u+0.5)/W*2-1 형태가 더 정확하지만,
    # 여기서는 기존 코드의 스케일과 호환되도록 u/(W-1) 기반으로 갑니다.
    gx = (u / (W - 1)) * 2 - 1
    gy = (v / (H - 1)) * 2 - 1
    uv_grid = torch.stack([gx, gy], dim=-1)  # (B,N,2)

    return u, v, uv_grid, cam_h[..., 2], valid


def zbuffer_visibility(u, v, depth, valid, H, W, eps=1e-4):
    """
    u,v,depth: (B,N)
    valid:     (B,N)
    returns visible: (B,N) bool
    방법:
      - 각 포인트를 정수 픽셀 bin으로 넣고 (round 또는 floor),
      - 같은 픽셀에 여러 포인트가 있으면 depth 최솟값만 visible.
    """
    B, Np = u.shape

    # 정수 픽셀 bin (가장 단순: round). 더 보수적으로 하려면 floor 사용 가능.
    ui = torch.round(u).long()
    vi = torch.round(v).long()

    # bounds clamp (valid 밖은 어차피 걸러짐)
    ui = ui.clamp(0, W - 1)
    vi = vi.clamp(0, H - 1)

    lin = vi * W + ui  # (B,N)

    # invalid는 scatter에 들어가도 min에 영향을 주지 않게 depth=+inf로 처리
    INF = torch.tensor(float('inf'), device=u.device, dtype=depth.dtype)
    depth_for_min = torch.where(valid, depth, INF)

    # 픽셀별 최소 depth
    min_depth = torch.full((B, H * W), float('inf'), device=u.device, dtype=depth.dtype)
    # torch>=1.12: scatter_reduce_ 지원
    min_depth.scatter_reduce_(1, lin, depth_for_min, reduce="amin", include_self=True)

    # 각 포인트가 속한 픽셀의 최소 depth와 비교
    min_d_at_pt = min_depth.gather(1, lin)
    visible = valid & (depth <= (min_d_at_pt + eps))

    return visible


# ----------------------------
# Main loop
# ----------------------------
patchtokens_list = []
uvgrid_list = []
visible_list = []

for i in range(0, n_images, batch_size):
    batch_images = images[i:i+batch_size]

    batch_extrinsics = torch.from_numpy(extrinsics[i:i+batch_size]).float().cuda()
    batch_intrinsics = torch.from_numpy(intrinsics[i:i+batch_size]).float().cuda()

    # preprocess images
    for j, img in enumerate(batch_images):
        image = img.resize((W, H), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        batch_images[j] = image

    batch_images = torch.stack(batch_images, dim=0).cuda()
    bs = batch_images.shape[0]

    # dino features
    with torch.no_grad():
        features = dinov2_model(batch_images, is_training=True)

    patchtokens = (
        features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
        .permute(0, 2, 1)
        .reshape(bs, 1024, n_patch, n_patch)
    )

    # positions: (N,3) -> (B,N,3)
    pts_world = positions.unsqueeze(0).expand(bs, -1, -1)

    # projection
    u_px, v_px, uv_grid, depth, valid = project_world_to_image(
        pts_world, batch_extrinsics, batch_intrinsics
    )

    # occlusion visibility by z-buffer
    visible = zbuffer_visibility(u_px, v_px, depth, valid, H, W, eps=eps)

    patchtokens_list.append(patchtokens)
    uvgrid_list.append(uv_grid)
    visible_list.append(visible)

patchtokens_all = torch.cat(patchtokens_list, dim=0)   # (V,1024,Hp,Wp)
uvgrid_all = torch.cat(uvgrid_list, dim=0)             # (V,N,2)
visible_all = torch.cat(visible_list, dim=0)           # (V,N)

# ----------------------------
# (Optional) visualize visible uv
# ----------------------------
os.makedirs("../debug/uv_test", exist_ok=True)
for i, (uvg, vis) in enumerate(zip(uvgrid_all, visible_all)):
    # uvg: [-1,1] -> pixel
    uv_img = (uvg + 1) / 2
    uv_img[:, 0] = uv_img[:, 0] * (W - 1)
    uv_img[:, 1] = uv_img[:, 1] * (H - 1)
    uv_img = uv_img.round().long().clamp_min(0)
    uv_img[:, 0] = uv_img[:, 0].clamp_max(W - 1)
    uv_img[:, 1] = uv_img[:, 1].clamp_max(H - 1)

    vis_img = np.zeros((H, W, 3), dtype=np.uint8)
    pts = uv_img[vis]  # visible만 표시
    if pts.numel() > 0:
        vis_img[pts[:, 1].cpu().numpy(), pts[:, 0].cpu().numpy()] = 255

    Image.fromarray(vis_img).save(f"../debug/uv_test/uv_visible_{i:03d}.png")


# ----------------------------
# Sample features only for visible points and average over views
# ----------------------------
# grid_sample input grid: (B, outH=1, outW=N, 2)
grid = uvgrid_all.unsqueeze(1)  # (V,1,N,2)

# (V,1024,Hp,Wp) -> sample -> (V,1024,1,N) -> (V,N,1024)
tokens_vnC = F.grid_sample(
    patchtokens_all, grid, mode='bilinear', align_corners=False
).squeeze(2).permute(0, 2, 1)  # (V,N,1024)

# mask occluded/outside
mask = visible_all.unsqueeze(-1)  # (V,N,1)
tokens_vnC = tokens_vnC * mask

# 평균 (보이는 뷰만)
count = visible_all.sum(dim=0).clamp(min=1).unsqueeze(-1)   # (N,1)
tokens_NC = tokens_vnC.sum(dim=0) / count                   # (N,1024)

tokens = tokens_NC.detach().cpu().numpy()
indices_np = indices

os.makedirs("../debug/features", exist_ok=True)
np.save("../debug/features/dinov2_features.npy", tokens)
np.save("../debug/features/voxel_indices.npy", indices_np)