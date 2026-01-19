import os
import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import iterative_closest_point

def save_pc(points, filepath, color=(255, 255, 255, 255)):
    """Save point cloud to a PLY file.

    Args:
        points (np.ndarray): Nx3 array of point coordinates.
        filepath (str): Path to save the PLY file.
        color (tuple): RGBA color of the points in the PLY file. (default: white)
    """
    num_points = points.shape[0]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"property uchar red\n")
        f.write(f"property uchar green\n")
        f.write(f"property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {color[3]}\n")

# save multiple point clouds
def save_multiple_pcs(point_clouds, filepath, colors=None, sample=-1):
    """Save multiple point clouds to a single PLY file

    Args:
        point_clouds (list of np.ndarray): List of Nx3 arrays of point coordinates.
        filepath (str): Path to save the PLY file.
        colors (list of tuple): List of RGBA colors for each point cloud. If None, all will be white.
        sample (int): If > 0, randomly sample this many points from each point cloud before saving.
    """
    assert isinstance(point_clouds, list), "point_clouds should be a list of numpy arrays"
    for pc in point_clouds:
        assert isinstance(pc, np.ndarray), "Each point cloud should be a numpy array"
        assert pc.ndim == 2, "Each point cloud should be a 2D array"
        assert pc.shape[1] == 3, "Each point cloud should have shape Nx3"

    if colors is None:
        colors = [(255, 255, 255, 255)] * len(point_clouds)

    total_points = sum(pc.shape[0] for pc in point_clouds)
    if sample > 0:
        new_point_clouds = []
        for pc in point_clouds:
            if pc.shape[0] > sample:
                indices = np.random.choice(pc.shape[0], sample, replace=False)
                new_point_clouds.append(pc[indices])
            else:
                new_point_clouds.append(pc)
        point_clouds = new_point_clouds
        total_points = sum(pc.shape[0] for pc in point_clouds)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"property uchar red\n")
        f.write(f"property uchar green\n")
        f.write(f"property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for pc, color in zip(point_clouds, colors):
            for point in pc:
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {color[3]}\n")

def estimate_rigid_transform_ransac_icp(
    source: torch.Tensor,   # (N,3) or (B,N,3)
    target: torch.Tensor,   # (M,3) or (B,M,3)
):
    """
    Torch ICP version (PyTorch3D).
    Fixes Half/AMP crash by running ICP in float32.

    Returns:
        R: (3,3) or (B,3,3)
        t: (3,)  or (B,3)
    """

    # keep original dtype/device to return consistent tensors
    out_dtype = source.dtype
    device = source.device

    # ensure batch dimension
    if source.dim() == 2:
        source_b = source.unsqueeze(0)
    else:
        source_b = source
    if target.dim() == 2:
        target_b = target.unsqueeze(0)
    else:
        target_b = target

    # ---- critical: run ICP in float32 (or float64) ----
    source_b_f = source_b.to(dtype=torch.float32)
    target_b_f = target_b.to(dtype=torch.float32)

    # AMP/autocast can still force fp16 unless disabled
    # So explicitly disable autocast for this block.
    with torch.cuda.amp.autocast(enabled=False):
        icp = iterative_closest_point(
            source_b_f, target_b_f,
            max_iterations=200,
            relative_rmse_thr=1e-6,
            estimate_scale=False,
            allow_reflection=False,
        )

    R = icp.RTs.R  # (B,3,3) float32
    t = icp.RTs.T  # (B,3)   float32

    # cast back to original dtype if you really want (optional)
    # NOTE: keeping R,t, s in float32 is often better numerically.
    R = R.to(device=device, dtype=out_dtype)
    t = t.to(device=device, dtype=out_dtype)

    if R.shape[0] == 1 and source.dim() == 2 and target.dim() == 2:
        R = R[0]
        t = t[0]

    return R, t

def unproject_depth_to_pointmap(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    if depth_map.ndim == 3:
        depth_map = depth_map[..., np.newaxis]  # (S, H, W, 1)

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array

def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix