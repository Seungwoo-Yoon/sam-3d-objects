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

