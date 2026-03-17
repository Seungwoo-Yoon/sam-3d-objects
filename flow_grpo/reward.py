"""Reward function: penalizes inter-object intersection via SDF."""

from typing import List, Dict

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix

from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult

def extract_sdf_surface(sdf: torch.Tensor, res: int, dtype: torch.dtype) -> torch.Tensor | None:
    """
    Extract sub-voxel-precise surface points from an SDF grid via zero-crossing
    interpolation along each axis.

    Args:
        sdf:   (res+1, res+1, res+1) SDF grid
        res:   grid resolution (number of voxels per axis)
        dtype: desired output dtype for the returned coordinates

    Returns:
        Tensor of shape (N, 3) with surface point coordinates in grid-index space,
        or None if no zero-crossings are found.
    """
    surface_pts = []
    for axis in range(3):
        s0 = sdf.narrow(axis, 0, res)   # all but last slice
        s1 = sdf.narrow(axis, 1, res)   # all but first slice
        cross = s0 * s1 < 0             # sign change → zero crossing
        gi, gj, gk = torch.where(cross) # integer grid coords of s0
        if gi.numel() == 0:
            continue
        v0 = s0[gi, gj, gk]
        v1 = s1[gi, gj, gk]
        t  = v0 / (v0 - v1)            # interpolation factor in [0,1]
        coords = torch.stack([gi, gj, gk], dim=-1).to(dtype)
        coords[:, axis] += t            # sub-voxel position along axis
        surface_pts.append(coords)

    if not surface_pts:
        return None
    return torch.cat(surface_pts, dim=0)  # (N, 3)


def object_intersection_reward(meshes: List[MeshExtractResult], scale, rotation, translation) -> float:
    """
    Compute scalar reward that penalizes inter-object intersection.

    For each ordered pair (i, j), the interior points of object i (SDF_i < 0)
    are transformed into object j's local coordinate frame.  The penetration
    depth at each point is max(0, -SDF_j), and its sum is the intersection
    penalty for that pair.

    Both SDF grids live in [-0.5, 0.5]^3 local space.
    Pose convention: world = (local * scale) @ R^T + t   (row-vector, R from quaternion_to_matrix)

    Args:
        meshes:      list of MeshExtractResult (one per object); each has .sdf_d (res+1)^3
        scale:       (N, 1, 3) per-object scale
        rotation:    (N, 1, 4) per-object quaternion  [w, x, y, z]
        translation: (N, 1, 3) per-object translation

    Returns:
        float: reward (≤ 0; less negative = less intersection = better).
    """
    # Build list of (original_index, mesh) for meshes that are valid.
    # sdf_d is stored as a flat ((res+1)^3,) tensor; we reshape to
    # (res+1, res+1, res+1) when we access it below.
    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if len(valid) < 2:
        return 0.0

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    # Precompute rotation matrices and cast poses to sdf dtype/device.
    # Keyed by original index so pose arrays (scale/rotation/translation)
    # are indexed correctly even when some objects are skipped.
    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    total_penalty = 0.0
    MAX_INTERIOR_PTS = 500_000   # cap to limit memory; sample randomly if exceeded

    for vi, mesh_i in valid:
        res_i  = mesh_i.res            # e.g. 64
        sdf_i  = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)

        pts_surface = extract_sdf_surface(sdf_i, res_i, dtype)
        if pts_surface is None:
            continue

        # Subsample to cap memory usage
        if pts_surface.shape[0] > MAX_INTERIOR_PTS:
            perm = torch.randperm(pts_surface.shape[0], device=device)[:MAX_INTERIOR_PTS]
            pts_surface = pts_surface[perm]

        # Map grid indices → local coordinates in [-0.5, 0.5]
        pts_interior = pts_surface / res_i - 0.5  # (M, 3)

        # ── forward transform: local_i  →  world ──
        # world = (local * s_i) @ R_i^T + t_i
        pts_world = (pts_interior * s_map[vi]) @ R_map[vi].T + t_map[vi]  # (M, 3)

        for vj, mesh_j in valid:
            if vi == vj:
                continue

            res_j = mesh_j.res
            sdf_j = mesh_j.sdf_d.reshape(res_j + 1, res_j + 1, res_j + 1)

            # ── inverse transform: world  →  local_j ──
            # local_j = (world - t_j) @ R_j / s_j
            pts_j = (pts_world - t_map[vj]) @ R_map[vj] / s_map[vj]   # (M, 3)

            # Discard points that fall outside j's grid  (cannot intersect)
            in_bounds = ((pts_j >= -0.5) & (pts_j <= 0.5)).all(dim=-1)  # (M,)
            if not in_bounds.any():
                continue
            pts_j_bounded = pts_j[in_bounds]   # (K, 3)

            # ── trilinear interpolation of sdf_j at pts_j_bounded ──
            # sdf_d axis layout: [dim0, dim1, dim2]  →  grid_sample axes [D, H, W]
            # grid_sample (x,y,z) maps to (W, H, D) = (dim2, dim1, dim0)
            # normalised coords: local ∈ [-0.5, 0.5]  →  grid_sample ∈ [-1, 1]  →  *2
            grid_xyz = pts_j_bounded[:, [2, 1, 0]] * 2.0          # (K, 3)  in (x,y,z) order
            grid     = grid_xyz[None, None, None, :, :]            # (1,1,1,K,3)

            sdf_j_vals = F.grid_sample(
                sdf_j[None, None].float(),   # (1, 1, D, H, W)
                grid.float(),
                mode='bilinear',             # trilinear for 5-D input
                padding_mode='border',
                align_corners=True,
            ).squeeze()                      # (K,)

            # Intersection penalty: penetration depth of i's interior into j
            penalty = F.relu(-sdf_j_vals).max().item()
            total_penalty += penalty

    # Higher reward = less intersection; negate so optimiser maximises it
    return -total_penalty


def ground_penetration_reward(
    meshes: List[MeshExtractResult],
    scale,
    rotation,
    translation,
    camera_transform: torch.Tensor,
) -> float:
    """
    Compute scalar reward that penalizes objects penetrating the ground plane (z = 0 in world space).

    For each object, surface points are extracted via SDF zero-crossings, transformed
    into world space via the camera-to-world matrix, and the deepest below-ground
    point (most negative world-space z) determines the per-object penetration penalty.

    Pose convention (in camera space): pts_cam = (local * scale) @ R^T + t
    camera_transform: (4, 4) world-to-camera matrix (column-vector convention).
                      Its inverse gives the camera-to-world transform used here.

    Args:
        meshes:           list of MeshExtractResult (one per object); each has .sdf_d (res+1)^3
        scale:            (N, 1, 3) per-object scale
        rotation:         (N, 1, 4) per-object quaternion [w, x, y, z]
        translation:      (N, 1, 3) per-object translation (in camera space)
        camera_transform: (4, 4) world-to-camera transformation matrix

    Returns:
        float: reward (≤ 0; 0 means no penetration).
    """
    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if not valid:
        return 0.0

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    # camera-to-world: p_world = cam_to_world @ p_cam  (homogeneous, column vectors)
    cam_to_world = torch.linalg.inv(camera_transform.to(device=device, dtype=dtype))  # (4, 4)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    total_penalty = 0.0
    MAX_SURFACE_PTS = 500_000

    for vi, mesh_i in valid:
        res_i = mesh_i.res
        sdf_i = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)

        pts_surface = extract_sdf_surface(sdf_i, res_i, dtype)
        if pts_surface is None:
            continue

        pts_surface = pts_surface.to(device)

        if pts_surface.shape[0] > MAX_SURFACE_PTS:
            perm = torch.randperm(pts_surface.shape[0], device=device)[:MAX_SURFACE_PTS]
            pts_surface = pts_surface[perm]

        # grid indices → local coords in [-0.5, 0.5]
        pts_local = pts_surface / res_i - 0.5   # (M, 3)

        # local → camera space:  pts_cam = (pts_local * s) @ R^T + t
        pts_cam = (pts_local * s_map[vi]) @ R_map[vi].T + t_map[vi]   # (M, 3)

        # camera → world space via homogeneous transform
        ones = torch.ones(pts_cam.shape[0], 1, device=device, dtype=dtype)
        pts_cam_h = torch.cat([pts_cam, ones], dim=-1)               # (M, 4)
        pts_world = (cam_to_world @ pts_cam_h.T).T[:, :3]            # (M, 3)

        # Ground penetration: z < 0 in world space
        z = pts_world[:, 2]                                           # (M,)
        below = z[z < 0]
        if below.numel() == 0:
            continue

        # Penalty = deepest penetration depth (most negative z → largest positive depth)
        max_depth = (-below.min()).item()
        total_penalty += max_depth

    return -total_penalty


def pointmap_coverage_reward(
    meshes: List[MeshExtractResult],
    scale,
    rotation,
    translation,
    pointmap: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    """
    Compute scalar reward that encourages each object's masked pointmap points
    to lie on its surface (SDF ≈ 0).

    For object i, pixels where masks[i] is non-zero are extracted from the pointmap
    to form per-object world-space point sets.  Each set is transformed into the
    object's local frame and the mean absolute SDF is used as the penalty.

    Both SDF grids live in [-0.5, 0.5]^3 local space.
    Pose convention: world = (local * scale) @ R^T + t   (row-vector, R from quaternion_to_matrix)

    Args:
        meshes:      list of N MeshExtractResult (one per object); each has .sdf_d (res+1)^3
        scale:       (N, 1, 3) per-object scale
        rotation:    (N, 1, 4) per-object quaternion  [w, x, y, z]
        translation: (N, 1, 3) per-object translation
        pointmap:    (H, W, 3) world-space 3-D coordinates for each pixel
        masks:       (N, H, W) per-object binary/boolean masks

    Returns:
        float: reward (≤ 0; less negative = closer to surface = better).
    """
    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if not valid:
        return 0.0

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    pointmap = pointmap.to(device=device, dtype=dtype)  # (H, W, 3)
    masks    = masks.to(device=device)                  # (N, H, W)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    total_penalty = 0.0

    for vi, mesh_i in valid:
        # Erode mask to exclude edge pixels.
        # A pixel is interior only if all pixels within a 7x7 neighbourhood are also masked
        # (3-pixel margin). This removes boundary artefacts from depth discontinuities.
        mask_f = masks[vi].float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_interior = (
            -F.max_pool2d(-mask_f, kernel_size=7, stride=1, padding=3)
        ).squeeze() > 0.5                                     # (H, W) bool

        pts_world = pointmap[mask_interior]                   # (K, 3)
        if pts_world.shape[0] == 0:
            continue

        res_i = mesh_i.res
        sdf_i = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)

        # world → local_i:  local = (world - t) @ R / s
        pts_local = (pts_world - t_map[vi]) @ R_map[vi] / s_map[vi]  # (K, 3)

        # Split points into in-bounds and out-of-bounds.
        # Out-of-bounds points are treated as |SDF| = 1 (maximally far from surface).
        in_bounds = ((pts_local >= -0.5) & (pts_local <= 0.5)).all(dim=-1)  # (K,)
        n_total   = pts_local.shape[0]

        abs_sdf_vals = torch.ones(n_total, device=device, dtype=dtype)  # default = 1

        if in_bounds.any():
            pts_bounded = pts_local[in_bounds]  # (M, 3)

            # Trilinear interpolation — same axis convention as object_intersection_reward:
            # local ∈ [-0.5, 0.5] → grid_sample ∈ [-1, 1]; (x,y,z) maps to (dim2, dim1, dim0)
            grid_xyz = pts_bounded[:, [2, 1, 0]] * 2.0          # (M, 3)
            grid     = grid_xyz[None, None, None, :, :]          # (1, 1, 1, M, 3)

            sdf_vals = F.grid_sample(
                sdf_i[None, None].float(),
                grid.float(),
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            ).squeeze()  # (M,)

            abs_sdf_vals[in_bounds] = sdf_vals.abs().to(dtype)

        # Penalty: mean |SDF| over all points (out-of-bounds → |SDF|=1)
        penalty = abs_sdf_vals.mean().item()
        total_penalty += penalty

    return -total_penalty


def compute_reward(meshes: List[MeshExtractResult], scale, rotation, translation, camera_transform, pointmap, masks) -> Dict[str, float]:
    # return object_intersection_reward(meshes, scale, rotation, translation) \
    #         + ground_penetration_reward(meshes, scale, rotation, translation, camera_transform) \
    #         + pointmap_coverage_reward(meshes, scale, rotation, translation, pointmap, masks)

    return {
        # "intersection": object_intersection_reward(meshes, scale, rotation, translation) * 10.0,
        # "ground_penetration": ground_penetration_reward(meshes, scale, rotation, translation, camera_transform),
        "pointmap_coverage": pointmap_coverage_reward(meshes, scale, rotation, translation, pointmap, masks),
    }