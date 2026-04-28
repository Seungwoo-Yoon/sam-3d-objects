"""Reward function: penalizes inter-object intersection via SDF."""

from typing import List, Dict

import trimesh
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np

from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult

THRESHOLD_GT_COVERAGE = -0.01  # hyperparameter: objects with GT coverage below this get all other rewards masked out to zero

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
    N = len(meshes)
    pose_device = scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if len(valid) < 2:
        return torch.zeros(N, device=pose_device)

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    # Precompute rotation matrices and cast poses to sdf dtype/device.
    # Keyed by original index so pose arrays (scale/rotation/translation)
    # are indexed correctly even when some objects are skipped.
    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    MAX_INTERIOR_PTS = 500_000   # cap to limit memory; sample randomly if exceeded

    # Fixed-size output: one slot per object (0 = no penalty)
    penalties = torch.zeros(N, device=device, dtype=torch.float32)

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
        # world = (local * s_i) @ (R_i @ M)^T + t_i
        pts_world = (pts_interior * s_map[vi]) @ (R_map[vi]).T + t_map[vi]  # (M, 3)

        object_penalty = 0.0

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
            object_penalty += penalty

        penalties[vi] = object_penalty / len(valid)

    # Higher reward = less intersection; negate so optimiser maximises it
    return -penalties.to(pose_device)


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
    N = len(meshes)
    pose_device = scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if not valid:
        return torch.zeros(N, device=pose_device)

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    # camera-to-world: p_world = cam_to_world @ p_cam  (homogeneous, column vectors)
    cam_to_world = torch.linalg.inv(camera_transform.to(device=device, dtype=dtype))  # (4, 4)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    MAX_SURFACE_PTS = 500_000

    # Fixed-size output: one slot per object (0 = no penalty)
    penalties = torch.zeros(N, device=device, dtype=torch.float32)

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

        # local → camera space:  pts_cam = (pts_local * s) @ (R @ M)^T + t
        pts_cam = (pts_local * s_map[vi]) @ (R_map[vi]).T + t_map[vi]   # (M, 3)

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
        penalties[vi] = (-below.min()).item()

    return -penalties.to(pose_device)



def object_intersection_reward_binary(
    meshes: List[MeshExtractResult],
    scale,
    rotation,
    translation,
) -> torch.Tensor:
    """
    Binary intersection reward using trimesh CollisionManager (FCL BVH).

    All meshes are registered with their world-space poses into a single
    CollisionManager.  A single in_collision_internal() call runs BVH-accelerated
    triangle-triangle intersection tests across all pairs at once — no SDF, no
    per-vertex containment checks.

    Pose convention: pts_world = (pts_local * scale) @ R^T + t
                     pts_local ∈ [−0.5, 0.5]³

    The 4×4 rigid transform passed to FCL encodes both rotation and scale:
        T[:3, :3] = R @ diag(s)
        T[:3,  3] = t

    Args:
        meshes:      list of MeshExtractResult; each needs .vertices (V,3) and .faces (F,3)
        scale:       (N, 1, 3) per-object scale
        rotation:    (N, 1, 4) per-object quaternion  [w, x, y, z]
        translation: (N, 1, 3) per-object translation

    Returns:
        Tensor of shape (N,) with values in {0.0, 1.0}.
    """
    N = len(meshes)
    pose_device = scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None
        and m.vertices is not None and m.vertices.shape[0] > 0
        and m.faces is not None and m.faces.shape[0] > 0
    ]
    if len(valid) < 2:
        return torch.ones(N, device=pose_device)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0] for vi, _ in valid}

    # ── Step 1: compute world-space AABB for each mesh ───────────────────────
    # v_world = v_local @ (R @ diag(s))^T + t  (row-vector convention)
    aabb_min: Dict[int, np.ndarray] = {}
    aabb_max: Dict[int, np.ndarray] = {}
    T_map:    Dict[int, np.ndarray] = {}

    for vi, mesh_i in valid:
        R_np = R_map[vi].cpu().numpy().astype(np.float64)
        s_np = scale[vi, 0].cpu().numpy().astype(np.float64)
        t_np = translation[vi, 0].cpu().numpy().astype(np.float64)

        T = np.eye(4)
        T[:3, :3] = R_np @ np.diag(s_np)
        T[:3, 3]  = t_np
        T_map[vi] = T

        verts_np    = mesh_i.vertices.cpu().numpy().astype(np.float64)  # (V, 3)
        verts_world = verts_np @ T[:3, :3].T + t_np
        aabb_min[vi] = verts_world.min(axis=0)
        aabb_max[vi] = verts_world.max(axis=0)

    # ── Step 2: AABB overlap pre-filter ─────────────────────────────────────
    # Keep only meshes that have at least one AABB-overlapping partner.
    # Two AABBs overlap iff they overlap along every axis simultaneously.
    def _aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
        return bool(np.all(a_min <= b_max) and np.all(a_max >= b_min))

    candidates: set = set()
    valid_indices = [vi for vi, _ in valid]
    for idx_i in range(len(valid_indices)):
        vi = valid_indices[idx_i]
        for idx_j in range(idx_i + 1, len(valid_indices)):
            vj = valid_indices[idx_j]
            if _aabb_overlap(aabb_min[vi], aabb_max[vi], aabb_min[vj], aabb_max[vj]):
                candidates.add(vi)
                candidates.add(vj)

    if len(candidates) < 2:
        return torch.ones(N, device=pose_device)

    # ── Step 3: FCL collision check on AABB candidates only ──────────────────
    manager = trimesh.collision.CollisionManager()

    for vi, mesh_i in valid:
        if vi not in candidates:
            continue
        tri_mesh = trimesh.Trimesh(
            vertices=mesh_i.vertices.cpu().numpy().astype(np.float64),
            faces=mesh_i.faces.cpu().numpy(),
            process=False,
        )
        manager.add_object(str(vi), tri_mesh, transform=T_map[vi])

    rewards = torch.ones(N, device=pose_device, dtype=torch.float32)

    is_collision, colliding_pairs = manager.in_collision_internal(return_names=True)
    if is_collision:
        for name_i, name_j in colliding_pairs:
            rewards[int(name_i)] = 0.0
            rewards[int(name_j)] = 0.0

    return rewards

    # ── ORIGINAL SDF-based implementation (kept for reference) ───────────────
    # N = len(meshes)
    # pose_device = scale.device
    # valid = [
    #     (i, m) for i, m in enumerate(meshes)
    #     if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    # ]
    # if len(valid) < 2:
    #     return torch.ones(N, device=pose_device)
    # device = valid[0][1].sdf_d.device
    # dtype  = valid[0][1].sdf_d.dtype
    # R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    # s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    # t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}
    # MAX_INTERIOR_PTS = 500_000
    # rewards = torch.ones(N, device=device, dtype=torch.float32)
    # for vi, mesh_i in valid:
    #     res_i = mesh_i.res
    #     sdf_i = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)
    #     pts_surface = extract_sdf_surface(sdf_i, res_i, dtype)
    #     if pts_surface is None:
    #         continue
    #     if pts_surface.shape[0] > MAX_INTERIOR_PTS:
    #         perm = torch.randperm(pts_surface.shape[0], device=device)[:MAX_INTERIOR_PTS]
    #         pts_surface = pts_surface[perm]
    #     pts_interior = pts_surface / res_i - 0.5  # (M, 3)
    #     pts_world = (pts_interior * s_map[vi]) @ R_map[vi].T + t_map[vi]  # (M, 3)
    #     intersects = False
    #     for vj, mesh_j in valid:
    #         if vi == vj:
    #             continue
    #         res_j = mesh_j.res
    #         sdf_j = mesh_j.sdf_d.reshape(res_j + 1, res_j + 1, res_j + 1)
    #         pts_j = (pts_world - t_map[vj]) @ R_map[vj] / s_map[vj]  # (M, 3)
    #         in_bounds = ((pts_j >= -0.5) & (pts_j <= 0.5)).all(dim=-1)
    #         if not in_bounds.any():
    #             continue
    #         pts_j_bounded = pts_j[in_bounds]
    #         grid_xyz = pts_j_bounded[:, [2, 1, 0]] * 2.0
    #         grid     = grid_xyz[None, None, None, :, :]
    #         sdf_j_vals = F.grid_sample(
    #             sdf_j[None, None].float(), grid.float(),
    #             mode='bilinear', padding_mode='border', align_corners=True,
    #         ).squeeze()
    #         if (sdf_j_vals < 0).any():
    #             intersects = True
    #             break
    #     if intersects:
    #         rewards[vi] = 0.0
    # return rewards.to(pose_device)


def ground_penetration_reward_binary(
    meshes: List[MeshExtractResult],
    scale,
    rotation,
    translation,
    camera_transform: torch.Tensor,
) -> torch.Tensor:
    """
    Binary ground-penetration reward using mesh vertices directly.

    For each object: reward = 0 if any mesh vertex is below z = 0 in world space,
    1 otherwise.  Uses mesh.vertices instead of SDF zero-crossing extraction.

    Pipeline per object:
        local [−0.5, 0.5]³  →  camera space  →  world space (via cam_to_world)

    Args:
        meshes:           list of MeshExtractResult; each needs .vertices (V, 3)
        scale:            (N, 1, 3) per-object scale
        rotation:         (N, 1, 4) per-object quaternion [w, x, y, z]
        translation:      (N, 1, 3) per-object translation (in camera space)
        camera_transform: (4, 4) world-to-camera transformation matrix

    Returns:
        Tensor of shape (N,) with values in {0.0, 1.0}.
    """
    N = len(meshes)
    pose_device = scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None
        and m.vertices is not None and m.vertices.shape[0] > 0
    ]
    if not valid:
        return torch.ones(N, device=pose_device)

    device = valid[0][1].vertices.device
    dtype  = valid[0][1].vertices.dtype

    cam_to_world = torch.linalg.inv(camera_transform.to(device=device, dtype=dtype))  # (4, 4)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    rewards = torch.ones(N, device=device, dtype=torch.float32)

    for vi, mesh_i in valid:
        pts_local = mesh_i.vertices.to(device=device, dtype=dtype)             # (V, 3)

        # local → camera space
        pts_cam = (pts_local * s_map[vi]) @ R_map[vi].T + t_map[vi]           # (V, 3)

        # camera → world space (homogeneous)
        ones      = torch.ones(pts_cam.shape[0], 1, device=device, dtype=dtype)
        pts_world = (cam_to_world @ torch.cat([pts_cam, ones], dim=-1).T).T[:, :3]  # (V, 3)

        if (pts_world[:, 2] < 0).any():
            rewards[vi] = 0.0

    return rewards.to(pose_device)

    # ── ORIGINAL SDF-based implementation (kept for reference) ───────────────
    # valid = [
    #     (i, m) for i, m in enumerate(meshes)
    #     if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    # ]
    # if not valid:
    #     return torch.ones(N, device=pose_device)
    # device = valid[0][1].sdf_d.device
    # dtype  = valid[0][1].sdf_d.dtype
    # cam_to_world = torch.linalg.inv(camera_transform.to(device=device, dtype=dtype))
    # R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    # s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    # t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}
    # MAX_SURFACE_PTS = 500_000
    # rewards = torch.ones(N, device=device, dtype=torch.float32)
    # for vi, mesh_i in valid:
    #     res_i = mesh_i.res
    #     sdf_i = mesh_i.sdf_d.reshape(res_i + 1, res_i + 1, res_i + 1)
    #     pts_surface = extract_sdf_surface(sdf_i, res_i, dtype)
    #     if pts_surface is None:
    #         continue
    #     pts_surface = pts_surface.to(device)
    #     if pts_surface.shape[0] > MAX_SURFACE_PTS:
    #         perm = torch.randperm(pts_surface.shape[0], device=device)[:MAX_SURFACE_PTS]
    #         pts_surface = pts_surface[perm]
    #     pts_local = pts_surface / res_i - 0.5
    #     pts_cam   = (pts_local * s_map[vi]) @ R_map[vi].T + t_map[vi]
    #     ones = torch.ones(pts_cam.shape[0], 1, device=device, dtype=dtype)
    #     pts_cam_h = torch.cat([pts_cam, ones], dim=-1)
    #     pts_world = (cam_to_world @ pts_cam_h.T).T[:, :3]
    #     z = pts_world[:, 2]
    #     if (z < 0).any():
    #         rewards[vi] = 0.0
    # return rewards.to(pose_device)


def _rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix (Zhou et al., 2019).

    Args:
        rot_6d: (..., 6) — first two columns of R, concatenated

    Returns:
        (..., 3, 3) rotation matrix with columns as orthonormal basis vectors
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def _sample_mesh_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Sample points uniformly from a triangle mesh surface via area-weighted sampling.

    Args:
        vertices: (V, 3) mesh vertices
        faces:    (F, 3) triangle face indices (long)
        num_samples: number of surface points to sample

    Returns:
        (num_samples, 3) points on the mesh surface
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    areas = torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1) * 0.5  # (F,)
    total_area = areas.sum()
    if total_area < 1e-10 or faces.shape[0] == 0:
        # Degenerate mesh — fall back to repeating existing vertices
        idx = torch.randint(0, max(vertices.shape[0], 1), (num_samples,), device=vertices.device)
        return vertices[idx]

    face_ids = torch.multinomial(areas / total_area, num_samples, replacement=True)

    r1 = torch.rand(num_samples, device=vertices.device, dtype=vertices.dtype).sqrt()
    r2 = torch.rand(num_samples, device=vertices.device, dtype=vertices.dtype)
    u = 1.0 - r1
    v = r1 * (1.0 - r2)
    w = r1 * r2

    pts = (
        u[:, None] * vertices[faces[face_ids, 0]]
        + v[:, None] * vertices[faces[face_ids, 1]]
        + w[:, None] * vertices[faces[face_ids, 2]]
    )
    return pts  # (num_samples, 3)


def _chamfer_distance(pts_a: torch.Tensor, pts_b: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance (squared L2).

    Args:
        pts_a: (M, 3)
        pts_b: (N, 3)

    Returns:
        Scalar — mean_a(min_b d²) + mean_b(min_a d²)
    """
    dist2 = torch.cdist(pts_a, pts_b) ** 2  # (M, N)
    return dist2.min(dim=1).values.mean() + dist2.min(dim=0).values.mean()


def gt_coverage_reward(
    meshes: List[MeshExtractResult],
    pred_scale,
    pred_rotation,
    pred_translation,
    gt_mesh_points: torch.Tensor,
    gt_scale: torch.Tensor,
    gt_rotation_6d: torch.Tensor,
    gt_translation: torch.Tensor,
    gt_mesh_available: torch.Tensor,
) -> torch.Tensor:
    """
    Reward based on Chamfer Distance between predicted and GT meshes in camera space.

    Predicted mesh surface points are uniformly sampled from the predicted mesh
    triangles (MeshExtractResult.vertices / .faces, local space [-0.5, 0.5]³)
    and transformed to camera space with the predicted pose.

    GT mesh points (pre-sampled from the GT mesh surface in raw local coords,
    supplied by the dataset) are transformed to camera space with the GT pose.

    Per-object reward = –Chamfer Distance (higher = more similar shapes).

    Pose conventions:
        predicted: pts_cam = (pts_local * s_pred) @ R_pred^T + t_pred
                   pts_local ∈ [–0.5, 0.5]³
        GT:        pts_cam = (pts_raw * s_gt) @ R_gt^T + t_gt
                   pts_raw are raw trimesh coordinates in the object's
                   local coordinate system.

    Args:
        meshes:            list of N MeshExtractResult (predicted)
        pred_scale:        (N, 1, 3) predicted per-object scale
        pred_rotation:     (N, 1, 4) predicted quaternion  [w, x, y, z]
        pred_translation:  (N, 1, 3) predicted translation (camera space)
        gt_mesh_points:    (N, M, 3) GT surface points in raw local coords
        gt_scale:          (N, 3)    GT scale
        gt_rotation_6d:    (N, 6)    GT 6D rotation
        gt_translation:    (N, 3)    GT translation (camera space)
        gt_mesh_available: (N,)      bool — which objects have valid GT meshes

    Returns:
        Tensor of shape (N,) with –Chamfer Distance per object
        (0.0 for objects without a valid GT mesh or predicted mesh).
    """
    N = len(meshes)
    pose_device = pred_scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.vertices is not None and m.vertices.shape[0] > 0
           and m.faces is not None and m.faces.shape[0] > 0
    ]
    if not valid:
        return torch.zeros(N, device=pose_device)

    device = valid[0][1].vertices.device
    dtype  = valid[0][1].vertices.dtype

    R_pred_map = {
        vi: quaternion_to_matrix(pred_rotation[vi, 0:1])[0].to(device=device, dtype=dtype)
        for vi, _ in valid
    }
    s_pred_map = {vi: pred_scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_pred_map = {vi: pred_translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    gt_scale       = gt_scale.to(device=device, dtype=dtype)
    gt_rotation_6d = gt_rotation_6d.to(device=device, dtype=dtype)
    gt_translation = gt_translation.to(device=device, dtype=dtype)
    gt_mesh_points = gt_mesh_points.to(device=device, dtype=dtype)
    gt_mesh_available = gt_mesh_available.to(device=device)

    R_gt = _rotation_6d_to_matrix(gt_rotation_6d)  # (N, 3, 3)

    NUM_SAMPLES = 4096

    rewards = torch.zeros(N, device=device, dtype=torch.float32)

    for vi, mesh_i in valid:
        if not gt_mesh_available[vi]:
            continue

        # ── Predicted mesh: surface sample → camera space ──
        pts_pred_local = _sample_mesh_surface(
            mesh_i.vertices.to(dtype=dtype),
            mesh_i.faces,
            NUM_SAMPLES,
        )
        pts_pred_cam = (pts_pred_local * s_pred_map[vi]) @ R_pred_map[vi].T + t_pred_map[vi]

        # ── GT mesh points → camera space ──
        pts_gt_raw = gt_mesh_points[vi]  # (M, 3)
        pts_gt_cam = (pts_gt_raw * gt_scale[vi]) @ R_gt[vi].T + gt_translation[vi]

        # ── Chamfer Distance ──
        cd = _chamfer_distance(pts_pred_cam, pts_gt_cam)
        rewards[vi] = -cd.item()

    return rewards.to(pose_device)


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
    N = len(meshes)
    pose_device = scale.device

    valid = [
        (i, m) for i, m in enumerate(meshes)
        if m is not None and m.sdf_d is not None and m.sdf_d.numel() > 0
    ]
    if not valid:
        return torch.zeros(N, device=pose_device)

    device = valid[0][1].sdf_d.device
    dtype  = valid[0][1].sdf_d.dtype

    pointmap = pointmap.to(device=device, dtype=dtype)  # (H, W, 3)
    masks    = masks.to(device=device)                  # (N, H, W)

    R_map = {vi: quaternion_to_matrix(rotation[vi, 0:1])[0].to(device=device, dtype=dtype) for vi, _ in valid}
    s_map = {vi: scale[vi, 0].to(device=device, dtype=dtype)       for vi, _ in valid}
    t_map = {vi: translation[vi, 0].to(device=device, dtype=dtype) for vi, _ in valid}

    # Fixed-size output: one slot per object (0 = no penalty)
    penalties = torch.zeros(N, device=device, dtype=torch.float32)

    for vi, mesh_i in valid:
        # Erode mask to exclude edge pixels.
        # A pixel is interior only if all pixels within a 3x3 neighbourhood are also masked
        # (1-pixel margin). This removes boundary artefacts from depth discontinuities.
        mask_f = masks[vi].float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_interior = (
            -F.max_pool2d(-mask_f, kernel_size=3, stride=1, padding=1)
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
            abs_sdf_vals[~in_bounds] = sdf_vals.abs().max().item()  # out-of-bounds treated as max distance from surface

        # Penalty: mean |SDF| over all points (out-of-bounds → |SDF|=max)
        penalties[vi] = abs_sdf_vals.mean().item() * scale[vi, 0].mean().item()  # scale penalty by object size to avoid bias towards smaller objects

    return -penalties.to(pose_device)


def compute_reward(
    meshes: List[MeshExtractResult],
    scale,
    rotation,
    translation,
    camera_transform,
    pointmap,
    masks,
    gt_mesh_points: torch.Tensor = None,
    gt_scale: torch.Tensor = None,
    gt_rotation_6d: torch.Tensor = None,
    gt_translation: torch.Tensor = None,
    gt_mesh_available: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    result = {
        #"intersection": object_intersection_reward(meshes, scale, rotation, translation),
        #"ground_penetration": ground_penetration_reward(meshes, scale, rotation, translation, camera_transform),
        "collision": object_intersection_reward_binary(meshes, scale, rotation, translation) + ground_penetration_reward_binary(meshes, scale, rotation, translation, camera_transform),
        # "pointmap_coverage": pointmap_coverage_reward(meshes, scale, rotation, translation, pointmap, masks),
    }

    gt_available = (
        gt_mesh_points is not None
        and gt_scale is not None
        and gt_rotation_6d is not None
        and gt_translation is not None
        and gt_mesh_available is not None
        and gt_mesh_available.any()
    )
    if gt_available:
        result["gt_coverage"] = gt_coverage_reward(
            meshes, scale, rotation, translation,
            gt_mesh_points, gt_scale, gt_rotation_6d, gt_translation, gt_mesh_available,
        )

        # print(result["gt_coverage"])

        for k in result:
            if k != "gt_coverage":
                result[k] = result[k] * (result["gt_coverage"] > THRESHOLD_GT_COVERAGE).float()  # mask out rewards when GT coverage is very poor (threshold is a hyperparameter)

    return result