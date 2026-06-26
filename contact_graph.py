"""Build contact graphs for FoundationPoseDataset samples.

This intentionally reuses FoundationPoseDataset's loading path. For each dataset
item/render, it saves:

    <render_dir>/contact_graph.json

The matrix order is stored in object_names. Mesh poses follow the GT debug path
in flow_grpo/trainer.py:

    pts_cam = (verts_local * scale) @ R.T + translation

Ground contact then uses the same camera_view_transform convention as
flow_grpo/reward.py: cam_to_world = inverse(camera_view_transform).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import trange

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

from foundation_pose_dataset import FoundationPoseDataset


def _extract_visible_objects(dataset: FoundationPoseDataset, frame_data: Dict[str, Any], states: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    visible_objects: List[str] = []
    object_datasets: Dict[str, str] = {}
    seen = set()

    id_mapping = frame_data.get("id_mapping", {})
    for mask_id in sorted(id_mapping.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
        obj_info = dataset._extract_object_name_from_path(id_mapping[mask_id])
        if obj_info is None:
            continue
        obj_name, dataset_type = obj_info
        if obj_name not in states.get("objects", {}) or obj_name in seen:
            continue
        visible_objects.append(obj_name)
        object_datasets[obj_name] = dataset_type
        seen.add(obj_name)

    return visible_objects, object_datasets


def _rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _transform_local_points(points: np.ndarray, scale: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return ((points * scale) @ rotation.T + translation).astype(np.float32, copy=False)


def _camera_to_world(points_cam: np.ndarray, camera_view_transform: np.ndarray) -> np.ndarray:
    # FoundationPoseDataset builds camera-space GT as:
    #   V = camera_view_transform.T
    #   points_p3d = diag(-1, 1, -1) @ (V @ points_world)
    # Undo that here so ground contact is tested in the original world frame.
    view_world_to_gl = camera_view_transform.astype(np.float64).T
    gl_to_pytorch3d = np.diag([-1.0, 1.0, -1.0, 1.0])
    pytorch3d_to_gl = gl_to_pytorch3d
    cam_to_world = np.linalg.inv(view_world_to_gl)

    ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    points_p3d_h = np.concatenate([points_cam.astype(np.float64), ones], axis=1)
    points_gl_h = (pytorch3d_to_gl @ points_p3d_h.T).T
    return (cam_to_world @ points_gl_h.T).T[:, :3].astype(np.float32, copy=False)


def _sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    ids = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[ids]


def _aabb_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    separated = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(separated))


def _min_distance_numpy(points_a: np.ndarray, points_b: np.ndarray, chunk_size: int) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return math.inf
    if points_b.shape[0] < points_a.shape[0]:
        points_a, points_b = points_b, points_a

    best = math.inf
    points_b = points_b.astype(np.float32, copy=False)
    for start in range(0, points_a.shape[0], chunk_size):
        chunk = points_a[start : start + chunk_size].astype(np.float32, copy=False)
        diff = chunk[:, None, :] - points_b[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        best = min(best, float(dist2.min()))
    return math.sqrt(best)


def _min_distance(points_a: np.ndarray, points_b: np.ndarray, tree_a: Any, tree_b: Any, chunk_size: int) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return math.inf
    if cKDTree is None:
        return _min_distance_numpy(points_a, points_b, chunk_size)
    if points_a.shape[0] <= points_b.shape[0]:
        distances, _ = tree_b.query(points_a, k=1)
    else:
        distances, _ = tree_a.query(points_b, k=1)
    return float(np.min(distances))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_sample_graph(
    dataset: FoundationPoseDataset,
    idx: int,
    contact_eps: float,
    ground_eps: float,
    max_pair_points: int,
    distance_chunk_size: int,
    seed: int,
) -> Dict[str, Any]:
    render_info = dataset.scenes[idx]
    states = dataset._load_states(render_info["scene_parent_dir"])
    camera_params = dataset._load_camera_params(render_info["render_dir"], frame_id=0)
    frame_data = dataset._load_frame_data(render_info["render_dir"], frame_id=0)

    visible_objects, object_datasets = _extract_visible_objects(dataset, frame_data, states)
    if len(visible_objects) > dataset.max_objects_per_scene:
        visible_objects = visible_objects[: dataset.max_objects_per_scene]
        object_datasets = {name: object_datasets[name] for name in visible_objects}

    pose_latents = [
        dataset._pose_to_latents(
            states["objects"][obj_name],
            camera_view_transform=camera_params["camera_view_transform"],
        )
        for obj_name in visible_objects
    ]

    np.random.seed(seed + idx)
    mesh_data = dataset._load_mesh_data(visible_objects, object_datasets)

    if pose_latents:
        scale = np.stack([_tensor_to_numpy(item["scale"]) for item in pose_latents]).astype(np.float32)
        rotation_6d = np.stack([_tensor_to_numpy(item["6drotation_normalized"]) for item in pose_latents]).astype(np.float32)
        translation = np.stack([_tensor_to_numpy(item["translation"]) for item in pose_latents]).astype(np.float32)
    else:
        scale = np.zeros((0, 3), dtype=np.float32)
        rotation_6d = np.zeros((0, 6), dtype=np.float32)
        translation = np.zeros((0, 3), dtype=np.float32)
    rotation = _rotation_6d_to_matrix(rotation_6d)
    mesh_available = _tensor_to_numpy(mesh_data["mesh_available"]).astype(bool)

    rng = np.random.default_rng(seed + idx)
    points_cam: List[Optional[np.ndarray]] = []
    points_world: List[Optional[np.ndarray]] = []
    aabb_min: List[Optional[np.ndarray]] = []
    aabb_max: List[Optional[np.ndarray]] = []
    min_z: List[Optional[float]] = []
    mesh_paths: List[Optional[str]] = []

    for obj_idx, obj_name in enumerate(visible_objects):
        if not mesh_available[obj_idx]:
            points_cam.append(None)
            points_world.append(None)
            aabb_min.append(None)
            aabb_max.append(None)
            min_z.append(None)
            mesh_paths.append(None)
            continue

        verts = mesh_data.get("mesh_verts", [None] * len(visible_objects))[obj_idx]
        if verts is None:
            verts_local = _tensor_to_numpy(mesh_data["mesh_points"][obj_idx]).astype(np.float32)
        else:
            verts_local = _tensor_to_numpy(verts).astype(np.float32)

        verts_cam = _transform_local_points(
            verts_local,
            scale[obj_idx],
            rotation[obj_idx],
            translation[obj_idx],
        )
        sample_cam = _sample_points(verts_cam, max_pair_points, rng)
        verts_world = _camera_to_world(verts_cam, camera_params["camera_view_transform"])

        points_cam.append(sample_cam)
        points_world.append(verts_world)
        aabb_min.append(verts_cam.min(axis=0))
        aabb_max.append(verts_cam.max(axis=0))
        min_z.append(float(verts_world[:, 2].min()))
        mesh_paths.append(_guess_mesh_path(dataset.gso_root, obj_name))

    n = len(visible_objects)
    contact_matrix = [[False for _ in range(n)] for _ in range(n)]
    distance_matrix = [[None for _ in range(n)] for _ in range(n)]
    contact_pairs = []

    trees = [cKDTree(p) if cKDTree is not None and p is not None and p.shape[0] > 0 else None for p in points_cam]

    for i in range(n):
        if points_cam[i] is None:
            continue
        for j in range(i + 1, n):
            if points_cam[j] is None:
                continue
            assert aabb_min[i] is not None and aabb_max[i] is not None
            assert aabb_min[j] is not None and aabb_max[j] is not None
            if _aabb_distance(aabb_min[i], aabb_max[i], aabb_min[j], aabb_max[j]) > contact_eps:
                continue
            dist = _min_distance(points_cam[i], points_cam[j], trees[i], trees[j], distance_chunk_size)
            distance_matrix[i][j] = distance_matrix[j][i] = dist
            if dist <= contact_eps:
                contact_matrix[i][j] = contact_matrix[j][i] = True
                contact_pairs.append({
                    "i": i,
                    "j": j,
                    "object_i": visible_objects[i],
                    "object_j": visible_objects[j],
                    "distance": dist,
                })

    ground_contact = [bool(z is not None and z <= ground_eps) for z in min_z]
    ground_contact_objects = [
        {"i": i, "object": visible_objects[i], "min_z": min_z[i]}
        for i, is_contact in enumerate(ground_contact)
        if is_contact
    ]

    return {
        "version": 1,
        "source": "contact_graph.py",
        "scene_id": render_info["scene_id"],
        "scene_parent_dir": str(render_info["scene_parent_dir"]),
        "render_dir": str(render_info["render_dir"]),
        "coordinate_frame": "camera_for_object_pairs_world_for_ground",
        "pose_convention": "FoundationPoseDataset + trainer.py GT debug transform",
        "thresholds": {
            "contact_eps": contact_eps,
            "ground_eps": ground_eps,
            "max_pair_points": max_pair_points,
        },
        "object_names": visible_objects,
        "object_datasets": [object_datasets.get(name, "unknown") for name in visible_objects],
        "mesh_available": [bool(v) for v in mesh_available.tolist()],
        "mesh_paths": mesh_paths,
        "contact_matrix": contact_matrix,
        "ground_contact": ground_contact,
        "contact_pairs": contact_pairs,
        "ground_contact_objects": ground_contact_objects,
        "sampled_distances": distance_matrix,
        "min_world_z": min_z,
        "stats": {
            "num_objects": n,
            "num_meshes": int(mesh_available.sum()),
            "num_contact_pairs": len(contact_pairs),
            "num_ground_contacts": len(ground_contact_objects),
        },
    }


def _guess_mesh_path(gso_root: Optional[Path], obj_name: str) -> Optional[str]:
    if gso_root is None:
        return None
    candidates = [
        gso_root / obj_name / "meshes" / "model.obj",
        gso_root / obj_name / "meshes" / "model.ply",
        gso_root / obj_name / "model.obj",
        gso_root / obj_name / "model.ply",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def build_contact_graphs(args: argparse.Namespace) -> Dict[str, int]:
    dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=args.max_objects_per_scene,
        load_images=False,
        load_depth=False,
        load_masks=True,
        precomputed_latents=args.precomputed_latents,
        num_renders_per_scene=args.num_renders_per_scene,
        gso_root=args.gso_root,
        load_meshes=True,
        mesh_num_samples=args.mesh_num_samples,
    )

    uncached_load_gso_mesh = dataset._load_gso_mesh
    dataset._load_gso_mesh = lru_cache(maxsize=None)(uncached_load_gso_mesh)

    ground_eps = args.ground_eps if args.ground_eps is not None else args.contact_eps
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    stats = {"seen": total, "written": 0, "skipped": 0, "failed": 0}

    for idx in trange(total):
        render_dir = dataset.scenes[idx]["render_dir"]
        output_path = render_dir / args.output_name
        if output_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue
        try:
            graph = _build_sample_graph(
                dataset,
                idx,
                contact_eps=args.contact_eps,
                ground_eps=ground_eps,
                max_pair_points=args.max_pair_points,
                distance_chunk_size=args.distance_chunk_size,
                seed=args.seed,
            )
            _write_json(output_path, graph)
            stats["written"] += 1
            if not args.quiet and (idx == 0 or (idx + 1) % args.log_every == 0 or idx + 1 == total):
                print(
                    f"[{idx + 1}/{total}] wrote {output_path} "
                    f"objects={graph['stats']['num_objects']} "
                    f"pairs={graph['stats']['num_contact_pairs']} "
                    f"ground={graph['stats']['num_ground_contacts']}"
                )
        except Exception as exc:
            stats["failed"] += 1
            if args.fail_fast:
                raise
            print(f"[WARN] failed idx={idx} render_dir={render_dir}: {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FoundationPoseDataset contact_graph.json files.")
    parser.add_argument("--data-root", type=str, default="foundationpose")
    parser.add_argument("--gso-root", type=str, default="gso/google_scanned_objects/models_normalized")
    parser.add_argument("--output-name", type=str, default="contact_graph.json")
    parser.add_argument("--contact-eps", type=float, default=0.01)
    parser.add_argument("--ground-eps", type=float, default=None)
    parser.add_argument("--mesh-num-samples", type=int, default=2048)
    parser.add_argument("--max-pair-points", type=int, default=2048)
    parser.add_argument("--distance-chunk-size", type=int, default=512)
    parser.add_argument("--max-objects-per-scene", type=int, default=32)
    parser.add_argument("--num-renders-per-scene", type=int, default=1)
    parser.add_argument("--precomputed-latents", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_contact_graphs(args)
    if not args.quiet:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
