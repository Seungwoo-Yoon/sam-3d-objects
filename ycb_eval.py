"""
YCB-V Evaluation: ADD-S metric for SAM3D (InferenceJoint)

For each frame in the given scene(s), runs InferenceJoint and computes
ADD-S (Average Distance - Symmetric) by comparing predicted vs GT poses
applied to the GT mesh model points.

Single coordinate convention throughout: BOP/OpenCV (X-right, Y-down, Z-forward).
Both GT and pred poses are converted to BOP right after extraction.
PLY vertices are used as-is (already in BOP model space).

  P_gt   = R_bop_gt   @ PLY_verts.T + t_bop_gt    [N, 3]
  P_pred = R_bop_pred @ PLY_verts.T + t_bop_pred  [N, 3]
  ADD-S  = mean_i( min_j ||P_gt_i - P_pred_j|| ) / diameter

Conversion from PyTorch3D (P3D) to BOP, M = diag([-1,-1,1]):
  R_bop = M @ R_p3d @ M
  t_bop = M @ t_p3d
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

from ycb_dataset import YCBVDataset

_M = np.diag([-1.0, -1.0, 1.0])   # P3D ↔ BOP: flip X and Y  (M^{-1} = M)


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D → 3×3 R (Zhou et al.).  Encoding stores first two COLUMNS of R."""
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def p3d_to_bop(R_p3d: np.ndarray, t_p3d: np.ndarray):
    """Convert P3D pose → BOP pose.  R_bop = M @ R_p3d @ M,  t_bop = M @ t_p3d."""
    # return (_M @ R_p3d @ _M).astype(np.float64), (_M @ t_p3d).astype(np.float64)
    return R_p3d @ _M, t_p3d


# ─── ADD-S (BOP convention throughout) ──────────────────────────────────────

def compute_adds(
    gt_pts: np.ndarray,      # [N, 3] GT model space (BOP), mm
    R_gt:   np.ndarray,      # [3, 3] BOP
    t_gt:   np.ndarray,      # [3]    BOP, mm
    R_pred: np.ndarray,      # [3, 3] BOP
    t_pred: np.ndarray,      # [3]    BOP, mm
    pred_pts: np.ndarray | None = None,  # [M, 3] pred model space (BOP), mm; falls back to gt_pts
) -> float:
    P_gt   = (R_gt   @ gt_pts.T).T + t_gt
    pts    = pred_pts if pred_pts is not None else gt_pts
    P_pred = (R_pred @ pts.T).T + t_pred
    dists  = np.linalg.norm(P_gt[:, None] - P_pred[None], axis=-1)
    return float(dists.min(axis=1).mean())


# ─── Mesh helpers (BOP convention throughout) ────────────────────────────────

def load_mesh_trimesh(models_root: Path, obj_id: int) -> trimesh.Trimesh | None:
    path = models_root / f"obj_{obj_id:06d}.ply"
    if not path.exists():
        return None
    return trimesh.load(str(path), force="mesh")


def apply_bop_pose_to_ply(
    model_mesh: trimesh.Trimesh,
    R_bop: np.ndarray,   # [3, 3]
    t_bop: np.ndarray,   # [3]  mm
) -> trimesh.Trimesh:
    """Transform PLY mesh (BOP model space) → BOP camera space."""
    mesh  = model_mesh.copy()
    verts = mesh.vertices.astype(np.float64)
    mesh.vertices = (R_bop @ verts.T).T + t_bop
    return mesh


def get_pred_model_pts(
    glb,
    scale: np.ndarray,   # [3]
    max_pts: int = 4096,
) -> np.ndarray | None:
    """
    Extract predicted mesh vertices in BOP model space (local, before pose).

    SAM3D local → BOP local:
      p_bop_local = M @ (scale * p_local)
    """
    if glb is None:
        return None
    if isinstance(glb, trimesh.Scene):
        parts = [g for g in glb.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None
        mesh = trimesh.util.concatenate([p.copy() for p in parts])
    elif isinstance(glb, trimesh.Trimesh):
        mesh = glb.copy()
    else:
        return None

    pts, _ = trimesh.sample.sample_surface(mesh, max_pts)
    pts = pts.astype(np.float64) * scale[None, :]  # apply scale
    pts = (_M @ pts.T).T                            # P3D local → BOP local
    return pts


def apply_bop_pose_to_sam3d_glb(
    glb,
    R_bop:  np.ndarray,   # [3, 3]
    t_bop:  np.ndarray,   # [3]  mm
    scale:  np.ndarray,   # [3]
) -> trimesh.Trimesh | None:
    """
    Transform SAM3D output GLB (P3D local space) → BOP camera space.

    SAM3D local → BOP camera:
      p_cam_bop = R_bop @ (M @ (scale * p_local)) + t_bop
    where M converts SAM3D local space (P3D convention) → BOP local space.
    """
    if glb is None:
        return None
    if isinstance(glb, trimesh.Scene):
        parts = [g for g in glb.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None
        mesh = trimesh.util.concatenate([p.copy() for p in parts])
    elif isinstance(glb, trimesh.Trimesh):
        mesh = glb.copy()
    else:
        return None

    verts = mesh.vertices.astype(np.float64) * scale[None, :]  # scale
    verts = (_M @ verts.T).T                                    # P3D local → BOP local
    verts = (R_bop @ verts.T).T + t_bop                        # BOP camera space
    mesh.vertices = verts
    return mesh


# ─── Collision score ────────────────────────────────────────────────────────

def compute_collision_score(pred_meshes_cam: list) -> float:
    """
    Collision score = (# objects involved in any mesh-mesh collision) / (total objects).

    Meshes must all be in the same coordinate frame (BOP camera space).
    Objects whose GLB was unavailable (None) are counted in the denominator
    but cannot be flagged as colliding.

    Uses AABB pre-filter then FCL BVH (trimesh CollisionManager).
    """
    n_total = len(pred_meshes_cam)
    valid = [(i, m) for i, m in enumerate(pred_meshes_cam) if m is not None]
    if n_total == 0 or len(valid) < 2:
        return 0.0

    # AABB pre-filter: keep only meshes that overlap with at least one other
    aabb_min = {vi: m.vertices.min(axis=0) for vi, m in valid}
    aabb_max = {vi: m.vertices.max(axis=0) for vi, m in valid}

    candidates: set = set()
    vis = [vi for vi, _ in valid]
    for a in range(len(vis)):
        for b in range(a + 1, len(vis)):
            vi, vj = vis[a], vis[b]
            if np.all(aabb_min[vi] <= aabb_max[vj]) and np.all(aabb_max[vi] >= aabb_min[vj]):
                candidates.add(vi)
                candidates.add(vj)

    if len(candidates) < 2:
        return 0.0

    manager = trimesh.collision.CollisionManager()
    for vi, mesh in valid:
        if vi in candidates:
            manager.add_object(str(vi), mesh)

    is_collision, colliding_pairs = manager.in_collision_internal(return_names=True)
    if not is_collision:
        return 0.0

    colliding_objects: set = set()
    for name_i, name_j in colliding_pairs:
        colliding_objects.add(int(name_i))
        colliding_objects.add(int(name_j))

    return len(colliding_objects) / n_total


# ─── Main eval loop ─────────────────────────────────────────────────────────

def run_eval(args):
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = YCBVDataset(
        data_root=args.data_root,
        scene_ids=args.scenes,
        load_depth=True,
        load_masks=True,
        ycb_models_root=args.models_root,
        load_meshes=False,   # we load meshes manually below
    )
    print(f"Dataset: {len(dataset)} frames")

    models_root = Path(args.models_root)

    # Load models_info for diameters
    models_info = {}
    info_path = models_root / "models_info.json"
    if info_path.exists():
        with open(info_path) as f:
            raw = json.load(f)
        models_info = {int(k): v for k, v in raw.items()}

    # Cache of GT meshes (for sampling + saving)
    mesh_cache: dict = {}

    def get_model_mesh(obj_id: int):
        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = load_mesh_trimesh(models_root, obj_id)
        return mesh_cache[obj_id]

    # ── Inference pipeline ───────────────────────────────────────────────────
    from inference_joint import InferenceJoint
    inference = InferenceJoint(args.config, compile=False)

    # ── Optional checkpoint fine-tuning ─────────────────────────────────────
    if args.checkpoint:
        from peft.peft_model import PeftModel
        backbone = inference._pipeline.models["ss_generator"].reverse_fn.backbone
        inference._pipeline.models["ss_generator"].reverse_fn.backbone = (
            PeftModel.from_pretrained(backbone, args.checkpoint, device_map="auto")
        )
        inference._pipeline.models["ss_generator"].reverse_fn.strength = 0.0
        print(f"Loaded LoRA checkpoint: {args.checkpoint}")

    # ── Output dirs ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metrics accumulator ──────────────────────────────────────────────────
    per_obj_adds: dict[int, list[float]] = defaultdict(list)  # obj_id → [adds_score, ...]
    all_adds: list[float] = []
    per_scene_collision: dict[str, list[float]] = defaultdict(list)  # scene_id → [collision_score, ...]
    all_collision: list[float] = []

    # ── Main loop ────────────────────────────────────────────────────────────
    for idx in range(len(dataset)):
        sample = dataset[idx]
        scene_id  = sample['scene_id']
        frame_id  = sample['frame_id']
        n_obj     = sample['num_objects']
        obj_ids   = sample['obj_ids'].tolist()

        if n_obj == 0:
            continue

        print(f"\n[{idx+1}/{len(dataset)}] scene={scene_id} frame={frame_id:06d}  objects={obj_ids}")

        # Prepare inputs for InferenceJoint
        image_np = (sample['conditionals']['image'].numpy() * 255).clip(0, 255).astype(np.uint8)
        masks_list = [
            sample['conditionals']['object_masks'][i].numpy().astype(bool)
            for i in range(n_obj)
        ]

        # GT pointmap: mask background (Z≈0) pixels to NaN so recover_focal_shift
        # doesn't encounter division-by-zero (Z=0 → infinite residuals).
        pointmap = sample['conditionals']['pointmap'].clone()  # [H, W, 3]
        bg_mask = pointmap[..., 2].abs() < 1e-3               # Z≈0 → background
        pointmap[bg_mask] = float('nan')

        print(f"inval pixels in pointmap: {bg_mask.sum().item()} / {bg_mask.numel()}")

        # Run SAM3D inference
        try:
            outputs = inference(image_np, masks_list, seed=args.seed, pointmap=pointmap)
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            continue

        if len(outputs) != n_obj:
            print(f"  [WARN] output count {len(outputs)} != n_obj {n_obj}, skipping frame")
            continue

        # Accumulators for scene-level GLB (one scene per frame)
        pred_parts: list[trimesh.Trimesh] = []
        gt_parts:   list[trimesh.Trimesh] = []
        pred_meshes_cam: list = []   # for collision detection (None if GLB unavailable)

        for i, (obj_id, output) in enumerate(zip(obj_ids, outputs)):
            # ── GT pose → BOP ─────────────────────────────────────────────
            r6d_gt   = sample['latents']['6drotation_normalized'][i].numpy()
            t_p3d_gt = sample['latents']['translation'][i].numpy()
            R_gt_bop, t_gt_bop = p3d_to_bop(rot6d_to_matrix(r6d_gt), t_p3d_gt)

            # ── Pred pose → BOP ───────────────────────────────────────────
            t_p3d_pred = output['translation'].detach().cpu().numpy().reshape(3)
            scale_pred  = output['scale'].detach().cpu().numpy().reshape(3)
            R_p3d_pred  = quaternion_to_matrix(
                output['rotation'].detach().cpu()
            ).squeeze(0).numpy()
            R_pred_bop, t_pred_bop = p3d_to_bop(R_p3d_pred, t_p3d_pred)

            # ── Pred mesh in BOP camera space (for collision + optional save) ──
            pred_cam = apply_bop_pose_to_sam3d_glb(
                output.get('glb'), R_pred_bop, t_pred_bop, scale_pred
            )
            pred_meshes_cam.append(pred_cam)   # None if GLB unavailable

            # ── ADD-S (GT model pts → P_gt, pred mesh pts → P_pred) ──────
            gt_mesh = get_model_mesh(obj_id)
            if gt_mesh is None:
                print(f"  [WARN] no mesh for obj {obj_id}, skipping")
                continue

            gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, 4096)
            gt_pts = gt_pts.astype(np.float64)  # BOP model space

            # Extract predicted mesh surface samples in BOP local space
            pred_pts = get_pred_model_pts(output.get('glb'), scale_pred, max_pts=4096)
            if pred_pts is None:
                print(f"  [WARN] no predicted mesh for obj {obj_id}, falling back to GT pts for P_pred")

            diameter = float(models_info.get(obj_id, {}).get('diameter', 1.0))
            adds_mm   = compute_adds(gt_pts, R_gt_bop, t_gt_bop, R_pred_bop, t_pred_bop, pred_pts)
            adds_norm = adds_mm / diameter

            per_obj_adds[obj_id].append(adds_norm)
            all_adds.append(adds_norm)

            passed = adds_norm < args.threshold
            print(f"  obj {obj_id:3d}: ADD-S={adds_norm:.4f} (diam={diameter:.1f}mm, "
                  f"raw={adds_mm:.1f}mm)  {'PASS' if passed else 'FAIL'}")

            # ── Accumulate meshes for scene GLB (BOP camera space) ────────
            if args.save_meshes:
                gt_parts.append(apply_bop_pose_to_ply(gt_mesh, R_gt_bop, t_gt_bop))
                if pred_cam is not None:
                    pred_parts.append(pred_cam)

        # ── Collision score for this frame ────────────────────────────────
        if n_obj > 1:
            col_score = compute_collision_score(pred_meshes_cam)
            per_scene_collision[str(scene_id)].append(col_score)
            all_collision.append(col_score)
            print(f"  collision score: {col_score:.4f}  "
                  f"({round(col_score * n_obj)}/{n_obj} objects colliding)")

        # ── Export per-frame scene GLBs (GT + pred combined) ──────────────
        if args.save_meshes and (gt_parts or pred_parts):
            frame_dir = out_dir / scene_id / f"{frame_id:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            if gt_parts:
                trimesh.util.concatenate(gt_parts).export(
                    str(frame_dir / "gt.glb")
                )
            if pred_parts:
                trimesh.util.concatenate(pred_parts).export(
                    str(frame_dir / "pred.glb")
                )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADD-S Results (normalized by diameter)")
    print(f"  Threshold: {args.threshold}")
    print("=" * 60)

    if not all_adds:
        print("No results computed.")
        return

    print(f"\n{'obj_id':>8}  {'#frames':>8}  {'mean ADD-S':>12}  {'AUC<thr':>10}")
    print("-" * 50)

    summary = {}
    for obj_id in sorted(per_obj_adds.keys()):
        scores = per_obj_adds[obj_id]
        mean_score = np.mean(scores)
        pass_rate  = np.mean([s < args.threshold for s in scores])
        summary[obj_id] = {'mean': mean_score, 'pass_rate': pass_rate, 'n': len(scores)}
        print(f"  {obj_id:6d}  {len(scores):8d}  {mean_score:12.4f}  {pass_rate:10.4f}")

    print("-" * 50)
    overall_mean = np.mean(all_adds)
    overall_pass = np.mean([s < args.threshold for s in all_adds])
    print(f"  {'ALL':>6}  {len(all_adds):8d}  {overall_mean:12.4f}  {overall_pass:10.4f}")
    print("=" * 60)

    # ── Collision report ─────────────────────────────────────────────────────
    if all_collision:
        print("\n" + "=" * 60)
        print("Collision Score  (colliding objects / total objects per frame)")
        print("=" * 60)
        print(f"\n{'scene_id':>10}  {'#frames':>8}  {'mean collision':>16}")
        print("-" * 42)
        for sid in sorted(per_scene_collision.keys()):
            scores = per_scene_collision[sid]
            print(f"  {sid:>8}  {len(scores):8d}  {np.mean(scores):16.4f}")
        print("-" * 42)
        print(f"  {'ALL':>8}  {len(all_collision):8d}  {np.mean(all_collision):16.4f}")
        print("=" * 60)

    # Save JSON summary
    collision_summary = {
        sid: {'mean': float(np.mean(v)), 'n': len(v)}
        for sid, v in per_scene_collision.items()
    }
    result = {
        'threshold': args.threshold,
        'overall': {'mean_adds': overall_mean, 'pass_rate': overall_pass, 'n': len(all_adds)},
        'per_object': {str(k): v for k, v in summary.items()},
        'collision': {
            'overall_mean': float(np.mean(all_collision)) if all_collision else None,
            'n_frames': len(all_collision),
            'per_scene': collision_summary,
        },
    }
    result_path = out_dir / "adds_results.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results → {result_path}")


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM3D on YCB-V with ADD-S metric")
    parser.add_argument("--data-root",   default="YCB-V",         help="YCB-V BOP data root")
    parser.add_argument("--models-root", default="YCB-models",    help="YCB-models directory")
    parser.add_argument("--config",      default="checkpoints/hf/pipeline_original.yaml")
    parser.add_argument("--checkpoint",  default=None,
                        help="Optional LoRA PEFT checkpoint dir to load into ss_generator")
    parser.add_argument("--scenes",      nargs="+", type=int, default=[48],
                        help="Scene IDs to evaluate (default: 48)")
    parser.add_argument("--threshold",   type=float, default=0.1,
                        help="ADD-S pass threshold (fraction of diameter, default 0.1)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--save-meshes", action="store_true",
                        help="Save predicted and GT meshes in BOP camera space as GLB")
    parser.add_argument("--output-dir",  default="debug/ycb_eval",
                        help="Directory for results and optional meshes")
    parser.add_argument("--min-visib",   type=float, default=0.1,
                        help="Minimum visibility fraction to include object (default 0.1)")

    args = parser.parse_args()
    run_eval(args)
