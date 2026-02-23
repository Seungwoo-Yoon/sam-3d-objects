#!/usr/bin/env python3
"""
YCB-V evaluation script for SAM3D.

Computes ADD / ADD-S metrics as defined in PoseCNN (Xiang et al., 2018).

Unit conventions:
  - YCB-V GT translation (cam_t_m2c): mm
  - YCB model vertices (PLY files):   mm
  - SAM3D translation output:         meters  →  × pred_translation_scale → mm
                                       (default scale = 1000)

Usage (quick test on 2 scenes):
    python evaluate_ycbv.py \\
        --config ../checkpoints/hf/pipeline.yaml \\
        --checkpoint ../outputs/joint_sam3d_3/step_00000200.pt \\
        --scenes 000048 000049 \\
        --max-frames 5 \\
        --visualize-every 1

Usage (full evaluation):
    python evaluate_ycbv.py \\
        --config ../checkpoints/hf/pipeline.yaml \\
        --checkpoint ../outputs/joint_sam3d_3/step_00000200.pt \\
        --output-json ./eval_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (set before pyplot import)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# YCB-V object IDs that are symmetric → use ADD-S instead of ADD
SYMMETRIC_OBJ_IDS = {1, 13, 16, 18, 19, 20, 21}

# YCB-V object name mapping (BOP numbering, 1-indexed)
OBJ_NAMES = {
    1:  "002_master_chef_can",
    2:  "003_cracker_box",
    3:  "004_sugar_box",
    4:  "005_tomato_soup_can",
    5:  "006_mustard_bottle",
    6:  "007_tuna_fish_can",
    7:  "008_pudding_box",
    8:  "009_gelatin_box",
    9:  "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class YCBVDataset:
    """
    Load YCB-V dataset stored in BOP format.

    Expected directory layout::

        ycbv_root/
          000048/
            rgb/          {frame_id:06d}.png
            depth/        {frame_id:06d}.png   (raw uint16, × depth_scale = mm)
            mask_visib/   {frame_id:06d}_{obj_idx:06d}.png
            scene_gt.json
            scene_gt_info.json
            scene_camera.json

        models_root/
          models_info.json
          obj_{obj_id:06d}.ply   (vertices in mm)
    """

    def __init__(
        self,
        ycbv_root: str,
        models_root: str,
        scenes: Optional[List[str]] = None,
        min_visib_fract: float = 0.1,
        model_num_points: int = 2048,
    ):
        self.ycbv_root = Path(ycbv_root)
        self.models_root = Path(models_root)
        self.min_visib_fract = min_visib_fract
        self.model_num_points = model_num_points

        # Discover scenes
        all_scenes = sorted(
            d for d in os.listdir(ycbv_root)
            if os.path.isdir(os.path.join(ycbv_root, d))
        )
        self.scenes = scenes if scenes is not None else all_scenes

        # Model point cloud cache: obj_id → (N, 3) mm
        self._model_cache: Dict[int, np.ndarray] = {}

        # Model metadata
        with open(self.models_root / "models_info.json") as f:
            self.model_info: Dict[int, dict] = {
                int(k): v for k, v in json.load(f).items()
            }

        # Build per-frame index
        self.frames = self._build_frame_index()

    # ── Index building ────────────────────────────────────────────────────────

    def _build_frame_index(self) -> List[Dict]:
        """Build list of frame dicts, each containing all visible objects."""
        frames = []
        for scene in self.scenes:
            scene_path = self.ycbv_root / scene

            with open(scene_path / "scene_gt.json") as f:
                scene_gt = {int(k): v for k, v in json.load(f).items()}
            with open(scene_path / "scene_gt_info.json") as f:
                scene_gt_info = {int(k): v for k, v in json.load(f).items()}
            with open(scene_path / "scene_camera.json") as f:
                scene_camera = {int(k): v for k, v in json.load(f).items()}

            for frame_id, obj_list in sorted(scene_gt.items()):
                if frame_id not in scene_camera:
                    continue

                cam_data = scene_camera[frame_id]
                cam_K = np.array(cam_data["cam_K"]).reshape(3, 3)
                # depth_scale: raw_depth_value × scale = depth_mm
                depth_scale = cam_data["depth_scale"]

                info_list = scene_gt_info.get(frame_id, [])

                objects = []
                for obj_idx, obj in enumerate(obj_list):
                    info = info_list[obj_idx] if obj_idx < len(info_list) else {}
                    visib_fract = info.get("visib_fract", 0.0)
                    if visib_fract < self.min_visib_fract:
                        continue
                    objects.append({
                        "obj_idx":   obj_idx,
                        "obj_id":    obj["obj_id"],
                        # BOP: R flattened row-major → shape (3,3)
                        "cam_R_m2c": np.array(obj["cam_R_m2c"]).reshape(3, 3),
                        "cam_t_m2c": np.array(obj["cam_t_m2c"]),   # mm
                        "visib_fract": visib_fract,
                        "bbox_visib":  info.get("bbox_visib"),
                    })

                if objects:
                    frames.append({
                        "scene":       scene,
                        "frame_id":    frame_id,
                        "cam_K":       cam_K,
                        "depth_scale": depth_scale,
                        "objects":     objects,
                    })
        return frames

    # ── Data loading ──────────────────────────────────────────────────────────

    def get_model_points(self, obj_id: int) -> np.ndarray:
        """Sampled 3-D model surface points in mm (cached)."""
        if obj_id not in self._model_cache:
            path = self.models_root / f"obj_{obj_id:06d}.ply"
            mesh = trimesh.load(str(path), process=False)
            pts = np.asarray(mesh.vertices, dtype=np.float64)
            if len(pts) > self.model_num_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(pts), self.model_num_points, replace=False)
                pts = pts[idx]
            self._model_cache[obj_id] = pts
        return self._model_cache[obj_id]

    def get_model_mesh(self, obj_id: int) -> trimesh.Trimesh:
        """Load full trimesh of the model (mm)."""
        path = self.models_root / f"obj_{obj_id:06d}.ply"
        return trimesh.load(str(path), process=False)

    def load_rgb(self, scene: str, frame_id: int) -> np.ndarray:
        path = self.ycbv_root / scene / "rgb" / f"{frame_id:06d}.png"
        return np.array(Image.open(path).convert("RGB"))

    def load_depth_mm(self, scene: str, frame_id: int, depth_scale: float) -> np.ndarray:
        """Return depth in mm (float32)."""
        path = self.ycbv_root / scene / "depth" / f"{frame_id:06d}.png"
        raw = np.array(Image.open(path)).astype(np.float32)
        return raw * depth_scale   # mm

    def load_mask_visib(
        self, scene: str, frame_id: int, obj_idx: int
    ) -> Optional[np.ndarray]:
        """Boolean visibility mask, or None if file is missing."""
        path = self.ycbv_root / scene / "mask_visib" / f"{frame_id:06d}_{obj_idx:06d}.png"
        if not path.exists():
            return None
        return (np.array(Image.open(path)) > 0)

    def depth_to_pointmap(
        self, depth_mm: np.ndarray, cam_K: np.ndarray
    ) -> np.ndarray:
        """
        Convert depth (mm) to an XYZ pointmap in **meters** (H, W, 3).

        SAM3D expects a pointmap in the same unit as its training data
        (typically metres from MoGE).
        """
        depth_m = depth_mm / 1000.0
        h, w = depth_m.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        X = (u - cx) / fx * depth_m
        Y = (v - cy) / fy * depth_m
        Z = depth_m
        return np.stack([X, Y, Z], axis=-1).astype(np.float32)   # (H, W, 3) m


# ─────────────────────────────────────────────────────────────────────────────
# ADD / ADD-S Metrics  (PoseCNN, Xiang et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

def compute_add(
    pts: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    R_pred: np.ndarray, t_pred: np.ndarray,
) -> float:
    """
    ADD metric.

    Mean distance between model points transformed by GT pose and predicted pose.
    Used for non-symmetric objects.

    Args:
        pts:    (N, 3) model points (mm)
        R_gt:   (3, 3) GT rotation   (model → camera)
        t_gt:   (3,)   GT translation (mm)
        R_pred: (3, 3) predicted rotation
        t_pred: (3,)   predicted translation (mm)
    Returns:
        Scalar ADD error in mm.
    """
    pts_gt   = pts @ R_gt.T   + t_gt
    pts_pred = pts @ R_pred.T + t_pred
    return float(np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1)))


def compute_add_s(
    pts: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    R_pred: np.ndarray, t_pred: np.ndarray,
) -> float:
    """
    ADD-S metric (symmetric version of ADD).

    Uses nearest-neighbour matching instead of index-to-index correspondence.
    Recommended for symmetric objects.

    Returns:
        Scalar ADD-S error in mm.
    """
    pts_gt   = pts @ R_gt.T   + t_gt
    pts_pred = pts @ R_pred.T + t_pred
    tree = cKDTree(pts_pred)
    dists, _ = tree.query(pts_gt, k=1)
    return float(np.mean(dists))


def compute_auc(
    errors: np.ndarray,
    max_thresh_mm: float,
    num_steps: int = 1000,
) -> float:
    """
    Area Under the Curve (AUC) of the ADD(-S) accuracy-vs-threshold plot,
    normalised to [0, 1].

    Args:
        errors:        1-D array of per-instance errors (mm).
        max_thresh_mm: upper integration limit (mm).
    """
    thresholds = np.linspace(0.0, max_thresh_mm, num_steps)
    accs = np.array([(errors < t).mean() for t in thresholds], dtype=np.float64)
    return float(np.trapz(accs, thresholds) / max_thresh_mm)


# ─────────────────────────────────────────────────────────────────────────────
# Pose extraction from SAM3D output
# ─────────────────────────────────────────────────────────────────────────────

def sam3d_output_to_pose_mm(
    output: dict,
    pred_translation_scale: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation matrix and translation in mm from a SAM3D output dict.

    SAM3D outputs (from InferencePipelineJoint):
      - ``rotation``:             quaternion tensor (w, x, y, z), shape (4,) or (1, 4)
      - ``translation``:          (3,) or (1, 3) tensor in metres
      - ``6drotation_normalized``: 6-D rotation if 'rotation' is absent

    Args:
        output:                  Dict returned by InferenceJoint.__call__
        pred_translation_scale:  Factor to convert SAM3D translation to mm.
                                 Default 1000 → metres to mm.
    Returns:
        R_pred: (3, 3) rotation matrix  (model canonical → camera)
        t_pred: (3,)   translation in mm
    """
    from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix

    # ── Rotation ──────────────────────────────────────────────────────────────
    if "rotation" in output:
        q = output["rotation"]
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().float()
            if q.dim() == 1:
                q = q.unsqueeze(0)
            R_pred = quaternion_to_matrix(q)[0].numpy()
        else:
            R_pred = np.asarray(q, dtype=np.float64).reshape(3, 3)
    elif "6drotation_normalized" in output:
        rot6d = output["6drotation_normalized"]
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.detach().cpu().float()
            if rot6d.dim() == 1:
                rot6d = rot6d.unsqueeze(0)
            R_pred = rotation_6d_to_matrix(rot6d)[0].numpy()
        else:
            raise KeyError("Cannot parse rotation from output dict.")
    else:
        raise KeyError("SAM3D output dict has no 'rotation' or '6drotation_normalized'.")

    # ── Translation ───────────────────────────────────────────────────────────
    t = output["translation"]
    if isinstance(t, torch.Tensor):
        t_pred_raw = t.detach().cpu().float().numpy().flatten()
    else:
        t_pred_raw = np.asarray(t, dtype=np.float64).flatten()

    t_pred_mm = t_pred_raw * pred_translation_scale   # → mm
    return R_pred.astype(np.float64), t_pred_mm.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SAM3D model loading & inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sam3d_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    compile: bool = False,
) -> "InferenceJoint":   # noqa: F821
    """
    Load InferenceJoint with optional SAM3D checkpoint.

    Args:
        config_path:     Path to ``pipeline.yaml``
        checkpoint_path: Path to a ``.pt`` file produced by the training loop.
                         The checkpoint is expected to hold a
                         ``model_state_dict`` key (as saved by train_dual_backbone_foundationpose.py).
        compile:         Enable ``torch.compile`` (slows first call, speeds up rest).
    """
    # Make sure the repo root is on sys.path
    repo_root = str(Path(config_path).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from inference_joint import InferenceJoint

    inference = InferenceJoint(config_path, compile=compile)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        inference._pipeline.models["ss_generator"].reverse_fn.backbone.load_state_dict(
            state_dict, strict=False
        )
        print(f"[SAM3D] Loaded checkpoint: {checkpoint_path}")

    return inference


def run_sam3d_inference(
    model,
    rgb: np.ndarray,
    masks: List[np.ndarray],
    pointmap: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Run SAM3D inference for one frame.

    Args:
        model:    InferenceJoint instance
        rgb:      (H, W, 3) uint8 RGB image
        masks:    List of (H, W) bool arrays, one per object
        pointmap: (H, W, 3) float32 XYZ in metres — fed to the pipeline
                  if available.  Pass None to let the model infer monocular depth.
        seed:     Random seed for reproducibility.
    Returns:
        List of output dicts (one per mask).
    """
    pm = None
    if pointmap is not None:
        pm = torch.from_numpy(pointmap.astype(np.float32)).cuda()

    with torch.no_grad():
        outputs = model(rgb, masks, seed=seed, pointmap=pm)

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _project_points(pts_3d_mm: np.ndarray, cam_K: np.ndarray):
    """
    Project 3-D camera-frame points (mm) to 2-D pixel coordinates.

    Returns:
        uv:    (N, 2) float array of pixel coords
        valid: (N,)   bool mask — True where depth > 0
    """
    valid = pts_3d_mm[:, 2] > 0.0
    uv = np.zeros((len(pts_3d_mm), 2), dtype=np.float32)
    if valid.any():
        p = pts_3d_mm[valid]                   # (M, 3)
        uvh = (cam_K @ p.T).T                  # (M, 3)
        uv[valid] = uvh[:, :2] / uvh[:, 2:3]
    return uv, valid


def visualize_pose_overlay(
    rgb: np.ndarray,
    cam_K: np.ndarray,
    gt_annotations: List[dict],
    predictions: Optional[List[Optional[dict]]],
    model_points_dict: Dict[int, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
    point_size: float = 1.5,
    alpha: float = 0.55,
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Overlay GT (green) and SAM3D predicted (red/orange) poses on the RGB image.

    Each object gets its own shade of green / red so multiple objects are
    distinguishable.

    Args:
        rgb:                (H, W, 3) uint8 RGB image.
        cam_K:              (3, 3) camera intrinsic matrix.
        gt_annotations:     List of dicts (keys: ``obj_id``, ``cam_R_m2c``, ``cam_t_m2c``).
        predictions:        List of dicts (keys: ``obj_id``, ``R_pred``, ``t_pred_mm``),
                            or None per entry if inference failed.
                            Pass ``None`` for the whole list to skip predictions.
        model_points_dict:  {obj_id → (N, 3) array in mm}.
        save_path:          If given, save the figure to this path.
        show:               If True, call ``plt.show()`` (requires display).
        point_size:         Scatter point size.
        alpha:              Point transparency.
        title:              Custom figure title.

    Returns:
        The input ``rgb`` array (unchanged).
    """
    n_obj = max(len(gt_annotations), 1)
    greens = plt.cm.Greens(np.linspace(0.5, 0.95, n_obj))
    reds   = plt.cm.Reds  (np.linspace(0.5, 0.95, n_obj))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb)
    ax.set_title(title or "GT (green) vs SAM3D (red)", fontsize=13)
    ax.axis("off")

    for i, ann in enumerate(gt_annotations):
        obj_id = ann["obj_id"]
        pts    = model_points_dict.get(obj_id)
        if pts is None:
            continue

        label_sfx = f" obj{obj_id}"

        # ── GT overlay ────────────────────────────────────────────────────────
        pts_cam_gt = pts @ ann["cam_R_m2c"].T + ann["cam_t_m2c"]
        uv_gt, valid_gt = _project_points(pts_cam_gt, cam_K)
        if valid_gt.any():
            ax.scatter(
                uv_gt[valid_gt, 0], uv_gt[valid_gt, 1],
                s=point_size, color=greens[i], alpha=alpha,
                label=f"GT{label_sfx}",
            )

        # ── Prediction overlay ────────────────────────────────────────────────
        if predictions is not None and i < len(predictions) and predictions[i] is not None:
            pred = predictions[i]
            pts_cam_pred = pts @ pred["R_pred"].T + pred["t_pred_mm"]
            uv_pred, valid_pred = _project_points(pts_cam_pred, cam_K)
            if valid_pred.any():
                ax.scatter(
                    uv_pred[valid_pred, 0], uv_pred[valid_pred, 1],
                    s=point_size, color=reds[i], alpha=alpha,
                    label=f"Pred{label_sfx}",
                )

    # Common legend entries
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=9, label="GT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=9, label="SAM3D Pred"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Vis] Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return rgb


def visualize_frame(
    dataset: YCBVDataset,
    scene: str,
    frame_id: int,
    predictions: Optional[List[Optional[dict]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Convenience function: load a frame from the dataset and visualise it.

    Useful for interactive exploration::

        from evaluate_ycbv import YCBVDataset, visualize_frame
        ds = YCBVDataset('./YCB-V', './YCB-models')
        visualize_frame(ds, '000048', 1, show=True)

    Args:
        dataset:     YCBVDataset instance.
        scene:       Scene name, e.g. ``'000048'``.
        frame_id:    Integer frame ID.
        predictions: Optional list of prediction dicts (same format as
                     returned by :func:`evaluate`).  Pass ``None`` to show
                     GT only.
        save_path:   Save figure to this path if given.
        show:        Call ``plt.show()`` (needs a display / interactive mode).
    """
    # Find frame metadata
    frame_meta = None
    for f in dataset.frames:
        if f["scene"] == scene and f["frame_id"] == frame_id:
            frame_meta = f
            break
    if frame_meta is None:
        raise ValueError(f"Frame {scene}/{frame_id} not found in dataset index.")

    rgb = dataset.load_rgb(scene, frame_id)
    objects = frame_meta["objects"]
    model_pts = {o["obj_id"]: dataset.get_model_points(o["obj_id"]) for o in objects}

    visualize_pose_overlay(
        rgb         = rgb,
        cam_K       = frame_meta["cam_K"],
        gt_annotations = objects,
        predictions    = predictions,
        model_points_dict = model_pts,
        save_path   = save_path,
        show        = show,
        title       = f"Scene {scene}  Frame {frame_id:06d}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_results(results: List[dict]) -> dict:
    """Aggregate per-instance results into per-class and overall summaries."""
    if not results:
        return {}

    all_add     = np.array([r["add"]          for r in results])
    all_add_s   = np.array([r["add_s"]        for r in results])
    all_adds_sel = np.array([r["add_s_select"] for r in results])
    all_diams   = np.array([r["diameter"]     for r in results])

    thresh = all_diams * 0.1   # 10% of object diameter

    # AUC upper limit: 50% of mean diameter (common convention)
    auc_max = float(all_diams.mean() * 0.5)

    overall = {
        "n_samples":           len(results),
        "ADD_acc_10pct":       float((all_add     < thresh).mean()),
        "ADD-S_acc_10pct":     float((all_add_s   < thresh).mean()),
        "ADD(-S)_acc_10pct":   float((all_adds_sel < thresh).mean()),
        "ADD_mean_mm":         float(all_add.mean()),
        "ADD-S_mean_mm":       float(all_add_s.mean()),
        "ADD_AUC":             compute_auc(all_add,   auc_max),
        "ADD-S_AUC":           compute_auc(all_add_s, auc_max),
        "ADD(-S)_AUC":         compute_auc(all_adds_sel, auc_max),
    }

    per_object = {}
    by_obj: Dict[int, List[dict]] = {}
    for r in results:
        by_obj.setdefault(r["obj_id"], []).append(r)

    for obj_id, obj_res in by_obj.items():
        adds     = np.array([r["add"]          for r in obj_res])
        add_ss   = np.array([r["add_s"]        for r in obj_res])
        adds_sel = np.array([r["add_s_select"] for r in obj_res])
        diam     = obj_res[0]["diameter"]
        thresh_o = diam * 0.1

        per_object[obj_id] = {
            "obj_name":          OBJ_NAMES.get(obj_id, f"obj_{obj_id:06d}"),
            "n_samples":         len(obj_res),
            "diameter_mm":       diam,
            "is_symmetric":      obj_res[0]["is_symmetric"],
            "ADD_acc_10pct":     float((adds     < thresh_o).mean()),
            "ADD-S_acc_10pct":   float((add_ss   < thresh_o).mean()),
            "ADD(-S)_acc_10pct": float((adds_sel < thresh_o).mean()),
            "ADD_mean_mm":       float(adds.mean()),
            "ADD-S_mean_mm":     float(add_ss.mean()),
        }

    return {"overall": overall, "per_object": per_object, "raw": results}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    dataset: YCBVDataset,
    model,
    pred_translation_scale: float = 1000.0,
    seed: int = 42,
    max_frames: Optional[int] = None,
    use_depth_pointmap: bool = True,
    visualize_every: int = 0,
    vis_output_dir: str = "./eval_vis",
) -> dict:
    """
    Run ADD / ADD-S evaluation over the dataset.

    Args:
        dataset:                YCBVDataset instance.
        model:                  Loaded InferenceJoint model.
        pred_translation_scale: SAM3D translation unit → mm.
                                1000 for metres → mm (default).
        seed:                   Random seed for SAM3D.
        max_frames:             Truncate evaluation at this many frames.
        use_depth_pointmap:     Compute XYZ pointmap from depth and pass it
                                to the model for conditioning.
        visualize_every:        Save overlay image every N frames (0 = off).
        vis_output_dir:         Directory for visualisation PNG files.

    Returns:
        Summary dict with ``overall``, ``per_object``, and ``raw`` entries.
    """
    results: List[dict] = []
    all_predictions: dict = {}   # (scene, frame_id) → list of pred dicts

    frames = dataset.frames[:max_frames] if max_frames else dataset.frames

    for frame_idx, frame in enumerate(tqdm(frames, desc="Evaluating")):
        scene    = frame["scene"]
        frame_id = frame["frame_id"]
        cam_K    = frame["cam_K"]
        objects  = frame["objects"]

        # ── Load data ─────────────────────────────────────────────────────────
        rgb = dataset.load_rgb(scene, frame_id)

        pointmap = None
        if use_depth_pointmap:
            depth_mm = dataset.load_depth_mm(scene, frame_id, frame["depth_scale"])
            pointmap = dataset.depth_to_pointmap(depth_mm, cam_K)

        masks         = []
        valid_objects = []
        for obj in objects:
            mask = dataset.load_mask_visib(scene, frame_id, obj["obj_idx"])
            if mask is None or mask.sum() == 0:
                continue
            masks.append(mask)
            valid_objects.append(obj)

        if not masks:
            continue

        # ── SAM3D inference ───────────────────────────────────────────────────
        try:
            outputs = run_sam3d_inference(model, rgb, masks,
                                          pointmap=pointmap, seed=seed)
        except Exception as e:
            print(f"[WARN] Inference failed for {scene}/{frame_id:06d}: {e}")
            continue

        # ── Compute metrics per object ────────────────────────────────────────
        frame_preds: List[Optional[dict]] = []

        for obj, output in zip(valid_objects, outputs):
            obj_id = obj["obj_id"]
            R_gt   = obj["cam_R_m2c"]   # (3,3)
            t_gt   = obj["cam_t_m2c"]   # (3,) mm

            try:
                R_pred, t_pred_mm = sam3d_output_to_pose_mm(
                    output, pred_translation_scale
                )
            except (KeyError, Exception) as e:
                print(f"[WARN] Pose extraction failed obj{obj_id}: {e}")
                frame_preds.append(None)
                continue

            pts      = dataset.get_model_points(obj_id)
            diameter = dataset.model_info[obj_id]["diameter"]
            is_sym   = obj_id in SYMMETRIC_OBJ_IDS

            add_val   = compute_add  (pts, R_gt, t_gt, R_pred, t_pred_mm)
            add_s_val = compute_add_s(pts, R_gt, t_gt, R_pred, t_pred_mm)

            results.append({
                "scene":        scene,
                "frame_id":     frame_id,
                "obj_id":       obj_id,
                "obj_idx":      obj["obj_idx"],
                "add":          add_val,
                "add_s":        add_s_val,
                # ADD(-S): ADD-S for symmetric objects, ADD otherwise
                "add_s_select": add_s_val if is_sym else add_val,
                "diameter":     diameter,
                "visib_fract":  obj["visib_fract"],
                "is_symmetric": is_sym,
                "R_pred":       R_pred.tolist(),
                "t_pred_mm":    t_pred_mm.tolist(),
            })

            frame_preds.append({
                "obj_id":    obj_id,
                "R_pred":    R_pred,
                "t_pred_mm": t_pred_mm,
            })

        all_predictions[(scene, frame_id)] = frame_preds

        # ── Optional visualisation ────────────────────────────────────────────
        if visualize_every > 0 and frame_idx % visualize_every == 0:
            model_pts_vis = {
                o["obj_id"]: dataset.get_model_points(o["obj_id"])
                for o in valid_objects
            }
            save_path = os.path.join(
                vis_output_dir, scene, f"{frame_id:06d}.png"
            )
            visualize_pose_overlay(
                rgb=rgb,
                cam_K=cam_K,
                gt_annotations=valid_objects,
                predictions=frame_preds,
                model_points_dict=model_pts_vis,
                save_path=save_path,
                show=False,
                title=f"Scene {scene}  Frame {frame_id:06d}",
            )

    summary = _aggregate_results(results)
    summary["_predictions"] = all_predictions   # keep for later visualisation
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print a formatted evaluation summary to stdout."""
    overall = summary.get("overall", {})

    print("\n" + "=" * 66)
    print("  YCB-V / SAM3D  –  Evaluation Results")
    print("=" * 66)
    print(f"\nOverall ({overall.get('n_samples', 0)} object instances):")
    print(f"  ADD      acc @ 10%d  : {overall.get('ADD_acc_10pct',     0) * 100:6.2f}%")
    print(f"  ADD-S    acc @ 10%d  : {overall.get('ADD-S_acc_10pct',   0) * 100:6.2f}%")
    print(f"  ADD(-S)  acc @ 10%d  : {overall.get('ADD(-S)_acc_10pct', 0) * 100:6.2f}%")
    print(f"  ADD      AUC         : {overall.get('ADD_AUC',            0) * 100:6.2f}%")
    print(f"  ADD-S    AUC         : {overall.get('ADD-S_AUC',          0) * 100:6.2f}%")
    print(f"  ADD(-S)  AUC         : {overall.get('ADD(-S)_AUC',        0) * 100:6.2f}%")
    print(f"  ADD      mean error  : {overall.get('ADD_mean_mm',        0):8.2f} mm")
    print(f"  ADD-S    mean error  : {overall.get('ADD-S_mean_mm',      0):8.2f} mm")

    per_obj = summary.get("per_object", {})
    if per_obj:
        print("\nPer-object breakdown (ADD / ADD-S accuracy @ 10% diameter):")
        hdr = (f"  {'ID':>4}  {'Name':<28}  {'N':>4}  {'Diam':>6}  "
               f"{'Sym':>3}  {'ADD%':>6}  {'ADDS%':>6}  {'ADD(-S)%':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for obj_id in sorted(per_obj.keys()):
            m = per_obj[obj_id]
            print(
                f"  {obj_id:>4}  {m['obj_name']:<28}  {m['n_samples']:>4}  "
                f"{m['diameter_mm']:>5.0f}mm  "
                f"{'Y' if m['is_symmetric'] else 'N':>3}  "
                f"{m['ADD_acc_10pct']     * 100:>5.1f}%  "
                f"{m['ADD-S_acc_10pct']   * 100:>5.1f}%  "
                f"{m['ADD(-S)_acc_10pct'] * 100:>7.1f}%"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YCB-V ADD/ADD-S evaluation for SAM3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # paths
    p.add_argument("--ycbv-root",    default="./YCB-V",
                   help="YCB-V dataset root directory")
    p.add_argument("--models-root",  default="./YCB-models",
                   help="Directory with obj_XXXXXX.ply + models_info.json")
    p.add_argument("--config",       default="../checkpoints/hf/pipeline.yaml",
                   help="SAM3D pipeline YAML config")
    p.add_argument("--checkpoint",   default=None,
                   help="SAM3D .pt checkpoint (optional)")
    # dataset
    p.add_argument("--scenes",       nargs="+", default=None,
                   help="Subset of scene IDs to evaluate (e.g. 000048 000049)")
    p.add_argument("--max-frames",   type=int, default=None,
                   help="Process at most this many frames (quick test)")
    p.add_argument("--min-visib",    type=float, default=0.1,
                   help="Minimum visibility fraction to include an object")
    # inference
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--pred-translation-scale", type=float, default=1000.0,
                   help="Multiply SAM3D translation output to get mm "
                        "(1000 → metres to mm)")
    p.add_argument("--no-pointmap",  action="store_true",
                   help="Disable depth-based pointmap conditioning")
    # output
    p.add_argument("--visualize-every", type=int, default=0,
                   help="Save an overlay PNG every N frames (0 = disabled)")
    p.add_argument("--vis-dir",      default="./eval_vis",
                   help="Output directory for visualisation images")
    p.add_argument("--output-json",  default="./eval_results.json",
                   help="Save aggregated results JSON to this path")
    return p


def main():
    args = build_arg_parser().parse_args()

    # Dataset
    dataset = YCBVDataset(
        ycbv_root      = args.ycbv_root,
        models_root    = args.models_root,
        scenes         = args.scenes,
        min_visib_fract= args.min_visib,
    )
    print(f"[Dataset]  {len(dataset.frames)} frames  |  "
          f"{len(dataset.scenes)} scenes")

    # Model
    model = load_sam3d_model(
        config_path     = args.config,
        checkpoint_path = args.checkpoint,
        compile         = False,
    )
    print("[Model]  SAM3D loaded")

    # Evaluate
    summary = evaluate(
        dataset                = dataset,
        model                  = model,
        pred_translation_scale = args.pred_translation_scale,
        seed                   = args.seed,
        max_frames             = args.max_frames,
        use_depth_pointmap     = not args.no_pointmap,
        visualize_every        = args.visualize_every,
        vis_output_dir         = args.vis_dir,
    )

    # Report
    print_summary(summary)

    # Persist
    saveable = {k: v for k, v in summary.items() if k != "_predictions"}
    with open(args.output_json, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"[Results]  Saved → {args.output_json}")


if __name__ == "__main__":
    main()
