"""
YCB-V dataset test: load scene 48, frame 0 and visualise.

Outputs (debug/ycb/):
  scene.glb          — 3D scene: meshes transformed to BOP camera space
  rgb.png            — raw RGB
  rgb_masks.png      — RGB + coloured mask overlay + legend
  mask_gallery.png   — per-object masks
  depth.png          — depth map (plasma colourmap, mm)
  reprojection.png   — mesh silhouettes projected onto RGB
  pose_table.txt     — per-object pose info
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from ycb_dataset import YCBVDataset

# ── constants ─────────────────────────────────────────────────────────────────
DATA_ROOT   = "YCB-V"
MODELS_ROOT = "YCB-models"
SCENE_ID    = 49
OUT_DIR     = Path("debug/ycb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# BOP ↔ PyTorch3D flip matrix  M = diag([-1,-1,1]),  M^{-1} = M
_M = np.diag([-1.0, -1.0, 1.0])


# ── helpers ───────────────────────────────────────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D rotation → 3×3 rotation matrix (Zhou et al., Gram–Schmidt)."""
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # columns = basis vectors


def latents_to_bop(r6d: np.ndarray, t_p3d: np.ndarray):
    """
    Recover BOP model-to-camera R and t from SAM3D latent representation.

    Latents are in PyTorch3D convention (M = diag[-1,-1,1]):
        R_p3d = M @ R_bop @ M
        t_p3d = M @ t_bop
    Inverse:
        R_bop = M @ R_p3d @ M   (M is its own inverse)
        t_bop = M @ t_p3d
    """
    R_p3d = rot6d_to_matrix(r6d)
    R_bop = (_M @ R_p3d @ _M).astype(np.float64)
    t_bop = (_M @ t_p3d).astype(np.float64)
    return R_bop, t_bop


def distinct_colors(n: int) -> np.ndarray:
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return np.array([cmap(i)[:3] for i in range(n)])


# ── visualisation helpers ─────────────────────────────────────────────────────

def save_rgb(image: np.ndarray, path: Path) -> None:
    """Save [H,W,3] float32 [0,1] as PNG."""
    plt.imsave(str(path), image.clip(0, 1))
    print(f"Saved {path}")


def save_rgb_masks(image: np.ndarray, masks: np.ndarray, obj_ids: list, path: Path) -> None:
    """RGB + coloured mask overlay."""
    colors = distinct_colors(len(obj_ids))
    vis = image.copy()
    alpha = 0.45
    for i, c in enumerate(colors):
        m = masks[i] > 0.5
        vis[m] = vis[m] * (1 - alpha) + c * alpha
    vis_u8 = (vis * 255).clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(vis_u8)
    ax.set_title(f"Scene {SCENE_ID} — RGB + Masks")
    ax.axis("off")
    patches = [mpatches.Patch(color=colors[i], label=f"obj {obj_ids[i]}") for i in range(len(obj_ids))]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_mask_gallery(masks: np.ndarray, obj_ids: list, path: Path) -> None:
    n = masks.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes) if n > 1 else np.array([[axes]])
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(masks[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"obj {obj_ids[i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Per-Object Visibility Masks", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_depth(depth: np.ndarray, path: Path) -> None:
    valid = depth > 0
    d_vis = np.zeros_like(depth)
    if valid.any():
        d_vis[valid] = (depth[valid] - depth[valid].min()) / (depth[valid].max() - depth[valid].min() + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(d_vis, cmap="plasma")
    ax.set_title("Depth (mm, normalised)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_reprojection(image: np.ndarray, sample: dict, path: Path) -> None:
    """Project mesh vertices into image plane and overlay on RGB."""
    latents = sample["latents"]
    conds   = sample["conditionals"]
    K       = conds["camera_K"].numpy()
    obj_ids = sample["obj_ids"].tolist()
    n       = sample["num_objects"]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W   = image.shape[:2]
    colors = distinct_colors(n)

    all_tris = []   # (avg_z, tri_px, color_rgb)

    for i in range(n):
        r6d   = latents["6drotation_normalized"][i].numpy()
        t_p3d = latents["translation"][i].numpy()
        R_bop, t_bop = latents_to_bop(r6d, t_p3d)

        # Try to get mesh from mesh_data; fall back to a unit sphere
        if "mesh_data" in sample and sample["mesh_data"]["mesh_available"][i]:
            pts = sample["mesh_data"]["mesh_points"][i].numpy()  # [S, 3] model space, mm
        else:
            continue

        # Transform: model space (mm) → BOP camera space (mm)
        pts_cam = (R_bop @ pts.T).T + t_bop   # [S, 3]

        z = pts_cam[:, 2]
        valid = z > 1.0
        if not valid.any():
            continue
        pts_cam = pts_cam[valid]
        z = z[valid]

        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy
        pts_2d = np.stack([u, v], axis=-1).astype(np.float32)

        # Convex hull of projected points → filled polygon
        pts_int = pts_2d.astype(np.int32)
        pts_int[:, 0] = pts_int[:, 0].clip(0, W - 1)
        pts_int[:, 1] = pts_int[:, 1].clip(0, H - 1)
        hull = cv2.convexHull(pts_int)
        if hull is not None and len(hull) >= 3:
            color_rgb = tuple(int(c * 255) for c in colors[i])
            all_tris.append((float(z.mean()), hull, color_rgb))

    render = np.zeros((H, W, 3), dtype=np.uint8)
    all_tris.sort(key=lambda x: -x[0])   # far → near
    for _, hull, color in all_tris:
        cv2.fillPoly(render, [hull], color)

    rgb_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    mask_px = render.sum(axis=-1) > 0
    vis = rgb_u8.copy()
    vis[mask_px] = (
        vis[mask_px].astype(np.float32) * 0.5
        + render[mask_px].astype(np.float32) * 0.5
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(rgb_u8);  axes[0].set_title("Original RGB");          axes[0].axis("off")
    axes[1].imshow(vis);     axes[1].set_title("Mesh Reprojection");     axes[1].axis("off")
    patches = [mpatches.Patch(color=colors[i], label=f"obj {obj_ids[i]}") for i in range(n)]
    axes[1].legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def build_and_save_glb(sample: dict, path: Path) -> None:
    """
    Reconstruct scene in BOP camera space and export as GLB.

    Each object's PLY mesh (model space, mm) is transformed via
    cam_R_m2c / cam_t_m2c (recovered from latents) into BOP camera space,
    then converted to metres for viewer compatibility.
    """
    latents     = sample["latents"]
    n           = sample["num_objects"]
    obj_ids     = sample["obj_ids"].tolist()
    colors      = distinct_colors(n)
    models_root = Path(MODELS_ROOT)

    scene = trimesh.Scene()

    for i in range(n):
        obj_id = obj_ids[i]
        r6d    = latents["6drotation_normalized"][i].numpy()
        t_p3d  = latents["translation"][i].numpy()
        R_bop, t_bop = latents_to_bop(r6d, t_p3d)

        # 4×4 model-to-camera transform (still in mm)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_bop
        T[:3,  3] = t_bop

        # Load actual PLY mesh
        ply_path = models_root / f"obj_{obj_id:06d}.ply"
        if ply_path.exists():
            mesh = trimesh.load(str(ply_path), force="mesh")
        else:
            print(f"  [warn] PLY not found for obj {obj_id}, using sphere placeholder")
            mesh = trimesh.creation.uv_sphere(radius=50.0)

        # mm → m (vertices and translation)
        mesh.vertices /= 1000.0
        T_m = T.copy()
        T_m[:3, 3] /= 1000.0

        mesh.apply_transform(T_m)

        rgba = (np.array([*colors[i], 0.85]) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
        scene.add_geometry(mesh, node_name=f"obj_{obj_id:06d}")

    scene.export(str(path))
    print(f"Saved {path}")


def save_pose_table(sample: dict, path: Path) -> None:
    latents = sample["latents"]
    obj_ids = sample["obj_ids"].tolist()
    n       = sample["num_objects"]

    lines = []
    lines.append(f"Scene {SCENE_ID}  frame {sample['frame_id']}  ({n} objects)")
    lines.append("-" * 70)
    lines.append(f"{'idx':<4} {'obj_id':<8} {'t_x (mm)':<12} {'t_y (mm)':<12} {'t_z (mm)':<12}")
    lines.append("-" * 70)
    for i in range(n):
        t = latents["translation"][i].numpy()
        lines.append(f"{i:<4} {obj_ids[i]:<8} {t[0]:+.2f}{'':>4} {t[1]:+.2f}{'':>4} {t[2]:+.2f}")
    lines.append("-" * 70)
    table = "\n".join(lines)
    print(table)
    path.write_text(table + "\n")
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading YCBVDataset — scene {SCENE_ID} ...")
    dataset = YCBVDataset(
        data_root=DATA_ROOT,
        scene_ids=[SCENE_ID],
        load_depth=True,
        load_masks=True,
        ycb_models_root=MODELS_ROOT,
        load_meshes=True,
        mesh_num_samples=4096,
    )
    print(f"  Frames in scene: {len(dataset)}")

    sample = dataset[0]
    conds  = sample["conditionals"]
    n      = sample["num_objects"]
    print(f"  scene_id={sample['scene_id']}  frame_id={sample['frame_id']}  num_objects={n}")
    print(f"  obj_ids: {sample['obj_ids'].tolist()}")

    # Print latent shapes
    print("\nlatents:")
    for k, v in sample["latents"].items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print("\nconditionals:")
    for k, v in conds.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    if "mesh_data" in sample:
        print("\nmesh_data:")
        for k, v in sample["mesh_data"].items():
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        print(f"  available: {sample['mesh_data']['mesh_available'].tolist()}")

    image   = conds["image"].numpy()                 # [H, W, 3]
    obj_ids = sample["obj_ids"].tolist()

    # 1. RGB
    save_rgb(image, OUT_DIR / "rgb.png")

    # 2. RGB + masks
    if "object_masks" in conds:
        masks = conds["object_masks"].numpy()
        save_rgb_masks(image, masks, obj_ids, OUT_DIR / "rgb_masks.png")
        save_mask_gallery(masks, obj_ids, OUT_DIR / "mask_gallery.png")

    # 3. Depth
    if "depth" in conds:
        save_depth(conds["depth"].numpy(), OUT_DIR / "depth.png")

    # 4. Pose table
    save_pose_table(sample, OUT_DIR / "pose_table.txt")

    # 5. Mesh reprojection onto image
    save_reprojection(image, sample, OUT_DIR / "reprojection.png")

    # 6. 3D scene GLB
    build_and_save_glb(sample, OUT_DIR / "scene.glb")

    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
