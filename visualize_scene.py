"""
Visualize a scene from FoundationPoseDataset.

Loads one scene's objects and poses, then produces:
  1. RGB image with colored segmentation masks overlaid
  2. Per-object mask gallery
  3. Camera-view mesh reprojection (3D meshes projected back onto the original image)
  4. 3D scene with GSO meshes placed at ground-truth poses

Usage:
    python visualize_scene.py --scene-idx 0
    python visualize_scene.py --scene-idx 0 --no-3d          # skip 3D viewer
    python visualize_scene.py --scene-idx 0 --save-dir vis/  # save figures
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import trimesh

from foundation_pose_dataset import FoundationPoseDataset


# ──────────────────────── helpers ────────────────────────

def distinct_colors(n: int) -> np.ndarray:
    """Return n visually distinct RGB colors in [0,1]."""
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    return np.array([cmap(i)[:3] for i in range(n)])


def overlay_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend colored per-object masks onto an RGB image.

    Args:
        image: [H, W, 3] float in [0,1]
        masks: [N, H, W] float (binary-ish)
        alpha: blending weight for mask color
    Returns:
        blended: [H, W, 3] uint8
    """
    colors = distinct_colors(masks.shape[0])
    vis = image.copy()
    for i, color in enumerate(colors):
        m = masks[i] > 0.5
        vis[m] = vis[m] * (1 - alpha) + np.array(color) * alpha
    return (vis * 255).clip(0, 255).astype(np.uint8)


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Convert [6] 6D rotation to [3,3] rotation matrix (Gram-Schmidt)."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def build_transform(translation: np.ndarray, rotation_6d: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from translation, 6D-rotation, scale."""
    R = rotation_6d_to_matrix(rotation_6d)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R * scale[np.newaxis, :]  # column-wise scaling
    T[:3, 3] = translation
    return T


# ──────────────────────── plotting ────────────────────────

def plot_rgb_with_masks(image: np.ndarray, masks: np.ndarray, obj_names: list, save_path: str | None = None):
    """Plot RGB with coloured mask overlay + legend."""
    blended = overlay_masks(image, masks)
    colors = distinct_colors(len(obj_names))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(blended)
    ax.set_title("Scene: RGB + Instance Masks")
    ax.axis("off")

    patches = [mpatches.Patch(color=colors[i], label=obj_names[i]) for i in range(len(obj_names))]
    ax.legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay  -> {save_path}")
    plt.close(fig)


def plot_mask_gallery(masks: np.ndarray, obj_names: list, save_path: str | None = None):
    """Show each object mask in a grid."""
    n = masks.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(masks[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(obj_names[i], fontsize=8)
        ax.axis("off")

    fig.suptitle("Per-Object Masks", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved gallery  -> {save_path}")
    plt.close(fig)


def plot_pose_table(latents: dict, obj_names: list, save_path: str | None = None):
    """Print a nice table of per-object pose information."""
    translations = latents["translation"].numpy()
    scales = latents["scale"].numpy()
    rotations = latents["6drotation_normalized"].numpy()

    header = f"{'#':<3} {'Object':<50} {'Translation (x,y,z)':<30} {'Scale (x,y,z)':<25}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for i, name in enumerate(obj_names):
        t = translations[i]
        s = scales[i]
        t_str = f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"
        s_str = f"({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})"
        lines.append(f"{i:<3} {name:<50} {t_str:<30} {s_str:<25}")
    lines.append(sep)
    table = "\n".join(lines)
    print(table)

    if save_path:
        with open(save_path, "w") as f:
            f.write(table + "\n")
        print(f"Saved table    -> {save_path}")


def build_3d_scene(
    obj_names: list,
    latents: dict,
    gso_root: Path,
) -> trimesh.Scene:
    """
    Load each object's GSO mesh, reconstruct world transforms from latents
    (translation, 6D-rotation, scale), and assemble into a trimesh.Scene.
    """
    scene = trimesh.Scene()
    colors = distinct_colors(len(obj_names))

    translations = latents["translation"].numpy()
    rotations_6d = latents["6drotation_normalized"].numpy()
    scales = latents["scale"].numpy()

    for i, name in enumerate(obj_names):
        # Reconstruct 4x4 transform from latents
        T = build_transform(translations[i], rotations_6d[i], scales[i])

        # Try to load mesh
        mesh_path = gso_root / name / "meshes" / "model.obj"
        if not mesh_path.exists():
            mesh_path = gso_root / name / "meshes" / "model.ply"
        if not mesh_path.exists():
            # Fallback: use a small sphere as placeholder
            mesh = trimesh.creation.uv_sphere(radius=0.02)
        else:
            try:
                mesh = trimesh.load(str(mesh_path), force="mesh")
            except Exception:
                mesh = trimesh.creation.uv_sphere(radius=0.02)

        # Apply color
        rgba = (np.array([*colors[i], 0.85]) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)

        mesh.apply_transform(T)
        scene.add_geometry(mesh, node_name=name)

    return scene


def render_camera_view(
    obj_names: list,
    latents: dict,
    gso_root: Path,
    camera_K: np.ndarray,
    image: np.ndarray,
    save_path: str | None = None,
):
    """
    Render 3D meshes reprojected onto the original camera view.

    Object poses in `latents` are in PyTorch3D camera coordinates
    (X-left, Y-up, Z-forward).  We convert to OpenCV convention
    (X-right, Y-down, Z-forward) and project with K to get pixel coords,
    then overlay coloured mesh silhouettes on the original RGB image.
    """
    H, W = image.shape[:2]
    colors = distinct_colors(len(obj_names))

    translations = latents["translation"].numpy()
    rotations_6d = latents["6drotation_normalized"].numpy()
    scales = latents["scale"].numpy()

    fx, fy = camera_K[0, 0], camera_K[1, 1]
    cx, cy = camera_K[0, 2], camera_K[1, 2]

    # Collect (avg_depth, triangle_2d, colour) for painter's algorithm
    all_tris: list[tuple[float, np.ndarray, tuple]] = []

    for i, name in enumerate(obj_names):
        T = build_transform(translations[i], rotations_6d[i], scales[i])

        # Load mesh
        mesh_path = gso_root / name / "meshes" / "model.obj"
        if not mesh_path.exists():
            mesh_path = gso_root / name / "meshes" / "model.ply"
        if not mesh_path.exists():
            continue
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh")
        except Exception:
            continue

        # Simplify heavy meshes for faster rendering
        max_faces = 2000
        if len(mesh.faces) > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            except Exception:
                # fallback: subsample faces
                idx = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=int)
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx])

        # Transform model vertices -> camera space (PyTorch3D coords)
        verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        verts_cam = (T @ verts_h.T).T[:, :3]

        # PyTorch3D (X-left, Y-up, Z-fwd) -> OpenCV (X-right, Y-down, Z-fwd)
        verts_cv = verts_cam * np.array([-1.0, -1.0, 1.0])

        # Perspective projection
        z = verts_cv[:, 2]
        u = fx * verts_cv[:, 0] / z + cx
        v = fy * verts_cv[:, 1] / z + cy
        pts_2d = np.stack([u, v], axis=-1)  # [N_verts, 2]

        color_rgb = tuple(int(c) for c in (colors[i] * 255).astype(np.uint8))

        for face in mesh.faces:
            if (z[face] <= 0.01).any():
                continue
            tri = pts_2d[face].astype(np.int32)  # [3, 2]
            avg_z = float(z[face].mean())
            all_tris.append((avg_z, tri, color_rgb))

    # Sort far → near (painter's algorithm)
    all_tris.sort(key=lambda x: -x[0])

    # Rasterize
    render = np.zeros((H, W, 3), dtype=np.uint8)
    for _, tri, color in all_tris:
        cv2.fillPoly(render, [tri], color)

    # Blend with original image
    alpha = 0.5
    rgb_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    mask = render.sum(axis=-1) > 0
    vis = rgb_u8.copy()
    vis[mask] = (vis[mask].astype(np.float32) * (1 - alpha)
                 + render[mask].astype(np.float32) * alpha).astype(np.uint8)

    # --- figure: side-by-side original vs overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(rgb_u8)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    axes[1].imshow(vis)
    axes[1].set_title("Camera-View Mesh Reprojection")
    axes[1].axis("off")

    patches = [mpatches.Patch(color=colors[i], label=obj_names[i])
               for i in range(len(obj_names))]
    axes[1].legend(handles=patches, loc="upper left", fontsize=6, framealpha=0.7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved cam view -> {save_path}")
    plt.close(fig)


def render_3d_topdown(scene: trimesh.Scene, save_path: str | None = None):
    """Render a top-down view of the 3D scene using matplotlib scatter of mesh vertices."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for name, geom in scene.geometry.items():
        if name == "ground_plane":
            continue
        verts = geom.vertices
        color = geom.visual.vertex_colors[:1, :3] / 255.0 if hasattr(geom.visual, "vertex_colors") else [0.5, 0.5, 0.5]
        # Downsample for faster plotting
        step = max(1, len(verts) // 500)
        ax.scatter(
            verts[::step, 0], verts[::step, 1], verts[::step, 2],
            s=1, alpha=0.6, color=color[0] if len(np.array(color).shape) > 1 else color,
            label=name,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scene (vertex scatter)")
    ax.legend(fontsize=6, loc="upper left", markerscale=5)

    # Set equal aspect ratio
    all_verts = np.concatenate([g.vertices for n, g in scene.geometry.items() if n != "ground_plane"], axis=0)
    mid = all_verts.mean(axis=0)
    span = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D view  -> {save_path}")
    plt.close(fig)


# ──────────────────────── main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize a FoundationPose scene")
    parser.add_argument("--data-root", type=str, default="/workspace/sam-3d-objects/foundationpose",
                        help="FoundationPose data root")
    parser.add_argument("--gso-root", type=str,
                        default="/workspace/sam-3d-objects/gso/google_scanned_objects/models_normalized",
                        help="GSO mesh root (models_orig/)")
    parser.add_argument("--scene-idx", type=int, default=0, help="Scene index to visualize")
    parser.add_argument("--save-dir", type=str, default="vis_scene", help="Output directory")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D mesh visualization")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    gso_root = Path(args.gso_root)

    # ── 1. Load dataset ──────────────────────────────────
    print("Loading FoundationPoseDataset ...")
    dataset = FoundationPoseDataset(
        data_root=args.data_root,
        load_images=True,
        load_depth=True,
        load_masks=True,
        load_meshes=True,
        gso_root=args.gso_root,
        mesh_num_samples=2048,
    )
    print(f"  Total renders: {len(dataset)}")

    if args.scene_idx >= len(dataset):
        print(f"Scene index {args.scene_idx} out of range (0..{len(dataset)-1})")
        sys.exit(1)

    # ── 2. Fetch one sample ──────────────────────────────
    sample = dataset[args.scene_idx]
    scene_id = sample["scene_id"]
    num_objects = sample["num_objects"]
    latents = sample["latents"]
    conditionals = sample["conditionals"]

    print(f"\n  Scene ID     : {scene_id}")
    print(f"  Num objects  : {num_objects}")

    # Recover visible object names from the render info
    render_info = dataset.scenes[args.scene_idx]
    object_dir = render_info["object_dir"]
    render_dir = render_info["render_dir"]
    states = dataset._load_states(object_dir)
    frame_data = dataset._load_frame_data(render_dir, frame_id=0)

    visible_objects = []
    if "id_mapping" in frame_data:
        for prim_path in frame_data["id_mapping"].values():
            obj_info = dataset._extract_object_name_from_path(prim_path)
            if obj_info is not None:
                name, _ = obj_info
                if name in states["objects"]:
                    visible_objects.append(name)
    visible_objects = list(set(visible_objects))
    print(f"  Objects      : {visible_objects}")

    # ── 3. Pose table ────────────────────────────────────
    print()
    plot_pose_table(latents, visible_objects, save_path=str(save_dir / "pose_table.txt"))

    # ── 4. RGB + mask overlay ────────────────────────────
    image = conditionals["image"].numpy()  # [H, W, 3]
    if "object_masks" in conditionals:
        masks = conditionals["object_masks"].numpy()  # [N, H, W]
        plot_rgb_with_masks(image, masks, visible_objects, save_path=str(save_dir / "rgb_masks_overlay.png"))
        plot_mask_gallery(masks, visible_objects, save_path=str(save_dir / "mask_gallery.png"))
    else:
        # No masks — just save the raw RGB
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.set_title(f"Scene: {scene_id}")
        ax.axis("off")
        fig.savefig(str(save_dir / "rgb.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved RGB      -> {save_dir / 'rgb.png'}")

    # ── 5. Depth map ─────────────────────────────────────
    if "depth" in conditionals:
        depth = conditionals["depth"].numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(depth, cmap="turbo")
        ax.set_title("Depth Map")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(str(save_dir / "depth.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved depth    -> {save_dir / 'depth.png'}")

    # ── 6. Camera-view mesh reprojection ────────────────
    if "camera_K" in conditionals and not args.no_3d:
        camera_K = conditionals["camera_K"].numpy()
        print("\nRendering camera-view mesh reprojection ...")
        render_camera_view(
            visible_objects, latents, gso_root, camera_K, image,
            save_path=str(save_dir / "camera_view_overlay.png"),
        )

    # ── 7. 3D scene ──────────────────────────────────────
    if not args.no_3d:
        print("\nBuilding 3D scene with GSO meshes ...")
        scene_3d = build_3d_scene(visible_objects, latents, gso_root)

        # Save as GLB for interactive viewing
        glb_path = save_dir / "scene.glb"
        scene_3d.export(str(glb_path))
        print(f"Saved GLB      -> {glb_path}  (open in 3D viewer / meshlab / blender)")

        # Matplotlib 3D scatter fallback
        render_3d_topdown(scene_3d, save_path=str(save_dir / "scene_3d_scatter.png"))

    # ── 8. Summary JSON ──────────────────────────────────
    summary = {
        "scene_id": scene_id,
        "num_objects": num_objects,
        "objects": {},
    }
    for i, name in enumerate(visible_objects):
        summary["objects"][name] = {
            "translation": latents["translation"][i].numpy().tolist(),
            "scale": latents["scale"][i].numpy().tolist(),
            "rotation_6d": latents["6drotation_normalized"][i].numpy().tolist(),
        }
    with open(save_dir / "scene_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary  -> {save_dir / 'scene_summary.json'}")

    print(f"\nDone! All outputs in: {save_dir}/")


if __name__ == "__main__":
    main()
