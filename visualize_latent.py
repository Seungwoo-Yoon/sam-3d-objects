"""
Visualize one FoundationPose shape latent against its GSO mesh.

The script loads a sample from FoundationPoseDataset, takes one object's
canonical GSO mesh and shape latent, decodes the shape latent with ss_decoder,
and writes an interactive Plotly HTML overlay:

    GT mesh              -> translucent mesh
    decoded shape latent -> point cloud from occupied sparse-structure voxels

Example:
    python visualize_latent.py \
        --data-root foundationpose_test \
        --gso-root gso/google_scanned_objects/models_normalized \
        --scene-idx 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
from hydra.utils import instantiate
from omegaconf import OmegaConf

from foundation_pose_dataset import FoundationPoseDataset


logger = logging.getLogger("visualize_latent")


def load_ss_decoder(config_path: str | Path, ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load ss_decoder the same way as inference_pipeline.py does."""
    cfg = OmegaConf.load(config_path)
    if "pretrained_ckpt_path" in cfg:
        del cfg["pretrained_ckpt_path"]

    model = instantiate(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def visible_object_names(dataset: FoundationPoseDataset, scene_idx: int) -> list[str]:
    """Mirror FoundationPoseDataset.__getitem__ enough to recover object names."""
    render_info = dataset.scenes[scene_idx]
    states = dataset._load_states(render_info["scene_parent_dir"])
    frame_data = dataset._load_frame_data(render_info["render_dir"], frame_id=0)

    names = []
    if "id_mapping" in frame_data:
        for prim_path in frame_data["id_mapping"].values():
            obj_info = dataset._extract_object_name_from_path(prim_path)
            if obj_info is None:
                continue
            obj_name, _ = obj_info
            if obj_name in states["objects"]:
                names.append(obj_name)

    # Keep the dataset's current behavior so indices line up with sample tensors.
    return list(set(names))


def choose_object_index(sample: dict[str, Any], requested_idx: int | None) -> int:
    num_objects = int(sample["num_objects"])
    if num_objects <= 0:
        raise ValueError("Sample has no visible objects.")

    if requested_idx is not None:
        if requested_idx < 0 or requested_idx >= num_objects:
            raise IndexError(f"--object-idx {requested_idx} out of range 0..{num_objects - 1}")
        return requested_idx

    mesh_available = sample.get("mesh_data", {}).get("mesh_available")
    if mesh_available is not None:
        available = torch.where(mesh_available)[0]
        if available.numel() > 0:
            return int(available[0].item())

    return 0


def shape_latent_to_cube(shape_latent: torch.Tensor) -> torch.Tensor:
    """
    Normalize one object's shape latent to [1, 8, 16, 16, 16].

    FoundationPoseDataset normally returns [8, 16, 16, 16], while generated
    latents in training/inference are often [4096, 8].
    """
    if shape_latent.ndim == 4:
        if shape_latent.shape[0] != 8:
            raise ValueError(f"Expected [8, 16, 16, 16], got {tuple(shape_latent.shape)}")
        return shape_latent.unsqueeze(0)

    if shape_latent.ndim == 2:
        if shape_latent.shape[-1] != 8:
            raise ValueError(f"Expected flat latent [L, 8], got {tuple(shape_latent.shape)}")
        k = round(shape_latent.shape[0] ** (1.0 / 3.0))
        if k ** 3 != shape_latent.shape[0]:
            raise ValueError(f"Flat latent length is not cubic: {shape_latent.shape[0]}")
        return shape_latent.view(k, k, k, 8).permute(3, 0, 1, 2).unsqueeze(0)

    if shape_latent.ndim == 5:
        if shape_latent.shape[0] != 1:
            raise ValueError(f"Expected one-object batch, got {tuple(shape_latent.shape)}")
        return shape_latent

    raise ValueError(f"Unsupported shape latent shape: {tuple(shape_latent.shape)}")


@torch.no_grad()
def decode_shape_points(
    ss_decoder: torch.nn.Module,
    shape_latent: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Decode shape latent into occupied voxel centers in canonical [-0.5, 0.5] space."""
    latent_cube = shape_latent_to_cube(shape_latent).float().to(device)
    ss_logits = ss_decoder(latent_cube)

    coords = torch.argwhere(ss_logits > threshold)
    if coords.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32), ss_logits

    if coords.shape[1] == 5:
        coords = coords[:, [0, 2, 3, 4]]
    elif coords.shape[1] != 4:
        raise ValueError(f"Unexpected ss_decoder output rank: coords shape {tuple(coords.shape)}")

    spatial = coords[:, 1:].float().cpu().numpy()
    resolution = np.array(ss_logits.shape[-3:], dtype=np.float32)
    points = (spatial + 0.5) / resolution - 0.5
    return points.astype(np.float32), ss_logits


def mesh_from_sample(sample: dict[str, Any], object_idx: int) -> trimesh.Trimesh:
    mesh_data = sample.get("mesh_data")
    if mesh_data is None:
        raise ValueError("Sample has no mesh_data. Did you instantiate the dataset with load_meshes=True?")

    verts = mesh_data["mesh_verts"][object_idx]
    faces = mesh_data["mesh_faces"][object_idx]
    if verts is None or faces is None:
        raise ValueError(f"Object index {object_idx} does not have an available GSO mesh.")

    return trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )


def simplify_for_plot(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh

    try:
        return mesh.simplify_quadric_decimation(max_faces)
    except Exception as exc:
        logger.warning("Mesh simplification failed (%s); plotting a deterministic face subset.", exc)
        face_idx = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=np.int64)
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[face_idx], process=False)


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def write_overlay_html(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    output_path: Path,
    title: str,
    point_size: int,
) -> None:
    fig = go.Figure()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="GSO mesh",
            color="rgb(0, 120, 255)",
            opacity=0.38,
            flatshading=True,
        )
    )

    if points.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name="decoded shape latent",
                marker={
                    "size": point_size,
                    "color": "rgb(220, 64, 56)",
                    "opacity": 0.9,
                },
            )
        )

    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def export_debug_assets(mesh: trimesh.Trimesh, points: np.ndarray, output_dir: Path, stem: str) -> None:
    mesh.export(output_dir / f"{stem}_mesh.ply")
    if points.shape[0] > 0:
        trimesh.PointCloud(points).export(output_dir / f"{stem}_decoded_points.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a FoundationPose shape latent decoded by ss_decoder.")
    parser.add_argument("--data-root", type=str, default="foundationpose_test")
    parser.add_argument("--gso-root", type=str, default="gso/google_scanned_objects/models_normalized")
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--object-idx", type=int, default=None, help="Object index in the scene. Defaults to first object with mesh.")
    parser.add_argument("--ss-decoder-config", type=str, default="checkpoints/hf/ss_decoder.yaml")
    parser.add_argument("--ss-decoder-checkpoint", type=str, default="checkpoints/hf/ss_decoder.ckpt")
    parser.add_argument("--output-dir", type=str, default="outputs/latent_vis")
    parser.add_argument("--threshold", type=float, default=0.0, help="Occupancy logit threshold; existing code uses 0.")
    parser.add_argument("--max-points", type=int, default=50000, help="Max decoded points to show in Plotly; <=0 keeps all.")
    parser.add_argument("--max-faces", type=int, default=50000, help="Max mesh faces to show in Plotly; <=0 keeps all.")
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--no-export-assets", action="store_true", help="Only write HTML; skip mesh/point PLY exports.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=9999,
        load_images=False,
        load_depth=False,
        load_masks=True,
        precomputed_latents=False,
        num_renders_per_scene=1,
        gso_root=args.gso_root,
        load_meshes=True,
        mesh_num_samples=2048,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No renders found under {args.data_root}")
    if args.scene_idx < 0 or args.scene_idx >= len(dataset):
        raise IndexError(f"--scene-idx {args.scene_idx} out of range 0..{len(dataset) - 1}")

    sample = dataset[args.scene_idx]
    object_names = visible_object_names(dataset, args.scene_idx)
    object_idx = choose_object_index(sample, args.object_idx)
    object_name = object_names[object_idx] if object_idx < len(object_names) else f"object_{object_idx}"

    logger.info("Scene: %s", sample["scene_id"])
    logger.info("Object %d/%d: %s", object_idx, sample["num_objects"] - 1, object_name)
    logger.info("Loading ss_decoder on %s", device)

    ss_decoder = load_ss_decoder(args.ss_decoder_config, args.ss_decoder_checkpoint, device)

    shape_latent = sample["latents"]["shape"][object_idx]
    if torch.count_nonzero(shape_latent).item() == 0:
        logger.warning("Selected shape latent is all zeros; check gso_root/latent_codes path.")

    points, ss_logits = decode_shape_points(
        ss_decoder=ss_decoder,
        shape_latent=shape_latent,
        threshold=args.threshold,
        device=device,
    )
    logger.info("ss_decoder output shape: %s", tuple(ss_logits.shape))
    logger.info("Decoded occupied points: %d", points.shape[0])

    mesh = mesh_from_sample(sample, object_idx)
    plot_mesh = simplify_for_plot(mesh.copy(), args.max_faces)
    plot_points = subsample_points(points, args.max_points)

    safe_scene = str(sample["scene_id"]).replace("/", "_")
    safe_name = object_name.replace("/", "_")
    stem = f"scene{args.scene_idx:04d}_obj{object_idx:02d}_{safe_name}"
    html_path = output_dir / f"{stem}.html"

    write_overlay_html(
        mesh=plot_mesh,
        points=plot_points,
        output_path=html_path,
        title=f"{safe_scene} / {object_name}",
        point_size=args.point_size,
    )

    if not args.no_export_assets:
        export_debug_assets(mesh, points, output_dir, stem)

    logger.info("Wrote overlay: %s", html_path)


if __name__ == "__main__":
    main()
