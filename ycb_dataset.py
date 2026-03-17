"""
YCB-Video (YCB-V) Dataset — SAM3D compatible

Returns the same latents/conditionals structure as FoundationPoseDataset.

BOP format directory layout:
  {scene}/
    rgb/              {frame_id:06d}.png
    depth/            {frame_id:06d}.png  (uint16, multiply by depth_scale → mm)
    mask/             {frame_id:06d}_{obj_idx:06d}.png
    mask_visib/       {frame_id:06d}_{obj_idx:06d}.png
    scene_camera.json per-frame: cam_K, cam_R_w2c, cam_t_w2c (mm), depth_scale
    scene_gt.json     per-frame list: obj_id, cam_R_m2c, cam_t_m2c (mm)
    scene_gt_info.json per-frame list: bbox_obj, bbox_visib, visib_fract

YCB-models directory layout:
  obj_{obj_id:06d}.ply   one mesh per object class (vertices in mm)
  models_info.json       diameter, bbox extents, symmetries per object

Coordinate convention
---------------------
BOP/OpenCV camera space : X right, Y down,  Z forward
PyTorch3D camera space  : X left,  Y up,    Z forward
Conversion (same as FoundationPoseDataset pointmap flip): M = diag([-1, -1, 1])
  t_p3d     = M @ t_bop
  R_p3d     = M @ R_bop @ M          (M is its own inverse)
Mesh vertices are kept in model space (BOP convention); the same M is applied
when the caller needs them in PyTorch3D convention.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

logger = logging.getLogger(__name__)

# BOP (OpenCV) → PyTorch3D: flip X and Y
_BOP_TO_P3D = np.diag([-1.0, -1.0, 1.0])


def _rotation_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R concatenated (Zhou et al. 6D rotation repr)."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _bop_pose_to_latents(
    R_m2c: np.ndarray,  # [3, 3] float32, BOP convention
    t_m2c: np.ndarray,  # [3]    float32, mm, BOP convention
) -> Dict[str, torch.Tensor]:
    """
    Convert BOP model-to-camera pose to SAM3D latent format.

    Converts from BOP camera (X-right, Y-down, Z-forward) to
    PyTorch3D camera (X-left, Y-up, Z-forward) by applying M = diag([-1,-1,1]).

    Returns:
        translation:            [3]  (mm, PyTorch3D convention)
        6drotation_normalized:  [6]  (Zhou et al., PyTorch3D convention)
        scale:                  [3]  (ones — YCB-V has no per-object scale)
    """
    M = _BOP_TO_P3D

    t_p3d = (M @ t_m2c).astype(np.float32)              # [3]
    R_p3d = (M @ R_m2c @ M).astype(np.float32)          # [3, 3]
    rot6d = _rotation_to_6d(R_p3d)                       # [6]
    scale = np.ones(3, dtype=np.float32)

    return {
        'translation':           torch.from_numpy(t_p3d),
        '6drotation_normalized': torch.from_numpy(rot6d),
        'scale':                 torch.from_numpy(scale),
    }


class YCBVDataset(Dataset):
    """
    YCB-Video dataset with SAM3D-compatible output format.

    Each sample is one frame from one scene.

    Returns
    -------
    latents : dict
        translation            [num_objects, 3]   mm, PyTorch3D convention
        6drotation_normalized  [num_objects, 6]   Zhou et al., PyTorch3D
        scale                  [num_objects, 3]   ones
        shape                  [num_objects, 8, 16, 16, 16]  zeros (no latent)

    conditionals : dict
        camera_K               [3, 3]
        camera_view_transform  [4, 4]  world-to-camera (standard row-major)
        image                  [H, W, 3]  float32 in [0, 1]
        depth                  [H, W]     float32, mm
        pointmap               [H, W, 3]  backprojected, PyTorch3D convention
        object_masks           [num_objects, H, W]  visible masks

    mesh_data : dict  (only when load_meshes=True)
        mesh_points    [num_objects, num_samples, 3]  surface point cloud, mm, model space
        mesh_available [num_objects]                  bool — False if .ply not found
        mesh_bbox_min  [num_objects, 3]               mm, model space
        mesh_bbox_max  [num_objects, 3]               mm, model space
        voxels         [num_objects, 64, 64, 64]      uint8 occupancy grid

    scene_id    : str
    frame_id    : int
    num_objects : int
    obj_ids     : [num_objects]  int64  (YCB object class IDs)
    """

    def __init__(
        self,
        data_root: str,
        scene_ids: Optional[List[Union[int, str]]] = None,
        load_depth: bool = True,
        load_masks: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        min_visib_fract: float = 0.0,
        ycb_models_root: Optional[str] = None,
        load_meshes: bool = False,
        mesh_num_samples: int = 2048,
    ):
        """
        Args:
            data_root:        Root directory (contains 000048/, 000049/, …)
            scene_ids:        Scene IDs to load (int or zero-padded string).
                              None → load all scenes found.
            load_depth:       Load depth images and compute pointmap.
            load_masks:       Load per-object visibility masks.
            image_size:       Resize target (H, W). None keeps original 480×640.
            min_visib_fract:  Skip objects below this visibility fraction.
            ycb_models_root:  Path to YCB-models directory (contains obj_000001.ply, …).
                              Required when load_meshes=True.
            load_meshes:      Load per-object PLY meshes and sample point clouds.
            mesh_num_samples: Number of surface points to sample per mesh.
        """
        self.data_root = Path(data_root)
        self.load_depth = load_depth
        self.load_masks = load_masks
        self.image_size = image_size
        self.min_visib_fract = min_visib_fract
        self.ycb_models_root = Path(ycb_models_root) if ycb_models_root else None
        self.load_meshes = load_meshes and TRIMESH_AVAILABLE and ycb_models_root is not None
        self.mesh_num_samples = mesh_num_samples

        if load_meshes and not TRIMESH_AVAILABLE:
            logger.warning("load_meshes=True but trimesh is not installed. Mesh loading disabled.")
        if load_meshes and ycb_models_root is None:
            logger.warning("load_meshes=True but ycb_models_root not provided. Mesh loading disabled.")

        # Load models_info.json (bbox / diameter metadata) — always, when root is given
        self._models_info: Dict[int, Dict] = {}
        if self.ycb_models_root is not None:
            info_path = self.ycb_models_root / "models_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    raw = json.load(f)
                self._models_info = {int(k): v for k, v in raw.items()}

        # Mesh cache: obj_id → trimesh.Trimesh (loaded lazily)
        self._mesh_cache: Dict[int, Any] = {}

        if scene_ids is None:
            scene_dirs = sorted(
                d for d in self.data_root.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        else:
            scene_dirs = []
            for sid in scene_ids:
                d = self.data_root / f"{int(sid):06d}"
                if d.exists():
                    scene_dirs.append(d)
                else:
                    logger.warning(f"Scene directory not found: {d}")

        self.samples: List[Dict[str, Any]] = []
        for scene_dir in scene_dirs:
            self._index_scene(scene_dir)

        logger.info(
            f"YCBVDataset: {len(self.samples)} frames "
            f"across {len(scene_dirs)} scene(s)"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_scene(self, scene_dir: Path) -> None:
        camera_file = scene_dir / "scene_camera.json"
        gt_file = scene_dir / "scene_gt.json"
        gt_info_file = scene_dir / "scene_gt_info.json"

        if not camera_file.exists() or not gt_file.exists():
            logger.warning(f"Missing JSON in {scene_dir}, skipping.")
            return

        with open(camera_file) as f:
            scene_camera = json.load(f)
        with open(gt_file) as f:
            scene_gt = json.load(f)
        gt_info = {}
        if gt_info_file.exists():
            with open(gt_info_file) as f:
                gt_info = json.load(f)

        for frame_key, cam_data in scene_camera.items():
            frame_id = int(frame_key)
            rgb_path = scene_dir / "rgb" / f"{frame_id:06d}.png"
            if not rgb_path.exists():
                continue
            self.samples.append({
                "scene_dir": scene_dir,
                "scene_id":  scene_dir.name,
                "frame_id":  frame_id,
                "camera":    cam_data,
                "gt":        scene_gt.get(frame_key, []),
                "gt_info":   gt_info.get(frame_key, []),
            })

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_camera(self, cam_data: Dict) -> Dict[str, Any]:
        K = np.array(cam_data["cam_K"], dtype=np.float32).reshape(3, 3)
        R_w2c = np.array(cam_data["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
        t_w2c = np.array(cam_data["cam_t_w2c"], dtype=np.float32)
        depth_scale = float(cam_data.get("depth_scale", 1.0))

        # Compose 4×4 world-to-camera matrix (standard row-major form)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R_w2c
        view[:3,  3] = t_w2c

        return {
            "camera_K":              K,
            "camera_view_transform": view,
            "depth_scale":           depth_scale,
        }

    def _filter_objects(
        self,
        gt_list: List[Dict],
        gt_info_list: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Return per-object dicts filtered by min_visib_fract."""
        objects = []
        for i, obj_gt in enumerate(gt_list):
            info = gt_info_list[i] if i < len(gt_info_list) else {}
            visib_fract = float(info.get("visib_fract", 1.0))
            if visib_fract < self.min_visib_fract:
                continue
            objects.append({
                "obj_id":      int(obj_gt["obj_id"]),
                "obj_idx":     i,
                "cam_R_m2c":   np.array(obj_gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                "cam_t_m2c":   np.array(obj_gt["cam_t_m2c"], dtype=np.float32),
                "visib_fract": visib_fract,
            })
        return objects

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_depth_mm(self, path: Path, depth_scale: float) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Depth not found: {path}")
        depth = raw.astype(np.float32) * depth_scale          # → mm
        if self.image_size is not None:
            depth = cv2.resize(
                depth, (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return depth

    def _load_mask(self, path: Path, H: int, W: int) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return np.zeros((H, W), dtype=np.float32)
        mask = (raw > 0).astype(np.float32)
        if self.image_size is not None:
            mask = cv2.resize(
                mask, (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask

    def _get_mesh(self, obj_id: int) -> Optional[Any]:
        """Load and cache trimesh for a given YCB obj_id (1-indexed)."""
        if obj_id in self._mesh_cache:
            return self._mesh_cache[obj_id]
        ply_path = self.ycb_models_root / f"obj_{obj_id:06d}.ply"
        mesh = None
        if ply_path.exists():
            try:
                mesh = trimesh.load(str(ply_path), force='mesh')
            except Exception as e:
                logger.warning(f"Failed to load mesh {ply_path}: {e}")
        else:
            logger.warning(f"Mesh not found: {ply_path}")
        self._mesh_cache[obj_id] = mesh
        return mesh

    def _load_mesh_data(self, objects: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Load mesh data for all objects in the current frame.

        Point clouds are in model space (mm, BOP convention).
        The voxel grid normalises vertices into a [-1, 1] cube before voxelising,
        matching the convention used in FoundationPoseDataset.

        Returns a dict matching FoundationPoseDataset's mesh_data:
            mesh_points    [N, mesh_num_samples, 3]
            mesh_available [N]  bool
            mesh_bbox_min  [N, 3]
            mesh_bbox_max  [N, 3]
            voxels         [N, 64, 64, 64]  uint8
        """
        mesh_points_list, mesh_available_list = [], []
        bbox_min_list, bbox_max_list, voxel_list = [], [], []

        for obj in objects:
            obj_id = obj["obj_id"]
            mesh = self._get_mesh(obj_id)

            if mesh is not None:
                # Sample surface points
                pts, _ = trimesh.sample.sample_surface(mesh, self.mesh_num_samples)
                pts = pts.astype(np.float32)
                mesh_points_list.append(torch.from_numpy(pts))
                mesh_available_list.append(True)

                # Bounding box from vertices
                verts = mesh.vertices.astype(np.float32)
                bbox_min_list.append(torch.from_numpy(verts.min(axis=0)))
                bbox_max_list.append(torch.from_numpy(verts.max(axis=0)))

                # Voxelise: normalise vertices to [-1, 1] then fill 64³ grid
                R = 64
                v_min = verts.min(axis=0, keepdims=True)
                v_max = verts.max(axis=0, keepdims=True)
                v_range = (v_max - v_min).clip(min=1e-6)
                pts_norm = (pts - v_min) / v_range * 2.0 - 1.0   # in [-1, 1]
                pts_norm = np.clip(pts_norm, -1.0, 1.0)
                ijk = np.floor((pts_norm + 1.0) * 0.5 * R).astype(np.int64)
                ijk = np.clip(ijk, 0, R - 1)
                voxel = np.zeros((R, R, R), dtype=np.uint8)
                voxel[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
                voxel_list.append(torch.from_numpy(voxel))
            else:
                # Fallback: use models_info bbox if available, otherwise zeros
                info = self._models_info.get(obj_id, {})
                mesh_points_list.append(torch.zeros(self.mesh_num_samples, 3))
                mesh_available_list.append(False)
                if info:
                    bmin = np.array([info['min_x'], info['min_y'], info['min_z']], dtype=np.float32)
                    bmax = bmin + np.array([info['size_x'], info['size_y'], info['size_z']], dtype=np.float32)
                else:
                    bmin = bmax = np.zeros(3, dtype=np.float32)
                bbox_min_list.append(torch.from_numpy(bmin))
                bbox_max_list.append(torch.from_numpy(bmax))
                voxel_list.append(torch.zeros(64, 64, 64, dtype=torch.uint8))

        return {
            'mesh_points':    torch.stack(mesh_points_list),   # [N, S, 3]
            'mesh_available': torch.tensor(mesh_available_list, dtype=torch.bool),
            'mesh_bbox_min':  torch.stack(bbox_min_list),      # [N, 3]
            'mesh_bbox_max':  torch.stack(bbox_max_list),      # [N, 3]
            'voxels':         torch.stack(voxel_list),         # [N, 64, 64, 64]
        }

    def _backproject_depth(
        self,
        depth: np.ndarray,  # [H, W] mm
        K: np.ndarray,      # [3, 3]
    ) -> np.ndarray:
        """
        Back-project depth into 3-D pointmap in PyTorch3D camera convention.

        BOP depth → standard camera (X-right, Y-down, Z-forward) → flip X,Y
        → PyTorch3D (X-left, Y-up, Z-forward).
        """
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pointmap = np.stack([x, y, z], axis=-1).astype(np.float32)  # [H, W, 3]

        # BOP camera (X-right, Y-down) → PyTorch3D (X-left, Y-up)
        pointmap[..., 0] *= -1   # flip X
        pointmap[..., 1] *= -1   # flip Y
        return pointmap

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        scene_dir: Path = meta["scene_dir"]
        frame_id:  int  = meta["frame_id"]
        scene_id:  str  = meta["scene_id"]

        cam = self._parse_camera(meta["camera"])
        depth_scale = cam.pop("depth_scale")
        K: np.ndarray = cam["camera_K"]

        objects = self._filter_objects(meta["gt"], meta["gt_info"])
        num_objects = len(objects)

        # ── RGB ──────────────────────────────────────────────────────────
        image = self._load_image(scene_dir / "rgb" / f"{frame_id:06d}.png")
        H, W = image.shape[:2]

        # ── Latents (per-object pose) ─────────────────────────────────────
        if num_objects > 0:
            latent_list = [
                _bop_pose_to_latents(o["cam_R_m2c"], o["cam_t_m2c"])
                for o in objects
            ]
            latents = {
                key: torch.stack([l[key] for l in latent_list], dim=0)
                for key in latent_list[0]
            }
        else:
            latents = {
                'translation':           torch.zeros(0, 3),
                '6drotation_normalized': torch.zeros(0, 6),
                'scale':                 torch.zeros(0, 3),
            }
        # Shape latents: zeros (no precomputed latents for YCB-V)
        latents['shape'] = torch.zeros(num_objects, 8, 16, 16, 16)

        # ── Conditionals ─────────────────────────────────────────────────
        conditionals: Dict[str, Any] = {
            'camera_K':              torch.from_numpy(cam["camera_K"]),
            'camera_view_transform': torch.from_numpy(cam["camera_view_transform"]),
            'image':                 torch.from_numpy(image),   # [H, W, 3]
        }

        if self.load_depth:
            depth_path = scene_dir / "depth" / f"{frame_id:06d}.png"
            if depth_path.exists():
                depth = self._load_depth_mm(depth_path, depth_scale)
                conditionals['depth'] = torch.from_numpy(depth)  # [H, W]

                pointmap = self._backproject_depth(depth, K)
                conditionals['pointmap'] = torch.from_numpy(pointmap)  # [H, W, 3]

        if self.load_masks and num_objects > 0:
            masks = np.stack([
                self._load_mask(
                    scene_dir / "mask_visib" / f"{frame_id:06d}_{o['obj_idx']:06d}.png",
                    H, W,
                )
                for o in objects
            ], axis=0)
            conditionals['object_masks'] = torch.from_numpy(masks)  # [num_objects, H, W]
        elif self.load_masks:
            conditionals['object_masks'] = torch.zeros(0, H, W)

        result = {
            'latents':      latents,
            'conditionals': conditionals,
            'scene_id':     scene_id,
            'frame_id':     frame_id,
            'num_objects':  num_objects,
            'obj_ids':      torch.tensor(
                [o['obj_id'] for o in objects], dtype=torch.long
            ),
        }

        if self.load_meshes and num_objects > 0:
            result['mesh_data'] = self._load_mesh_data(objects)

        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """Pass-through collate for batch_size=1 (scene-level training)."""
    assert len(batch) == 1, "YCBVDataset requires batch_size=1"
    return batch[0]


# ── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="YCB-V")
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--load-meshes", action="store_true")
    args = parser.parse_args()

    dataset = YCBVDataset(
        data_root=args.data_root,
        scene_ids=args.scenes,
        load_depth=True,
        load_masks=True,
        ycb_models_root=args.models_root,
        load_meshes=args.load_meshes,
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"scene_id={sample['scene_id']}  frame_id={sample['frame_id']}  "
              f"num_objects={sample['num_objects']}")

        print("\nlatents:")
        for k, v in sample['latents'].items():
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

        print("\nconditionals:")
        for k, v in sample['conditionals'].items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")

        plt.imsave("ycbv_rgb.png", sample['conditionals']['image'].numpy())
        if 'depth' in sample['conditionals']:
            d = sample['conditionals']['depth'].numpy()
            valid = d > 0
            d_vis = np.zeros_like(d)
            d_vis[valid] = (d[valid] - d[valid].min()) / (d[valid].max() - d[valid].min() + 1e-8)
            plt.imsave("ycbv_depth.png", d_vis, cmap="plasma")
        print("Saved ycbv_rgb.png / ycbv_depth.png")

        if 'mesh_data' in sample:
            print("\nmesh_data:")
            for k, v in sample['mesh_data'].items():
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            n_avail = sample['mesh_data']['mesh_available'].sum().item()
            print(f"  meshes loaded: {n_avail}/{sample['num_objects']}")
