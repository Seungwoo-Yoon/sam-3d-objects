"""
FoundationPose Dataset for Scene-Level Training

This dataset loads FoundationPose synthetic training data where each scene contains
multiple objects rendered from Objaverse/GSO. Compatible with the train_shortcut_scene.py script.

FoundationPose Training Dataset Structure:
- Top-level directories are object IDs (e.g., 3874429731/)
- Each object directory contains:
  - scene_XXXXXXXX/: Intermediate scene directory
    - states.json: Object poses and transformations for all objects in scene
    - scene-xxxxx-xxxx/: Actual scene directory with renders
      - RenderProduct_Replicator/: Default camera view
        - rgb/: RGB images
        - instance_segmentation/: Segmentation masks and mappings
        - distance_to_image_plane/: Depth maps (numpy)
        - camera_params/: Camera parameters (JSON)
        - bounding_box_2d_loose/: 2D bounding boxes
      - RenderProduct_Replicator_01/: Additional camera views (optional)

GSO Mesh Loading:
To load GSO mesh data, provide the gso_root parameter pointing to the GSO dataset directory:

    dataset = FoundationPoseDataset(
        data_root='/path/to/foundationpose',
        gso_root='/path/to/google_scanned_objects',
        load_meshes=True,
        mesh_num_samples=2048,
    )

The dataset will automatically load mesh point clouds for GSO objects. Expected GSO structure:
    google_scanned_objects/
      ├── object_name_1/
      │   ├── meshes/
      │   │   └── model.obj (or model.ply)
      ├── object_name_2/
      │   ├── meshes/
      │   │   └── model.obj
      ...

Each sample will include a 'mesh_data' dict with:
- 'mesh_points': [num_objects, num_samples, 3] - sampled point clouds from mesh surfaces
- 'mesh_available': [num_objects] - boolean mask indicating which objects have meshes
- 'mesh_bbox_min': [num_objects, 3] - minimum x, y, z bounds of each mesh
- 'mesh_bbox_max': [num_objects, 3] - maximum x, y, z bounds of each mesh
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logging.warning("trimesh not available. Mesh loading will be disabled.")

from sam3d_objects.data.utils import tree_tensor_map


logger = logging.getLogger(__name__)



class FoundationPoseDataset(Dataset):
    """
    Dataset that loads FoundationPose synthetic training data for scene-level training.

    Each sample contains all objects from a rendered scene as a single batch:
    - latents: Dict of tensors with shape [num_objects_in_scene, ...]
    - conditionals: Conditioning information (images, masks, depth, camera_K, etc.)
    - scene_id: Identifier for the scene
    - num_objects: Number of objects in the scene
    """

    def __init__(
        self,
        data_root: str,
        max_objects_per_scene: int = 32,
        load_images: bool = True,
        load_depth: bool = True,
        load_masks: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        vae_encoder: Optional[Any] = None,
        precomputed_latents: bool = False,
        num_renders_per_scene: int = 1,
        gso_root: Optional[str] = None,
        load_meshes: bool = False,
        mesh_num_samples: int = 2048,
    ):
        """
        Args:
            data_root: Root directory containing FoundationPose data
            max_objects_per_scene: Maximum number of objects per scene
            load_images: Whether to load RGB images
            load_depth: Whether to load depth maps
            load_masks: Whether to load object masks
            image_size: Target image size (H, W) for resizing, None for original
            vae_encoder: Optional VAE encoder for encoding objects online
            precomputed_latents: If True, load pre-computed latents from disk
            num_renders_per_scene: Number of render products to use per scene (default 1)
            gso_root: Root directory containing GSO mesh files (e.g., /path/to/google_scanned_objects)
            load_meshes: Whether to load 3D mesh data for GSO objects
            mesh_num_samples: Number of points to sample from mesh surface (default 2048)
        """
        self.data_root = Path(data_root)
        self.max_objects_per_scene = max_objects_per_scene
        self.load_images = load_images
        self.load_depth = load_depth
        self.load_masks = load_masks
        self.image_size = image_size
        self.vae_encoder = vae_encoder
        self.precomputed_latents = precomputed_latents
        self.num_renders_per_scene = num_renders_per_scene
        self.gso_root = Path(gso_root) if gso_root is not None else None
        self.load_meshes = load_meshes
        self.mesh_num_samples = mesh_num_samples

        if self.load_meshes and not TRIMESH_AVAILABLE:
            logger.warning("load_meshes=True but trimesh is not available. Mesh loading disabled.")
            self.load_meshes = False

        if self.load_meshes and self.gso_root is None:
            logger.warning("load_meshes=True but gso_root is not provided. Mesh loading disabled.")
            self.load_meshes = False

        # Load scene metadata
        self.scenes = self._load_scenes()

        logger.info(f"Loaded {len(self.scenes)} renders across all scenes")

    def _load_scenes(self) -> List[Dict[str, Any]]:
        """
        Load list of scenes and their render products.

        Returns:
            List of render dictionaries containing:
            - object_dir: Path to top-level numbered directory
            - scene_parent_dir: Path to scene_XXXXXXXX directory (contains states.json)
            - scene_dir: Path to scene-XXX-XXX directory (contains render products)
            - render_dir: Path to specific render product
            - scene_id: Scene identifier
        """
        renders = []

        # Find all top-level object directories (numbered directories)
        object_dirs = sorted([d for d in self.data_root.iterdir()
                            if d.is_dir() and d.name.isdigit()])

        for object_dir in object_dirs:
            # Find intermediate scene directories (scene_00000000, scene_00000001, etc.)
            scene_parent_dirs = sorted([d for d in object_dir.iterdir()
                                       if d.is_dir() and d.name.startswith('scene_')])

            for scene_parent_dir in scene_parent_dirs:
                # Check if states.json exists at this level
                states_file = scene_parent_dir / 'states.json'
                if not states_file.exists():
                    continue

                # Find actual scene directories (scene-XXX-XXX)
                scene_dirs = sorted([d for d in scene_parent_dir.iterdir()
                                   if d.is_dir() and d.name.startswith('scene-')])

                for scene_dir in scene_dirs:
                    # Find render products (RenderProduct_Replicator, RenderProduct_Replicator_01, etc.)
                    render_dirs = sorted([d for d in scene_dir.iterdir()
                                        if d.is_dir() and d.name.startswith('RenderProduct_')])

                    # Limit number of renders per scene
                    render_dirs = render_dirs[:self.num_renders_per_scene]

                    for render_dir in render_dirs:
                        # Check if required data exists
                        rgb_dir = render_dir / 'rgb'
                        camera_params_dir = render_dir / 'camera_params'

                        if not rgb_dir.exists() or not camera_params_dir.exists():
                            continue

                        scene_id = f"{object_dir.name}_{scene_parent_dir.name}_{scene_dir.name}_{render_dir.name}"

                        renders.append({
                            'object_dir': object_dir,
                            'scene_parent_dir': scene_parent_dir,
                            'scene_dir': scene_dir,
                            'render_dir': render_dir,
                            'scene_id': scene_id,
                        })

        return renders

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_states(self, scene_parent_dir: Path) -> Dict[str, Any]:
        """
        Load states.json containing object poses and transformations.

        Args:
            scene_parent_dir: Path to scene_XXXXXXXX directory containing states.json

        Returns:
            Dictionary with object information keyed by object UID.
        """
        states_file = scene_parent_dir / 'states.json'
        with open(states_file, 'r') as f:
            states = json.load(f)
        return states

    def _load_camera_params(self, render_dir: Path, frame_id: int = 0) -> Dict[str, Any]:
        """
        Load camera parameters for a specific frame.

        Returns:
            Dictionary with camera intrinsics and parameters.
        """
        camera_file = render_dir / 'camera_params' / f'camera_params_{frame_id:06d}.json'

        with open(camera_file, 'r') as f:
            camera_params = json.load(f)

        # Parse camera intrinsic matrix from projection parameters
        # FoundationPose uses: focal length, aperture, and resolution
        focal_length = camera_params['cameraFocalLength']
        aperture = camera_params['cameraAperture']  # [width, height] in mm
        resolution = camera_params['renderProductResolution']  # [width, height] in pixels

        # Calculate intrinsic matrix
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = focal_length * resolution[0] / aperture[0]
        fy = focal_length * resolution[1] / aperture[1]
        cx = resolution[0] / 2.0
        cy = resolution[1] / 2.0

        camera_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return {
            'camera_K': camera_K,
            'camera_view_transform': np.array(camera_params['cameraViewTransform']).reshape(4, 4),
            'resolution': resolution,
        }

    def _load_instance_segmentation(
        self,
        render_dir: Path,
        frame_id: int = 0
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Load instance segmentation mask and mapping.

        Returns:
            Tuple of (segmentation_mask, id_to_path_mapping)
        """
        seg_dir = render_dir / 'instance_segmentation'

        # Load segmentation mask
        seg_file = seg_dir / f'instance_segmentation_{frame_id:06d}.png'
        seg_mask = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)

        # Load mapping from mask IDs to object paths
        mapping_file = seg_dir / f'instance_segmentation_mapping_{frame_id:06d}.json'
        with open(mapping_file, 'r') as f:
            id_mapping = json.load(f)

        return seg_mask, id_mapping

    def _extract_object_name_from_path(self, prim_path: str) -> Optional[Tuple[str, str]]:
        """
        Extract object name and dataset type from prim path.
        Handles multiple dataset prefixes: objaverse_, gso_, etc.

        Examples:
        - '/World/objects/gso_adizero_5Tool_25/model/mesh' -> ('adizero_5Tool_25', 'gso')
        - '/World/objects/objaverse_1871970a007c414b92d4e77c1d38ea7b/mesh' -> ('1871970a007c414b92d4e77c1d38ea7b', 'objaverse')

        Returns:
            Tuple of (object_name, dataset_type) or None if not found
        """
        # Common dataset prefixes
        prefixes = ['objaverse_', 'gso_', 'shapenet_', 'ycb_']

        for prefix in prefixes:
            if f'/{prefix}' in prim_path:
                parts = prim_path.split(f'/{prefix}')
                if len(parts) > 1:
                    # Extract name after prefix, before next '/'
                    name = parts[1].split('/')[0]
                    dataset_type = prefix.rstrip('_')
                    return (name, dataset_type)

        # Fallback: try to extract from /World/objects/NAME pattern
        if '/World/objects/' in prim_path:
            parts = prim_path.split('/World/objects/')
            if len(parts) > 1:
                # Get the next component after /World/objects/
                name = parts[1].split('/')[0]
                return (name, 'unknown')

        return None

    def _load_gso_mesh(self, object_name: str) -> Optional[trimesh.Trimesh]:
        """
        Load GSO mesh for a given object name.

        Args:
            object_name: GSO object name (e.g., 'adizero_5Tool_25')

        Returns:
            Trimesh object or None if not found
        """
        if not TRIMESH_AVAILABLE or self.gso_root is None:
            return None

        # Common GSO mesh file patterns
        possible_paths = [
            self.gso_root / object_name / 'meshes' / 'model.obj',
            self.gso_root / object_name / 'meshes' / 'model.ply',
            self.gso_root / object_name / 'model.obj',
            self.gso_root / object_name / 'model.ply',
            self.gso_root / object_name / f'{object_name}.obj',
            self.gso_root / object_name / f'{object_name}.ply',
        ]

        for mesh_path in possible_paths:
            if mesh_path.exists():
                try:
                    mesh = trimesh.load(str(mesh_path), force='mesh')
                    logger.debug(f"Loaded GSO mesh: {mesh_path}")
                    return mesh
                except Exception as e:
                    logger.warning(f"Failed to load mesh from {mesh_path}: {e}")
                    continue

        logger.warning(f"GSO mesh not found for object: {object_name}")
        return None

    def _sample_mesh_points(self, mesh: trimesh.Trimesh, num_samples: int) -> np.ndarray:
        """
        Sample points from mesh surface.

        Args:
            mesh: Trimesh object
            num_samples: Number of points to sample

        Returns:
            Points array of shape [num_samples, 3]
        """
        if mesh is None:
            return np.zeros((num_samples, 3), dtype=np.float32)

        # Sample points on mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        return points.astype(np.float32)

    def _pose_to_latents(
        self,
        obj_data: Dict[str, Any],
        camera_view_transform: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert object state data to latent representation.

        Args:
            obj_data: Object data from states.json with transform_matrix_world, scale, etc.
            camera_view_transform: 4x4 camera view matrix (world-to-camera) stored in
                USD row-vector convention (translation in last row). If provided, the
                returned rotation and translation are expressed in camera coordinates.

        Returns:
            Dictionary with:
            - translation: [3]
            - 6drotation_normalized: [6] (first two columns of rotation matrix, concatenated)
            - scale: [3]
        """
        # transform_matrix_world is stored column-major; transpose to row-major
        # giving the standard form [[R@S | t], [0 0 0 | 1]]
        transform = np.array(obj_data['transform_matrix_world'], dtype=np.float64).T

        # Convert from world coordinates to camera coordinates if view matrix provided
        if camera_view_transform is not None:
            # cameraViewTransform uses the same USD column-major convention;
            # transpose to standard [R_v | t_v ; 0 0 0 1] form
            V = camera_view_transform.astype(np.float64).T
            # T_cam = V @ T_world  ->  rotation becomes R_v @ R_w, translation becomes R_v @ t_w + t_v
            transform = V @ transform
            # Convert OpenGL camera convention (X-right, Y-up, looking down -Z) to
            # PyTorch3D convention (X-left, Y-up, looking down +Z) by flipping X and Z
            gl_to_pytorch3d = np.diag([-1.0, 1.0, -1.0, 1.0])
            transform = gl_to_pytorch3d @ transform

        # Translation from the last column
        translation = transform[:3, 3].astype(np.float32)

        # The upper-left 3x3 = R @ diag(scale); extract scale as column norms
        RS = transform[:3, :3]
        scale = np.linalg.norm(RS, axis=0).astype(np.float32)  # [3]

        # Pure rotation: normalize each column
        R = RS / scale[np.newaxis, :]

        # 6D rotation = column 0 then column 1 of R (Zhou et al.)
        rotation_6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)  # [6]

        return {
            'translation': torch.from_numpy(translation),
            '6drotation_normalized': torch.from_numpy(rotation_6d),
            'scale': torch.from_numpy(scale) * 2,
        }

    def _load_scene_latents(
        self,
        scene_parent_dir: Path,
        states: Dict[str, Any],
        visible_objects: List[str],
        object_datasets: Dict[str, str],
        camera_view_transform: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load or compute latent representations for all visible objects in the scene.

        Args:
            scene_parent_dir: Path to scene_XXXXXXXX directory (contains states.json)
            states: Parsed states.json
            visible_objects: List of object names that are visible in this render
                           (matches keys in states['objects'])
            object_datasets: Mapping of object_name -> dataset_type (e.g., 'gso', 'objaverse')
            camera_view_transform: 4x4 camera view matrix for converting poses to
                camera coordinates. If None, poses remain in world coordinates.

        Returns:
            Dictionary with keys like:
            - 'shape': [num_objects, 8, 16, 16, 16]
            - 'translation': [num_objects, 3]
            - 'scale': [num_objects, 3]
            - '6drotation_normalized': [num_objects, 6]
        """
        if self.precomputed_latents:
            # Load pre-computed latents from disk if available
            latents_file = scene_parent_dir / 'latents.pt'
            if latents_file.exists():
                all_latents = torch.load(latents_file)
                # Filter to only visible objects
                # This assumes latents are keyed by object UID
                latents = {}
                for key in all_latents.keys():
                    latents[key] = torch.stack([all_latents[key][uid]
                                               for uid in visible_objects], dim=0)
                return latents

        # Extract pose-based latents for each visible object
        objects_data = states['objects']
        latents_list = []

        for obj_name in visible_objects:
            if obj_name in objects_data:
                obj_latents = self._pose_to_latents(
                    objects_data[obj_name],
                    camera_view_transform=camera_view_transform,
                )
                latents_list.append(obj_latents)
            else:
                logger.warning(f"Object {obj_name} not found in states.json")

        if len(latents_list) == 0:
            # Return empty latents
            return {
                'translation': torch.zeros(0, 3),
                '6drotation_normalized': torch.zeros(0, 6),
                'scale': torch.zeros(0, 3),
                'shape': torch.zeros(0, 256),
            }

        # Stack into batch format: [num_objects, ...]
        latents = {}
        for key in latents_list[0].keys():
            latents[key] = torch.stack([l[key] for l in latents_list], dim=0)

        # Load shape latents from precomputed npy files
        shape_latents = []

        for obj_name in visible_objects:
            dataset_type = object_datasets.get(obj_name, 'unknown')
            shape_latent = None

            if dataset_type == 'gso' and self.gso_root is not None:
                # Load precomputed latent from npy file
                latent_path = self.gso_root / 'latent_codes' / f'{obj_name}.npy'
                if latent_path.exists():
                    try:
                        # Load latent: shape (1, 8, 16, 16, 16)
                        loaded_latent = np.load(latent_path)
                        # Remove batch dimension: (8, 16, 16, 16)
                        shape_latent = torch.from_numpy(loaded_latent).squeeze(0)
                        logger.debug(f"Loaded shape latent for {obj_name} from {latent_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load latent for {obj_name}: {e}")
                else:
                    logger.warning(f"Latent file not found for {obj_name}: {latent_path}")

            # Fallback to zero latent if not loaded
            if shape_latent is None:
                if self.vae_encoder is not None:
                    # TODO: Implement shape encoding using VAE encoder
                    shape_latent = torch.randn(8, 16, 16, 16)
                else:
                    shape_latent = torch.zeros(8, 16, 16, 16)

            shape_latents.append(shape_latent)

        # Stack shape latents: [num_objects, 8, 16, 16, 16]
        latents['shape'] = torch.stack(shape_latents, dim=0)

        return latents

    def _load_frame_data(
        self,
        render_dir: Path,
        frame_id: int = 0
    ) -> Dict[str, Any]:
        """
        Load visual data for a specific frame (RGB, depth, masks).
        """
        data = {}

        # Load RGB image
        if self.load_images:
            rgb_file = render_dir / 'rgb' / f'rgb_{frame_id:06d}.png'
            if rgb_file.exists():
                image = Image.open(rgb_file).convert('RGB')
                if self.image_size is not None:
                    image = image.resize((self.image_size[1], self.image_size[0]))
                image = np.array(image).astype(np.float32) / 255.0
                data['image'] = torch.from_numpy(image)  # [H, W, 3]

        # Load depth map
        if self.load_depth:
            depth_file = render_dir / 'distance_to_image_plane' / f'distance_to_image_plane_{frame_id:06d}.npy'
            if depth_file.exists():
                depth = np.load(depth_file)
                if self.image_size is not None:
                    depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
                depth = depth.astype(np.float32)
                data['depth'] = torch.from_numpy(depth)  # [H, W]

        # Load instance segmentation
        if self.load_masks:
            seg_mask, id_mapping = self._load_instance_segmentation(render_dir, frame_id)
            if self.image_size is not None:
                seg_mask = cv2.resize(seg_mask, (self.image_size[1], self.image_size[0]),
                                    interpolation=cv2.INTER_NEAREST)
            data['seg_mask'] = seg_mask
            data['id_mapping'] = id_mapping

        return data

    def _load_mesh_data(
        self,
        visible_objects: List[str],
        object_datasets: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Load mesh data for visible GSO objects.

        Args:
            visible_objects: List of object names
            object_datasets: Mapping of object_name -> dataset_type

        Returns:
            Dictionary with:
            - 'mesh_points': [num_objects, num_samples, 3] - sampled point clouds
            - 'mesh_available': [num_objects] - boolean mask indicating which objects have meshes
            - 'mesh_bbox_min': [num_objects, 3] - minimum x, y, z bounds of mesh
            - 'mesh_bbox_max': [num_objects, 3] - maximum x, y, z bounds of mesh
        """
        mesh_points_list = []
        voxel_list = []
        mesh_available_list = []
        mesh_bbox_min_list = []
        mesh_bbox_max_list = []

        for obj_name in visible_objects:
            dataset_type = object_datasets.get(obj_name, 'unknown')

            # Currently only supports GSO meshes
            if dataset_type == 'gso':
                mesh = self._load_gso_mesh(obj_name)
                if mesh is not None:
                    points = self._sample_mesh_points(mesh, self.mesh_num_samples)
                    mesh_points_list.append(torch.from_numpy(points))
                    mesh_available_list.append(True)

                    # voxelize mesh into [64, 64, 64] grid when points are in [-1, 1] cube
                    # copilot generate:
                    R = 64  # resolution
                    pts = points  # (N,3), float

                    # 1) 범위 오차 보정 (약간 벗어난 값 clip)
                    pts = np.clip(pts, -1.0, 1.0)

                    # 2) [-1,1] -> [0, R) 연속좌표로 매핑 후 voxel index로 변환
                    ijk = np.floor((pts + 1.0) * 0.5 * R).astype(np.int64)
                    ijk = np.clip(ijk, 0, R - 1)  # (N,3), each in [0,63]

                    # 3) occupancy 채우기
                    voxel = np.zeros((R, R, R), dtype=np.uint8)
                    voxel[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1 
                    voxel_list.append(torch.from_numpy(voxel))

                    # Calculate bounding box from mesh vertices
                    bbox_min = mesh.vertices.min(axis=0).astype(np.float32)
                    bbox_max = mesh.vertices.max(axis=0).astype(np.float32)
                    mesh_bbox_min_list.append(torch.from_numpy(bbox_min))
                    mesh_bbox_max_list.append(torch.from_numpy(bbox_max))
                else:
                    # Mesh not found, use zeros
                    mesh_points_list.append(torch.zeros(self.mesh_num_samples, 3))
                    mesh_available_list.append(False)
                    mesh_bbox_min_list.append(torch.zeros(3))
                    mesh_bbox_max_list.append(torch.zeros(3))
                    voxel_list.append(torch.zeros(64, 64, 64, dtype=torch.uint8))   
            else:
                # Non-GSO objects, no mesh available
                mesh_points_list.append(torch.zeros(self.mesh_num_samples, 3))
                mesh_available_list.append(False)
                mesh_bbox_min_list.append(torch.zeros(3))
                mesh_bbox_max_list.append(torch.zeros(3))
                voxel_list.append(torch.zeros(64, 64, 64, dtype=torch.uint8))   

        return {
            'mesh_points': torch.stack(mesh_points_list, dim=0),  # [num_objects, num_samples, 3]
            'mesh_available': torch.tensor(mesh_available_list, dtype=torch.bool),  # [num_objects]
            'mesh_bbox_min': torch.stack(mesh_bbox_min_list, dim=0),  # [num_objects, 3]
            'mesh_bbox_max': torch.stack(mesh_bbox_max_list, dim=0),  # [num_objects, 3]
            'voxels': torch.stack(voxel_list, dim=0),  # [num_objects, 64, 64, 64]
        }

    def _load_scene_conditionals(
        self,
        frame_data: Dict[str, Any],
        camera_params: Dict[str, Any],
        visible_objects: List[str],
    ) -> Dict[str, Any]:
        """
        Load conditioning information for the scene.
        """
        conditionals = {}

        # Add camera intrinsics
        conditionals['camera_K'] = torch.from_numpy(camera_params['camera_K']).float()

        # Add image as conditioning
        if 'image' in frame_data:
            conditionals['image'] = frame_data['image']

        # Add depth as conditioning
        if 'depth' in frame_data:
            conditionals['depth'] = frame_data['depth']

            # Add pointmap by backprojecting depth using camera intrinsics
            depth = frame_data['depth'].numpy()  # [H, W]

            # set inf depth values to the max finite depth for backprojection
            finite_mask = np.isfinite(depth)
            if np.any(finite_mask):
                max_finite_depth = np.max(depth[finite_mask])
                depth[~finite_mask] = max_finite_depth

            H, W = depth.shape
            fx = camera_params['camera_K'][0, 0]
            fy = camera_params['camera_K'][1, 1]
            cx = camera_params['camera_K'][0, 2]
            cy = camera_params['camera_K'][1, 2]
            i_coords, j_coords = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            x = (i_coords - cx) * depth / fx
            y = (j_coords - cy) * depth / fy
            z = depth
            pointmap = np.stack([x, y, z], axis=-1).astype(np.float32)  # [H, W, 3]

            # convert coordinates from OpenGL to PyTorch3D convention
            pointmap[..., 0] *= -1  # Flip X
            pointmap[..., 1] *= -1  # Flip Y

            conditionals['pointmap'] = torch.from_numpy(pointmap)

        # Add per-object masks if available
        if 'seg_mask' in frame_data:
            seg_mask = frame_data['seg_mask']
            id_mapping = frame_data['id_mapping']

            # Build reverse mapping: object_name -> mask_id
            name_to_mask_id = {}
            for mask_id_str, prim_path in id_mapping.items():
                obj_info = self._extract_object_name_from_path(prim_path)
                if obj_info is not None:
                    obj_name, _ = obj_info
                    name_to_mask_id[obj_name] = int(mask_id_str)

            # Extract individual object masks
            object_masks = []
            for obj_name in visible_objects:
                if obj_name in name_to_mask_id:
                    mask_id = name_to_mask_id[obj_name]
                    obj_mask = (seg_mask == mask_id).astype(np.float32)
                    object_masks.append(torch.from_numpy(obj_mask))
                else:
                    # Object not in this render, use empty mask
                    h, w = seg_mask.shape
                    object_masks.append(torch.zeros(h, w))

            if len(object_masks) > 0:
                conditionals['object_masks'] = torch.stack(object_masks, dim=0)  # [num_objects, H, W]

        return conditionals

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a scene sample.

        Returns all objects from a rendered scene as a single batch.
        """
        render_info = self.scenes[idx]
        object_dir = render_info['object_dir']
        scene_parent_dir = render_info['scene_parent_dir']
        scene_dir = render_info['scene_dir']
        render_dir = render_info['render_dir']
        scene_id = render_info['scene_id']

        # Load states.json
        states = self._load_states(scene_parent_dir)

        # Load camera parameters
        camera_params = self._load_camera_params(render_dir, frame_id=0)

        # Load frame data (image, depth, segmentation)
        frame_data = self._load_frame_data(render_dir, frame_id=0)

        # Determine which objects are visible in this render
        # Store both object name and dataset type for mesh loading
        visible_objects = []
        object_datasets = {}  # Maps object_name -> dataset_type
        if 'id_mapping' in frame_data:
            id_mapping = frame_data['id_mapping']
            for prim_path in id_mapping.values():
                obj_info = self._extract_object_name_from_path(prim_path)
                if obj_info is not None:
                    obj_name, dataset_type = obj_info
                    if obj_name in states['objects']:
                        visible_objects.append(obj_name)
                        object_datasets[obj_name] = dataset_type

        # Remove duplicates
        visible_objects = list(set(visible_objects))

        # Get number of objects
        num_objects = len(visible_objects)

        # Filter if too many objects
        if num_objects > self.max_objects_per_scene:
            logger.warning(
                f"Scene {scene_id} has {num_objects} objects, "
                f"exceeding max {self.max_objects_per_scene}. Sampling subset."
            )
            selected_indices = np.random.choice(
                num_objects,
                self.max_objects_per_scene,
                replace=False
            )
            visible_objects = [visible_objects[i] for i in selected_indices]
            num_objects = self.max_objects_per_scene

        # Load latents for all visible objects (in camera coordinates)
        latents = self._load_scene_latents(
            scene_parent_dir, states, visible_objects, object_datasets,
            camera_view_transform=camera_params['camera_view_transform'],
        )

        # Load conditioning information
        conditionals = self._load_scene_conditionals(
            frame_data, camera_params, visible_objects
        )

        # Load mesh data if requested
        mesh_data = None
        if self.load_meshes:
            mesh_data = self._load_mesh_data(visible_objects, object_datasets)

        result = {
            'latents': latents,
            'conditionals': conditionals,
            'scene_id': scene_id,
            'num_objects': num_objects,
        }

        if mesh_data is not None:
            result['mesh_data'] = mesh_data

        return result

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for FoundationPose dataset.

    Since each scene can have a different number of objects,
    we process one scene at a time (batch_size=1).
    """
    assert len(batch) == 1, "Scene-level training requires batch_size=1"
    return batch[0]


# Example usage and testing
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Test FoundationPose dataset with GSO mesh loading')
    parser.add_argument('--data-root', type=str, default='/workspace/sam-3d-objects/foundationpose/',
                        help='Root directory containing FoundationPose data')
    parser.add_argument('--gso-root', type=str, default='/workspace/sam-3d-objects/gso/google_scanned_objects/',
                        help='Root directory containing GSO mesh files')
    parser.add_argument('--load-meshes', action='store_true',
                        help='Load GSO mesh data')
    args = parser.parse_args()

    # Test the dataset
    data_root = args.data_root

    if not Path(data_root).exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    dataset = FoundationPoseDataset(
        data_root=data_root,
        max_objects_per_scene=32,
        precomputed_latents=True,
        num_renders_per_scene=1,
        gso_root=args.gso_root,
        load_meshes=args.load_meshes,
        mesh_num_samples=2048,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"GSO mesh loading: {'enabled' if args.load_meshes else 'disabled'}")
    if args.gso_root:
        print(f"GSO root: {args.gso_root}")

    if len(dataset) > 0:
        # Load a sample
        sample = dataset[1]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Scene ID: {sample['scene_id']}")
        print(f"Number of objects: {sample['num_objects']}")
        print(f"\nLatent keys: {sample['latents'].keys()}")
        for key, value in sample['latents'].items():
            print(f"  {key}: {value.shape}")
        print(f"\nConditional keys: {sample['conditionals'].keys()}")
        for key, value in sample['conditionals'].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

        # Print mesh data if available
        if 'mesh_data' in sample:
            print(f"\nMesh data keys: {sample['mesh_data'].keys()}")
            for key, value in sample['mesh_data'].items():
                print(f"  {key}: {value.shape}")
            # Count how many objects have meshes
            num_with_meshes = sample['mesh_data']['mesh_available'].sum().item()
            print(f"\nObjects with meshes: {num_with_meshes}/{sample['num_objects']}")

        # save image, masks for visualization and scale, translation, rotation in a json file
        import matplotlib.pyplot as plt
        image = sample['conditionals']['image'].numpy()
        plt.imsave('test_image.png', image)

        if 'object_masks' in sample['conditionals']:
            masks = sample['conditionals']['object_masks'].numpy()
            for i in range(masks.shape[0]):
                plt.imsave(f'{i}.png', masks[i], cmap='gray')

        # visualize depthmap
        if 'depth' in sample['conditionals']:
            depth = sample['conditionals']['depth'].numpy()
            # normalize depth for visualization (except for infinite values)
            depth_min = np.min(depth[np.isfinite(depth)])
            depth_max = np.max(depth[np.isfinite(depth)])
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

            # set infinite values to 1.0 (white)
            depth[~np.isfinite(depth)] = 1.0

            plt.imsave('test_depth.png', depth, cmap='gray')

        if 'pointmap' in sample['conditionals']:
            pointmap = sample['conditionals']['pointmap'].numpy()
            np.save('test_pointmap.npy', pointmap)
        
        # save latents to json
        latents_json = {}
        latents_json['scale'] = sample['latents']['scale'].numpy().tolist()
        latents_json['translation'] = sample['latents']['translation'].numpy().tolist()
        latents_json['6drotation_normalized'] = sample['latents']['6drotation_normalized'].numpy().tolist()

        # Add mesh bbox data if available
        if 'mesh_data' in sample:
            latents_json['mesh_available'] = sample['mesh_data']['mesh_available'].numpy().tolist()
            latents_json['mesh_bbox_min'] = sample['mesh_data']['mesh_bbox_min'].numpy().tolist()
            latents_json['mesh_bbox_max'] = sample['mesh_data']['mesh_bbox_max'].numpy().tolist()

        with open('test_latents.json', 'w') as f:
            json.dump(latents_json, f, indent=4)