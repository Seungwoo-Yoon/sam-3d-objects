# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional
from copy import deepcopy
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image
import open3d as o3d
import os

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d, rotation_6d_to_matrix

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area, layout_post_optimization
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

from romatch import roma_indoor
from romatch.utils.utils import tensor_to_pil
import torch.nn.functional as F
from sam3d_objects.custom.utils import *

def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=layout_post_optimization, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        ss_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)

        return item

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped



    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None

        points_tensor = points_tensor.permute(2, 0, 1)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask)
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]
        else:
            point_map_tensor["intrinsics"] = intrinsics.to(self.device)

        return point_map_tensor

    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal
        revised_quat, revised_t, revised_scale, final_iou, _, _ = (
            self.layout_post_optimization_method(
                mesh_glb,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_pointmap"][0].permute(1, 2, 0),
                intrinsics,
                min_size=518,
            )
        )
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=2,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
    ) -> dict:
        image = self.merge_image_and_mask(image, mask)
        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            if estimate_plane:
                return self.estimate_plane2(pointmap_dict, image)

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )

            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )


            # We could probably use the decoder from the models themselves
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            rotation_6d = ss_return_dict["6drotation_normalized"]
            rotation_matrix = rotation_6d_to_matrix(rotation_6d)
            translation = ss_return_dict["translation"]
            # translation[:, :] -= pointmap_shift / pointmap_scale[:, 2]

            extrinsic = torch.eye(4, device=rotation_matrix.device).unsqueeze(0).repeat(rotation_matrix.shape[0], 1, 1)
            
            extrinsic[:, :3, :3] = rotation_matrix

            normalized_pointmap_scale = torch.mean(pointmap_scale.squeeze(1), dim=1)  # (B,)

            # translation *= normalized_pointmap_scale
            # translation -= pointmap_shift
            extrinsic[:, :3, 3] = translation.squeeze(1)

            intrinsic = pointmap_dict["intrinsics"]

            # apply scale to intrinsic and extrinsic
            scale = torch.mean(ss_return_dict["scale"].squeeze(1), dim=1)  # (B,)

            intrinsic = intrinsic.clone()
            # intrinsic[:2, :2] = intrinsic[:2, :2] * (scale[:, None] * normalized_pointmap_scale[:, None])
            extrinsic = extrinsic
            extrinsic[:, :3, 3] = extrinsic[:, :3, 3] / scale[:, None]

            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                }
                # return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)

            render = render_utils.render_frames(
                outputs['gaussian'][0],
                extrinsic,
                [intrinsic],
                {"resolution": 518, "bg_color": (0, 0, 0), "backend": "gsplat"}
            )['color'][0]
            
            depth = render_utils.render_frames(
                outputs['mesh'][0],
                extrinsic,
                [intrinsic],
                {"resolution": 518, "bg_color": (0, 0, 0), "backend": "gsplat"}
            )['depth'][0]

            pred_pointmap = unproject_depth_to_pointmap(
                np.expand_dims(depth, axis=0),
                np.eye(4)[None, :, :][:, :3, :],
                intrinsic.unsqueeze(0)
            )[0]

            depth_img = np.rot90(depth, k=2)
            # depth_img = (depth_img / np.nanmax(depth_img) * 255).astype(np.uint8) <- 이거 대신 mask에서의 max로 나누기
            depth_mask = ss_input_dict['rgb_image_mask'][0,0].cpu().numpy()
            max_depth = np.nanmax(depth_img[depth_mask > 0.5])
            depth_img = (depth_img / max_depth * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_img)
            os.makedirs("../debug/rendering_test", exist_ok=True)
            depth_img.save(f"../debug/rendering_test/test_depth.png")

            '''
            img1 = np.rot90(render, k=2)
            img1 = Image.fromarray(img1)
            img1.save(f"../debug/rendering_test/test.png")

            img2 = ss_input_dict['rgb_image'][0] * ss_input_dict['rgb_image_mask'][0]
            img2 = (img2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img2 = Image.fromarray(img2)
            img2.save('../debug/rendering_test/original.png')

            roma_model = roma_indoor(device=self.device)
            H, W = roma_model.get_output_resolution()

            img1 = img1.resize((W,H))
            img2 = img2.resize((W,H))

            warp, certainty = roma_model.match(img1, img2, device=self.device)
            matches, certainty = roma_model.sample(warp, certainty)
            kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, 518, 518, 518, 518)

            # mask out keypoints outside the mask
            mask_np = ss_input_dict['rgb_image_mask'][0,0].cpu().numpy()

            img1 = img1.resize((518,518))
            img2 = img2.resize((518,518))

            valid_kpts = []
            for i in range(kpts1.shape[0]):
                x2, y2 = int(kpts2[i,0]), int(kpts2[i,1])
                if x2 >=0 and x2 < 518 and y2 >=0 and y2 < 518:
                    if mask_np[y2, x2] > 0.5:
                        valid_kpts.append(i)
            kpts1 = kpts1[valid_kpts]
            kpts2 = kpts2[valid_kpts]

            # visualize keypoints with opencv 
            # visualize two images side by side with keypoints and lines connecting them
            # random colors for each keypoint
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            vis_img = np.zeros((518, 2*518, 3), dtype=np.uint8)
            vis_img[:, :518, :] = img1_cv
            vis_img[:, 518:, :] = img2_cv
            for i in range(kpts1.shape[0]):
                color = (int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255)))
                pt1 = (int(kpts1[i, 0]), int(kpts1[i, 1]))
                pt2 = (int(kpts2[i, 0] + 518), int(kpts2[i, 1]))
                cv2.circle(vis_img, pt1, 3, color, -1)
                cv2.circle(vis_img, pt2, 3, color, -1)
                cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)
            cv2.imwrite('../debug/rendering_test/matches.png', vis_img)

            gt_keypoint = ss_input_dict['rgb_pointmap'][0].permute(1, 2, 0).cpu().numpy()
            gt_keypoint = gt_keypoint[kpts2[:,0].long().cpu().numpy(), kpts2[:,1].long().cpu().numpy(), :]

            pred_keypoint = pred_pointmap[kpts1[:,0].long().cpu().numpy(), kpts1[:,1].long().cpu().numpy(), :]

            # point cloud registration using open3d including scaling
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_keypoint)
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(pred_keypoint)
            threshold = 0.05  # 5cm distance threshold

            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_pred, pcd_gt, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            print(reg_p2p.transformation)
            '''

            try:
                if (
                    with_layout_postprocess
                    and self.layout_post_optimization_method is not None
                ):
                    assert glb is not None, "require mesh to run postprocessing"
                    logger.info("Running layout post optimization method...")
                    postprocessed_pose = self.run_post_optimization(
                        deepcopy(glb),
                        pointmap_dict["intrinsics"],
                        ss_return_dict,
                        ss_input_dict,
                    )
                    ss_return_dict.update(postprocessed_pose)
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            # glb.export("sample.glb")
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
                "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
            }

    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        # img_3chw: (3, H, W)
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0

        # heuristics
        if max_side > 3800:
            scale_factor = 0.125
        if max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # -> (1, 3, H/4, W/4)
        return x.squeeze(0)

    def estimate_plane(self, pointmap_dict, image, ground_area_threshold=0.25, min_points=100):
        assert image.shape[-1] == 4  # rgba format
        # Extract mask from alpha channel
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # Get all points in 3D space (H, W, 3)
        pts_hwc = pts.cpu().permute((1, 2, 0))

        valid_mask_points = floor_mask.cpu().numpy()
        # Extract points that fall within the mask
        if valid_mask_points.any():
            # Get points within mask
            masked_points = pts_hwc[valid_mask_points]
            # Filter out invalid points (zero points from depth estimation failures)
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.numpy()
        else:
            points = np.array([]).reshape(0, 3)
     
        # Calculate area coverage and check num of points
        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")
        if overlap_area > ground_area_threshold and has_enough_points:
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh")
            except Exception as e:
                logger.error(f"Failed to estimate plane: {e}")
                mesh = None
        else:
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }

    def estimate_plane2(
        self,
        pointmap_dict,
        image,
        ground_area_threshold=0.25,
        min_points=100,
        # ---- robust defaults ----
        max_planes=6,
        ransac_n=3,
        num_iterations=2500,
        distance_threshold=0.02,
        min_inliers=300,
        # scoring weights
        w_support=1.0,      # inlier ratio
        w_flatness=0.8,     # how planar the inliers are (PCA thickness)
        w_lowness=1.4,      # prefer plane near a global "lower extreme" (auto from candidate sign)
    ):
        """
        View-agnostic ground plane estimation.
        Keeps SAM3D IO shape/keys compatible.

        Returns SAM3D-style dict:
        glb(mesh or None), translation(1,3), scale(1,3), rotation(1,4 quat)
        """
        assert image.shape[-1] == 4  # rgba format

        # Mask from alpha channel (same as original)
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # (H,W,3)
        pts_hwc = pts.detach().cpu().permute((1, 2, 0))
        valid_mask_points = floor_mask.detach().cpu().numpy()

        if valid_mask_points.any():
            masked_points = pts_hwc[valid_mask_points]  # (N,3) torch
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.detach().cpu().numpy().astype(np.float64)
        else:
            points = np.empty((0, 3), dtype=np.float64)

        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")

        if not (overlap_area > ground_area_threshold and has_enough_points):
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            return {
                "glb": None,
                "translation": torch.tensor([[0.0, 0.0, 0.0]]),
                "scale": torch.tensor([[1.0, 1.0, 1.0]]),
                "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            }

        # Open3D point cloud
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(points)

        # Precompute global scale for normalization
        # We'll use global projection ranges along candidate normals.
        best = None
        remaining = pcd0

        def pca_flatness(inlier_pts: np.ndarray):
            """
            Returns thickness ratio: smallest eigenvalue / sum eigenvalues (smaller => flatter plane).
            """
            if inlier_pts.shape[0] < 10:
                return 1.0
            X = inlier_pts - inlier_pts.mean(axis=0, keepdims=True)
            C = (X.T @ X) / max(X.shape[0] - 1, 1)
            evals = np.linalg.eigvalsh(C)
            s = float(np.sum(evals))
            if s < 1e-12:
                return 1.0
            return float(evals[0] / s)  # thickness ratio

        # Find multiple plane candidates
        for k in range(max_planes):
            if len(remaining.points) < max(min_inliers, ransac_n):
                break

            try:
                plane_model, inliers = remaining.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations,
                )
            except Exception as e:
                logger.error(f"Open3D plane segmentation failed: {e}")
                break

            inliers = np.asarray(inliers, dtype=np.int64)
            if inliers.size < min_inliers:
                break

            a, b, c, d = [float(x) for x in plane_model]
            n = np.array([a, b, c], dtype=np.float64)
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                remaining = remaining.select_by_index(inliers.tolist(), invert=True)
                continue
            n = n / nn

            pts_rem = np.asarray(remaining.points)
            inlier_pts = pts_rem[inliers]

            # Score components
            support = inliers.size / max(len(remaining.points), 1)  # local support
            thickness = pca_flatness(inlier_pts)                   # smaller is better
            flatness_score = 1.0 - np.clip(thickness * 50.0, 0.0, 1.0)  # heuristic mapping

            # "Lowness" WITHOUT known up:
            # We evaluate both normal directions (+n and -n), pick direction where the plane sits
            # closer to the "lower extreme" (smaller projection) of all masked points.
            # For each direction u, compare mean projection of plane inliers vs global min projection.
            # floor should be near the min in that axis direction.
            all_pts = points
            proj_all_pos = all_pts @ n
            proj_all_neg = all_pts @ (-n)

            # global mins
            min_pos = float(np.min(proj_all_pos))
            min_neg = float(np.min(proj_all_neg))

            mean_pos = float(np.mean(inlier_pts @ n))
            mean_neg = float(np.mean(inlier_pts @ (-n)))

            range_pos = max(float(np.max(proj_all_pos) - min_pos), 1e-6)
            range_neg = max(float(np.max(proj_all_neg) - min_neg), 1e-6)

            # closeness to global minimum (0 is best), convert to score (1 best)
            clos_pos = (mean_pos - min_pos) / range_pos
            clos_neg = (mean_neg - min_neg) / range_neg
            low_score_pos = 1.0 - np.clip(clos_pos, 0.0, 1.0)
            low_score_neg = 1.0 - np.clip(clos_neg, 0.0, 1.0)

            if low_score_neg > low_score_pos:
                n_used = -n
                d_used = -d
                low_score = low_score_neg
            else:
                n_used = n
                d_used = d
                low_score = low_score_pos

            score = w_support * support + w_flatness * flatness_score + w_lowness * low_score

            cand = {
                "k": k,
                "score": float(score),
                "n": n_used,
                "d": float(d_used),
                "inlier_pts": inlier_pts,
                "support": float(support),
                "flatness_score": float(flatness_score),
                "low_score": float(low_score),
                "inliers": int(inliers.size),
            }

            if (best is None) or (cand["score"] > best["score"]):
                best = cand

            # Remove this plane and continue
            remaining = remaining.select_by_index(inliers.tolist(), invert=True)

        if best is None:
            logger.info("No plane candidate found; falling back to original estimation.")
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh (fallback)")
            except Exception as e:
                logger.error(f"Failed to estimate plane (fallback): {e}")
                mesh = None
        else:
            logger.info(
                f"Selected plane k={best['k']} score={best['score']:.3f} "
                f"inliers={best['inliers']} support={best['support']:.3f} "
                f"flat={best['flatness_score']:.3f} low={best['low_score']:.3f}"
            )
            # Build mesh from inliers (more robust)
            try:
                mesh = o3d_plane_estimation(best["inlier_pts"])
                logger.info("Successfully estimated plane mesh (improved, view-agnostic)")
            except Exception as e:
                logger.error(f"Failed to estimate plane (improved): {e}")
                mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
