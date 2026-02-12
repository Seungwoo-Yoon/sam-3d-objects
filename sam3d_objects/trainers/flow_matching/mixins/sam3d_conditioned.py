from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os

import torchvision
from sam3d_objects.pipeline import preprocess_utils
from hydra.utils import instantiate
from safetensors.torch import load_file
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import look_at_view_transform
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from ....utils import dist_utils
from moge.model.v1 import MoGeModel
from sam3d_objects.pipeline.depth_models.moge import MoGe


class Sam3DConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, ss_generator_config_path: str = './checkpoints/hf/ss_generator.yaml', *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = OmegaConf.load('./checkpoints/hf/pipeline.yaml')["ss_preprocessor"]
        self.preprocessor = instantiate(config)
        
        conf = OmegaConf.load(
            ss_generator_config_path
        )
        self.condition_embedder = self.instantiate_and_load_from_pretrained(
            conf["module"]["condition_embedder"]["backbone"],
            ss_generator_config_path.replace('.yaml', '.ckpt'),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                "_base_models.condition_embedder."
            ),
            device='cuda',
        )
        
        self.depth_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl')
        self.depth_model = MoGe(self.depth_model)
    
    @staticmethod
    def instantiate_and_load_from_pretrained(
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cuda", 
    ):
        model = instantiate(config)

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cuda")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
            )
        model = model.to(device)

        return model

    @staticmethod
    def prepare_for_training(**kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(Sam3DConditionedMixin, Sam3DConditionedMixin), 'prepare_for_training'):
            super(Sam3DConditionedMixin, Sam3DConditionedMixin).prepare_for_training(**kwargs)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    @staticmethod
    def merge_image_and_mask(
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)

            mask = np.array(mask)
            if mask.ndim == 2:
                mask = mask[..., None]

            assert mask.shape[:2] == image.shape[:2]
            image = np.concatenate([image[..., :3], mask], axis=-1)

        image = np.array(image)
        return image

    @staticmethod
    def get_mask(
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        mask_source: str,
    ) -> torch.Tensor:
        """
        Extract a mask from either the alpha channel of an RGB image or a depth image.

        Args:
            rgb_image: Tensor of shape (B, C, H, W) or (C, H, W) where C >= 4 if using alpha channel
            depth_image: Tensor of shape (B, 1, H, W) or (1, H, W) containing depth information
            mask_source: Source of the mask, either "ALPHA_CHANNEL" or "DEPTH"

        Returns:
            mask: Tensor of shape (B, 1, H, W) or (1, H, W) containing the extracted mask
        """
        # Handle unbatched inputs (add batch dimension if needed)
        is_batched = len(rgb_image.shape) == 4

        if not is_batched:
            rgb_image = rgb_image.unsqueeze(0)
            if depth_image is not None:
                depth_image = depth_image.unsqueeze(0)

        if mask_source == "ALPHA_CHANNEL":
            if rgb_image.shape[1] != 4:
                logger.warning(f"No ALPHA CHANNEL for the image, cannot read mask.")
                mask = None
            else:
                mask = rgb_image[:, 3:4, :, :]
        elif mask_source == "DEPTH":
            mask = depth_image
        else:
            raise ValueError(f"Invalid mask source: {mask_source}")

        # Remove batch dimension if input was unbatched
        if not is_batched:
            mask = mask.squeeze(0)

        return mask

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
        rgb_image_mask = (self.get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].cuda(),
            "image": _item["image"][None].cuda(),
            "rgb_image": _item["rgb_image"][None].cuda(),
            "rgb_image_mask": _item["rgb_image_mask"][None].cuda(),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].cuda()
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].cuda()
            item["pointmap_scale"] = _item["pointmap_scale"][None].cuda()
            item["pointmap_shift"] = _item["pointmap_shift"][None].cuda()
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].cuda()
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].cuda()

        return item
    
    @staticmethod
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

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(self.camera_to_pytorch3d_camera(device='cuda').rotation)
                .cuda()
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.cuda()
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
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        return point_map_tensor

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return pointmap

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        image = cond['image']
        masks = cond['masks']

        images = [self.merge_image_and_mask(image, mask) for mask in masks]
        pointmap = self.compute_pointmap(image)

        input_dicts = [self.preprocess_image(img, self.preprocessor, pointmap=pointmap['pointmap']) for img in images]

        cond = [self.get_condition_input(
            self.condition_embedder, input_dict, []
        )[0][0] for input_dict in input_dicts]

        cond = torch.concat(cond, dim=0)

        # cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}
