from typing import Union, Optional, List
import os
from loguru import logger
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

class InferencePipelineJoint(InferencePipelinePointMap):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        masks: Union[None, List[Image.Image], List[np.ndarray]] = None,
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
        num_objects = len(masks) if masks is not None else 0
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]

        

        images = [self.merge_image_and_mask(image, mask) for mask in masks]

        for i in range(len(images)):
            images[i][:, :, 3] = images[i][:, :, 3] * 255

        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            if estimate_plane:
                return self.estimate_plane2(pointmap_dict, image)

            ss_input_dicts = [self.preprocess_image(
                img, self.ss_preprocessor, pointmap=pointmap
            ) for img in images]

            slat_input_dicts = [self.preprocess_image(img, self.slat_preprocessor) for img in images]
            if seed is not None:
                torch.manual_seed(seed)

            # ss_return_dict = self.sample_sparse_structure(
            #     ss_input_dict,
            #     inference_steps=stage1_inference_steps,
            #     use_distillation=use_stage1_distillation,
            # )

            cond = [self.get_condition_input(self.condition_embedders["ss_condition_embedder"], input_dict, [])[0][0]
                    for input_dict in ss_input_dicts]
            cond = torch.concat(cond, dim=0)

            latent_shape_dict = {
                k: (num_objects,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
            }

            ss_return_dict = ss_generator(
                latent_shape_dict,
                self.device,
                cond,
            )

            ss_return_dicts = [{k: v[i : i + 1] for k, v in ss_return_dict.items()} for i in range(num_objects)]

            ret = []

            for ss_input_dict, ss_return_dict, slat_input_dict in zip(ss_input_dicts, ss_return_dicts, slat_input_dicts):
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

                #logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
                #ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

                if stage1_only:
                    logger.info("Finished!")
                    ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                    ret.append({
                        **ss_return_dict,
                        "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                        "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                    })
                    continue
                    # return ss_return_dict

                shape_latent = ss_return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

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

                ret.append({
                    **ss_return_dict,
                    **outputs,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                })

            return ret
    