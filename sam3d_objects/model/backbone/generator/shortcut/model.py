# Copyright (c) Meta Platforms, Inc. and affiliates.
import random
from typing import Callable, Sequence, Union
import torch
import numpy as np
from functools import partial
import optree
import math
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
import open3d as o3d

from sam3d_objects.model.backbone.generator.base import Base
from sam3d_objects.data.utils import right_broadcasting
from sam3d_objects.data.utils import tree_tensor_map, tree_reduce_unique
from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching, _get_device
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import copy

from sam3d_objects.custom.utils import *

obj_count = 0


# https://arxiv.org/pdf/2410.12557
class ShortCut(FlowMatching):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_shortcut = no_shortcut
        self.self_consistency_prob = self_consistency_prob
        self.shortcut_loss_weight = shortcut_loss_weight
        self.self_consistency_cfg_strength = self_consistency_cfg_strength
        self.ratio_cfg_samples_in_self_consistency_target = ratio_cfg_samples_in_self_consistency_target
        self.fm_in_shortcut_target_prob = fm_in_shortcut_target_prob
        self.fm_eps_max = fm_eps_max
        self.batch_mode = batch_mode
        self.cfg_modalities = cfg_modalities

    def _generate_d(self, x1):
        """
        Generate shortcut step sizes d with binary-time schedule.
        
        This method ensures deterministic behavior for distributed training:
        - Exactly self_consistency_prob fraction of samples will have d > 0 (self-consistency)
        - Remaining samples will have d = 0 (flow matching)
        - All distributed ranks will have consistent counts, preventing deadlocks
        
        Args:
            x1: Input tensor or tree of tensors
            
        Returns:
            d: Tensor of step sizes with shape [batch_size]
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        # Use binary-time schedule: d ∈ {1/2^i for i in range(8)}
        base = [1 / 2**i for i in range(8)]
        
        # Deterministic approach: exactly self_consistency_prob fraction will have d>0
        # This ensures all distributed ranks have consistent behavior
        if self.batch_mode:
            num_self_consistency_samples = int(random.random() < self.self_consistency_prob) * batch_size
        else:
            num_self_consistency_samples = int(batch_size * self.self_consistency_prob)
        num_flow_matching_samples = batch_size - num_self_consistency_samples
        
        # Create deterministic d values
        d = torch.zeros(batch_size, device=device)
        
        if num_self_consistency_samples > 0:
            # Randomly select d values for self-consistency samples
            selected_elements = random.choices(base, k=1) * num_self_consistency_samples
            d[:num_self_consistency_samples] = torch.FloatTensor(selected_elements).to(device)
        
        # Shuffle the d values to randomize which samples get which d values
        # This maintains the deterministic count while randomizing positions
        shuffle_indices = torch.randperm(batch_size, device=device)
        d = d[shuffle_indices]

        return d

    @torch.no_grad()
    def compute_self_consistency_target(self, x_t, t, d, *args_conditionals, **kwargs_conditionals):
        """
        Compute self-consistency target for shortcut model's self-consistency objective.
        
        This method uses a mixed approach where:
        - First 25% of samples (num_cfg_samples) use CFG blending with strength 7.0
        - Remaining 75% of samples use conditional-only targets
        
        Safety guarantees:
        - Ensures at least 1 sample in CFG part (num_cfg_samples >= 1 for batch_size >= 2)
        - For batch_size < 2, falls back to all conditional-only (no CFG)
        - Handles edge cases where batch size is too small for mixed approach
        
        The process involves:
        1. Forward all samples through conditional model to get s_t_cond and s_td_cond
        2. Forward first num_cfg_samples through unconditional model to get s_t_uncond and s_td_uncond  
        3. Apply CFG blending: (1 + strength) * cond - strength * uncond for first num_cfg_samples
        4. Concatenate CFG results with conditional-only results for remaining samples
        5. Average the two velocities to get final self-consistency target
        """
        # CFG strength for self-consistency target computation
        self_consistency_cfg_strength = self.self_consistency_cfg_strength
        
        # Mixed approach: configurable ratio of CFG:conditional-only samples
        batch_size = x_t.shape[0] if not isinstance(x_t, dict) else next(iter(x_t.values())).shape[0]
        if self.batch_mode:
            num_cfg_samples = int(random.random() < self.ratio_cfg_samples_in_self_consistency_target) * batch_size
        else:
            num_cfg_samples = int(batch_size * self.ratio_cfg_samples_in_self_consistency_target)  # Configurable ratio for CFG
        num_cond_only_samples = batch_size - num_cfg_samples  # Remaining for conditional-only
        use_fm_in_shortcut_target = random.random() < self.fm_in_shortcut_target_prob

        # Handle edge case where batch_size < 2 (fallback to all conditional-only)
        # if batch_size < 2:
        #     num_cfg_samples = 0
        #     num_cond_only_samples = batch_size
        
        
        # ### DEBUG ###############################
        # num_cfg_samples = 0
        # num_cond_only_samples = batch_size
        # # ### DEBUG ###############################
        
        # Step 1: Get velocity predictions at current time t
        # Forward all samples through conditional model
        s_t_cond = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )
        
        # Handle CFG and conditional-only parts
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model
            if isinstance(x_t, dict):
                x_t_cfg = {k: v[:num_cfg_samples] for k, v in x_t.items()}
            else:
                x_t_cfg = x_t[:num_cfg_samples]
            
            s_t_uncond = self.reverse_fn(
                x_t_cfg,
                t[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )
            
            # Apply CFG blending for first num_cfg_samples using our standard formula
            s_t_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_t_cond), s_t_uncond
            )
            
            # Combine CFG results with conditional-only results for remaining samples
            if num_cond_only_samples > 0:
                s_t = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_t_cfg, s_t_cond
                )
            else:
                # All samples use CFG
                s_t = s_t_cond
                if isinstance(s_t_cond, dict):
                    for modality in self.cfg_modalities:
                        s_t[modality] = s_t_cfg[modality]
                else:
                    s_t = s_t_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_t = s_t_cond
        
        # Step 2: Take a step of size d using current velocity
        x_td = tree_tensor_map(lambda x, v: x + v * d[..., None, None], x_t, s_t)
        
        # Step 3: Get velocity predictions at time t+d
        # Forward all samples through conditional model at t+d
        s_td_cond = self.reverse_fn(
            x_td,
            (t + d) * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )
        
        # Handle CFG and conditional-only parts at t+d
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model at t+d
            if isinstance(x_td, dict):
                x_td_cfg = {k: v[:num_cfg_samples] for k, v in x_td.items()}
            else:
                x_td_cfg = x_td[:num_cfg_samples]
            
            s_td_uncond = self.reverse_fn(
                x_td_cfg,
                (t + d)[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )
            
            # Apply CFG blending for first num_cfg_samples at t+d using our standard formula
            s_td_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_td_cond), s_td_uncond
            )
            
            # Combine CFG results with conditional-only results for remaining samples at t+d
            if num_cond_only_samples > 0:
                s_td = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_td_cfg, s_td_cond
                )
            else:
                # All samples use CFG
                s_td = s_td_cond
                if isinstance(s_td_cond, dict):
                    for modality in self.cfg_modalities:
                        s_td[modality] = s_td_cfg[modality]
                else:
                    s_td = s_td_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_td = s_td_cond
        
        # Step 4: Compute self-consistency target as average of two velocities
        s_target = tree_tensor_map(lambda a, b: (a + b).detach() / 2, s_t, s_td)
        
        return s_target

    def _generate_t_and_d(self, x1):
        """
        Generate t and d together according to shortcut models paper.
        
        According to the paper: "During training, we first sample d, then sample t only at the discrete
        points for which the shortcut model will be queried, i.e. multiples of d. We train the 
        self-consistency objective only at these timesteps."
        
        This ensures that when d > 0 (self-consistency samples), t is sampled at multiples of d.
        When d = 0 (flow matching samples), t can be sampled normally.
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        
        # First sample d
        d = self._generate_d(x1)
        
        # Then sample t based on d
        t = torch.zeros(batch_size, device=device)
        
        # For flow matching samples (d = 0), sample t normally
        flow_matching_mask = (d == 0)
        if flow_matching_mask.any():
            num_flow_samples = flow_matching_mask.sum().item()
            t_flow = self.training_time_sampler_fn(
                size=(num_flow_samples,),
                generator=self.random_generator,
            ).to(device)
            t[flow_matching_mask] = t_flow
        
        # For self-consistency samples (d > 0), sample t at multiples of d
        self_consistency_mask = (d > 0)
        if self_consistency_mask.any():
            d_nonzero = d[self_consistency_mask]
            # Sample how many multiples of d to use for each sample
            # We want t to be k*d where k is a random integer such that t ∈ [0, 1-d]
            # This ensures t + d ≤ 1
            max_multiples = torch.floor((1.0 - d_nonzero) / d_nonzero).long()
            # Ensure max_multiples is at least 0 to avoid empty range
            max_multiples = torch.clamp(max_multiples, min=0)
            
            # For each sample, randomly choose k from [0, max_multiples] - vectorized
            # Generate random values [0, 1) for all samples
            random_vals = torch.rand_like(d_nonzero)
            # Scale to [0, max_multiples + 1) and floor to get integers [0, max_multiples]
            k_values = torch.floor(random_vals * (max_multiples.float() + 1))
            # Compute t = k * d for all samples
            t_self_consistency = k_values * d_nonzero
            
            t[self_consistency_mask] = t_self_consistency
        
        return t, d

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        """Compute shortcut model loss with mixed flow matching and self-consistency objectives"""
        # t, d = self._generate_t_and_d(x1)
        # TODO: use equal samples for t and d across the batch
        t = self._generate_t(x1)
        d = self._generate_d(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)
        
        # Determine which samples use flow matching vs  self-consistency
        flow_matching_indices = (d == 0).nonzero(as_tuple=False).squeeze(-1)  # 75% of the time use d=0 (flow matching), 25% use self-consistency
        self_consistency_indices = (d > 0).nonzero(as_tuple=False).squeeze(-1)
        d[d == 0] = torch.rand_like(d[d == 0]) * self.fm_eps_max
        
        # Clear autocast cache for gradient computation
        torch.clear_autocast_cache()
        
        # Get model prediction
        s = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=2 * d * self.time_scale,
            **kwargs_conditionals,
        )

        # Compute component losses separately by selecting relevant indices
        flow_matching_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)
        self_consistency_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)
        
        # Flow matching component (for d=0 samples)
        if len(flow_matching_indices) > 0:
            # Select samples where d=0 and compute flow matching target only for these samples
            x0_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x0)
            x1_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x1)
            s_flow = tree_tensor_map(lambda x: x[flow_matching_indices], s)
            
            # Compute flow matching target only for selected samples
            flow_matching_target = self._generate_target(x0_flow, x1_flow)
            
            flow_matching_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_flow,
                flow_matching_target,
            )
            flow_matching_loss_val = sum(optree.tree_flatten(flow_matching_loss)[0])
        
        # Shortcut self-consistency component (for d>0 samples)
        if len(self_consistency_indices) > 0:
            # Select samples where d>0 and compute self-consistency target only for these samples
            x_t_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], x_t)
            t_shortcut = t[self_consistency_indices]
            d_shortcut = d[self_consistency_indices]
            s_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], s)
            
            # Create conditional arguments for selected samples
            if self.batch_mode:
                args_conditionals_shortcut = args_conditionals
                kwargs_conditionals_shortcut = kwargs_conditionals
            else:
                args_conditionals_shortcut = tuple(
                    tree_tensor_map(lambda x: x[self_consistency_indices], arg) if torch.is_tensor(arg) else arg
                    for arg in args_conditionals
                )
                kwargs_conditionals_shortcut = {
                    k: (tree_tensor_map(lambda x: x[self_consistency_indices], v) if torch.is_tensor(v) else v)
                    for k, v in kwargs_conditionals.items()
                }
            
            # Compute self-consistency target only for selected samples
            self_consistency_target = self.compute_self_consistency_target(
                x_t_shortcut, t_shortcut, d_shortcut, 
                *args_conditionals_shortcut, **kwargs_conditionals_shortcut
            )
            
            self_consistency_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_shortcut,
                self_consistency_target,
            )
            self_consistency_loss_val = sum(optree.tree_flatten(self_consistency_loss)[0])
        
        # Total loss is the sum of both components (linear combination)
        total_loss = flow_matching_loss_val + self.shortcut_loss_weight * self_consistency_loss_val
        
        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": flow_matching_loss_val,
            "self_consistency_loss": self_consistency_loss_val,
        }
        return total_loss, detail_losses
        
    def _prepare_t_and_d(self, steps=None):
        """Prepare time sequence and step size for inference"""
        steps = self.inference_steps if steps is None else steps
        t_seq = np.linspace(0, 1, steps + 1)

        if self.no_shortcut:
            d = 0
        else:
            # Use uniform step size for inference
            d = 1 / steps

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq, d

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()
        
        global obj_count
        obj_count += 1

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        t_copy = t

        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d_pred = torch.tensor(
            [(1 - t_copy) * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )

        pointmap = kwargs_conditionals.pop("pointmap", None)
        mask = kwargs_conditionals.pop("mask", None)
        ss_decoder = kwargs_conditionals.pop("ss_decoder", None)
        
        if pointmap is None:
            ret = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
        else:
            velocity = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals) 

            # TODO: measure GPU time from here
            pred_latent = tree_tensor_map(
                lambda x, v: x + v * (1 - t_copy), x_t, velocity
            ) # consistency model 이어서 정확하지 않음

            ss = ss_decoder(
                pred_latent['shape'].permute(0, 2, 1)
                .contiguous()
                .view(pred_latent['shape'].shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]]
            ss_pcs = []
            
            for i in range(pred_latent['shape'].shape[0]):
                ss_pc = coords[coords[:, 0] == i][:, 1:].float()
                ss_pc = ss_pc / 64.0 - 0.5  # normalize to [-0.5, 0.5]
                
                scale = pred_latent['scale'][i]        # (1, 3)
                rotation = pred_latent['6drotation_normalized'][i]  # (1, 6) 6D rotation
                translation = pred_latent['translation'][i] # (1, 3)

                scale = torch.exp(scale)

                scale = scale.mean(dim=-1, keepdim=True).expand_as(scale)
                R = rotation_6d_to_matrix(rotation)[0]  # (3, 3)

                ss_pc = (ss_pc * scale) @ R.T + translation  # Apply scale, rotation, translation

                ss_pcs.append(ss_pc)

            # (M, 3, H, W) -> (M, HW, 3)
            #pointmap_pc = pointmap.permute(0, 2, 3, 1).reshape(pointmap.shape[0], -1, 3)
            
            pointmap_pc = []
            for i in range(pointmap.shape[0]):
                pm = pointmap[i].permute(1, 2, 0).reshape(-1, 3)  # (HW, 3)
                msk = mask[i].permute(1, 2, 0).reshape(-1)        # (HW,)
                pm = pm[msk > 0.5]                                # Apply mask
                pointmap_pc.append(pm)
            
            # save the first batch's point clouds for debugging
            global obj_count
            save_multiple_pcs(
                [
                    ss_pcs[0].cpu().numpy(),
                    pointmap_pc[0].cpu().numpy()
                ],
                f"../debug/pc_readout/{obj_count}/{t_copy:.4f}.ply",
                colors=[
                    (255, 0, 0, 255),
                    (0, 255, 0, 255)
                ]
            )

            # ICP
            # if t_copy > 0.5:
            if False:
                for i in range(len(ss_pcs)):
                    # R_icp, t_icp, s_icp = estimate_rigid_transform_ransac_icp(ss_pcs[i], pointmap_pc[i])
                    R_icp, t_icp = estimate_rigid_transform_ransac_icp(pointmap_pc[i], ss_pcs[i])
                    R_icp = torch.tensor(R_icp, device=_get_device(x_t), dtype=torch.float32)
                    R_icp = R_icp.T  # inverse
                    
                    t_icp = torch.tensor(t_icp, device=_get_device(x_t), dtype=torch.float32)
                    t_icp = - (R_icp @ t_icp.unsqueeze(-1)).squeeze(-1)

                    rotation = R_icp @ rotation_6d_to_matrix(pred_latent['6drotation_normalized'][i])[0]
                    translation = pred_latent['translation'][i] @ R_icp.T + t_icp

                    pred_latent['6drotation_normalized'][i] = matrix_to_rotation_6d(rotation)
                    pred_latent['translation'][i] = translation

                ret = tree_tensor_map(
                    lambda x, pred: (pred - x) / (1 - t_copy), x_t, pred_latent
                )
            else:
                ret = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
            # TODO: measure GPU time until here
            
        kwargs_conditionals["pointmap"] = pointmap
        kwargs_conditionals["mask"] = mask
        kwargs_conditionals["ss_decoder"] = ss_decoder
        
        return ret
