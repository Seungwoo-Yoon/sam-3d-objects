from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.sam3d_conditioned import Sam3DConditionedMixin


class SSGeneratorTrainer(BasicTrainer):
    """
    Trainer for ShortCut (ss_generator) flow matching model.

    This trainer handles the specific training requirements for the ShortCut model,
    which combines flow matching with self-consistency objectives for faster inference.

    The ShortCut model implements the approach from https://arxiv.org/pdf/2410.12557
    where the model learns to predict larger steps during generation while maintaining
    consistency across different step sizes.

    Args:
        models (dict[str, nn.Module]): Models to train. Should contain 'generator' key.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def training_losses(
        self,
        x_1: Union[torch.Tensor, Dict[str, torch.Tensor]],
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for the ShortCut model.

        The ShortCut model combines two objectives:
        1. Flow matching loss: Standard flow matching objective for d=0 samples
        2. Self-consistency loss: Consistency across different step sizes for d>0 samples

        Args:
            x_1: The [N x C x ...] tensor or dict of tensors representing the clean data.
            cond: The conditioning information (e.g., image features, masks).
            kwargs: Additional arguments to pass to the model.

        Returns:
            terms: Dict containing loss values and metrics
            status: Dict containing status information
        """
        # Prepare conditioning
        cond = self.get_cond(cond, **kwargs)

        # Call the generator's loss function
        # The ShortCut model handles the flow matching + self-consistency internally
        total_loss, detail_losses = self.training_models['generator'](
            x_1,
            cond,
            **kwargs
        )

        # Build loss terms for logging
        terms = edict()
        terms["loss"] = total_loss

        # Add detailed losses for monitoring
        for key, value in detail_losses.items():
            if isinstance(value, torch.Tensor):
                terms[key] = value
            else:
                terms[key] = value

        # Compute additional metrics for logging
        status = edict()

        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run inference snapshot for visualization and evaluation.

        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing generated samples and ground truth
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # Collect samples
        sample_gt = []
        sample = []
        cond_vis = []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))

            # Move data to GPU
            data = {
                k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch]
                for k, v in data.items()
            }

            # Get ground truth
            if 'x_1' in data:
                x_1 = data['x_1']
                del data['x_1']
            elif 'latent' in data:
                x_1 = data['latent']
                del data['latent']
            else:
                raise ValueError("Dataset must provide 'x_1' or 'latent' key")

            sample_gt.append(x_1)

            # Visualize conditioning
            cond_vis.append(self.vis_cond(**data))

            # Prepare conditioning for inference
            cond = self.get_cond(data.get('cond'), **data)

            # Determine noise shape
            if isinstance(x_1, dict):
                x_shape = {k: v.shape for k, v in x_1.items()}
            else:
                x_shape = x_1.shape

            # Generate samples
            x_device = x_1.device if isinstance(x_1, torch.Tensor) else next(iter(x_1.values())).device

            # Use the generator's generate_iter method
            generation_results = []
            for t, x_t, _ in self.models['generator'].generate_iter(
                x_shape,
                x_device,
                cond,
                **data,
            ):
                # Store final result
                if t >= 1.0 - 1e-5:  # Final timestep
                    generation_results.append(x_t)

            if generation_results:
                sample.append(generation_results[-1])

        # Concatenate results
        if isinstance(sample_gt[0], dict):
            sample_gt_dict = {}
            sample_dict_result = {}
            for key in sample_gt[0].keys():
                sample_gt_dict[key] = torch.cat([s[key] for s in sample_gt], dim=0)
                if sample:
                    sample_dict_result[key] = torch.cat([s[key] for s in sample], dim=0)

            sample_dict = {
                'sample_gt': {'value': sample_gt_dict, 'type': 'sample'},
            }
            if sample:
                sample_dict['sample'] = {'value': sample_dict_result, 'type': 'sample'}
        else:
            sample_gt_tensor = torch.cat(sample_gt, dim=0)
            sample_dict = {
                'sample_gt': {'value': sample_gt_tensor, 'type': 'sample'},
            }
            if sample:
                sample_tensor = torch.cat(sample, dim=0)
                sample_dict['sample'] = {'value': sample_tensor, 'type': 'sample'}

        # Add conditioning visualizations
        if cond_vis and cond_vis[0]:
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: torch.cat(x, dim=0) if isinstance(x[0], torch.Tensor) else x,
                'type': lambda x: x[0],
            }))

        return sample_dict


class SSGeneratorCFGTrainer(ClassifierFreeGuidanceMixin, SSGeneratorTrainer):
    """
    Trainer for ShortCut flow matching model with classifier-free guidance.

    This trainer extends SSGeneratorTrainer with classifier-free guidance support,
    allowing the model to be trained with conditioning dropout and used with
    guidance strength during inference.

    Args:
        Same as SSGeneratorTrainer, plus:
        p_uncond (float): Probability of dropping conditions during training.
    """
    pass


class Sam3DConditionedSSGeneratorCFGTrainer(Sam3DConditionedMixin, SSGeneratorCFGTrainer):
    """
    Trainer for ShortCut flow matching model with SAM3D conditioning and CFG.

    This trainer combines:
    - ShortCut flow matching with self-consistency
    - Classifier-free guidance
    - SAM3D conditioning (image + mask based)

    This is the main trainer for the ss_generator model as configured in
    checkpoints/hf/ss_generator.yaml

    Args:
        Same as SSGeneratorCFGTrainer, plus SAM3D conditioning parameters.
    """
    pass
