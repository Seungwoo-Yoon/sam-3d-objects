"""
Flow-GRPO-Fast RL finetuning of MidiSparseStructureFlowTdfyWrapper (LoRA).

Algorithm: Flow-GRPO-Fast (arxiv:2505.05470 + yifan123/flow_grpo), adapted for
3D sparse structure generation.

Flow-GRPO-Fast vs Flow-GRPO:
  Regular Flow-GRPO applies SDE sampling and GRPO training across ALL T_train steps.
  Flow-GRPO-Fast confines stochasticity to only T_sde steps (T_sde < T_train),
  randomly placed within the full T_train-step trajectory.
  The remaining (T_train − T_sde) steps use deterministic ODE sampling.
  This yields comparable reward with significantly less memory and compute.

Training overview:
  For each batch (one scene / one object conditioning):
    1. [Generation phase — ODE pre-branch, no_grad, single trajectory]
       Randomly sample branch_idx ∈ [0, T_train − T_sde].
       Run branch_idx deterministic ODE steps to reach x_branch:
         x_{t+dt} = x_t + v_θ(x_t, t) · dt       ← pure Euler ODE (no noise)

    2. [Generation phase — SDE window, no_grad, G trajectories]
       From x_branch, branch G diverse samples via T_sde SDE steps:
         x_{t+dt} = x_t
                  + [v_θ - σ_t²/(2(1-t)) · (x_t + t·v_θ)] · dt   ← drift
                  + σ_t · √dt · ε                                   ← diffusion
       where σ_t = a · √((1-t)/t)  (high at t≈0, zero at t=1)
       Store trajectories for the SDE window only.

    3. [Generation phase — ODE post-branch, no_grad, G trajectories]
       Continue each of the G trajectories with ODE for the remaining
       (T_train − branch_idx − T_sde) steps (not stored, only final x_1 used).

    4. [Reward phase, no_grad]
       Decode each x_1 through the frozen pipeline:
         shape_latent → ss_decoder → coords → slat_generator → slat_decoder → SDF
       Compute scalar reward r_g for each sample g.

    5. [Update phase, with grad for LoRA params]
       Group-relative advantage: A_g = (r_g − mean(r)) / (std(r) + ε)
       GRPO loss per SDE-window step t, per sample g:
         log_ratio   = [‖x_{t+dt} − μ_θ_old‖² − ‖x_{t+dt} − μ_θ_new‖²]
                       / (2 · σ_t² · dt)
         ratio       = exp(log_ratio)
         pg_term     = min(ratio·A, clip(ratio, 1−ε_clip, 1+ε_clip)·A)
         kl_term     = ‖v_θ_new − v_ref‖²  (simplified KL, closed-form proxy)
         L           = −pg_term + β·kl_term
       Backprop L, update LoRA adapters.

Only the LoRA adapters inside MidiSparseStructureFlowTdfyWrapper are trained.
ss_decoder, slat_generator, slat_decoder are frozen (eval mode, no grad).

Usage:
    python train_flow_grpo_foundationpose.py \\
        --config                      checkpoints/hf/midi_ss_generator.yaml \\
        --ss_generator_checkpoint     checkpoints/hf/ss_generator.ckpt \\
        --lora_checkpoint             ./outputs/midi_lora_fp/latest_peft \\
        --ss_decoder_config           checkpoints/hf/ss_decoder.yaml \\
        --ss_decoder_checkpoint       checkpoints/hf/ss_decoder.ckpt \\
        --slat_generator_config       checkpoints/hf/slat_generator.yaml \\
        --slat_generator_checkpoint   checkpoints/hf/slat_generator.ckpt \\
        --slat_decoder_mesh_config    checkpoints/hf/slat_decoder_mesh.yaml \\
        --slat_decoder_mesh_checkpoint checkpoints/hf/slat_decoder_mesh.ckpt \\
        --data_root /path/to/foundationpose_data \\
        --output_dir ./outputs/flow_grpo_fp

Model loading mirrors inference_pipeline.py exactly:
  - ss_decoder:              direct config + ckpt  (state_dict_key=None)
  - slat_generator:          config["module"]["generator"]["backbone"]
                             ckpt prefix "_base_models.generator."
  - slat_decoder_mesh:       direct config + ckpt  (state_dict_key=None)
  - slat_condition_embedder: config["module"]["condition_embedder"]["backbone"]
                             ckpt prefix "_base_models.condition_embedder."
                             (from slat_generator_checkpoint)
  - slat_preprocessor:       config["tdfy"]["val_preprocessor"]
                             (from slat_generator_config)
"""

import os

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from peft.peft_model import PeftModel

from foundation_pose_dataset import FoundationPoseDataset, collate_fn
from sam3d_objects.utils.dist_utils import setup_dist, unwrap_dist

# Reuse helpers from LoRA training script
from train_midi_lora_foundationpose import (
    build_model_from_config,
    load_ss_generator_checkpoint,
    apply_lora,
    freeze_condition_embedder,
    freeze_non_lora_params,
    get_parameter_groups,
    save_checkpoint,
    load_checkpoint,
)
from train_dual_backbone_foundationpose import setup_logger

from flow_grpo import (
    load_ss_decoder,
    load_slat_generator,
    load_slat_decoder_mesh,
    load_condition_embedder,
    load_slat_preprocessor,
    quantize_backbone_4bit,
    enable_gradient_checkpointing,
    train_epoch_grpo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(args):
    # -------------------------------------------------------------------------
    # Distributed setup
    # -------------------------------------------------------------------------
    is_distributed = "RANK" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_dist(
            rank=rank,
            local_rank=local_rank,
            world_size=int(os.environ["WORLD_SIZE"]),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=os.environ.get("MASTER_PORT", "12355"),
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(output_dir / "grpo_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        shutil.copy2(args.config, output_dir / "model_config.yaml")

    exp_logger = setup_logger(
        logger_type=args.logger,
        output_dir=output_dir,
        config=vars(args),
        rank=rank,
    )

    # -------------------------------------------------------------------------
    # Build trainable SS generator model + LoRA  (optionally QLoRA)
    # -------------------------------------------------------------------------
    logger.info("Building trainable ss_generator with %s...",
                "QLoRA" if args.qlora else "LoRA")
    model, ss_preprocessor = build_model_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)

    if args.ss_generator_checkpoint:
        load_ss_generator_checkpoint(
            checkpoint_path=args.ss_generator_checkpoint,
            model=model,
            device=device,
        )

    if args.freeze_embedder:
        freeze_condition_embedder(model)

    if args.lora_checkpoint:
        # Load LoRA config (rank/alpha/target_modules) AND weights from checkpoint.
        # PeftModel.from_pretrained reads adapter_config.json automatically,
        # so --lora_rank / --lora_alpha are ignored when a checkpoint is given.
        logger.info(f"Loading pretrained LoRA from {args.lora_checkpoint}")
        raw_model = unwrap_dist(model)
        backbone = raw_model.reverse_fn.backbone
        peft_backbone = PeftModel.from_pretrained(backbone, args.lora_checkpoint)
        raw_model.reverse_fn.backbone = peft_backbone
        logger.info("Loaded LoRA adapter (config + weights) via PeftModel.from_pretrained")
    else:
        apply_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # -------------------------------------------------------------------------
    # Reference backbone — deepcopy NOW (fp16, before any quantization).
    # This captures the exact starting-point policy for KL regularisation.
    # -------------------------------------------------------------------------
    logger.info("Creating frozen reference backbone...")
    raw_backbone = peft_backbone if args.lora_checkpoint else unwrap_dist(model).reverse_fn.backbone
    ref_backbone = copy.deepcopy(raw_backbone)
    ref_backbone.eval()
    for p in ref_backbone.parameters():
        p.requires_grad_(False)
    logger.info("Reference backbone created and frozen (fp16).")

    # -------------------------------------------------------------------------
    # QLoRA: quantize training backbone base weights to 4-bit AFTER ref copy.
    # LoRA adapters (lora_A / lora_B) remain in fp16/bf16.
    # -------------------------------------------------------------------------
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        logger.info("Applying QLoRA %d-bit quantization to training backbone...", args.qlora_bits)
        train_backbone = unwrap_dist(model).reverse_fn.backbone
        quantize_backbone_4bit(
            train_backbone,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            double_quant=True,
        )
        # Move back to GPU to materialise the Params4bit buffers, then prepare
        train_backbone.to(device)
        prepare_model_for_kbit_training(train_backbone, use_gradient_checkpointing=False)
        logger.info("QLoRA quantization complete.")

        model.reverse_fn.backbone = train_backbone

    freeze_non_lora_params(model)

    # -------------------------------------------------------------------------
    # Gradient checkpointing (reduces peak activation memory during backward)
    # -------------------------------------------------------------------------
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(unwrap_dist(model))

    # -------------------------------------------------------------------------
    # Load frozen downstream models
    # Mirrors inference_pipeline.py's init_* methods exactly.
    # -------------------------------------------------------------------------
    logger.info("Loading frozen downstream models...")

    # ss_decoder: direct checkpoint, no prefix stripping (state_dict_key=None)
    ss_decoder = load_ss_decoder(
        args.ss_decoder_config, args.ss_decoder_checkpoint, device
    )

    # slat_generator: config["module"]["generator"]["backbone"] + prefix "_base_models.generator."
    slat_generator = load_slat_generator(
        args.slat_generator_config, args.slat_generator_checkpoint, device
    )

    # slat_decoder_mesh: direct checkpoint, no prefix stripping
    slat_decoder_mesh = load_slat_decoder_mesh(
        args.slat_decoder_mesh_config, args.slat_decoder_mesh_checkpoint, device
    )

    # slat_condition_embedder: config["module"]["condition_embedder"]["backbone"]
    #   + prefix "_base_models.condition_embedder."  (from slat generator ckpt)
    logger.info("Loading SLAT condition embedder...")
    slat_condition_embedder = load_condition_embedder(
        args.slat_generator_config, args.slat_generator_checkpoint, device
    )

    # slat_preprocessor: config["tdfy"]["val_preprocessor"]  (from slat generator config)
    logger.info("Loading SLAT preprocessor...")
    slat_preprocessor = load_slat_preprocessor(args.pipeline_config)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    logger.info("Creating dataset...")
    train_dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=args.max_objects_per_scene,
        load_images=True,
        load_depth=True,
        load_masks=True,
        image_size=(args.image_height, args.image_width) if args.image_width > 0 else None,
        precomputed_latents=False,   # GRPO needs raw images for decoding
        num_renders_per_scene=args.num_renders_per_scene,
        gso_root=args.gso_root or None,
        load_meshes=args.load_meshes,
    )
    logger.info(f"Dataset size: {len(train_dataset)} scenes")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------------------------------------------
    # DDP
    # -------------------------------------------------------------------------
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_params,
        )

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    param_groups = get_parameter_groups(model, args.learning_rate, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_reward = -float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_reward = ckpt.get("metrics", {}).get("reward_mean", -float("inf"))

        raw_model = unwrap_dist(model)
        current_backbone = raw_model.reverse_fn.backbone
        # Unwrap existing PEFT wrapper (from lora_checkpoint) to get the base model
        base_backbone = current_backbone.base_model.model if isinstance(current_backbone, PeftModel) else current_backbone
        peft_backbone = PeftModel.from_pretrained(base_backbone, args.resume.replace(".pt", "_peft"), device=device)
        raw_model.reverse_fn.backbone = peft_backbone

        # Recreate optimizer so it tracks the new PEFT backbone's lora params
        param_groups = get_parameter_groups(model, args.learning_rate, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    assert args.t_sde_steps < args.t_train_steps, (
        f"--t_sde_steps ({args.t_sde_steps}) must be strictly less than "
        f"--t_train_steps ({args.t_train_steps})"
    )

    logger.info(
        f"Starting Flow-GRPO-Fast: {args.num_epochs} epochs, "
        f"G={args.group_size}, T_train={args.t_train_steps}, T_sde={args.t_sde_steps}, "
        f"sde_a={args.sde_a}, kl_coeff={args.kl_coeff}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_metrics, best_reward = train_epoch_grpo(
            model=model,
            ref_backbone=ref_backbone,
            ss_decoder=ss_decoder,
            slat_generator=slat_generator,
            slat_decoder_mesh=slat_decoder_mesh,
            slat_preprocessor=slat_preprocessor,
            slat_condition_embedder=slat_condition_embedder,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            ss_preprocessor=ss_preprocessor,
            G=args.group_size,
            T_train=args.t_train_steps,
            T_sde=args.t_sde_steps,
            sde_a=args.sde_a,
            clip_epsilon=args.clip_epsilon,
            kl_coeff=args.kl_coeff,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            scheduler=scheduler,
            exp_logger=exp_logger if rank == 0 else None,
            global_step=global_step,
            save_interval_steps=args.save_interval_steps,
            output_dir=output_dir,
            is_distributed=is_distributed,
            best_reward=best_reward,
            cfg_strength=args.cfg_strength,
        )
        global_step += len(train_loader)

        if rank == 0:
            from train_dual_backbone_foundationpose import log_metrics
            logger.info(
                f"Epoch {epoch}: loss={epoch_metrics['loss']:.4f}, "
                f"pg={epoch_metrics['pg_loss']:.4f}, "
                f"kl={epoch_metrics['kl_loss']:.4f}, "
                f"reward={epoch_metrics['reward_mean']:.4f}"
            )
            if exp_logger is not None:
                log_metrics(exp_logger, {
                    "epoch/loss": epoch_metrics["loss"],
                    "epoch/reward_mean": epoch_metrics["reward_mean"],
                }, global_step)

            save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step, epoch_metrics,
                str(output_dir / "latest.pt"),
                is_distributed,
            )

        if args.save_interval_epochs > 0 and (epoch + 1) % args.save_interval_epochs == 0:
            if rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, epoch_metrics,
                    str(output_dir / f"epoch_{epoch:04d}.pt"),
                    is_distributed,
                )

    logger.info("Flow-GRPO training completed.")
    if rank == 0 and exp_logger is not None:
        if hasattr(exp_logger, "finish"):
            exp_logger.finish()
        elif hasattr(exp_logger, "close"):
            exp_logger.close()

    if is_distributed:
        dist.destroy_process_group()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flow-GRPO RL finetuning of MidiSparseStructureFlowTdfyWrapper (LoRA)"
    )

    # ---- SS generator (trainable) ----
    parser.add_argument("--config", type=str, required=True,
                        help="SS generator config YAML")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--ss_generator_checkpoint", type=str, default=None,
                        help="Pretrained ss_generator.ckpt (base weights)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to a PEFT adapter directory from train_midi_lora "
                             "(e.g. ./outputs/midi_lora/latest_peft)")

    # ---- LoRA / QLoRA ----
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--qlora", action="store_true", default=False,
                        help="Enable QLoRA: quantize base model weights to 4-bit (NF4) "
                             "via bitsandbytes. LoRA adapters remain in fp16. "
                             "Requires: pip install bitsandbytes>=0.41")
    parser.add_argument("--qlora_bits", type=int, default=4, choices=[4],
                        help="Quantization bit-width for QLoRA (currently only 4 supported)")

    # ---- Frozen downstream models ----
    parser.add_argument("--ss_decoder_config", type=str, required=True)
    parser.add_argument("--ss_decoder_checkpoint", type=str, required=True)
    parser.add_argument("--slat_generator_config", type=str, required=True)
    parser.add_argument("--slat_generator_checkpoint", type=str, required=True)
    parser.add_argument("--slat_decoder_mesh_config", type=str, required=True)
    parser.add_argument("--slat_decoder_mesh_checkpoint", type=str, required=True)
    parser.add_argument("--pipeline_config", type=str, required=True)
    # NOTE: slat_preprocessor and slat_condition_embedder are loaded directly
    # from slat_generator_config/checkpoint – no separate pipeline_yaml needed.

    # ---- Data ----
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--gso_root", type=str, default=None)
    parser.add_argument("--max_objects_per_scene", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=0)
    parser.add_argument("--num_renders_per_scene", type=int, default=1)
    parser.add_argument("--load_meshes", action="store_true")

    # ---- Flow-GRPO-Fast hyperparameters ----
    parser.add_argument("--group_size", type=int, default=8,
                        help="Number of trajectories per input (G)")
    parser.add_argument("--t_train_steps", type=int, default=10,
                        help="Total denoising steps for the full trajectory (T_train)")
    parser.add_argument("--t_sde_steps", type=int, default=2,
                        help="Number of SDE steps randomly sampled for training (T_sde). "
                             "Must be strictly less than --t_train_steps. "
                             "Remaining steps use deterministic ODE sampling.")
    parser.add_argument("--sde_a", type=float, default=0.2,
                        help="SDE noise amplitude: σ_t = a·√((1−t)/t)")
    parser.add_argument("--kl_coeff", type=float, default=0.04,
                        help="KL divergence penalty coefficient (β)")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--cfg_strength", type=float, default=7.0,
                        help="CFG strength during SDE generation")

    # ---- Memory optimizations ----
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing on transformer blocks. "
                             "Reduces peak activation memory during backward at the cost "
                             "of ~2x more compute (activations are recomputed on backward).")

    # ---- Optimizer ----
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--freeze_embedder", action="store_true", default=True)
    parser.add_argument("--no_freeze_embedder", dest="freeze_embedder", action="store_false")
    parser.add_argument("--find_unused_params", action="store_true")

    # ---- Logging & checkpoints ----
    parser.add_argument("--logger", type=str, default="wandb",
                        choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval_steps", type=int, default=50)
    parser.add_argument("--save_interval_epochs", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./outputs/flow_grpo_fp")

    # ---- Resume ----
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
