export CUDA_VISIBLE_DEVICES=1

python train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root ./foundationpose \
        --output_dir ./outputs/dual_backbone_fp \
        --precomputed_latents \
        --gso_root ./gso/google_scanned_objects \
        --init_from_sparse_flow_checkpoint ./checkpoints/hf/ss_generator.ckpt \
        --validation_dirs ./notebook/images/segment1
        # --resume ./