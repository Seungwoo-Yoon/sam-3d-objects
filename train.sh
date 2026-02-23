export CUDA_VISIBLE_DEVICES=1

python train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root ./foundationpose \
        --output_dir ./outputs/joint_sam3d_5 \
        --precomputed_latents \
        --gso_root ./gso \
        --init_from_sparse_flow_checkpoint ./checkpoints/hf/ss_generator.ckpt \
        --warmup_ratio 0.005 \
        --save_interval_steps 400 \
        --num_epochs 1 \
        --learning_rate 1e-2 \
        # --validation_dirs ./notebook/images/segment1
        # --resume ./