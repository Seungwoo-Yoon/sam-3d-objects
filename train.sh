export CUDA_VISIBLE_DEVICES=0

python train_dual_backbone_foundationpose.py \
        --config checkpoints/hf/dual_backbone_generator.yaml \
        --data_root ./foundationpose \
        --output_dir ./outputs/dual_backbone_fp