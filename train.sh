export CUDA_VISIBLE_DEVICES=1

# python train_dual_backbone_foundationpose.py \
#         --config checkpoints/hf/dual_backbone_generator.yaml \
#         --data_root ./foundationpose \
#         --output_dir ./outputs/joint_sam3d_5 \
#         --precomputed_latents \
#         --gso_root ./gso \
#         --init_from_sparse_flow_checkpoint ./checkpoints/hf/ss_generator.ckpt \
#         --warmup_ratio 0.005 \
#         --save_interval_steps 400 \
#         --num_epochs 1 \
#         --learning_rate 1e-2 \
#         # --validation_dirs ./notebook/images/segment1
#         # --resume ./

# Single GPU
python train_midi_lora_foundationpose.py \
        --config checkpoints/hf/midi_ss_generator.yaml \
        --data_root ./foundationpose \
        --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
        --precomputed_latents \
        --output_dir ./outputs/midi_sam3d_2 \
        --gso_root ./gso \
        --warmup_ratio 0.005 \
        --save_interval_steps 500 \
        --num_epochs 1 \
        --learning_rate 1e-4 \
        --resume ./outputs/midi_sam3d_2/step_00006500.pt \

# python train_midi_lora_foundationpose.py \
#         --config checkpoints/hf/midi_ss_generator.yaml \
#         --data_root ./foundationpose_test \
#         --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
#         --precomputed_latents \
#         --output_dir ./outputs/midi_sam3d_test \
#         --gso_root ./gso \
#         --warmup_ratio 0.005 \
#         --save_interval_steps 500 \
#         --num_epochs 1 \
#         --learning_rate 1e-4 \
#         --output_dir ./outputs/midi_sam3d_test \
#         --resume ./outputs/midi_sam3d_test/step_00000005.pt \