python train_flow_grpo_foundationpose.py \
    --config            checkpoints/hf/midi_ss_generator.yaml \
    --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
    --lora_checkpoint   ./outputs/midi_sam3d_2/step_00006500_peft \
    --ss_decoder_config checkpoints/hf/ss_decoder.yaml \
    --ss_decoder_checkpoint checkpoints/hf/ss_decoder.ckpt \
    --slat_generator_config  checkpoints/hf/slat_generator.yaml \
    --slat_generator_checkpoint checkpoints/hf/slat_generator.ckpt \
    --slat_decoder_mesh_config  checkpoints/hf/slat_decoder_mesh.yaml \
    --slat_decoder_mesh_checkpoint checkpoints/hf/slat_decoder_mesh.ckpt \
    --data_root ./foundationpose_test \
    --output_dir ./outputs/flow_grpo_pointmap \
    --pipeline_config checkpoints/hf/pipeline.yaml \
    --group_size 16 \
    --t_train_steps 10 \
    --qlora \
    --max_objects_per_scene 10 \
    --gradient_checkpointing \
    --num_epochs 1 \
    --warmup_ratio 0.0 \
    --learning_rate 3e-4 \
    --t_sde_steps 2 \
    --sde_a 0.4 \
    --kl_coeff 0.01 \
    # --resume ./outputs/flow_grpo_test/step_00000250.pt \
    # --save_interval_steps 1 \