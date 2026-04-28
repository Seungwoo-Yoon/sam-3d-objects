python train_flow_grpo_foundationpose.py \
    --config            checkpoints/hf/midi_ss_generator.yaml \
    --ss_generator_checkpoint checkpoints/hf/ss_generator.ckpt \
    --ss_decoder_config checkpoints/hf/ss_decoder.yaml \
    --ss_decoder_checkpoint checkpoints/hf/ss_decoder.ckpt \
    --slat_generator_config  checkpoints/hf/slat_generator.yaml \
    --slat_generator_checkpoint checkpoints/hf/slat_generator.ckpt \
    --slat_decoder_mesh_config  checkpoints/hf/slat_decoder_mesh.yaml \
    --slat_decoder_mesh_checkpoint checkpoints/hf/slat_decoder_mesh.ckpt \
    --data_root ./foundationpose_test \
    --gso_root ./gso/google_scanned_objects/models_normalized \
    --output_dir ./outputs/flow_grpo_conditioned \
    --pipeline_config checkpoints/hf/pipeline.yaml \
    --group_size 2 \
    --t_train_steps 10 \
    --max_objects_per_scene 16 \
    --gradient_checkpointing \
    --num_epochs 1 \
    --warmup_ratio 0.02 \
    --learning_rate 1e-4 \
    --t_sde_steps 2 \
    --sde_a 0.4 \
    --kl_coeff 0.04 \
    --lora_checkpoint ./outputs/midi_sam3d_2/step_00006500_peft \
    # --resume ./outputs/flow_grpo_conditioned/step_00000250.pt \
    # --save_interval_steps 1 \
    # --qlora \