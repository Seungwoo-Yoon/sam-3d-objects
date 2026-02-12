#!/bin/bash

# Training script for ShortCut Scene-Level Model
# This script provides convenient presets for common training configurations

set -e  # Exit on error

# Default parameters
DATA_ROOT=""
OUTPUT_DIR="./outputs/shortcut_scene"
NUM_GPUS=1
NUM_EPOCHS=100
LEARNING_RATE=1e-4
BATCH_SIZE=1  # Always 1 for scene-level training
MAX_OBJECTS=32

# ShortCut parameters
SELF_CONSISTENCY_PROB=0.25
SHORTCUT_LOSS_WEIGHT=1.0
CFG_STRENGTH=3.0
RATIO_CFG_SAMPLES=0.5

# Training parameters
GRAD_CLIP=1.0
WEIGHT_DECAY=0.01
NUM_WORKERS=4
LOG_INTERVAL=10
SAVE_INTERVAL=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_objects)
            MAX_OBJECTS="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_root PATH          Path to training data (required)"
            echo "  --output_dir PATH         Output directory (default: ./outputs/shortcut_scene)"
            echo "  --num_gpus N              Number of GPUs (default: 1)"
            echo "  --num_epochs N            Number of epochs (default: 100)"
            echo "  --learning_rate LR        Learning rate (default: 1e-4)"
            echo "  --max_objects N           Max objects per scene (default: 32)"
            echo "  --preset NAME             Use preset configuration (small|medium|large)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Presets:"
            echo "  small    - Small model for quick experimentation"
            echo "  medium   - Medium model for balanced performance"
            echo "  large    - Large model for best quality (requires more GPU memory)"
            echo ""
            echo "Examples:"
            echo "  # Single GPU training"
            echo "  $0 --data_root /data/scenes --output_dir ./outputs/exp1"
            echo ""
            echo "  # Multi-GPU training with 4 GPUs"
            echo "  $0 --data_root /data/scenes --num_gpus 4 --preset large"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_ROOT" ]; then
    echo "Error: --data_root is required"
    echo "Use --help for usage information"
    exit 1
fi

# Apply preset configurations
case $PRESET in
    small)
        echo "Using 'small' preset configuration"
        MAX_OBJECTS=16
        LEARNING_RATE=2e-4
        ;;
    medium)
        echo "Using 'medium' preset configuration"
        MAX_OBJECTS=32
        LEARNING_RATE=1e-4
        ;;
    large)
        echo "Using 'large' preset configuration"
        MAX_OBJECTS=64
        LEARNING_RATE=5e-5
        WEIGHT_DECAY=0.05
        ;;
    "")
        echo "No preset specified, using default configuration"
        ;;
    *)
        echo "Error: Unknown preset '$PRESET'"
        echo "Valid presets: small, medium, large"
        exit 1
        ;;
esac

# Print configuration
echo "================================"
echo "ShortCut Scene-Level Training"
echo "================================"
echo "Data root:         $DATA_ROOT"
echo "Output dir:        $OUTPUT_DIR"
echo "Number of GPUs:    $NUM_GPUS"
echo "Number of epochs:  $NUM_EPOCHS"
echo "Learning rate:     $LEARNING_RATE"
echo "Max objects/scene: $MAX_OBJECTS"
echo "--------------------------------"
echo "ShortCut config:"
echo "  Self-consistency prob:  $SELF_CONSISTENCY_PROB"
echo "  Shortcut loss weight:   $SHORTCUT_LOSS_WEIGHT"
echo "  CFG strength:           $CFG_STRENGTH"
echo "  Ratio CFG samples:      $RATIO_CFG_SAMPLES"
echo "================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python train_shortcut_scene.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_objects_per_scene $MAX_OBJECTS \
    --self_consistency_prob $SELF_CONSISTENCY_PROB \
    --shortcut_loss_weight $SHORTCUT_LOSS_WEIGHT \
    --cfg_strength $CFG_STRENGTH \
    --ratio_cfg_samples $RATIO_CFG_SAMPLES \
    --grad_clip $GRAD_CLIP \
    --weight_decay $WEIGHT_DECAY \
    --num_workers $NUM_WORKERS \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL"

# Run training
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training..."
    echo "Command: $CMD"
    echo ""
    eval $CMD
else
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    echo "Command: torchrun --nproc_per_node=$NUM_GPUS $CMD"
    echo ""
    torchrun --nproc_per_node=$NUM_GPUS $CMD
fi

echo ""
echo "================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "================================"