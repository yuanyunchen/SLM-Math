#!/bin/bash

################################################################################
# QLoRA Training Script for Qwen2.5-Math-1.5B
# Memory-efficient training with 4-bit quantization
#
# QLoRA Benefits:
# - ~75% less GPU memory than full LoRA
# - Can use higher rank (r=64) for better performance
# - Requires CUDA GPU (not compatible with MPS/CPU)
################################################################################

set -e  # Exit on error

################################################################################
# Training Settings

# Model to train
MODEL="Qwen2.5-Math-1.5B"

# Run name (results will be saved to: results/sft_checkpoints/<RUN_NAME>_<MMDD_HHMM>/)
RUN_NAME="qwen25math_1.5b_cot_qlora"

# Training data
DATASET="gsm8k"
DATA_FILE="data/cot_generated/cot_x_ai_grok_4_1_fast_reasoning_gsm8k_train_7473_1120_2035/cot_data.json"

# Training duration
NUM_EPOCHS=3

# QLoRA mode (true/false)
USE_QLORA="true"

# Configuration file
# - sft_config_qlora.yaml: QLoRA configuration (recommended for limited GPU memory)
# - sft_config.yaml: Standard LoRA configuration
CONFIG_FILE="configs/sft_config_qlora.yaml"

################################################################################
# Advanced Settings (Optional)

# Override batch size (optional, comment out to use config default)
# BATCH_SIZE=2

# Override learning rate (optional, comment out to use config default)
# LEARNING_RATE=2e-4

# Override LoRA rank (optional, comment out to use config default)
# LORA_RANK=64

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                QLoRA Training - Qwen2.5-Math-1.5B                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if CUDA is available (QLoRA requirement)
if [ "$USE_QLORA" = "true" ]; then
    echo "Checking CUDA availability for QLoRA..."
    python -c "import torch; assert torch.cuda.is_available(), 'QLoRA requires CUDA GPU'; print('✓ CUDA available')" || {
        echo "✗ Error: QLoRA requires CUDA GPU"
        echo "  Your system: $(python -c 'import torch; print(\"MPS\" if torch.backends.mps.is_available() else \"CPU\")')"
        echo ""
        echo "Options:"
        echo "  1. Use regular LoRA instead (set USE_QLORA='false')"
        echo "  2. Run on a system with CUDA GPU"
        exit 1
    }
    echo ""
fi

# Check if model exists
MODEL_PATH="pretrained_models/$MODEL"
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    echo "Please download the model first."
    exit 1
fi
echo "✓ Model found: $MODEL_PATH"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "✗ Error: Data file not found: $DATA_FILE"
    exit 1
fi
echo "✓ Data file found: $DATA_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file: $CONFIG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Run name: $RUN_NAME"
echo "  Dataset: $DATASET"
echo "  Epochs: $NUM_EPOCHS"
echo "  QLoRA: $USE_QLORA"
echo "  Config: $CONFIG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

################################################################################
# Run Training

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Starting Training...                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Build command
CMD="python -m models.train_SFT --config $CONFIG_FILE"

# Add QLoRA flag if enabled
if [ "$USE_QLORA" = "true" ]; then
    CMD="$CMD --qlora"
fi

# Add optional overrides
if [ ! -z "$BATCH_SIZE" ]; then
    echo "Note: Batch size override not implemented in this version"
    echo "      Please modify the config file directly"
fi

# Run training
echo "Command: $CMD"
echo ""
$CMD

################################################################################
# Training Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Training Complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/sft_checkpoints/${RUN_NAME}_*/"
echo ""
echo "Next steps:"
echo "  1. View training logs:"
echo "     tensorboard --logdir=logs"
echo ""
echo "  2. Visualize training metrics:"
echo "     python -m utils.visualize_training --checkpoint_dir results/sft_checkpoints/${RUN_NAME}_*"
echo ""
echo "  3. Test the trained model:"
echo "     # (Add inference script command here)"
echo ""


