#!/bin/bash

################################################################################
# SFT Training Script for Qwen2.5-Math Models
# Trains Chain-of-Thought reasoning models in DeepSeek-R1 style
################################################################################

################################################################################
# Training Configuration

# Base model to fine-tune
# Options: Qwen2.5-Math-1.5B, Qwen2.5-Math-1.5B-Instruct, etc.
MODEL="Qwen2.5-Math-1.5B"

# Training data file(s)
# Can be single file or space-separated list of files
DATA_FILES=(
    "data/cot_generated/cot_x_ai_grok_4_1_fast_reasoning_gsm8k_train_7473_1120_2035/cot_data.json"
    # Add more data files here if needed
    # "data/cot_generated/other_cot_data.json"
)

# Run name (used for organizing output directory)
# Results will be saved to: results/sft_checkpoints/<RUN_NAME>_<MMDD_HHMM>/
RUN_NAME="qwen25math_1.5b_cot_gsm8k"

# Configuration file path
CONFIG_FILE="configs/sft_config.yaml"

# Resume from checkpoint (optional, leave empty for new training)
# Set to checkpoint path to resume training
RESUME_CHECKPOINT=""

# Training mode
# Set to true to disable LoRA (use full fine-tuning instead)
NO_LORA=false

################################################################################
# Advanced Settings (optional overrides)

# Override batch size (if needed for your GPU memory)
# BATCH_SIZE=2

# Override learning rate
# LEARNING_RATE=2e-4

# Override number of epochs
# NUM_EPOCHS=3

# Override max sequence length
# MAX_SEQ_LENGTH=1536

################################################################################
# Environment Setup

# Set CUDA device(s) to use
# Single GPU: export CUDA_VISIBLE_DEVICES=0
# Multiple GPUs: export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# Set number of threads for CPU operations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

################################################################################
# Conda Environment

# Activate conda environment (if using)
# Uncomment and modify if you're using a specific conda environment
# conda activate slm_math_env

################################################################################
# Build Command

CMD="python -m models.train_SFT"
CMD="$CMD --config $CONFIG_FILE"

# Add model override if specified
if [ ! -z "$MODEL" ]; then
    CMD="$CMD --model_name $MODEL"
fi

# Add data files
if [ ${#DATA_FILES[@]} -gt 0 ]; then
    CMD="$CMD --data_files ${DATA_FILES[@]}"
fi

# Add resume checkpoint if specified
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# Add no-lora flag if specified
if [ "$NO_LORA" = true ]; then
    CMD="$CMD --no_lora"
fi

################################################################################
# Pre-flight Checks

echo "=============================================================================="
echo "SFT Training - Pre-flight Checks"
echo "=============================================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python or activate your environment."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Training will use CPU (very slow)."
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi
echo "Configuration file: $CONFIG_FILE"

# Check if data files exist
echo ""
echo "Data files:"
for DATA_FILE in "${DATA_FILES[@]}"; do
    if [ ! -f "$DATA_FILE" ]; then
        echo "  ERROR: Data file not found: $DATA_FILE"
        exit 1
    fi
    FILE_SIZE=$(du -h "$DATA_FILE" | cut -f1)
    echo "  ✓ $DATA_FILE ($FILE_SIZE)"
done

# Check if model directory exists
MODEL_DIR="pretrained_models/$MODEL"
if [ ! -d "$MODEL_DIR" ]; then
    echo ""
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Please download the model first."
    exit 1
fi
echo "Model directory: $MODEL_DIR"

echo ""
echo "Training configuration:"
echo "  Run name: $RUN_NAME"
echo "  Model: $MODEL"
echo "  LoRA enabled: $([ "$NO_LORA" = true ] && echo "NO (full fine-tuning)" || echo "YES")"
echo "  CUDA devices: $CUDA_VISIBLE_DEVICES"
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    echo "  Resume from: $RESUME_CHECKPOINT"
fi

echo ""
echo "=============================================================================="
echo "Starting Training"
echo "=============================================================================="
echo "Command: $CMD"
echo ""

################################################################################
# Run Training

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Generate log file name with timestamp
TIMESTAMP=$(date +"%m%d_%H%M")
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_${TIMESTAMP}.log"

# Run training with output to both console and log file
echo "Training log will be saved to: $LOG_FILE"
echo ""

# Run the command and capture exit code
$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

################################################################################
# Post-training Summary

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Completed Successfully!"
    echo "=============================================================================="
    echo ""
    echo "Outputs saved to: results/sft_checkpoints/${RUN_NAME}_*/"
    echo "Training log: $LOG_FILE"
    echo ""
    echo "To monitor training with TensorBoard:"
    echo "  tensorboard --logdir=logs"
    echo ""
    echo "To evaluate the trained model:"
    echo "  bash scripts/run_evaluation.sh"
else
    echo "Training Failed (Exit code: $EXIT_CODE)"
    echo "=============================================================================="
    echo ""
    echo "Check the log file for details: $LOG_FILE"
    echo ""
fi

exit $EXIT_CODE


