#!/bin/bash

################################################################################
# Generate Self-Correction Training Data
# 
# This script calls API to generate multi-turn self-correction trajectories
# for fine-tuning the interactive code agent.
################################################################################

# API Configuration
API_KEY="your-api-key-here"  # Replace with your actual API key
BASE_URL="https://api.x.ai/v1"  # For Grok API
MODEL="grok-3-fast"  # or grok-3, gpt-4, etc.

# Generation Settings
N_SAMPLES=10000  # Number of samples to generate
OUTPUT_DIR="data/agent_sft"
OUTPUT_FILE="${OUTPUT_DIR}/correction_trajectories.json"

################################################################################
# Run generation

echo "=== Self-Correction Data Generation ==="
echo "API: ${BASE_URL}"
echo "Model: ${MODEL}"
echo "Samples: ${N_SAMPLES}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the generation script
python -m data_collection.generate_correction_data \
    --api-key "${API_KEY}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --n-samples ${N_SAMPLES} \
    --output "${OUTPUT_FILE}"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Generation Complete ==="
    echo "Output file: ${OUTPUT_FILE}"
    
    # Convert to ChatML format for training
    echo ""
    echo "Converting to ChatML format..."
    python -c "
from data_collection.generate_correction_data import convert_to_sft_format
convert_to_sft_format(
    '${OUTPUT_FILE}',
    '${OUTPUT_DIR}/correction_trajectories_chatml.json',
    'chatml'
)
"
    echo "Done!"
else
    echo "Generation failed!"
    exit 1
fi

