# GRPO RL Training Guide

## Overview

This guide provides instructions for training Qwen2.5-Math-1.5B using GRPO (Group Relative Policy Optimization) with a rule-based binary reward verifier.

## Files Created

### 1. Core Training Files

- **`models/train_rl_base.py`**: Main GRPO training script
  - Implements custom GRPO trainer
  - Binary reward computation (correct/wrong)
  - LoRA for parameter-efficient training
  - Reference model for KL divergence

- **`utils/reward_utils.py`**: Reward computation utilities
  - Extract answers from `\boxed{}` format
  - Normalize answers for comparison
  - Compute binary rewards: +1 (correct), -1 (wrong), -0.5 (no answer)

- **`configs/rl_grpo_config.yaml`**: Full training configuration
  - Model and LoRA settings
  - GRPO hyperparameters
  - Data and training settings

- **`configs/rl_grpo_config_test.yaml`**: Test configuration (small-scale)

- **`scripts/train_rl_baseline.sh`**: Training launch script

## Requirements

### Software Dependencies

```bash
# Core dependencies (already in requirements.txt)
torch>=2.1.0
transformers==4.37.2
accelerate>=0.20.0
trl==0.7.10
peft==0.7.1
datasets>=2.0.0
pyyaml>=6.0
```

### Hardware Requirements

**Minimum Requirements:**
- GPU: 24GB VRAM (e.g., RTX 3090, RTX 4090)
- RAM: 32GB
- Storage: 50GB free space

**Recommended:**
- GPU: 40GB+ VRAM (e.g., A100, A6000)
- RAM: 64GB+
- Storage: 100GB free space

**Memory Usage Breakdown:**
- Base Model (Qwen2.5-Math-1.5B): ~3GB (fp16)
- Reference Model: ~3GB (fp16)
- LoRA Adapters: ~0.5GB
- Activations (batch=4): ~4-6GB
- **Total: ~10-13GB per GPU**

## Training Configuration

### Key Hyperparameters

```yaml
# Model
model_path: "<your_sft_checkpoint>"  # Replace with your SFT checkpoint

# LoRA Configuration
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# GRPO Parameters
num_return_sequences: 2  # Generate 2 responses per prompt
reward_correct: 1.0
reward_wrong: -1.0
reward_no_answer: -0.5
kl_coef: 0.05  # KL penalty coefficient

# Training
num_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # Effective batch size = 16
learning_rate: 5e-6

# Generation
max_new_tokens: 2048
temperature: 0.7
top_p: 0.9
```

### Dataset

- **File**: `data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json`
- **Size**: 18,946 samples (all correct)
- **Format**:
  ```json
  {
    "question": "...",
    "ground_truth": "42",
    "predicted_answer": "42",
    "correct": true,
    ...
  }
  ```

## Usage

### Step 1: Prepare Your SFT Checkpoint

Update the model path in the configuration file:

```yaml
# configs/rl_grpo_config.yaml
model:
  path: "checkpoints/your_sft_checkpoint"  # Replace this
```

### Step 2: Update the Launch Script

Edit `scripts/train_rl_baseline.sh`:

```bash
# Model checkpoint (SFT checkpoint)
MODEL_PATH="checkpoints/your_sft_checkpoint"  # Update this line
```

### Step 3: Run Training

```bash
# Full training
bash scripts/train_rl_baseline.sh

# Or run directly with Python
python models/train_rl_base.py \
    --config configs/rl_grpo_config.yaml \
    --model_path checkpoints/your_sft_checkpoint \
    --output_dir results/rl_grpo_$(date +%Y%m%d_%H%M%S)
```

### Step 4: Monitor Training

Training outputs:
- **Checkpoints**: `results/rl_grpo_checkpoints/checkpoint-*/`
- **Logs**: `results/rl_grpo_checkpoints/training.log`
- **Metrics**: Logged every 10 steps

Key metrics to watch:
- **reward**: Average reward (target: >0.7)
- **accuracy**: Percentage of correct answers (target: >70%)
- **kl_penalty**: KL divergence from reference model (target: <0.5)
- **loss**: Combined policy loss + KL penalty

## Testing

### Quick Test (Small Sample)

```bash
# Test with 10 samples
python models/train_rl_base.py \
    --config configs/rl_grpo_config_test.yaml \
    --max_samples 10
```

### Verify Reward Computation

```bash
# Run reward utils tests
python utils/reward_utils.py
```

Expected output:
```
Test 1 ✓
  Generated: The answer is \boxed{42}
  Truth: 42
  Predicted: 42
  Correct: True
  Reward: 1.0 (expected: 1.0)
...
```

## Training Time Estimation

With the default configuration:
- **Samples**: 18,946 × 2 generations = 37,892 training samples
- **Batch size**: 16 (4 × 4)
- **Steps per epoch**: ~2,370
- **Total steps**: ~7,110 (3 epochs)
- **Time per step**: ~2-3 seconds (on A100)
- **Total time**: **4-6 hours** (on A100 40GB)

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 2  # From 4
   gradient_accumulation_steps: 8  # From 4
   ```

2. Enable gradient checkpointing (already enabled by default):
   ```yaml
   gradient_checkpointing: true
   ```

3. Use smaller max_new_tokens:
   ```yaml
   max_new_tokens: 1024  # From 2048
   ```

4. Reduce num_return_sequences:
   ```yaml
   num_return_sequences: 2  # From 4
   ```

### Issue 2: Training is Slow

**Solutions**:
1. Increase batch size (if you have memory):
   ```yaml
   per_device_train_batch_size: 8
   ```

2. Use FP16 (if supported):
   ```yaml
   fp16: true
   ```

3. Reduce logging frequency:
   ```yaml
   logging_steps: 50  # From 10
   ```

### Issue 3: Model Not Learning (Reward Not Increasing)

**Check**:
1. Learning rate might be too low/high
2. KL coefficient might be too high (preventing exploration)
3. Data quality issues

**Solutions**:
1. Adjust learning rate:
   ```yaml
   learning_rate: 1e-5  # Try higher
   ```

2. Reduce KL penalty:
   ```yaml
   kl_coef: 0.01  # From 0.05
   ```

3. Check data distribution:
   ```python
   python -c "
   import json
   data = json.load(open('data/...json'))
   print(f'Correct: {sum(d[\"correct\"] for d in data)}')
   print(f'Total: {len(data)}')
   "
   ```

### Issue 4: Tokenizer/Model Loading Issues

If you encounter tokenizer errors, make sure you're using a valid checkpoint with all required files:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors` or `pytorch_model.bin`

## Expected Results

After 3 epochs of training:
- **Average Reward**: 0.7-0.9
- **Accuracy**: 70-90% (depending on initial SFT quality)
- **KL Divergence**: 0.1-0.3

The model should:
- Generate more accurate answers
- Follow the `\boxed{}` format more consistently
- Show improved reasoning quality

## Advanced Usage

### Custom Reward Function

Edit `utils/reward_utils.py` to customize reward computation:

```python
def compute_reward(
    generated_text: str,
    ground_truth: str,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_no_answer: float = -0.5
) -> Tuple[float, dict]:
    # Add your custom logic here
    # Example: partial credit for close answers
    ...
```

### Multi-GPU Training

```bash
# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Run with accelerate
accelerate launch --multi_gpu models/train_rl_base.py \
    --config configs/rl_grpo_config.yaml \
    --model_path checkpoints/your_sft_checkpoint
```

### Resume from Checkpoint

```bash
python models/train_rl_base.py \
    --config configs/rl_grpo_config.yaml \
    --model_path results/rl_grpo_checkpoints/checkpoint-1000
```

## Notes

- **GRPO vs PPO**: GRPO is a simplified version of PPO that doesn't require a value network, making it easier to implement and train.

- **Reference Model**: The reference model is a frozen copy of the initial SFT model, used to compute KL divergence and prevent the policy from deviating too far.

- **Binary Rewards**: We use simple binary rewards (+1/-1) based on answer correctness. This is sufficient for mathematical reasoning where answers are either correct or incorrect.

- **LoRA**: We use LoRA (Low-Rank Adaptation) for memory efficiency. Only ~0.28% of parameters are trained.

## Contact & Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review training logs: `results/rl_grpo_checkpoints/training.log`
3. Test reward computation: `python utils/reward_utils.py`
4. Validate configuration: Check all paths in YAML files

## References

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **TRL Library**: [Transformer Reinforcement Learning](https://github.com/huggingface/trl)
















