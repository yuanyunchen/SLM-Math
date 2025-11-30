# Scripts Directory

This directory contains scripts for training, evaluation, and data processing.

## Directory Structure

```
scripts/
├── archive/                          # Archived old scripts
├── template/                         # Script templates
│   ├── eval.sh                      # Standard evaluation template
│   ├── eval_agent.sh                # Agent evaluation template
│   ├── build_cot_data.sh            # Data building template
│   └── train.sh                     # Training template
├── eval_lora_checkpoint.sh          # Example: Evaluate LoRA checkpoint
├── eval_sft_checkpoint.sh           # Example: Evaluate SFT checkpoint
├── eval_agent_lora_checkpoint.sh    # Example: Agent eval with LoRA
├── compare_agents.py                # Compare agent performance
└── download_data_and_models.py      # Download datasets and models
```

## Quick Start

### 1. Evaluate a Checkpoint

**For LoRA checkpoint:**
```bash
bash scripts/eval_lora_checkpoint.sh
```

**For SFT checkpoint:**
```bash
bash scripts/eval_sft_checkpoint.sh
```

### 2. Create Your Own Evaluation Script

```bash
# Copy template
cp scripts/template/eval.sh scripts/my_eval.sh

# Edit the configuration variables
nano scripts/my_eval.sh
# Set: MODEL, CHECKPOINT, DATASET, etc.

# Run
bash scripts/my_eval.sh
```

### 3. Agent Evaluation

```bash
# Copy template
cp scripts/template/eval_agent.sh scripts/my_agent_eval.sh

# Edit configuration
nano scripts/my_agent_eval.sh

# Run
bash scripts/my_agent_eval.sh
```

## Available Templates

### eval.sh
Standard evaluation template for direct model inference.

**Configuration variables:**
- `MODEL`: Base model name
- `CHECKPOINT`: Path to checkpoint (optional)
- `DATASET`: Dataset name (gsm8k, math, math500)
- `COUNT`: Number of test cases
- `MODE`: Evaluation mode (standard)
- `DETAILED`: Output verbosity (true/false)

### eval_agent.sh
Agent evaluation template for multi-agent workflows.

**Configuration variables:**
- `MODEL`: Base model name
- `CHECKPOINT`: Path to checkpoint (optional)
- `AGENT`: Agent method (solver_checker, solver_checker_chat, majority_vote, etc.)
- `MAX_ITERATIONS`: Maximum iterations for iterative agents
- `DATASET`: Dataset name
- `COUNT`: Number of test cases

## Example Scripts

### eval_lora_checkpoint.sh
Ready-to-use example for evaluating LoRA checkpoints. Uses:
- Base model: Qwen2.5-Math-1.5B
- LoRA checkpoint: checkpoints/lora_r16_20251124_130958/checkpoint-1188
- Dataset: GSM8K (100 samples)

### eval_sft_checkpoint.sh
Ready-to-use example for evaluating SFT checkpoints. Uses:
- SFT checkpoint: checkpoints/sft_20251124_131423/checkpoint-891
- Dataset: GSM8K (100 samples)

### eval_agent_lora_checkpoint.sh
Ready-to-use example for agent evaluation with LoRA checkpoint. Uses:
- Base model + LoRA adapter
- Agent: solver_checker_chat
- Dataset: GSM8K (10 samples)

## Python Utilities

### compare_agents.py
Compare performance across different agent methods.

```bash
python scripts/compare_agents.py \
    --results_dirs results/round1_* results/round2_* \
    --output comparison_report.txt
```

### download_data_and_models.py
Download datasets and pretrained models.

```bash
python scripts/download_data_and_models.py --dataset gsm8k --model Qwen2.5-Math-1.5B
```

## Checkpoint Support

All evaluation scripts now support checkpoint loading:

**LoRA Checkpoints:**
- Requires base model in `pretrained_models/`
- Adapter loaded on top of base model
- Path: `checkpoints/lora_*/checkpoint-*/`

**SFT Checkpoints:**
- Self-contained full model
- No base model needed
- Path: `checkpoints/sft_*/checkpoint-*/`

See [CHECKPOINT_EVALUATION.md](../docs/CHECKPOINT_EVALUATION.md) for details.

## Archived Scripts

Old scripts have been moved to `archive/` directory:
- GPU-specific evaluation scripts (eval_gpu*.sh)
- Batch runner scripts (run_*.sh)
- Old training scripts (train_*.sh)

These are kept for reference but are no longer maintained.

## Documentation

- [Checkpoint Evaluation Guide](../docs/CHECKPOINT_EVALUATION.md)
- [Checkpoint Quick Start](../docs/CHECKPOINT_QUICK_START.md)
- [Training Guide](../models/train_sft_baseline.py)

## Tips

1. **Start with examples**: Use the provided example scripts and modify them
2. **Use templates**: Copy templates for new evaluation tasks
3. **Check documentation**: See docs/ for detailed guides
4. **Test small first**: Use small COUNT (e.g., 10) to test before full evaluation
5. **Check results**: Results saved to `results/` directory

## Need Help?

- See `docs/CHECKPOINT_QUICK_START.md` for quick commands
- Check `archive/README.md` for old script migration guide
- Review example scripts for common usage patterns

