# Experiments

This folder contains experiment scripts for training SLM-Math models.

## Overview

| Script | GPU | Description |
|--------|-----|-------------|
| `exp_gpu0_sft_full.sh` | GPU 0 | Full SFT training (5 epochs) |
| `exp_gpu1_rl_baseline.sh` | GPU 1 | RL baseline with GRPO |
| `exp_gpu2_lora_then_rl.sh` | GPU 2 | LoRA SFT -> RL pipeline |

All experiments log to the same W&B project: `slm_math_experiments`

## Run All Experiments

To run all three experiments in parallel on different GPUs:

```bash
# Run all in background
cd /root/autodl-tmp/SLM-Math

# GPU 0: Full SFT
nohup bash scripts/monday_night_experiments/exp_gpu0_sft_full.sh > logs/exp_gpu0.log 2>&1 &

# GPU 1: RL Baseline
nohup bash scripts/monday_night_experiments/exp_gpu1_rl_baseline.sh > logs/exp_gpu1.log 2>&1 &

# GPU 2: LoRA -> RL
nohup bash scripts/monday_night_experiments/exp_gpu2_lora_then_rl.sh > logs/exp_gpu2.log 2>&1 &
```

Or run the master script:
```bash
bash scripts/monday_night_experiments/run_all.sh
```

## Experiment Details

### GPU 0: Full SFT (`exp_gpu0_sft_full.sh`)
- **Mode**: Full fine-tuning (no LoRA)
- **Epochs**: 5
- **Eval**: Every 1 epoch
- **Save**: Final model only
- **W&B Run**: `gpu0_sft_full_5ep`

### GPU 1: RL Baseline (`exp_gpu1_rl_baseline.sh`)
- **Mode**: GRPO RL with LoRA
- **Epochs**: 3
- **Log**: Every 10 steps
- **Eval**: Every 100 steps
- **Save**: Every 300 steps
- **W&B Run**: `gpu1_rl_grpo_baseline`

### GPU 2: LoRA -> RL (`exp_gpu2_lora_then_rl.sh`)
- **Phase 1**: LoRA SFT (5 epochs, eval every 1 epoch, save final model only)
- **Phase 2**: RL on LoRA checkpoint (3 epochs, save every 300 steps)
- **W&B Runs**: `gpu2_lora_sft_5ep`, `gpu2_rl_on_lora`

## Monitoring

```bash
# Watch experiment logs
tail -f logs/exp_gpu0.log
tail -f logs/exp_gpu1.log
tail -f logs/exp_gpu2.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## W&B Dashboard

All experiments are logged to: https://wandb.ai/<your-username>/slm_math_experiments

