# Monday Night Experiments (2024-11-24)

W&B Project: `1124_monday_night_experiments`

## Experiments Overview

| GPU | Script | Experiments |
|-----|--------|-------------|
| 0 | `gpu0_sft_two_lr.sh` | Full SFT with 2 learning rates |
| 1 | `gpu1_lora_two_ranks.sh` | LoRA with rank 16 and 32 |
| 2 | `gpu2_rl_experiments.sh` | RL from base + LoRA then RL |

## Common Settings

- **Model**: Qwen2.5-Math-1.5B
- **Data**: `data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json`
- **Eval Samples**: 500 per dataset (GSM8K + MATH500)
- **Save**: Final model only

---

## GPU 0: Full SFT Training

**Script**: `gpu0_sft_two_lr.sh`

| Run | Learning Rate | Epochs | W&B Run Name |
|-----|---------------|--------|--------------|
| 1 | 1e-5 (conservative) | 2 | gpu0_sft_lr1e5 |
| 2 | 5e-5 (aggressive) | 2 | gpu0_sft_lr5e5 |

**Settings**:
- Batch size: 16 x 4 = 64
- Eval: final only

---

## GPU 1: LoRA Training

**Script**: `gpu1_lora_two_ranks.sh`

| Run | LoRA Rank | Epochs | W&B Run Name |
|-----|-----------|--------|--------------|
| 1 | 16 | 2 | gpu1_lora_r16 |
| 2 | 32 | 2 | gpu1_lora_r32 |

**Settings**:
- Batch size: 24 x 4 = 96
- Eval: final only

---

## GPU 2: RL Experiments

**Script**: `gpu2_rl_experiments.sh`

| Run | Description | W&B Run Name |
|-----|-------------|--------------|
| 1 | RL directly from base model | gpu2_rl_from_base |
| 2 | LoRA 16 (1 epoch) -> RL | gpu2_lora_r16_for_rl + gpu2_rl_on_lora |

**RL Settings**:
- Batch size: 4 x 4 = 16
- Learning rate: 5e-6
- Total steps: 400
- Eval every: 200 steps
- Save: step 400 only (final)
- KL penalty: disabled

---

## Usage

Run each script in a separate terminal/screen:

```bash
# Terminal 1 (GPU 0)
bash scripts/1124_monday_night_experiments/gpu0_sft_two_lr.sh

# Terminal 2 (GPU 1)
bash scripts/1124_monday_night_experiments/gpu1_lora_two_ranks.sh

# Terminal 3 (GPU 2)
bash scripts/1124_monday_night_experiments/gpu2_rl_experiments.sh
```

Or run all in background:

```bash
nohup bash scripts/1124_monday_night_experiments/gpu0_sft_two_lr.sh > logs/gpu0_sft.out 2>&1 &
nohup bash scripts/1124_monday_night_experiments/gpu1_lora_two_ranks.sh > logs/gpu1_lora.out 2>&1 &
nohup bash scripts/1124_monday_night_experiments/gpu2_rl_experiments.sh > logs/gpu2_rl.out 2>&1 &
```

