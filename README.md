# SLM-Math: Small Language Models for Mathematical Reasoning

A comprehensive framework for training and evaluating Small Language Models on mathematical reasoning tasks. Features Chain-of-Thought (CoT) supervised fine-tuning, multi-agent evaluation workflows, and support for GSM8K, MATH, and MATH-500 benchmarks.

**Project**: Columbia University COMS4705 NLP Final Project  
**Date**: November 2025

---

## Experimental Results

### Training Results

We evaluate different training configurations on the Qwen2.5-Math-1.5B base model:

| Configuration | GSM8K | MATH-500 |
|---------------|-------|----------|
| **Base (Qwen2.5-Math-1.5B)** | 70.0% (350/500) | 53.2% (266/500) |
| **SFT lr=1e-5** | 81.4% (407/500) | 65.4% (327/500) |
| **SFT lr=5e-5** | 81.6% (408/500) | 67.0% (335/500) |
| **LoRA r=16** | 80.0% (400/500) | 67.2% (336/500) |
| **LoRA r=32** | 80.4% (402/500) | 66.2% (331/500) |

**Key Findings**:
- SFT with lr=5e-5 achieves the best overall performance (+11.6% on GSM8K, +13.8% on MATH-500)
- LoRA is parameter-efficient with comparable performance (80.4% on GSM8K, 67.2% on MATH-500)
- MATH-500 shows larger improvements than GSM8K, indicating better generalization to complex problems

### Agent Evaluation Results

We compare 9 different agent strategies using Qwen2.5-Math-1.5B:

| Rank | Agent | GSM8K | MATH-500 |
|------|-------|-------|----------|
| 1 | **solver_checker_with_tools** | 81.4% | 49.8% |
| 2 | majority_vote | 70.2% | 54.8% |
| 3 | agent_with_python_tools | 72.6% | 45.2% |
| 4 | solver_checker_summarizer | 59.4% | 45.2% |
| 5 | plan_and_reflection | 48.0% | 48.2% |
| 6 | solver_checker_summarizer_chat | 59.0% | 36.8% |
| 7 | solver_checker_stateless | 43.4% | 45.4% |
| 8 | solver_checker_chat | 43.4% | 42.8% |
| 9 | solver_checker_trivial_chat | 46.8% | 28.6% |

**Key Findings**:
- solver_checker_with_tools achieves the highest GSM8K accuracy (81.4%) with Python code execution
- majority_vote shows the best MATH-500 performance (54.8%) through ensemble voting
- Code execution tools significantly improve performance on computational problems

---

## Quick Start

### 1. Setup Environment

```bash
conda create -y --name slm_math python=3.10
conda activate slm_math
pip install -r requirements.txt
```

### 2. Download Datasets and Models

Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
python dataset/download_data_and_models.py --hf_token YOUR_HF_TOKEN
```

### 3. Quick Test

```bash
conda activate slm_math
python -m evaluation.eval --model "Qwen2.5-Math-1.5B" --round "test" --dataset "gsm8k" --count 10 --mode standard
```

---

## Evaluation

### Base Model Evaluation

```bash
# GSM8K evaluation (100 samples)
python -m evaluation.eval \
    --model "Qwen2.5-Math-1.5B" \
    --round "test" \
    --dataset "gsm8k" \
    --count 100 \
    --mode standard

# Full dataset evaluation
python -m evaluation.eval \
    --model "Qwen2.5-Math-1.5B" \
    --round "full_eval" \
    --dataset "math500" \
    --count 0 \
    --mode standard
```

### Agent Evaluation

```bash
# Solver-Checker with Tools (best performing)
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "solver_checker_with_tools" \
    --round "test" \
    --dataset "gsm8k" \
    --count 100 \
    --max_iterations 5

# Majority Vote ensemble
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "majority_vote" \
    --round "test" \
    --dataset "math500" \
    --count 100 \
    --num_runs 5 \
    --temperature 0.7
```

### Evaluate Fine-tuned Checkpoints

```bash
# Evaluate LoRA checkpoint
python -m evaluation.eval \
    --model "Qwen2.5-Math-1.5B" \
    --checkpoint "checkpoints/lora_r16_ckpt298_80.0acc" \
    --round "lora_eval" \
    --dataset "gsm8k" \
    --count 500 \
    --mode standard

# Evaluate SFT checkpoint
python -m evaluation.eval \
    --model "Qwen2.5-Math-1.5B" \
    --checkpoint "checkpoints/sft_full_lr5e5_ckpt298_81.6acc" \
    --round "sft_eval" \
    --dataset "math500" \
    --count 0 \
    --mode standard
```

**Output Structure**: `results/<round>_<model>_<dataset>_<count>_<MMDD>/`
- `log/` - Detailed execution logs
- `answers/` - Predictions JSON
- `metrics.csv` - Structured metrics
- `analysis_report.txt` - Summary report

---

## Training

### Supervised Fine-Tuning (SFT)

```bash
# Full fine-tuning
python -m models.train_SFT --config configs/sft_config.yaml

# LoRA fine-tuning
python -m models.train_SFT --config configs/sft_config.yaml

# QLoRA (4-bit quantization)
python -m models.train_SFT --config configs/sft_config_qlora.yaml --qlora
```

### Generate Chain-of-Thought Data

```bash
# Generate CoT data using GPT-4
python dataset/build_CoT_data.py \
    --dataset gsm8k \
    --output data/cot_generated/gsm8k_cot.json \
    --api_key YOUR_OPENAI_KEY
```

### Configuration Files

| Config | Description |
|--------|-------------|
| `configs/sft_config.yaml` | Full SFT configuration |
| `configs/sft_config_qlora.yaml` | QLoRA configuration |
| `configs/rl_config.yaml` | Reinforcement learning |
| `configs/distill_config.yaml` | Knowledge distillation |

---

## Available Agents

| Agent | Description | Best For |
|-------|-------------|----------|
| `solver_checker_with_tools` | Solver + Checker with Python execution | Computational problems |
| `majority_vote` | Ensemble voting over multiple runs | Robust answers |
| `agent_with_python_tools` | Single-shot with code execution | Simple calculations |
| `solver_checker_summarizer` | Solver + Checker + Summarizer | Complex reasoning |
| `plan_and_reflection` | Multi-phase planning agent | Multi-step problems |
| `solver_checker_chat` | Chat-based solver-checker | Iterative refinement |
| `solver_checker_stateless` | Stateless solver-checker | Independent verification |

---

## Supported Models

| Model | Parameters | Status |
|-------|------------|--------|
| Qwen2.5-Math-1.5B | 1.5B | Primary |
| Qwen2.5-Math-1.5B-Instruct | 1.5B | Supported |
| Qwen3-0.6B | 0.6B | Lightweight |
| Qwen3-1.7B | 1.7B | Supported |
| Qwen3-4B-Thinking-2507 | 4B | Advanced |

---

## Supported Datasets

| Dataset | Description | Test Samples |
|---------|-------------|--------------|
| **GSM8K** | Grade school math problems | 1,319 |
| **MATH** | Competition math problems | ~5,000 |
| **MATH-500** | Curated subset of MATH | 500 |

---

## Project Structure

```
SLM-Math/
├── agent/                      # Agent workflows
│   ├── solver_checker_*.py     # Solver-checker variants
│   ├── plan_and_reflection.py  # Planning agent
│   ├── majority_vote.py        # Ensemble voting
│   └── unified_config.py       # Unified generation configs
├── configs/                    # Training configurations
│   ├── sft_config.yaml
│   ├── sft_config_qlora.yaml
│   └── rl_config.yaml
├── data/                       # Datasets (excluded from git)
│   ├── gsm8k/
│   ├── math/
│   ├── math500/
│   └── cot_generated/          # Generated CoT data
├── dataset/                    # Data utilities
│   ├── dataloader.py
│   ├── build_CoT_data.py
│   └── download_data_and_models.py
├── evaluation/                 # Evaluation scripts
│   ├── eval.py                 # Base evaluation
│   ├── eval_agent.py           # Agent evaluation
│   └── inference_adapter.py
├── models/                     # Model utilities
│   ├── inference.py            # Model loading & generation
│   ├── inference_engine.py     # Unified inference engine
│   ├── train_SFT.py            # SFT training
│   └── train_RL.py             # RL training
├── pretrained_models/          # Model checkpoints (excluded from git)
├── checkpoints/                # Training checkpoints
├── results/                    # Evaluation results (excluded from git)
├── scripts/                    # Shell scripts
│   └── template/               # Script templates
├── utils/                      # Utility functions
│   ├── prompt_utils.py         # Prompts & answer extraction
│   ├── python_code_execution.py
│   └── train_utils.py
├── requirements.txt
└── README.md
```

---

## Evaluation Parameters

### Base Evaluation (`evaluation.eval`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name | Required |
| `--checkpoint` | Checkpoint path | None |
| `--round` | Test round identifier | Required |
| `--dataset` | Dataset name | Required |
| `--count` | Number of samples (0=all) | Required |
| `--mode` | Evaluation mode | `standard` |
| `--detailed` | Verbose output | `false` |
| `--batch_size` | Batch size | 1 |
| `--greedy` | Greedy decoding | `true` |

### Agent Evaluation (`evaluation.eval_agent`)

All base parameters plus:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--agent` | Agent method | Required |
| `--max_iterations` | Max iterations | 5 |
| `--num_runs` | Runs for majority_vote | 5 |
| `--temperature` | Sampling temperature | 0.7 |
| `--enable_solver_tools` | Enable solver tools | `true` |
| `--enable_checker_tools` | Enable checker tools | `true` |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for GPU training)
- 16GB+ GPU memory (for training)
- 8GB+ GPU memory (for inference)

See `requirements.txt` for complete dependencies.

---

## Troubleshooting

**Model not found**
- Ensure model is downloaded to `pretrained_models/<model_name>/`
- Check for `config.json` and `model.safetensors`

**CUDA out of memory**
- Reduce batch size
- Enable gradient checkpointing
- Use LoRA/QLoRA for training

**Dataset not found**
- Run download script: `python dataset/download_data_and_models.py`
- Verify dataset in `data/<dataset_name>/`

**Import errors**
- Activate conda environment: `conda activate slm_math`
- Reinstall: `pip install -r requirements.txt --upgrade`

---

## License

This project is for academic research purposes. Model weights are subject to their respective licenses (Qwen models under Apache 2.0).
