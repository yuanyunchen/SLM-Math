# SLM-Math: Empowering Small Language Models for Mathematical Reasoning

A comprehensive framework for enhancing small language models (1.5B parameters) on mathematical reasoning through supervised fine-tuning, reinforcement learning, and agentic workflows.

**Columbia University COMS4705 NLP Final Project**  
**Team**: Roger Wang, Jinzi Luo, Yunchen Yuan  
**Date**: December 2025

---

## Overview

This project investigates methods to enhance the mathematical reasoning capabilities of Qwen2.5-Math-1.5B through:

1. **Chain-of-Thought Data Generation**: Two-round cascade using Grok-4.1-Fast and MiniMax-M2
2. **Supervised Fine-Tuning**: Full SFT and LoRA configurations
3. **Reinforcement Learning**: GRPO applied to both Solver and Verifier components
4. **Agentic Workflows**: Solver-Verifier with integrated code execution and Code Feedback architectures

### Key Results

| Method | GSM8K-test | MATH-500 | Improvement |
|--------|------------|----------|-------------|
| **Base Model** | 65.8% | 53.2% | - |
| **SFT LoRA (r=16)** | 80.0% | 67.2% | +14.2 pp |
| **SFT Full** | 81.6% | 67.0% | +15.8 pp |
| **GRPO (RL)** | 82.4% | 68.2% | +16.6 pp |
| **Solver-Verifier (SFT Both)** | 86.4% | 68.0% | +20.6 pp |
| **Solver-Verifier (SFT+RL)** | **86.8%** | **68.8%** | **+21.0 pp** |
| **Code Feedback (SFT)** | 82.8% | 66.0% | +17.0 pp |
| **Code Feedback (SFT+RL)** | 84.6% | 67.8% | +18.8 pp |

### Key Findings

1. **SFT provides the largest gains** (+14.2 pp on GSM8K) by teaching structured reasoning patterns
2. **GRPO adds incremental refinement** (+2.4 pp) by optimizing both solver correctness and verifier reliability
3. **Agentic workflows enable error correction** (+6.4 pp beyond SFT) through inference-time verification
4. **Emergent code generation**: The model spontaneously generates Python code but fails to mentally execute it correctly—our code-executing agents address this by extracting and running the model's own code
5. **Solver quality > Verifier quality**: SFT on solver alone (+2.6 pp) provides 4× higher returns than SFT on verifier alone (+0.6 pp)
6. **Iteration 2 is optimal**: Highest accuracy (91.7% on GSM8K) due to feedback benefits without context overload

---

## Quick Start

### 1. Setup Environment

```bash
conda create -y --name slm_math python=3.10
conda activate slm_math
pip install -r requirements.txt
```

### 2. Download Datasets and Models

```bash
python dataset/download_data_and_models.py --hf_token YOUR_HF_TOKEN
```

### 3. Run Evaluation

```bash
# Base model evaluation
python -m evaluation.eval \
    --model "Qwen2.5-Math-1.5B" \
    --round "test" \
    --dataset "gsm8k" \
    --count 500 \
    --mode standard

# Solver-Verifier agent (best performing)
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "solver_verifier" \
    --round "test" \
    --dataset "gsm8k" \
    --count 500 \
    --max_iterations 5
```

---

## Training

### Supervised Fine-Tuning

```bash
# LoRA fine-tuning (rank=16, lr=1e-4, 2 epochs)
bash scripts/train_sft_lora.sh

# Full fine-tuning (lr=5e-5, 2 epochs)
bash scripts/train_sft_full.sh
```

**Key Hyperparameters**:
- **LoRA**: rank 16, learning rate 1e-4, batch size 128, 2 epochs
- **Full SFT**: learning rate 5e-5, batch size 128, 2 epochs
- **Training Data**: 18,946 CoT samples (97.3% success rate from 19,473 problems)

### Reinforcement Learning (GRPO)

GRPO is applied to both the **Solver** and **Verifier** components to optimize:
- **Solver**: Answer correctness, format quality, and parsability
- **Verifier**: Decision reliability in the Solver-Verifier loop

```bash
# GRPO training (initialized from SFT LoRA)
bash scripts/train_rl_grpo.sh
```

**Key Hyperparameters**:
- Learning rate: 5e-6 with cosine decay to 1e-7
- Batch size: 16 prompts, K=2 responses per prompt
- KL coefficient: β=0.05
- Training duration: 1 epoch on GSM8K
- Reward structure: Solver reward (correctness + format + parsing), Verifier reward (decision accuracy)

### Generate Chain-of-Thought Data

Our data pipeline uses a two-round cascade:
1. **Grok-4.1-Fast** (primary, 3 attempts)
2. **MiniMax-M2** (backup, 5 attempts for failed problems)

```bash
python dataset/build_CoT_data.py \
    --dataset gsm8k \
    --output data/cot_generated/gsm8k_cot.json
```

**Statistics**:
- GSM8K: 7,298/7,473 (97.7% success), avg 320 tokens, 45% with code
- MATH: 11,648/12,000 (97.1% success), avg 480 tokens, 72% with code

---

## Agentic Workflows

### Available Agents

| Agent | Description | GSM8K | MATH-500 |
|-------|-------------|-------|----------|
| **Solver-Verifier (SFT+RL)** | Two-model architecture with iterative feedback | **86.8%** | **68.8%** |
| **Solver-Verifier (SFT Both)** | Solver and verifier both SFT-trained | 86.4% | 68.0% |
| **Solver-Verifier (SFT Solver)** | Only solver is SFT-trained | 86.0% | 67.0% |
| **Solver-Verifier (SFT Verifier)** | Only verifier is SFT-trained | 84.0% | 67.4% |
| **Solver-Verifier (Base)** | Both models use base checkpoint | 83.4% | 66.2% |
| **Code Feedback (SFT+RL)** | Two-step: generate code → execute → answer | 84.6% | 67.8% |
| **Code Feedback (SFT)** | Code feedback with SFT model | 82.8% | 66.0% |
| **Code Feedback (Base)** | Code feedback with base model | 76.4% | 60.0% |
| **Solver Checker with Tools** | Solver + checker with Python execution | 81.4% | 49.8% |
| **Majority Vote** | Ensemble voting over 5 runs | 70.2% | 54.8% |
| **Agent with Python Tools** | Single-pass with code execution | 72.6% | 45.2% |

### Solver-Verifier Architecture

The Solver-Verifier workflow uses two models with integrated code execution:
1. **Solver**: Generates solutions with Python code generation and execution capability
2. **Verifier**: Validates with verdicts (CORRECT/INCORRECT/UNCLEAR)

**Workflow**:
- Solver can generate and execute code (self-loop)
- Execution results feed back to Solver
- Solver output goes to Verifier
- Verifier provides feedback for iterative refinement

Supports up to 5 iterations with feedback loops. 88% of problems are solved in the first two iterations. Iteration 2 achieves highest accuracy (91.7% on GSM8K); later iterations degrade due to context length limitations.

```bash
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "solver_verifier" \
    --solver_checkpoint "checkpoints/sft_lora_r16" \
    --verifier_checkpoint "checkpoints/sft_lora_r16" \
    --max_iterations 5
```

### Code Feedback Architecture

Two-step workflow that addresses the model's emergent code generation:
1. **Generate reasoning with code**
2. **Execute code** in sandbox
3. **Inject execution results** into context
4. **Generate final answer** based on feedback

```bash
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "code_feedback" \
    --checkpoint "checkpoints/sft_lora_r16"
```

**Why it works**: The model generates Python code spontaneously but cannot mentally execute it correctly. External execution achieves 89.8% accuracy when code runs successfully.

---

## Project Structure

```
SLM-Math/
├── agent/                          # Agent workflow implementations
│   ├── solver_verifier.py          # Solver-Verifier architecture
│   ├── code_feedback.py            # Code Feedback agent
│   ├── majority_vote.py            # Ensemble voting
│   └── unified_config.py           # Unified generation configs
├── configs/                        # Training configurations
│   ├── sft_config.yaml             # Full SFT (lr=5e-5, 2 epochs)
│   ├── sft_config_qlora.yaml       # LoRA (rank=16, lr=1e-4)
│   ├── rl_grpo_config.yaml         # GRPO (β=0.05, K=2)
│   └── rl_multi_agent_config.yaml  # Multi-agent RL
├── data/                           # Datasets
│   ├── gsm8k/                      # GSM8K dataset
│   ├── math/                       # MATH dataset
│   ├── math500/                    # MATH-500 subset
│   └── cot_generated/              # Generated CoT data (18,946 samples)
├── dataset/                        # Data utilities
│   ├── build_CoT_data.py           # CoT data generation
│   └── download_data_and_models.py # Setup script
├── evaluation/                     # Evaluation scripts
│   ├── eval.py                     # Base evaluation
│   └── eval_agent.py               # Agent evaluation
├── models/                         # Model training
│   ├── train_sft_lora.py           # LoRA training
│   ├── train_sft_full.py           # Full SFT training
│   ├── train_rl_grpo.py            # GRPO training
│   ├── train_sft_verifier.py       # Verifier training
│   ├── train_sft_solver.py         # Solver training
│   └── __training_scripts_index__.py # Training script documentation
├── scripts/                        # Shell scripts
│   ├── train_sft_lora.sh           # LoRA training script
│   ├── train_sft_full.sh           # Full SFT script
│   ├── train_rl_grpo.sh            # GRPO script
│   └── train_rl_code_feedback.sh   # Code feedback RL
├── results/                        # Evaluation results (29 configs)
├── utils/                          # Utility functions
│   ├── prompt_utils.py             # Prompts & answer extraction
│   └── python_code_execution.py    # Code sandbox
└── FinalReportDocs/                # Final report (LaTeX)
    └── FinalProjectDoc/
        └── finalreport_version3.tex
```

---

## Configuration Files

All configurations match the final report specifications:

| Config | Learning Rate | Epochs | Batch Size | Special Parameters |
|--------|--------------|--------|------------|-------------------|
| `sft_config.yaml` | 5e-5 | 2 | 128 | Full fine-tuning |
| `sft_config_qlora.yaml` | 1e-4 | 2 | 128 | LoRA rank=16 |
| `rl_grpo_config.yaml` | 5e-6 | 1 | 16 prompts | β=0.05, K=2 |
| `rl_multi_agent_config.yaml` | 5e-6 | 1 | 16 prompts | Multi-agent setup |

---

## Datasets

| Dataset | Train | Test | Description |
|---------|-------|------|-------------|
| **GSM8K** | 7,473 | 500 | Grade-school arithmetic (2-8 steps) |
| **MATH** | 12,000 | - | Competition-level problems |
| **MATH-500** | - | 500 | Curated MATH subset |

**CoT Data Statistics**:
- Total: 18,946 verified samples (97.3% success rate)
- Average length: 320 tokens (GSM8K), 480 tokens (MATH)
- Code generation: 45% (GSM8K), 72% (MATH)

---

## Key Insights

### What Works

1. **SFT is foundational**: Provides +14.2 pp by teaching structured reasoning
2. **GRPO refines**: Adds +2.4 pp by optimizing for correctness
3. **Solver quality > Verifier quality**: SFT on solver alone (+2.6 pp) beats SFT on verifier alone (+0.6 pp)
4. **Simple workflows win**: Solver-Verifier (5 iterations max) outperforms complex tool pipelines
5. **Code execution helps**: When code runs successfully, accuracy reaches 89.8%

### What Doesn't Work

1. **Long contexts hurt**: Agents with >2000 tokens degrade performance
2. **Tool-only approaches fail**: Python Tools without verification drops to 45.2% on MATH-500
3. **Majority Vote underperforms**: Only 70.2% vs 80.0% for SFT single-pass (small models have high variance)
4. **Summarizers lose information**: 15-20 pp drops from compressing reasoning traces

### Design Recommendations

1. Start with **LoRA SFT** (efficient, 80.0% accuracy)
2. Add **Solver-Verifier** with code execution for +6.4 pp gain
3. Apply **GRPO to both Solver and Verifier** after SFT gains saturate
4. Use **2-3 iterations** for Solver-Verifier (optimal trade-off)
5. Consider **Code Feedback** when models naturally generate code
6. **Avoid long-context approaches** (>2000 tokens) for 1.5B models

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (training)
- 8GB+ GPU memory (inference)

**Hardware Used**:
- NVIDIA H20 (96GB) for training
- RTX 4090 (24GB) for evaluation

See `requirements.txt` for complete dependencies.

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{wang2025slmmath,
  title={SLM-Math: Empowering Small Language Models for Mathematical Reasoning},
  author={Wang, Roger and Luo, Jinzi and Yuan, Yunchen},
  journal={Columbia University COMS4705 Final Project},
  year={2025}
}
```

---

## License

This project is for academic research purposes. Model weights are subject to their respective licenses:
- Qwen models: Apache 2.0
- Datasets: See individual dataset licenses

---

## Acknowledgments

- **TA Mentor**: Melody Ma
- **Course**: Columbia University COMS4705 Natural Language Processing
- **Models**: Qwen2.5-Math-1.5B by Alibaba Cloud
- **Datasets**: GSM8K (OpenAI), MATH (Hendrycks et al.)
- **APIs**: Grok-4.1-Fast (xAI), MiniMax-M2

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Roger Wang: lw3240@columbia.edu
- Jinzi Luo: jl7199@columbia.edu
- Yunchen Yuan: yy3610@columbia.edu
