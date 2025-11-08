# rStar-Math Setup Guide

**Status**: ✅ Fully Configured (Ready to run on GPU machine)  
**Date**: November 6, 2025  
**Environment**: slm_math (Python 3.10)

> **Note**: For a detailed technical setup status report, see [RSTAR_SETUP_STATUS.md](./RSTAR_SETUP_STATUS.md)

---

## 📦 What's Been Installed

### ✅ Core Dependencies
All installed in `slm_math` conda environment:

- **Core Python packages**:
  - antlr4-python3-runtime==4.11.1
  - omegaconf==2.3.0
  - termcolor, jsonlines, pebble
  - matplotlib, pyparsing, scipy
  - word2number, trl==0.24.0
  - func_timeout, editdistance

- **Evaluation Toolkit**:
  - ✅ MARIO_EVAL (installed from https://github.com/MARIO-Math-Reasoning/MARIO_EVAL)
  - ✅ latex2sympy2==1.9.0
  - ✅ math_evaluation==0.3.1
  - ✅ timeout-decorator==0.5.0

### ⚠️ GPU-Only Dependencies (NOT Installed - requires CUDA)
These **MUST be installed on your GPU machine**:

```bash
pip install vllm==0.6.6.post1  # Requires CUDA 12.4+
pip install flash-attn --no-build-isolation  # Optional but recommended
```

---

## 🖥️ Hardware Requirements

### Minimum Requirements (from README)
- **GPU**: A100 80GB (or equivalent high-memory GPU)
- **CUDA**: 12.4 or higher
- **RAM**: 128GB+ recommended
- **Storage**: 50GB+ for models

### Why GPU is Required
- `vllm` framework only works with CUDA GPUs
- MCTS (Monte Carlo Tree Search) is computationally intensive
- Tensor parallel execution across multiple GPUs for larger models

---

## 📁 Project Structure

```
rStar-Math/
├── config/               # Configuration files for training and evaluation
│   ├── sample_mcts.yaml          # MCTS sampling configuration
│   ├── sft_sample_mcts.yaml      # SFT + MCTS configuration
│   ├── sft_eval_mcts.yaml        # Evaluation with MCTS
│   ├── sft_eval_bs.yaml          # Evaluation with beam search
│   └── sft_eval_greedy.yaml      # Greedy decoding evaluation
│
├── eval_data/            # Evaluation datasets
│   ├── aime2024_test.json
│   ├── amc23_test.json
│   ├── collegemath_test.json
│   ├── gaokao2023en_test.json
│   ├── gsm8k_test.json
│   ├── math500_test.json
│   ├── math_test.json
│   ├── olympiadbench_test.json
│   └── omni-math_test.json
│
├── rstar_deepthink/      # Core rStar implementation
│   ├── __init__.py
│   ├── mcts.py                   # MCTS algorithm
│   ├── reward_model.py           # Process reward model
│   └── ...
│
├── train/                # Training scripts
│   ├── train_SFT.py              # Supervised fine-tuning
│   ├── train_RM.py               # Reward model training
│   ├── save_rm.py                # RM format conversion
│   ├── sample_sft_data.py
│   └── sample_rm_data.py
│
├── utils/                # Utility functions
│
├── MARIO_EVAL/           # ✅ Evaluation toolkit (installed)
│
├── main.py               # Main entry point
├── eval.py               # Evaluation script
├── eval_output.py        # Output evaluation
├── run_example.sh        # Example run script
└── requirements.txt      # Python dependencies

```

---

## 🎯 Required Models (NOT INCLUDED - You Must Provide)

### Policy Model (SFT-trained)
rStar-Math requires a **math-specialized policy model**. According to the paper:

**Recommended base models**:
- Qwen2.5-Math-7B (or similar math-specialized models)
- DeepSeek-Math-7B
- Your own fine-tuned model

**Where to get**:
- Fine-tune yourself using the provided training scripts
- Or use pre-trained models from Hugging Face
- Paper suggests starting with: `Qwen/Qwen2.5-Math-7B`

### Reward Model (PPM-trained)
A **process-based reward model** trained to evaluate step-level correctness.

**Training data provided**:
- [rStar-Math-SFT dataset](https://huggingface.co/datasets/microsoft/rStar-Math-SFT)
- [rStar-Math-PPM dataset](https://huggingface.co/datasets/microsoft/rStar-Math-PPM)

---

## 🚀 How to Run (On GPU Machine)

### 1. Transfer This Setup

```bash
# On your GPU machine, copy the entire rStar-Math directory
scp -r /path/to/rStar-Math user@gpu-machine:/path/to/destination/
```

### 2. Install GPU Dependencies

```bash
conda activate slm_math  # or create new environment
pip install vllm==0.6.6.post1
pip install flash-attn --no-build-isolation  # optional but recommended

# If CUDA version < 12.4, set library path:
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

### 3. Prepare Models

Download or place your trained models:

```bash
# Example: Download Qwen2.5-Math-7B as policy model
MODEL_DIR="/path/to/models"
mkdir -p $MODEL_DIR

# Option A: Use huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir $MODEL_DIR/policy_model

# Option B: In Python
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B')
model.save_pretrained('$MODEL_DIR/policy_model')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B')
tokenizer.save_pretrained('$MODEL_DIR/policy_model')
"
```

### 4. Run Evaluation

#### Greedy Decoding (Simplest)
```bash
MODEL="/path/to/policy_model"
CUDA_VISIBLE_DEVICES="0" python eval.py --model "$MODEL" --device 0 --task gsm8k
```

#### MCTS Inference (Full rStar)
```bash
MODEL="/path/to/policy_model"
RM="/path/to/reward_model"
QAF="eval_data/gsm8k_test.json"
CFG="config/sft_eval_mcts.yaml"

CUDA_VISIBLE_DEVICES="0" python main.py \
    --qaf $QAF \
    --custom_cfg $CFG \
    --model_dir $MODEL \
    --reward_model_dir $RM
```

#### Beam Search (Faster alternative)
```bash
MODEL="/path/to/policy_model"
RM="/path/to/reward_model"
QAF="eval_data/gsm8k_test.json"
CFG="config/sft_eval_bs.yaml"

CUDA_VISIBLE_DEVICES="0" python main.py \
    --qaf $QAF \
    --custom_cfg $CFG \
    --model_dir $MODEL \
    --reward_model_dir $RM
```

---

## 📊 Available Evaluation Tasks

rStar-Math includes built-in evaluation on:

1. **gsm8k** - Grade School Math (8K problems)
2. **math** - MATH dataset (challenging HS math)
3. **math500** - MATH subset (500 problems)
4. **aime2024** - American Invitational Mathematics Examination 2024
5. **amc23** - American Mathematics Competitions 2023
6. **collegemath** - College-level mathematics
7. **gaokao2023en** - Chinese Gaokao (English version)
8. **olympiadbench** - Mathematical olympiad problems
9. **omni-math** - Comprehensive math benchmark

---

## ⚙️ Configuration Files

### MCTS Configuration (`config/sft_eval_mcts.yaml`)
Key parameters to adjust:
- `n_generate_sample`: Number of samples per node (default: 32)
- `iterations`: MCTS iterations (default: 16)
- `llm_gpu_memory_utilization`: GPU memory usage (0.0-1.0)
- `tp`: Tensor parallelism degree

### Beam Search Configuration (`config/sft_eval_bs.yaml`)
Faster but slightly less accurate than MCTS.

### Greedy Configuration (`config/sft_eval_greedy.yaml`)
Single forward pass, no search (fastest, baseline).

---

## 🔧 Training Your Own Models

### 1. Generate Training Data

```bash
MODEL="deepseek-ai/DeepSeek-Coder-V2-Instruct"
QAF="path/to/train_set.json"
CFG="config/sample_mcts.yaml"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python main.py \
    --qaf $QAF \
    --custom_cfg $CFG \
    --model_dir $MODEL
```

### 2. Extract SFT Data

```bash
python extra_sft_file.py \
    --data_dir "MCTS_output_dir" \
    --output_file "sft_data.jsonl"

python train/sample_sft_data.py \
    --data_file "sft_data.jsonl" \
    --output_file "sft_train.json" \
    --n 2
```

### 3. Train Policy Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
    --nproc_per_node=8 train/train_SFT.py \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --data_path "sft_train.json" \
    --output_dir "path_to_save_policy" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --learning_rate 7e-6
```

### 4. Extract RM Data

```bash
python extra_rm_file.py \
    --data_dir "MCTS_output_dir" \
    --output_file "rm_data.jsonl"

python train/sample_rm_data.py \
    --data_file "rm_data.jsonl" \
    --output_file "rm_train.json"
```

### 5. Train Reward Model

```bash
accelerate launch --num_processes=8 train/train_RM.py \
    --model_name_or_path="path_to_policy_model" \
    --output_dir="path_to_save_rm" \
    --pair_json_path "rm_train.json" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=2 \
    --learning_rate=7e-6
```

### 6. Convert RM Format

```bash
python train/save_rm.py \
    --sft_model_path "path_to_policy_model" \
    --rm_ckpt_path "path_to_rm_checkpoint" \
    --rm_save_path "path_to_save_rm"
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use gradient accumulation:
```yaml
# In config file:
llm_gpu_memory_utilization: 0.8  # Reduce from 0.95
tp: 2  # Use tensor parallelism across 2 GPUs
```

### Issue: "undefined symbol: __nvJitLinkComplete_12_4"
**Solution**: Set library path:
```bash
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

### Issue: vllm Installation Fails
**Solution**: Ensure CUDA 12.4+ is installed:
```bash
nvcc --version  # Check CUDA version
pip install vllm==0.6.6.post1 --no-build-isolation
```

---

## 📚 References

- **Paper**: [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://huggingface.co/papers/2501.04519)
- **GitHub**: https://github.com/microsoft/rStar/tree/rStar-math
- **Datasets**:
  - [SFT Training Data](https://huggingface.co/datasets/microsoft/rStar-Math-SFT)
  - [PPM Training Data](https://huggingface.co/datasets/microsoft/rStar-Math-PPM)
- **Evaluation Toolkit**: [MARIO_EVAL](https://github.com/MARIO-Math-Reasoning/MARIO_EVAL)

---

## ✅ Checklist for GPU Machine

Before running on GPU machine, ensure:

- [ ] CUDA 12.4+ installed
- [ ] A100 80GB GPU (or equivalent)
- [ ] Install `vllm==0.6.6.post1`
- [ ] Install `flash-attn` (optional but recommended)
- [ ] Download/prepare policy model
- [ ] Download/prepare reward model (for MCTS)
- [ ] Adjust config files for your GPU setup
- [ ] Set `CUDA_VISIBLE_DEVICES` appropriately

---

## 💡 Quick Start Summary

**For Baseline Evaluation** (No training needed):
1. Get Qwen2.5-Math-7B model
2. Run: `python eval.py --model Qwen2.5-Math-7B --task gsm8k`

**For Full rStar** (Requires trained models):
1. Train policy + reward models (or use provided training data)
2. Run MCTS inference with both models
3. Compare results with baseline

---

**Status**: ✅ All dependencies installed, ready to transfer to GPU machine!

