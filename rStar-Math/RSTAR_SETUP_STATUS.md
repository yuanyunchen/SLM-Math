# rStar-Math Setup Status

## Overview
This document describes the current setup status of the rStar-Math project on macOS and the steps needed to complete the setup on a CUDA-enabled Linux machine (A100 GPU with CUDA 12.4).

## Current Machine (macOS)
- **OS**: macOS Darwin 24.6.0
- **Python**: 3.10.18 (conda environment: `slm_math`)
- **Limitations**: No NVIDIA GPU, no CUDA support

## Installation Status

### ✅ Successfully Installed Packages

The following packages from `requirements.txt` have been successfully installed:

| Package | Version | Status |
|---------|---------|--------|
| antlr4-python3-runtime | 4.11.1 | ✅ Installed |
| omegaconf | 2.3.0 | ✅ Installed (minor conflict with antlr4 version) |
| termcolor | 1.1.0 | ✅ Installed |
| jsonlines | 4.0.0 | ✅ Installed |
| pebble | 5.1.3 | ✅ Installed |
| matplotlib | 3.10.0 | ✅ Installed |
| pyparsing | 3.2.0 | ✅ Installed |
| scipy | 1.13.0 | ✅ Installed |
| word2number | 1.1 | ✅ Installed |
| transformers | 4.45.2 | ✅ Installed |
| trl | 0.9.6 | ✅ Installed |
| func_timeout | 4.3.5 | ✅ Installed |
| editdistance | 0.8.1 | ✅ Installed |
| datasets | 4.4.1 | ✅ Installed |
| latex2sympy2 | 1.9.0 | ✅ Installed (MARIO_EVAL component) |
| math_evaluation | 0.3.1 | ✅ Installed (MARIO_EVAL) |
| torch | 2.2.2 | ✅ Installed (CPU version only) |

### ❌ Failed to Install (CUDA Required)

| Package | Required Version | Issue |
|---------|------------------|-------|
| vllm | >=0.6.6.post1 | **Requires torch 2.5.1 with CUDA 12.4** |
| torch | 2.5.1 | macOS version not available, requires CUDA |

**Error Message**:
```
ERROR: Could not find a version that satisfies the requirement torch==2.5.1
(from versions: 1.7.1, ..., 2.2.2)
```

### Package Conflicts (Non-Critical)

```
omegaconf 2.3.0 requires antlr4-python3-runtime==4.9.*, but installed is 4.11.1
```
- This conflict is acceptable as rStar-Math's requirements specify 4.11.1
- omegaconf still functions correctly with 4.11.1

## Evaluation Data

All evaluation datasets are present in `eval_data/`:

```
eval_data/
├── aime2024_test.json       - AIME 2024 test set
├── amc23_test.json          - AMC 2023 test set  
├── collegemath_test.json    - College Math test set
├── gaokao2023en_test.json   - Gaokao 2023 English test set
├── GSM8K_test.json          - GSM8K test set
├── MATH_test.json           - MATH dataset test set
├── math500_test.json        - MATH-500 test set
├── olympiadbench_test.json  - OlympiadBench test set
└── omni_test.json           - Omni-Math test set
```

## MARIO_EVAL Installation

✅ **Status**: Successfully Installed

- latex2sympy component installed
- math_evaluation package installed in editable mode
- All required dependencies available

## Project Structure

```
rStar-Math/
├── config/                  - Configuration files (YAML)
├── eval_data/              - Evaluation datasets (9 datasets)
├── MARIO_EVAL/             - Evaluation toolkit (installed)
├── rstar_deepthink/        - Main package source code
├── train/                  - Training scripts and utilities
├── utils/                  - Utility functions
├── main.py                 - Main execution script
├── eval.py                 - Evaluation script
├── run_example.sh          - Example run script
└── requirements.txt        - Dependencies list
```

## Steps to Complete Setup on Linux Machine (A100 + CUDA 12.4)

### 1. Transfer Files

Transfer the entire `rStar-Math/` directory to the Linux machine:

```bash
# On target machine
rsync -avz --progress /path/to/SLM-Math/rStar-Math/ target_machine:/path/to/rStar-Math/
```

### 2. Create Conda Environment

```bash
# On Linux machine with CUDA 12.4
conda create -y --name rstar python=3.11
conda activate rstar
```

### 3. Install PyTorch with CUDA 12.4

```bash
pip install --upgrade pip
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install vLLM

```bash
pip install vllm==0.6.6.post1
```

If you encounter the error `undefined symbol: __nvJitLinkComplete_12_4`, run:

```bash
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

### 5. Install Other Requirements

```bash
cd /path/to/rStar-Math
pip install -r requirements.txt
```

### 6. Install MARIO_EVAL (Should Already Be There)

```bash
cd MARIO_EVAL/latex2sympy
pip install .
cd ..
pip install -e .
cd ..
```

### 7. Optional: Install Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

### 8. Verify Installation

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"
python -c "import vllm; print('vLLM Version:', vllm.__version__)"
python -c "from math_evaluation.core import evaluations; print('MARIO_EVAL Ready')"
```

## Usage Examples

### Test with Greedy Decoding

```bash
MODEL="deepseek-ai/DeepSeek-Coder-V2-Instruct"
CUDA_VISIBLE_DEVICES="0" python eval.py --model "$MODEL" --device 0 --task gsm8k
```

### Generate Training Data with MCTS

```bash
MODEL="deepseek-ai/DeepSeek-Coder-V2-Instruct"  
QAF="eval_data/GSM8K_test.json"
CFG="config/sample_mcts.yaml"
CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py --qaf $QAF --custom_cfg $CFG --model_dir $MODEL
```

### Run MCTS Inference with Policy and Reward Models

```bash
MODEL="path/to/policy_model"
RM="path/to/reward_model" 
QAF="eval_data/MATH_test.json"
CFG="config/sft_eval_mcts.yaml"
CUDA_VISIBLE_DEVICES="0" python main.py --qaf $QAF --custom_cfg $CFG --model_dir $MODEL --reward_model_dir $RM
```

## Important Configuration Notes

1. **GPU Memory**: Ensure you modify `llm_gpu_memory_utilization` and `tp` parameters in the config files based on your available GPU memory.

2. **CUDA Devices**: Adjust `CUDA_VISIBLE_DEVICES` based on available GPUs.

3. **Batch Size**: For limited VRAM, reduce `n_generate_sample` and `iterations` parameters.

## Training Datasets

rStar-Math uses data from:
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)

Open-source training datasets:
- [SFT-Trainset](https://huggingface.co/datasets/ElonTusk2001/rstar_sft)
- [PPM-Trainset](https://huggingface.co/datasets/ElonTusk2001/rstar_ppm)

## Known Issues & Solutions

### Issue 1: `undefined symbol: __nvJitLinkComplete_12_4`

**Solution**:
```bash
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

Add this to your `~/.bashrc` or `~/.bash_profile` for persistence.

### Issue 2: Out of Memory Errors

**Solution**: Reduce parameters in config files:
- Lower `n_generate_sample` from 32 to 16
- Lower `iterations` from 16 to 8
- Use beam search instead of MCTS with `config/sft_eval_bs.yaml`

### Issue 3: Slow Inference

**Solution**: 
- Enable tensor parallelism by setting `tp` parameter
- Use flash-attn 2 with `--attn_impl flash_attention_2`
- Reduce max sequence length in configs

## References

- Paper: [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://huggingface.co/papers/2501.04519)
- GitHub: [rStar-Math Repository](https://github.com/microsoft/rStar-Math)
- MARIO Evaluation: [MARIO_EVAL Paper](https://arxiv.org/abs/2404.13925)

## Summary

✅ **Ready for Transfer**:
- All code and data files are present
- Python packages compatible with macOS are installed
- MARIO_EVAL toolkit is configured
- Documentation is complete

❌ **Requires Linux + CUDA**:
- torch 2.5.1 with CUDA 12.4
- vllm 0.6.6.post1
- Actual model inference and training

**Next Steps**: Transfer to A100 machine and complete CUDA-dependent installations.


