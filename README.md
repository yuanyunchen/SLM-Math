# SLM-Math: Small Language Models for Mathematical Reasoning

Evaluate Small Language Models (Qwen2.5-Math and Qwen3) on mathematical reasoning benchmarks (GSM8K, MATH, and MATH-500 datasets).

## Quick Start

### 1. Setup Environment

```bash
conda create -y --name slm_math python=3.10
conda activate slm_math
pip install -r requirements.txt
```

### 2. Download Datasets and Models

Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
python dataset/download_data_and_models.py --hf_token YOUR_HF_TOKEN
```

### 3. Quick Test

```bash
conda activate slm_math
python -m evaluation.eval --model "Qwen2.5-Math-1.5B" --round "test" --dataset "gsm8k" --count 10
```

### 4. Run Evaluation

**Base Evaluation:**
```bash
python -m evaluation.eval --model "Qwen2.5-Math-1.5B" --round "test" --dataset "gsm8k" --count 100
```

**Agent Evaluation:**
```bash
python -m evaluation.eval_agent --model "Qwen2.5-Math-1.5B" --agent "solver_checker_chat" --round "test" --dataset "gsm8k" --count 100
```

Results saved to: `results/<round>_<model>_<dataset>_<count>_<MMDD>/`

---

## Detailed Documentation

### Environment Configuration

**Environment Name**: `slm_math` (required)

**System Requirements**:
- Python 3.10+ (tested with 3.10.18)
- CUDA optional (for GPU acceleration, CUDA 11.8+ recommended)
- Disk space: ~3GB per model, ~2GB for datasets, ~5GB for results

**Key Dependencies**:
- `torch>=2.0.0` - PyTorch core
- `transformers>=4.30.0` - HuggingFace model library
- `datasets>=2.0.0` - Dataset handling
- `pandas>=2.0.0` - Data processing
- `sympy>=1.12` - Mathematical expression checking
- `huggingface_hub>=0.20.0` - For downloading models and datasets

See `requirements.txt` for complete dependency list.

### Download Script Options

The download script supports selective downloads:

```bash
# Download only datasets
python dataset/download_data_and_models.py --hf_token YOUR_TOKEN --skip_models

# Download only models
python dataset/download_data_and_models.py --hf_token YOUR_TOKEN --skip_datasets

# Download specific items
python dataset/download_data_and_models.py --hf_token YOUR_TOKEN \
    --datasets gsm8k math \
    --models Qwen2.5-Math-1.5B
```

**What it downloads**:
- **Datasets**: GSM8K, MATH, MATH-500 → `data/` directory
- **Models**: Qwen2.5-Math-1.5B, Qwen3-1.7B → `pretrained_models/` directory

**Manual Download** (if preferred):

- **GSM8K**: `gsm8k` → `data/gsm8k/`
- **MATH**: `hendrycks/competition_math` → `data/math/`
- **MATH-500**: `lighteval/MATH` → `data/math500/`
- **Qwen2.5-Math-1.5B**: `Qwen/Qwen2.5-Math-1.5B` → `pretrained_models/Qwen2.5-Math-1.5B/`
- **Qwen3-1.7B**: `Qwen/Qwen3-1.7B-Instruct` → `pretrained_models/Qwen3-1.7B/`

**Note**: Models and datasets are excluded from git (see `.gitignore`). You need to download them separately.

### Evaluation Parameters

**Base Evaluation (`evaluation.eval`)**:
- `--model`: Model name (e.g., `Qwen2.5-Math-1.5B`, `Qwen3-1.7B`)
- `--round`: Test round identifier
- `--dataset`: Dataset name (`gsm8k`, `math`, `math500`)
- `--count`: Number of samples (0 = entire dataset)
- `--mode`: Evaluation mode (`standard`)
- `--detailed`: Output verbosity (`true`/`false`, default: `false`)

**Agent Evaluation (`evaluation.eval_agent`)**:
- All base parameters plus:
- `--agent`: Agent method (see available agents below)
- `--max_iterations`: Max iterations per problem (default: 5)

**Available Agents**:
- `solver_checker`: Base solver-checker (independent models)
- `solver_checker_chat`: Chat-based solver-checker (shared model) - **Recommended**
- `solver_checker_with_tools`: With Python code execution
- `plan_and_reflection`: Multi-phase planning agent
- `majority_vote`: Majority voting ensemble

### Supported Models

| Model | Parameters | Recommended Use | Status |
|-------|------------|-----------------|--------|
| Qwen2.5-Math-1.5B | 1.5B | Primary model for evaluation | ✅ Recommended |
| Qwen3-1.7B | 1.7B | Alternative model | ✅ Supported |
| Qwen3-0.6B | 0.6B | Lightweight testing | ✅ Supported |
| Qwen3-4B-Thinking-2507 | 4B | Advanced reasoning | Optional |
| Qwen3-8B | 8B | Large model | Optional |

### Supported Datasets

| Dataset | Description | Test Samples | Location |
|---------|-------------|--------------|----------|
| **GSM8K** | Grade school math word problems | 1,319 | `data/gsm8k/test/` |
| **MATH** | High school competition math | ~5,000 | `data/math/train/` |
| **MATH-500** | Subset of MATH dataset | 500 | `data/math500/test/` |

All datasets are preprocessed to HuggingFace datasets format with question IDs for error analysis.

### Project Structure

```
SLM-Math/
├── pretrained_models/       # Model checkpoints (excluded from git)
│   ├── Qwen2.5-Math-1.5B/  # Primary model
│   ├── Qwen3-1.7B/         # Alternative model
│   └── ...
├── data/                    # Datasets (excluded from git)
│   ├── gsm8k/              # GSM8K dataset
│   ├── math/               # MATH dataset
│   └── math500/            # MATH-500 dataset
├── evaluation/             # Evaluation scripts
│   ├── eval.py            # Base evaluation
│   └── eval_agent.py      # Agent evaluation
├── models/                 # Model utilities
│   ├── inference.py        # Model loading and generation
│   └── inference_engine.py # Unified inference engine
├── agent/                  # Agent workflows
│   ├── solver_checker_*.py
│   ├── plan_and_reflection.py
│   └── majority_vote.py
├── utils/                  # Utility functions
│   ├── prompt_utils.py    # Prompt formatting, answer extraction
│   └── python_code_execution.py
├── dataset/                # Dataset utilities
│   ├── dataloader.py      # Dataset loading
│   └── download_data_and_models.py  # Download script
└── results/               # Evaluation results (excluded from git)
    └── <round>_<model>_<dataset>_<count>_<MMDD>/
        ├── log/           # Detailed logs
        ├── answers/       # Predictions JSON
        ├── metrics.csv    # Metrics table
        └── analysis_report.txt
```

### Output Structure

**Directory naming**: `<round>_<model>_<dataset>_<count>_<MMDD>_[HHMM]`

Example: `test_Qwen2.5-Math-1.5B_gsm8k_100_1123/`

**Files**:
- `log/*.log` - Full execution logs
- `answers/*_answers.json` - All predictions with analysis
- `metrics.csv` - Structured metrics data
- `analysis_report.txt` - Human-readable summary (for agent evaluation)

### Key Features

✅ **Automatic timestamping** - No overwriting results  
✅ **Progress tracking** - Real-time accuracy updates  
✅ **Two output modes** - Detailed (streaming tokens) or compact (progress bar)  
✅ **Full dataset support** - Set `--count 0` to run entire dataset  
✅ **Resume support** - Continue from interrupted evaluations  
✅ **CSV export** - Easy manual error analysis with question IDs  

### Troubleshooting

**Model not found**
- Ensure model is downloaded and placed in `pretrained_models/<model_name>/`
- Check model directory contains `config.json` and `model.safetensors`

**Dataset not found**
- Ensure datasets are downloaded and placed in `data/<dataset_name>/`
- Verify dataset directory contains `train/` or `test/` subdirectories

**CUDA out of memory**
- Reduce batch size or use smaller model
- Enable CPU mode if GPU memory is insufficient

**Environment issues**
- Always activate conda environment: `conda activate slm_math`
- Verify Python version: `python --version` (should be 3.10+)
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

### Sample Results

```
Model: Qwen2.5-Math-1.5B
Dataset: GSM8K
Samples: 50
Accuracy: 42.0% (21/50)
Avg Time/Sample: 45.3s
```

### Tips

1. **Start small**: Use `--count 10` for quick tests
2. **Monitor progress**: Non-detailed mode shows running accuracy
3. **Check logs**: All output saved to log files regardless of display mode
4. **Manual analysis**: Use CSV files in results with question IDs

---

**Project**: Columbia COMS4705 NLP Final Project  
**Date**: November 2025
