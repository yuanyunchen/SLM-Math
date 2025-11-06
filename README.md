# SLM-Math: Small Language Models for Mathematical Reasoning

Evaluate Qwen3 Small Language Models on mathematical reasoning benchmarks (GSM8K and MATH datasets).

## Setup

### 1. Create Environment

```bash
# Create conda environment with Python 3.10
conda create -y --name slm_math python=3.10
conda activate slm_math
```

### 2. Install Packages

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Activate environment
conda activate slm_math

# Run evaluation (50 samples on GSM8K with Qwen3-0.6B)
bash shells/run_evaluation.sh
```

Results saved to: `results/<round>_<model>_<dataset>_<count>_<MMDD>/`

## Project Structure

```
SLM-Math/
├── models/
│   └── Qwen3-0.6B/          # Only 0.6B model tracked in git
├── data/
│   ├── gsm8k/               # GSM8K dataset (1,319 test samples)
│   ├── math/                # MATH dataset (competition math)
│   └── csv/                 # Preprocessed CSV for analysis
├── scripts/
│   ├── evaluate_batch.py    # Main evaluation script
│   └── utils.py             # Helper functions
├── shells/
│   ├── run_evaluation.sh    # Single evaluation runner
│   └── run_full_test.sh     # Batch evaluation
├── results/                 # Timestamped evaluation results
│   └── <round>_<model>_<dataset>_<count>_<MMDD>/
│       ├── log/             # Detailed logs
│       ├── answers/         # Predictions JSON
│       ├── metrics_*.csv    # Metrics table
│       └── metrics_*.txt    # Summary report
└── rStar-Math/              # Advanced reasoning (requires A100 GPU)
```

## Evaluation Modes

### Standard Mode
Direct answer generation with step-by-step reasoning.
```bash
# In shells/run_evaluation.sh, set:
MODE="standard"
```

### Thinking Mode
Deep reasoning with Chain-of-Thought analysis.
```bash
MODE="thinking"
```

## Parameters

Edit `shells/run_evaluation.sh`:

```bash
MODEL="Qwen3-0.6B"           # Model to evaluate
ROUND_NAME="test"            # Test round name
DATASET="gsm8k"              # gsm8k or math
COUNT=50                     # Number of samples (0 = entire dataset)
MODE="standard"              # standard or thinking
DETAILED="true"              # true = show tokens, false = progress bar only
```

## Output Structure

**Directory naming**: `<round>_<model>_<dataset>_<count>_<MMDD>_[HHMM]`

Example: `test_Qwen3-0.6B_gsm8k_100_1106/`

**Files**:
- `log/*.log` - Full execution logs
- `answers/*_answers.json` - All predictions with analysis
- `metrics_*.csv` - Structured metrics data
- `metrics_*.txt` - Human-readable summary

## Key Features

✅ **Automatic timestamping** - No overwriting results  
✅ **Progress tracking** - Real-time accuracy updates  
✅ **Two output modes** - Detailed (streaming tokens) or compact (progress bar)  
✅ **Full dataset support** - Set `COUNT=0` to run entire dataset  
✅ **CSV export** - Easy manual error analysis with question IDs  

## Models Available

| Model | Parameters | Status |
|-------|------------|--------|
| Qwen3-0.6B | 0.6B | ✅ In Git |
| Qwen3-1.7B | 1.7B | Local only |
| Qwen3-4B-Thinking | 4B | Local only |
| Qwen3-8B | 8B | Local only |

*Only Qwen3-0.6B is tracked in git to keep repo size manageable.*

## Datasets

**GSM8K**: Grade school math word problems (1,319 test samples)  
**MATH**: High school competition math (challenging)

Both preprocessed to CSV with question IDs for error analysis.

## Environment Requirements

- **Python**: 3.10.18
- **Conda environment**: `slm_math`
- **Key packages**: torch, transformers (4.57+), datasets, pandas, tqdm
- **Disk space**: ~2GB for Qwen3-0.6B model, ~10GB for datasets

## Advanced: rStar-Math

Deep reasoning with Monte Carlo Tree Search (MCTS).

**Requirements**:
- A100 80GB GPU
- CUDA 12.4
- vLLM

Setup complete in `rStar-Math/` directory. See `rStar-Math/RSTAR_SETUP_STATUS.md` for transfer instructions to GPU machine.

## Sample Results

```
Model: Qwen3-0.6B
Dataset: GSM8K
Samples: 50
Accuracy: 42.0% (21/50)
Avg Time/Sample: 45.3s
```

## Tips

1. **Start small**: Use `COUNT=10` for quick tests
2. **Monitor progress**: Non-detailed mode shows running accuracy
3. **Check logs**: All output saved to log files regardless of display mode
4. **Manual analysis**: Use CSV files in `data/csv/` with question IDs from results

---

**Project**: Columbia COMS4705 NLP Final Project  
**Date**: November 2025
