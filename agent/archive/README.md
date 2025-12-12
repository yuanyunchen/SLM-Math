# Multi-Agent Workflow

Pulled from the `dev/multiagent` branch: a solver-checker iterative evaluation system.

## 📁 Files

```
agent/
├── README.md                       # This file
├── run_multi_agent_eval.sh         # ⭐ Runner script
├── analyze_results.py              # ⭐ Result analysis tool
│
└── Core modules for the multiagent variant (modified from main project):
    ├── eval_pipeline_multiagent.py     # evaluation/eval_pipeline.py
    ├── prompt_utils_multiagent.py      # utils/prompt_utils.py
    └── inference_multiagent.py         # models/inference.py
```

## 🚀 Quick Start

### 1. Run evaluation

```bash
cd agent
./run_multi_agent_eval.sh
```

### 2. Optional: adjust config

Edit `run_multi_agent_eval.sh`:

```bash
MODEL="Qwen2.5-Math-1.5B"      # Model name
DATASET="gsm8k"                # Dataset (gsm8k/math)
COUNT=20                       # Number of samples (0 = all)
```

### 3. Analyze results

```bash
python analyze_results.py
```

## 💡 Multi-Agent Flow

```
Problem
 ↓
Solver generates an answer
 ↓
Checker verifies → CORRECT / INCORRECT / UNCLEAR
 ↓
If CORRECT: done ✓
If incorrect: provide feedback → Solver retries (up to 5 times)
```

## 📊 Analysis Report — 4 Case Types

`python analyze_results.py` produces a CSV with:

| Type | Description | Meaning |
|------|-------------|---------|
| **Type 1: Improved** | First wrong → later correct | ✅ System helpful |
| **Type 2: Degraded** | First correct → later wrong | ⚠️ Needs improvement |
| **Type 3: First Try** | Success in one shot | 🎯 Efficient |
| **Type 4: Unnecessary** | Correct but checker missed it | 🔍 Checker can improve |

## 🔧 Key Parameters

| Param | Description | Example |
|-------|-------------|---------|
| `MODEL` | Solver model | Qwen2.5-Math-1.5B |
| `CHECKER_MODEL` | Optional checker model | Qwen2.5-Math-1.5B-Instruct |
| `DATASET` | Dataset | gsm8k, math |
| `COUNT` | Sample count | 20 (0 = all) |
| `MODE` | Must be multi_agent | multi_agent |

## 📈 Output Files

### Evaluation results
```
results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>/
├── log/*.log          # Detailed logs (per-iteration dialogue)
├── metrics.csv        # Accuracy and other metrics
├── summary.txt        # Summary
└── answer.json        # Detailed answers
```

### Analysis report
```
summary/<dataset>_<model>_<count>problems_<timestamp>_analysis.csv
```

## 🆚 Differences vs Main Branch

| Feature | Main | Multiagent |
|---------|------|------------|
| Eval modes | standard, thinking | **+ multi_agent** |
| Iteration | none | **Solver-Checker loop** |
| Analysis | basic | **Automatic 4-case analysis** |

## 💻 Usage Examples

### Example 1: Basic evaluation

```bash
./run_multi_agent_eval.sh
```

### Example 2: Custom parameters

```bash
# Edit the script
nano run_multi_agent_eval.sh

# Change values:
MODEL="Qwen3-1.7B"
COUNT=100
DATASET="math"

# Run
./run_multi_agent_eval.sh
```

### Example 3: Different checker

```bash
# Uncomment in run_multi_agent_eval.sh:
CHECKER_MODEL="Qwen2.5-Math-1.5B-Instruct"
```

## 📝 Key Files

### Python modules

| File | Source | Main change |
|------|--------|-------------|
| `eval_pipeline_multiagent.py` | evaluation/eval_pipeline.py | Adds multi_agent mode |
| `prompt_utils_multiagent.py` | utils/prompt_utils.py | Adds Solver/Checker prompts |
| `inference_multiagent.py` | models/inference.py | Tuning inference params |
| `analyze_results.py` | new | 4-case analysis tool |

### Key functions (prompt_utils_multiagent.py)

```python
format_prompt_solver(question, checker_feedback=None)    # Solver prompt
format_prompt_checker(question, solver_response)         # Checker prompt
parse_checker_verdict(checker_response)                  # Extract verdict
parse_checker_tip(checker_response)                      # Extract feedback
```

## 🐛 FAQ

### Q: How to run?
```bash
./run_multi_agent_eval.sh
```

### Q: Where are results?
- Evaluation: `../results/<latest>/`
- Analysis: `../summary/*.csv`

### Q: How to analyze?
```bash
python analyze_results.py
```

### Q: Checker always returns UNCLEAR?
Refine `format_prompt_checker()` in `prompt_utils_multiagent.py`.

## 🎯 Optimization Tips

Based on the analysis report:

1. **Many Improved cases** → system is effective; keep using.
2. **Many Degraded cases** → improve the checker prompt.
3. **Many Unnecessary iterations** → strengthen checker detection.
4. **Low First Try rate** → improve solver prompt.

## ✅ Checklist

Before running:
- [ ] Models in `../pretrained_models/`
- [ ] Datasets in `../data/`
- [ ] `run_multi_agent_eval.sh` configured

After running:
- [ ] Check `../results/<dir>/summary.txt`
- [ ] Run `python analyze_results.py`
- [ ] Review 4-case statistics

---

**Quick start**: `./run_multi_agent_eval.sh`  
**Source**: `dev/multiagent` branch
