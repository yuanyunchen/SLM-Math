# Analytics Quick Start Guide

## Congratulations! 🎉

Your multi-agent system achieved **13/20 (65%)** accuracy, beating the baseline of **11/20 (55%)**!

## Quick Usage

### Analyze Latest Results
```bash
python scripts/analyze_results.py
```

### Analyze Specific Run
```bash
python scripts/analyze_results.py --log-dir results/testFixedOutput_Qwen2.5-Math-1.5B_gsm8k_20_1120_2131
```

## Key Findings from Latest Run

### Overall Performance
- **Accuracy**: 65.00% (13/20 correct)
- **Baseline**: 55.00% (11/20 correct)
- **Improvement**: +10 percentage points! ✓

### Iteration Statistics
- **Single iteration correct**: 8 problems (40%)
- **Multi-iteration correct**: 5 problems (25%)
- **Problems improved through iterations**: 5 (25%)

### Checker Performance
- **True Positives** (First CORRECT → Final Correct): 8
- **False Positives** (First CORRECT → Final Wrong): 3
- **Improved** (First NOT CORRECT → Final Correct): 5 ⭐

### Verdict Distribution
- CORRECT: 64.5%
- UNCLEAR: 32.3%
- INCORRECT: 3.2%

## Key Insights

### What's Working Well ✓
1. **Iteration loop is effective**: 5 problems improved from first iteration not correct to final correct
2. **Checker is balanced**: Mix of CORRECT (64.5%), UNCLEAR (32.3%), and INCORRECT (3.2%)
3. **Multi-iteration success**: 5 problems needed multiple iterations and got correct

### Areas for Improvement
1. **False Positives**: 3 problems where checker said CORRECT but answer was wrong
   - Sample 1: Janet's eggs (10 vs 18) - Revenue calculation error
   - Sample 3: House flipping (97500 vs 70000) - Profit interpretation error
   - Sample 9: John's drive (210 vs 45) - Turnaround calculation error

2. **Checker could be more critical**: Only 3.2% INCORRECT verdicts - might be too lenient

## Output Files

The script generates two CSV files in `summary/`:

### 1. Summary CSV
Contains aggregated statistics:
- Metadata (model, dataset, timestamp)
- Overall performance (accuracy, correct answers)
- Iteration statistics
- Checker performance metrics
- Improvement counts

### 2. Detailed CSV
Per-question analysis with:
- Question info (number, preview, ground truth)
- Answer correctness
- Iteration count
- First/last verdicts and answers
- Classification (e.g., "Improved - Final Correct", "False Positive")

## Example Analyses

### Find Improved Questions
```bash
# Run analysis
python scripts/analyze_results.py

# Open detailed CSV and filter by:
# classification = "Improved - Final Correct"
```

### Find False Positives
```bash
# Filter by:
# classification contains "False Positive"
```

### Compare Verdict Changes
```bash
# Filter by:
# verdict_changed = "Yes"
```

## Next Steps

1. **Reduce False Positives**: Focus on the 3 problems where checker incorrectly said CORRECT
2. **Analyze Improved Questions**: Study what made the checker initially uncertain
3. **Compare Multiple Runs**: Run with different parameters and compare CSV outputs
4. **Track Progress**: Keep CSV files to see improvement over time

## CSV File Naming
Format: `{dataset}_{model}_{count}problems_{timestamp}_{type}.csv`

Example: `unknown_unknown_20problems_20251120_215551_detailed.csv`

## Advanced Usage

### Compare Two Runs
```bash
# Run 1
python scripts/analyze_results.py --log-dir results/run1 --output-dir summary/run1

# Run 2
python scripts/analyze_results.py --log-dir results/run2 --output-dir summary/run2

# Compare the summary CSVs
```

### Analyze Specific Dataset
```bash
python scripts/analyze_results.py --log-dir results/testFixedOutput_Qwen2.5-Math-1.5B_gsm8k_20_1120_2131
```

---

For more details, see `scripts/README_analytics.md`

