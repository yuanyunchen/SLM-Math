# Analytics Script for Multi-Agent Evaluation

## Overview
`analyze_results.py` is a comprehensive analytics tool for analyzing multi-agent evaluation results. It generates detailed CSV reports and visualizations to help understand checker behavior, iteration patterns, and improvement opportunities.

## Features

### Automatic Latest Results Detection
- Automatically finds and analyzes the most recent results directory
- No need to specify paths manually

### Key Metrics Tracked
1. **Overall Performance**: Accuracy, correct answers, total problems
2. **Iteration Statistics**: Single vs multi-iteration success rates
3. **Checker Performance**: True positives, false positives, improvement tracking
4. **Verdict Distribution**: How often checker says CORRECT/INCORRECT/UNCLEAR

### Special Analysis Categories
- **Improved Questions**: First iteration not correct, but final answer correct (learning effect)
- **False Positives**: Checker said CORRECT but answer is wrong
- **False Negatives**: Checker said INCORRECT but answer is correct
- **Degraded Questions**: First iteration correct, but final answer wrong

## Usage

### Basic Usage (analyzes latest results)
```bash
python scripts/analyze_results.py
```

### Specify Results Directory
```bash
python scripts/analyze_results.py --log-dir results/testFixedOutput_Qwen2.5-Math-1.5B_gsm8k_20_1120_2131
```

### Specify Output Directory
```bash
python scripts/analyze_results.py --output-dir my_summaries
```

## Output Files

The script generates two CSV files in the `summary/` directory:

### 1. Detailed CSV (`*_detailed.csv`)
Per-question analysis with columns:
- `sample_num`: Question number
- `dataset_idx`: Dataset index
- `question_preview`: First 100 chars of question
- `ground_truth`: Expected answer
- `predicted_answer`: Model's answer
- `is_correct`: Whether answer matches ground truth
- `total_iterations`: Number of solver-checker iterations
- `first_verdict`: Checker's first verdict
- `last_verdict`: Checker's final verdict
- `first_answer`: Solver's first answer
- `last_answer`: Solver's final answer
- `verdict_changed`: Whether checker changed its mind
- `answer_changed`: Whether solver changed its answer
- `classification`: Category (e.g., "Improved - Final Correct", "False Positive")

### 2. Summary CSV (`*_summary.csv`)
Aggregated statistics:
- Metadata (model, dataset, timestamp)
- Overall performance metrics
- Iteration statistics
- Checker performance metrics
- Improvement counts

## Example Output

```
================================================================================
MULTI-AGENT EVALUATION ANALYTICS
================================================================================

Metadata:
  Model: Qwen2.5-Math-1.5B
  Dataset: GSM8K Split: test
  Round: testFixedOutput
  Timestamp: 2024-11-20 21:31:45

Overall Performance:
  Total Problems: 20
  Correct Answers: 13
  Accuracy: 65.00%

Iteration Statistics:
  Single Iteration - Correct: 10
  Single Iteration - Wrong: 5
  Multi Iteration - Correct: 3
  Multi Iteration - Wrong: 2

Checker Performance (First Iteration):
  First CORRECT → Final Correct: 10 (True Positives)
  First CORRECT → Final Wrong: 5 (False Positives)
  First NOT CORRECT → Final Correct: 3 (Improved)
  First NOT CORRECT → Final Wrong: 2 (Remained Wrong)

✓ Improved Questions (First iter NOT correct → Final correct): 3
  - Sample 5: A gym charges a $10 membership fee and $2 per class...
    First verdict: INCORRECT → Last verdict: CORRECT
    Answer: 34 (GT: 34) ✓
```

## Interpreting Results

### Key Performance Indicators

1. **Accuracy**: Overall correctness rate
   - Target: > baseline (11/20 = 55%)
   - Current: 13/20 = 65% ✓

2. **Improved Questions**: Shows learning effect
   - First iteration wrong → Final answer correct
   - Indicates checker+iteration loop is working

3. **False Positives**: Critical metric
   - Checker says CORRECT but answer is WRONG
   - High false positive rate means checker is too permissive

4. **Single vs Multi-Iteration Success**:
   - Single iteration correct: Checker correctly identified good solution early
   - Multi-iteration correct: Iteration loop helped improve solution

### What to Look For

**Good Signs:**
- Improved questions > 0 (iteration helping)
- Low false positive rate (< 30%)
- High single-iteration correct rate (checker efficient)

**Problem Signs:**
- High false positive rate (checker too lenient)
- Zero improved questions (iteration not helping)
- All problems taking 5 iterations (checker too strict)

## Advanced Analysis

### Finding Specific Issues
```bash
# Analyze specific run
python scripts/analyze_results.py --log-dir results/testFixedOutput_Qwen2.5-Math-1.5B_gsm8k_20_1120_2131

# Check the detailed CSV for patterns
# Look for questions with verdict_changed='Yes' and is_correct='No'
# These are cases where iteration might have hurt performance
```

### Comparing Runs
```bash
# Analyze multiple runs
python scripts/analyze_results.py --log-dir results/run1 --output-dir summary/run1
python scripts/analyze_results.py --log-dir results/run2 --output-dir summary/run2

# Then compare the summary CSVs
```

## Future Enhancements
- Visualization plots (iteration distribution, verdict trends)
- Comparison mode (compare two runs side-by-side)
- LaTeX report generation
- Interactive HTML dashboard

