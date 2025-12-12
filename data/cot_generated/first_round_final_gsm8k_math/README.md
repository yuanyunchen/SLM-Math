# Chain-of-Thought Data Generation

## Overview
This directory contains chain-of-thought (CoT) reasoning data generated using a two-round cascade approach with state-of-the-art teacher models.

## Generation Pipeline

### Teacher Models
1. **Primary: Grok-4.1-Fast**
   - 3 attempts per problem
   - Fast inference (~0.5s/problem)
   - 92% accuracy on GSM8K

2. **Backup: MiniMax-M2**
   - 5 attempts for problems where Grok fails
   - Provides complementary coverage on problems requiring different reasoning styles

### Data Sources
- **GSM8K Training Set**: 7,473 problems (grade-school arithmetic, 2-8 reasoning steps)
- **MATH Training Set**: 12,000 problems (competition-level, algebra/geometry/number theory/probability)

### Output Format
Each attempt produces structured output:
```
<think>
[step-by-step reasoning]
</think>
Therefore, the answer is \boxed{answer}.
```

### Quality Control
- Only solutions matching ground truth are retained
- Automatic answer extraction and normalization
- Multiple attempts increase success rate while maintaining quality

## Statistics

### Overall
- **Total Problems**: 19,473
- **Successful Generations**: 18,946
- **Success Rate**: 97.3%

### GSM8K
- **Total**: 7,473 problems
- **Successful**: 7,298 (97.7%)
- **Average Length**: 320 tokens
- **With Python Code**: 45%

### MATH
- **Total**: 12,000 problems
- **Successful**: 11,648 (97.1%)
- **Average Length**: 480 tokens
- **With Python Code**: 72%

## Files
- `first_round_final_gsm8k_math.json` - Main training data (18,946 samples)
- `first_round_final_gsm8k_math.csv` - CSV format for analysis
- `final_statistics.json` - Detailed statistics
- `cleanup_report.json` - Data cleaning and validation report

## Usage
This data is used for supervised fine-tuning (SFT) of Qwen2.5-Math-1.5B to teach structured mathematical reasoning patterns.

## Data Format
Each sample contains:
- `question`: Original problem text
- `ground_truth`: Correct answer
- `thinking_process`: CoT reasoning from teacher model
- `predicted_answer`: Extracted answer
- `teacher_model`: Which model generated this solution
- `dataset`: Source dataset (gsm8k or math)
- `generation_round`: Which attempt succeeded (1, 2, or 3)

