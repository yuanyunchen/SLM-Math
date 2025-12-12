# SFT Verifier Training Dataset

This directory contains training data for fine-tuning a verifier model to classify mathematical solutions as CORRECT, INCORRECT, or UNCLEAR.

## Dataset Overview

- **Total Examples**: 500
- **Format**: JSONL and CSV
- **Purpose**: Train a model to verify mathematical reasoning and code execution

## Dataset Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| CORRECT | 300 | 60.0% |
| INCORRECT | 175 | 35.0% |
| UNCLEAR | 25 | 5.0% |

### Error Categories (for INCORRECT examples)

| Category | Count | Percentage |
|----------|-------|------------|
| Arithmetic Error | 100 | 20.0% |
| Logic Error | 50 | 10.0% |
| Code Error | 25 | 5.0% |

## Files

### 1. Raw Data (with reasoning explanations)
- `sft_verifier_training_data.jsonl` - Original format with verification reasoning
- `sft_verifier_training_data.csv` - CSV version of raw data

**Format**:
```json
{
  "index": 1,
  "question": "...",
  "ground_truth": "...",
  "solution": "...",
  "verdict": "CORRECT|INCORRECT|UNCLEAR",
  "reasoning": "Explanation of why this verdict was assigned",
  "category": "correct|arithmetic_error|logic_error|code_error|unclear"
}
```

### 2. Training-Ready Format (recommended for SFT)
- `sft_verifier_training_formatted.jsonl` - **Use this for training**
- `sft_verifier_training_formatted.csv` - CSV version

**Format**:
```json
{
  "index": 1,
  "input": "Question: ...\n\nSolution:\n...",
  "output": "CORRECT|INCORRECT|UNCLEAR",
  "question": "...",
  "ground_truth": "...",
  "verdict": "CORRECT|INCORRECT|UNCLEAR",
  "category": "..."
}
```

## Training Format

The model should be trained to:
- **Input**: Question + Complete Solution (Chain-of-Thought reasoning + Python code + Final answer)
- **Output**: Single verdict token (CORRECT/INCORRECT/UNCLEAR)

### Example 1: CORRECT Solution

**Input**:
```
Question: If a train travels at 45 mph for 2 hours, how far does it travel?

Solution:
To find the distance traveled, I use the formula: distance = speed × time.

Given:
- Speed: 45 mph
- Time: 2 hours

Calculation:
Distance = 45 × 2 = 90 miles

Let me verify with code:
```python
speed = 45
time = 2
distance = speed * time
final_answer = distance
print(final_answer)
```

The code outputs 90, confirming the answer is \boxed{90}
```

**Output**: `CORRECT`

### Example 2: INCORRECT Solution (Logic Error)

**Input**:
```
Question: Tom scored 88, 93, and 85 on three tests. What is his average score?

Solution:
To find the average, I sum all scores.

Step-by-step reasoning:
1. Test 1: 88
2. Test 2: 93
3. Test 3: 85
4. Average = 88 + 93 + 85 = 266

Let me write code to compute this:
```python
scores = [88, 93, 85]
average = sum(scores)
final_answer = average
print(final_answer)
```

The code outputs 266, so the answer is \boxed{266}
```

**Output**: `INCORRECT`

**Reason**: Forgot to divide by the number of tests. Correct answer should be 88.67.

### Example 3: UNCLEAR Solution

**Input**:
```
Question: If 4 workers can complete a job in 13 days, how many days will it take 8 workers?

Solution:
To solve this work rate problem, I need to consider the relationship between workers and time.

Step-by-step reasoning:
1. Original: 4 workers, 13 days
2. New: 8 workers, ? days
3. More workers means less time...

Let me write code to compute this:
```python
workers1 = 4
days1 = 13
workers2 = 8
# Not sure about the exact formula here
days2 = (workers1 * days1) / workers2
final_answer = round(days2, 2)
print(final_answer)
```

The code outputs 6.5, so the answer is \boxed{6.5}

But I'm not entirely confident about this formula.
```

**Output**: `UNCLEAR`

**Reason**: Solution expresses uncertainty and lacks confidence in the approach.

## Error Types Covered

### 1. Arithmetic Errors (20%)
- Incorrect addition, subtraction, multiplication, or division
- Manual calculation doesn't match code output
- Off-by-one errors in counting

### 2. Logic Errors (10%)
- Wrong formula applied (e.g., perimeter = length + width instead of 2×(length + width))
- Missing steps in calculation (e.g., forgetting to divide when computing average)
- Conceptual misunderstanding of the problem

### 3. Code Errors (5%)
- Off-by-one errors in range() function
- Using wrong operator (// instead of /)
- Logic bugs in code implementation

### 4. Unclear Solutions (5%)
- Self-contradictory reasoning
- Expressing uncertainty or doubt
- Missing critical steps
- Ambiguous final answer format

## Usage

### For Training

```python
import json

# Load training data
with open('sft_verifier_training_formatted.jsonl', 'r', encoding='utf-8') as f:
    training_data = [json.loads(line) for line in f]

# Each example has:
# - input: Question + Solution (what the model sees)
# - output: Verdict (what the model should predict)

for example in training_data:
    prompt = example['input']
    label = example['output']  # CORRECT, INCORRECT, or UNCLEAR
    # Use for your SFT training pipeline
```

### For Evaluation

The dataset includes metadata fields for analysis:
- `category`: Type of example (correct, arithmetic_error, logic_error, etc.)
- `ground_truth`: Correct numerical answer
- `question`: Original question text

## Generation

The dataset was synthetically generated using `generate_verifier_data.py` with:
- Diverse problem types (arithmetic, algebra, word problems)
- Realistic error patterns observed in small language models
- Balanced distribution of verdict types

## Related Files

- **Solver Training Data**: `../sft_solver_training/` - Data for training the solution generator
- **Paper Reference**: See `FinalReportDocs/FinalProjectDoc/finalreport_version3.tex` Section 3.3 for methodology

## Notes

- This dataset is designed to train a verifier that can operate on any checkpoint (base, SFT, or RL)
- The verifier should learn to detect common failure modes in small language models
- Focus is on mathematical reasoning verification, not general code verification

