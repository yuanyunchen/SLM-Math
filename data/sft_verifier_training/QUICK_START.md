# Quick Start Guide - SFT Verifier Training

## 🚀 TL;DR

**Use this file for training**: `sft_verifier_training_formatted.jsonl`

**Format**:
- **Input**: Question + Solution (CoT + Python code + answer)
- **Output**: CORRECT | INCORRECT | UNCLEAR

**Dataset**: 500 examples (300 CORRECT, 175 INCORRECT, 25 UNCLEAR)

## 📥 Load Data (Python)

```python
import json

# Load training data
with open('data/sft_verifier_training/sft_verifier_training_formatted.jsonl', 'r', encoding='utf-8') as f:
    training_data = [json.loads(line) for line in f]

print(f"Loaded {len(training_data)} examples")

# Example structure
example = training_data[0]
print(f"Input: {example['input'][:100]}...")
print(f"Output: {example['output']}")
```

## 🎯 Training Example

```python
# Pseudocode for SFT training
for example in training_data:
    # Input: Question + Solution
    prompt = example['input']
    
    # Output: Single verdict token
    label = example['output']  # "CORRECT" or "INCORRECT" or "UNCLEAR"
    
    # Train your model
    loss = model.train(prompt, label)
```

## 📊 What's in the Dataset?

| Type | Count | What It Teaches |
|------|-------|----------------|
| CORRECT | 300 | Recognize valid solutions |
| Arithmetic Error | 100 | Detect calculation mistakes |
| Logic Error | 50 | Catch wrong formulas |
| Code Error | 25 | Find code bugs |
| Unclear | 25 | Flag uncertain reasoning |

## 🔍 Sample Training Pairs

### Pair 1: CORRECT Solution
**Input**:
```
Question: If a train travels at 45 mph for 2 hours, how far does it travel?

Solution:
To find the distance traveled, I use the formula: distance = speed × time.
Given: Speed: 45 mph, Time: 2 hours
Calculation: Distance = 45 × 2 = 90 miles

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

---

### Pair 2: INCORRECT Solution (Logic Error)
**Input**:
```
Question: A rectangle has length 12 cm and width 8 cm. What is its perimeter?

Solution:
To find the perimeter of a rectangle, I need to add all four sides.
Step-by-step reasoning:
1. Length: 12 cm
2. Width: 8 cm
3. Perimeter = length + width = 12 + 8 = 20 cm

```python
length = 12
width = 8
perimeter = length + width
final_answer = perimeter
print(final_answer)
```

The code outputs 20, so the answer is \boxed{20}
```

**Output**: `INCORRECT`

*Why?* Should be `2 × (length + width) = 40`, not `20`

---

### Pair 3: UNCLEAR Solution
**Input**:
```
Question: What is 15 divided by 3?

Solution:
To solve this division problem:
1. Dividend: 15
2. Divisor: 3
3. Result: 15 / 3 = 5.0

```python
dividend = 15
divisor = 3
result = dividend / divisor
final_answer = result
print(final_answer)
```

The code outputs 5.0, so the answer is 5.0 or maybe 5? 
Should I box the exact decimal or rounded version?
```

**Output**: `UNCLEAR`

*Why?* Shows uncertainty about answer format

## 🎓 Training Tips

1. **Balance**: Dataset is pre-balanced (60% correct, 40% incorrect/unclear)
2. **Batch Size**: Start with 8-16 for 1.5B models
3. **Learning Rate**: Try 1e-4 to 5e-5 for LoRA
4. **Epochs**: 2-3 epochs should be sufficient
5. **Validation**: Hold out 10% for validation

## 📁 File Structure

```
data/sft_verifier_training/
├── sft_verifier_training_formatted.jsonl  ← USE THIS
├── sft_verifier_training_formatted.csv    ← CSV version
├── README.md                              ← Full documentation
├── SUMMARY.md                             ← Dataset overview
└── QUICK_START.md                         ← This file
```

## 🔗 Integration with Solver-Verifier

```python
# 1. Train Solver (generates solutions)
solver_model = train_on('sft_solver_training/...')

# 2. Train Verifier (validates solutions)
verifier_model = train_on('sft_verifier_training/sft_verifier_training_formatted.jsonl')

# 3. Use in Solver-Verifier workflow
for problem in test_set:
    solution = solver_model.generate(problem)
    verdict = verifier_model.classify(problem + solution)
    
    if verdict == "INCORRECT":
        # Retry with feedback
        solution = solver_model.generate(problem, feedback=verdict)
```

## ❓ FAQ

**Q: Why 500 examples?**  
A: Sufficient for fine-tuning a verifier while keeping generation cost manageable.

**Q: Can I add more examples?**  
A: Yes! Run `generate_verifier_data.py` with different parameters.

**Q: What if my model outputs wrong verdicts?**  
A: Check if you're using the formatted version. Also try training for more epochs or adjusting learning rate.

**Q: Do I need both CSV and JSONL?**  
A: No, use JSONL for training. CSV is for spreadsheet analysis.

## 📞 Questions?

See `README.md` for detailed documentation or check the paper:
`FinalReportDocs/FinalProjectDoc/finalreport_version3.tex`

