# SFT Verifier Training Dataset - Summary

## ✅ Dataset Created Successfully

### Files Generated

1. **Training-Ready Format (USE THESE)**:
   - `sft_verifier_training_formatted.jsonl` - 500 examples in JSONL format
   - `sft_verifier_training_formatted.csv` - Same data in CSV format

2. **Raw Data (with explanations)**:
   - `sft_verifier_training_data.jsonl` - Includes verification reasoning
   - `sft_verifier_training_data.csv` - CSV version

3. **Documentation**:
   - `README.md` - Comprehensive documentation
   - `SUMMARY.md` - This file

4. **Scripts**:
   - `generate_verifier_data.py` - Generate raw data
   - `reformat_verifier_data.py` - Convert to training format

## 📊 Dataset Statistics

- **Total Examples**: 500
- **Format**: Input (Question + Solution) → Output (Verdict)

### Verdict Distribution
| Verdict | Count | Percentage |
|---------|-------|------------|
| CORRECT | 300 | 60.0% |
| INCORRECT | 175 | 35.0% |
| UNCLEAR | 25 | 5.0% |

### Error Type Distribution (INCORRECT examples)
| Error Type | Count | Description |
|------------|-------|-------------|
| Arithmetic Error | 100 | Wrong calculations (e.g., 24-8=14) |
| Logic Error | 50 | Wrong formula (e.g., perimeter = L+W) |
| Code Error | 25 | Bugs in code (e.g., range(1,n) missing n) |
| Unclear | 25 | Uncertain/ambiguous reasoning |

## 🎯 Training Format

### Input Structure
```
Question: [Math problem]

Solution:
[Chain-of-thought reasoning]
[Python code block]
[Final answer in \boxed{} format]
```

### Output Structure
```
CORRECT | INCORRECT | UNCLEAR
```

## 📝 Example Breakdown

### Example 1: CORRECT (60% of dataset)
- Proper reasoning steps
- Code matches manual calculation
- Correct final answer
- Proper formatting

### Example 2: INCORRECT - Arithmetic Error (20% of dataset)
- Manual calculation has arithmetic mistake
- Code output differs from stated answer
- Final answer doesn't match ground truth

### Example 3: INCORRECT - Logic Error (10% of dataset)
- Wrong formula or approach
- Code implements incorrect logic
- Conceptual misunderstanding

### Example 4: INCORRECT - Code Error (5% of dataset)
- Reasoning is correct
- Code has bugs (off-by-one, wrong operator)
- Execution produces wrong result

### Example 5: UNCLEAR (5% of dataset)
- Self-contradictory statements
- Expresses uncertainty
- Missing critical steps
- Ambiguous answer format

## 🔧 How to Use

### For SFT Training

```python
import json

# Load formatted data
with open('sft_verifier_training_formatted.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Training loop
for example in data:
    input_text = example['input']      # Question + Solution
    target_label = example['output']   # CORRECT/INCORRECT/UNCLEAR
    
    # Your SFT training code here
    # model.train(input_text, target_label)
```

### Integration with Solver-Verifier Workflow

This dataset trains the **Verifier** component:
- **Solver** (trained on `../sft_solver_training/`) generates solutions
- **Verifier** (trained on this dataset) classifies them as CORRECT/INCORRECT/UNCLEAR
- If INCORRECT/UNCLEAR, feedback is sent back to Solver for retry

## 🎓 Key Design Decisions

1. **60% CORRECT examples**: Model should recognize good solutions
2. **35% INCORRECT examples**: Model must detect various error types
3. **5% UNCLEAR examples**: Model should flag uncertain reasoning
4. **Diverse error patterns**: Covers arithmetic, logic, and code errors
5. **Realistic mistakes**: Based on common small model failures

## 📈 Expected Performance

After training on this dataset, the verifier should:
- ✅ Detect arithmetic calculation errors
- ✅ Identify wrong formulas or approaches
- ✅ Catch code execution bugs
- ✅ Flag uncertain or ambiguous reasoning
- ✅ Validate correct solutions with confidence

## 🔗 Related Components

- **Solver Training**: `../sft_solver_training/run_1209_1011/`
- **Evaluation Scripts**: `../../evaluation/`
- **Agent Implementation**: `../../agent/solver_verifier.py`
- **Paper Reference**: `../../FinalReportDocs/FinalProjectDoc/finalreport_version3.tex`

## 📌 Notes

- Dataset is synthetically generated but reflects realistic error patterns
- Designed specifically for math reasoning verification
- Compatible with any model checkpoint (base, SFT, RL)
- Can be extended with more examples using `generate_verifier_data.py`

---

**Generated**: December 2024  
**Purpose**: SFT training for verifier model in Solver-Verifier architecture  
**Project**: SLM-Math (COMS4705 Final Project)

