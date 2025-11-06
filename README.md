# SLM-Math: Small Language Models for Mathematical Reasoning

This project evaluates the performance of Qwen3 Small Language Models on mathematical reasoning benchmarks (GSM8K and MATH).

## Project Overview

This repository contains scripts and tools to:
- Download and prepare mathematical reasoning datasets (GSM8K, MATH)
- Download Qwen3 models (0.6B, 1.7B, 4B, 8B parameters)
- Evaluate model performance on benchmarks
- Generate detailed performance reports with metrics and timing

## Project Structure

```
SLM-Math/
├── data/                    # Dataset storage
│   ├── gsm8k/              # GSM8K dataset
│   └── math/               # MATH dataset
├── models/                  # Downloaded models
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   ├── Qwen3-4B-Instruct-2507/
│   └── Qwen3-8B/
├── scripts/                 # Python scripts
│   ├── download_datasets.py    # Download datasets
│   ├── download_models.py      # Download models
│   ├── evaluate.py            # Evaluation script
│   ├── utils.py              # Utility functions
│   └── test_setup.py         # Setup verification
├── results/                 # Evaluation results
│   ├── metrics/            # Individual model results
│   └── summary_report.txt  # Overall summary
├── requirements.txt         # Python dependencies
└── run_evaluation.sh       # Evaluation runner script
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- 20GB+ disk space for models
- Hugging Face account and API token

### Installation

1. Clone the repository:
```bash
cd /Users/yuanyunchen/Desktop/GitHub/SLM-Math
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set Hugging Face token (already configured in scripts):
```bash
export HF_TOKEN="your_token_here"
```

### Download Data and Models

1. Download datasets:
```bash
cd scripts
python download_datasets.py
```

2. Download models:
```bash
python download_models.py
```

3. Verify setup:
```bash
python test_setup.py
```

## Running Evaluations

### Quick Start

Run the complete evaluation pipeline:
```bash
./run_evaluation.sh
```

### Manual Evaluation

Run evaluation for specific models:
```bash
cd scripts
python evaluate.py
```

The script will:
- Test each model on GSM8K and MATH datasets
- Generate predictions for test samples
- Calculate accuracy metrics
- Record inference time
- Save results to `../results/metrics/`

## Evaluation Metrics

For each model-dataset combination, we report:
- **Accuracy**: Percentage of correctly solved problems
- **Total Time**: Complete evaluation duration
- **Avg Time per Sample**: Average inference time per problem
- **Correct/Total**: Number of correct predictions

## Results

Results are saved in two formats:

1. **Individual Results** (`results/metrics/`):
   - JSON files with detailed predictions
   - Format: `{model}_{dataset}_results.json`

2. **Summary Report** (`results/summary_report.txt`):
   - Aggregated performance metrics
   - Comparison across all models

## Models

### Qwen3 Series

| Model | Parameters | Size | Description |
|-------|------------|------|-------------|
| Qwen3-0.6B | 0.6B | ~1.2GB | Smallest model |
| Qwen3-1.7B | 1.7B | ~3.4GB | Small model |
| Qwen3-4B-Instruct | 4B | ~8GB | Medium model (Instruct) |
| Qwen3-8B | 8B | ~16GB | Large model |

## Datasets

### GSM8K
- **Description**: Grade School Math 8K problems
- **Size**: 7,473 training + 1,319 test samples
- **Focus**: Multi-step arithmetic reasoning
- **Format**: Natural language math word problems

### MATH
- **Description**: High school mathematics competition problems
- **Difficulty**: Challenging problems across multiple topics
- **Topics**: Algebra, geometry, calculus, probability, etc.

## Technical Details

### Answer Extraction
The evaluation uses sophisticated answer extraction to handle various formats:
- Extract numerical answers from text
- Normalize fractions and decimals
- Handle mathematical notation
- Support multiple answer formats

### Inference Configuration
- **Temperature**: 0.7
- **Max New Tokens**: 512
- **Sampling**: Greedy decoding
- **Precision**: FP16 (float16)

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `MAX_SAMPLES` in `evaluate.py`
   - Use smaller models (0.5B, 1.5B)

2. **Slow Download**:
   - Check internet connection
   - Use HF mirror if available
   - Download models one at a time

3. **Import Errors**:
   - Reinstall dependencies: `pip install -r requirements.txt --upgrade`
   - Check Python version: `python --version`

## Performance Expectations

Based on preliminary testing:
- **GSM8K**: Models show strong performance on grade school problems
- **MATH**: More challenging, performance varies by problem difficulty
- **Speed**: Larger models are slower but generally more accurate

## Citation

If you use this code or results, please cite:

```bibtex
@misc{slm-math-2025,
  title={SLM-Math: Evaluating Small Language Models on Mathematical Reasoning},
  author={Your Name},
  year={2025}
}
```

## License

This project uses:
- Qwen2.5 models: Apache 2.0 License
- GSM8K dataset: MIT License
- Code: MIT License

## Acknowledgments

- Qwen Team for the excellent Qwen2.5 models
- OpenAI for the GSM8K dataset
- Hendrycks et al. for the MATH dataset

