#!/usr/bin/env python3
"""
One-click download script for SLM-Math datasets and models.
Downloads datasets and models from HuggingFace Hub to the correct locations.

Usage:
    python scripts/download_data_and_models.py --hf_token YOUR_TOKEN
    python scripts/download_data_and_models.py --hf_token YOUR_TOKEN --datasets gsm8k math math500
    python scripts/download_data_and_models.py --hf_token YOUR_TOKEN --models Qwen2.5-Math-1.5B Qwen3-1.7B
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import login, snapshot_download
    from datasets import load_dataset
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install huggingface_hub datasets")
    sys.exit(1)


# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "gsm8k",
        "local_path": "data/gsm8k",
        "splits": ["train", "test"],
        "description": "GSM8K grade school math problems"
    },
    "math": {
        "hf_path": "hendrycks/competition_math",
        "local_path": "data/math",
        "splits": ["train"],
        "description": "MATH competition math problems"
    },
    "math500": {
        "hf_path": "lighteval/MATH",
        "local_path": "data/math500",
        "splits": ["test"],
        "description": "MATH-500 subset"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "Qwen2.5-Math-1.5B": {
        "hf_path": "Qwen/Qwen2.5-Math-1.5B",
        "local_path": "pretrained_models/Qwen2.5-Math-1.5B",
        "description": "Qwen2.5-Math-1.5B model"
    },
    "Qwen3-1.7B": {
        "hf_path": "Qwen/Qwen3-1.7B-Instruct",  # Update if correct path differs
        "local_path": "pretrained_models/Qwen3-1.7B",
        "description": "Qwen3-1.7B model"
    }
}


def download_dataset(dataset_name: str, base_path: Path, hf_token: Optional[str] = None):
    """Download a dataset from HuggingFace Hub"""
    if dataset_name not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        return False
    
    config = DATASET_CONFIGS[dataset_name]
    local_path = base_path / config["local_path"]
    
    print(f"\n{'='*80}")
    print(f"Downloading dataset: {dataset_name}")
    print(f"Description: {config['description']}")
    print(f"HuggingFace path: {config['hf_path']}")
    print(f"Local path: {local_path}")
    print(f"{'='*80}\n")
    
    # Check if already exists
    if local_path.exists():
        print(f"⚠️  Dataset already exists at {local_path}")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print(f"✓ Skipping {dataset_name}")
            return True
    
    try:
        # Download dataset splits
        for split in config["splits"]:
            print(f"Downloading {split} split...")
            dataset = load_dataset(
                config["hf_path"],
                split=split,
                token=hf_token,
                cache_dir=str(base_path / "cache")
            )
            
            # Save to disk in HuggingFace format
            split_path = local_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(split_path))
            print(f"✓ Saved {split} split to {split_path}")
        
        # Create dataset_dict.json for compatibility
        dataset_dict = {
            "splits": config["splits"],
            "dataset_name": dataset_name
        }
        import json
        with open(local_path / "dataset_dict.json", "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"\n✓ Successfully downloaded {dataset_name} to {local_path}")
        return True
        
    except (OSError, ValueError, ConnectionError) as e:
        print(f"\n✗ Error downloading {dataset_name}: {e}")
        return False


def download_model(model_name: str, base_path: Path, hf_token: Optional[str] = None):
    """Download a model from HuggingFace Hub"""
    if model_name not in MODEL_CONFIGS:
        print(f"ERROR: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_name]
    local_path = base_path / config["local_path"]
    
    print(f"\n{'='*80}")
    print(f"Downloading model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"HuggingFace path: {config['hf_path']}")
    print(f"Local path: {local_path}")
    print(f"{'='*80}\n")
    
    # Check if already exists
    if local_path.exists():
        model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
        if model_files:
            print(f"⚠️  Model already exists at {local_path}")
            response = input("Do you want to re-download? (y/N): ").strip().lower()
            if response != 'y':
                print(f"✓ Skipping {model_name}")
                return True
    
    try:
        # Download model using snapshot_download
        print("Downloading model files...")
        snapshot_download(
            repo_id=config["hf_path"],
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=str(base_path / "cache")
        )
        
        print(f"\n✓ Successfully downloaded {model_name} to {local_path}")
        return True
        
    except (OSError, ValueError, ConnectionError) as e:
        print(f"\n✗ Error downloading {model_name}: {e}")
        print(f"   Make sure the model path '{config['hf_path']}' is correct.")
        print("   Some models may require authentication or have different paths.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and models for SLM-Math project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets and models
  python scripts/download_data_and_models.py --hf_token YOUR_TOKEN

  # Download only datasets
  python scripts/download_data_and_models.py --hf_token YOUR_TOKEN --datasets gsm8k math math500

  # Download only models
  python scripts/download_data_and_models.py --hf_token YOUR_TOKEN --models Qwen2.5-Math-1.5B Qwen3-1.7B

  # Download specific items
  python scripts/download_data_and_models.py --hf_token YOUR_TOKEN --datasets gsm8k --models Qwen2.5-Math-1.5B
        """
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace access token (get from https://huggingface.co/settings/tokens)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to download (default: all)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help="Models to download (default: all)"
    )
    
    parser.add_argument(
        "--skip_datasets",
        action="store_true",
        help="Skip downloading datasets"
    )
    
    parser.add_argument(
        "--skip_models",
        action="store_true",
        help="Skip downloading models"
    )
    
    args = parser.parse_args()
    
    # Get project root
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent
    
    print("="*80)
    print("SLM-Math: Dataset and Model Download Script")
    print("="*80)
    print(f"Project root: {base_path}")
    print(f"Datasets to download: {args.datasets if not args.skip_datasets else 'None'}")
    print(f"Models to download: {args.models if not args.skip_models else 'None'}")
    print("="*80)
    
    # Login to HuggingFace
    try:
        login(token=args.hf_token)
        print("✓ Logged in to HuggingFace Hub\n")
    except (ValueError, ConnectionError) as e:
        print(f"✗ Error logging in to HuggingFace Hub: {e}")
        print("Please check your token at https://huggingface.co/settings/tokens")
        return 1
    
    # Download datasets
    dataset_success = True
    if not args.skip_datasets:
        print("\n" + "="*80)
        print("DOWNLOADING DATASETS")
        print("="*80)
        for dataset_name in args.datasets:
            success = download_dataset(dataset_name, base_path, args.hf_token)
            if not success:
                dataset_success = False
    else:
        print("\n⏭️  Skipping datasets (--skip_datasets flag set)")
    
    # Download models
    model_success = True
    if not args.skip_models:
        print("\n" + "="*80)
        print("DOWNLOADING MODELS")
        print("="*80)
        for model_name in args.models:
            success = download_model(model_name, base_path, args.hf_token)
            if not success:
                model_success = False
    else:
        print("\n⏭️  Skipping models (--skip_models flag set)")
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    if not args.skip_datasets:
        print(f"Datasets: {'✓ Success' if dataset_success else '✗ Some failed'}")
    if not args.skip_models:
        print(f"Models: {'✓ Success' if model_success else '✗ Some failed'}")
    print("="*80)
    
    if dataset_success and model_success:
        print("\n✓ All downloads completed successfully!")
        print("\nNext steps:")
        print("1. Verify files in data/ and pretrained_models/ directories")
        print("2. Run evaluation: python -m evaluation.eval --model Qwen2.5-Math-1.5B --dataset gsm8k --count 10")
        return 0
    else:
        print("\n⚠️  Some downloads failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

