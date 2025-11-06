import os
import sys
from pathlib import Path

def check_setup():
    print("Checking SLM-Math Project Setup")
    print("=" * 60)
    
    checks = {
        "datasets_installed": False,
        "transformers_installed": False,
        "torch_installed": False,
        "gsm8k_downloaded": False,
        "math_downloaded": False,
        "models_directory": False,
        "models_downloaded": 0
    }
    
    try:
        import datasets
        checks["datasets_installed"] = True
        print("✓ datasets library installed")
    except ImportError:
        print("✗ datasets library not installed")
    
    try:
        import transformers
        checks["transformers_installed"] = True
        print("✓ transformers library installed")
    except ImportError:
        print("✗ transformers library not installed")
    
    try:
        import torch
        checks["torch_installed"] = True
        print("✓ torch library installed")
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("  Running on CPU")
    except ImportError:
        print("✗ torch library not installed")
    
    gsm8k_path = Path("../data/gsm8k")
    if gsm8k_path.exists():
        checks["gsm8k_downloaded"] = True
        print(f"✓ GSM8K dataset found at {gsm8k_path}")
    else:
        print(f"✗ GSM8K dataset not found at {gsm8k_path}")
    
    math_path = Path("../data/math")
    if math_path.exists():
        checks["math_downloaded"] = True
        print(f"✓ MATH dataset found at {math_path}")
    else:
        print(f"✗ MATH dataset not found at {math_path}")
    
    models_path = Path("../models")
    if models_path.exists():
        checks["models_directory"] = True
        models = list(models_path.iterdir())
        checks["models_downloaded"] = len([m for m in models if m.is_dir()])
        print(f"✓ Models directory found")
        print(f"  Found {checks['models_downloaded']} model(s)")
        for model in models:
            if model.is_dir():
                print(f"    - {model.name}")
    else:
        print(f"✗ Models directory not found at {models_path}")
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print(f"  Required libraries: {sum([checks['datasets_installed'], checks['transformers_installed'], checks['torch_installed']])}/3")
    print(f"  Datasets ready: {sum([checks['gsm8k_downloaded'], checks['math_downloaded']])}/2")
    print(f"  Models downloaded: {checks['models_downloaded']}")
    
    if all([checks['datasets_installed'], checks['transformers_installed'], checks['torch_installed'], checks['gsm8k_downloaded']]) and checks['models_downloaded'] > 0:
        print("\n✓ Setup is complete! Ready to run evaluations.")
        return True
    else:
        print("\n⚠ Setup is incomplete. Please complete the missing steps.")
        return False

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)

