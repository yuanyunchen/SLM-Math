import os
from huggingface_hub import snapshot_download
from pathlib import Path

os.environ['HF_TOKEN'] = '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'

QWEN3_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B"
]

def download_model(model_name):
    print(f"\nDownloading {model_name}...")
    try:
        model_short_name = model_name.split('/')[-1]
        model_dir = Path(f"../models/{model_short_name}")
        
        print(f"  Downloading to {model_dir}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            token=os.environ['HF_TOKEN'],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        
        print(f"  ✓ Successfully downloaded {model_name}")
        
        size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"  Total size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {model_name}: {e}")
        return False

def main():
    print("Starting Qwen3 model downloads using Hugging Face Hub")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for model_name in QWEN3_MODELS:
        if download_model(model_name):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    print("\n" + "=" * 60)
    print(f"Successfully downloaded {len(successful)}/{len(QWEN3_MODELS)} models")
    if successful:
        print("\n✓ Successfully downloaded:")
        for model in successful:
            print(f"  - {model}")
    if failed:
        print("\n✗ Failed to download:")
        for model in failed:
            print(f"  - {model}")

if __name__ == "__main__":
    main()

