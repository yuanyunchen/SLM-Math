import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

os.environ['HF_TOKEN'] = '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'

QWEN3_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B"
]

def download_model(model_name):
    print(f"\nDownloading {model_name}...")
    try:
        model_short_name = model_name.split('/')[-1]
        model_dir = Path(f"../models/{model_short_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN']
        )
        
        print(f"  Saving tokenizer to {model_dir}...")
        tokenizer.save_pretrained(str(model_dir))
        
        print(f"  Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print(f"  Saving model to {model_dir}...")
        model.save_pretrained(str(model_dir))
        
        print(f"  Successfully downloaded {model_name}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  Error downloading {model_name}: {e}")
        print(f"  Continuing with next model...")
        return False
    
    return True

def main():
    print("Starting Qwen model downloads...")
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
        print("Successfully downloaded:")
        for model in successful:
            print(f"  - {model}")
    if failed:
        print("\nFailed to download:")
        for model in failed:
            print(f"  - {model}")

if __name__ == "__main__":
    main()

