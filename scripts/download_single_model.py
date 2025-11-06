import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

os.environ['HF_TOKEN'] = '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'

model_name = "Qwen/Qwen2.5-0.5B"

print(f"Downloading {model_name}...")
print("=" * 60)

model_short_name = model_name.split('/')[-1]
model_dir = Path(f"../models/{model_short_name}")
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Tokenizer already downloaded, checking model...")

model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
if model_files:
    print(f"Found {len(model_files)} model file(s) already downloaded")
    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")
else:
    print("No model files found, downloading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print(f"Saving model to {model_dir}...")
        model.save_pretrained(str(model_dir))
        
        print(f"Successfully downloaded {model_name}")
        
        model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
        total_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        print(f"Total model size: {total_size:.2f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

