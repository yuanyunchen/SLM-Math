import os
from huggingface_hub import snapshot_download
from pathlib import Path

os.environ['HF_TOKEN'] = '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'

model_name = "Qwen/Qwen3-4B-Thinking-2507"

print(f"\n{'='*60}")
print(f"Downloading {model_name}")
print(f"{'='*60}\n")

model_short_name = model_name.split('/')[-1]
model_dir = Path(f"../models/{model_short_name}")

snapshot_download(
    repo_id=model_name,
    local_dir=str(model_dir),
    token=os.environ['HF_TOKEN'],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
)

print(f"\n✓ Successfully downloaded {model_name}")
size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
print(f"Total size: {size_mb:.2f} MB")

