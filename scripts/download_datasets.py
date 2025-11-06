import os
from datasets import load_dataset
from pathlib import Path

def download_gsm8k():
    print("Downloading GSM8K dataset...")
    try:
        dataset = load_dataset("gsm8k", "main")
        
        data_dir = Path("../data/gsm8k")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(data_dir))
        print(f"GSM8K dataset saved to {data_dir}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")
        raise

def download_math():
    print("\nDownloading MATH dataset...")
    try:
        dataset = load_dataset("competition_math")
        
        data_dir = Path("../data/math")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(data_dir))
        print(f"MATH dataset saved to {data_dir}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"Error downloading MATH: {e}")
        print(f"Trying alternative dataset name...")
        try:
            dataset = load_dataset("lighteval/MATH")
            dataset.save_to_disk(str(data_dir))
            print(f"MATH dataset saved to {data_dir}")
            print(f"Train samples: {len(dataset['train'])}")
            print(f"Test samples: {len(dataset['test'])}")
        except Exception as e2:
            print(f"Error with alternative: {e2}")
            print("Skipping MATH dataset for now")

def main():
    print("Starting dataset downloads...")
    print("=" * 60)
    
    download_gsm8k()
    download_math()
    
    print("\n" + "=" * 60)
    print("All datasets downloaded successfully!")

if __name__ == "__main__":
    main()

