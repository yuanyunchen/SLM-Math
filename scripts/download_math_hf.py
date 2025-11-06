#!/usr/bin/env python3
"""
Download the MATH dataset from Hugging Face Hub.
"""

from datasets import load_dataset
from pathlib import Path

def download_math():
    print("=" * 80)
    print("Downloading MATH dataset from Hugging Face")
    print("=" * 80)
    
    data_dir = Path("../data/math")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\nAttempt 1: Trying 'qwedsacf/competition_math'...")
        dataset = load_dataset("qwedsacf/competition_math")
        dataset.save_to_disk(str(data_dir))
        print(f"✓ MATH dataset downloaded and saved to {data_dir}")
        
        # Print dataset info
        for split in dataset.keys():
            print(f"  {split}: {len(dataset[split])} samples")
        
        # Show a sample
        print("\nSample from train set:")
        sample = dataset['train'][0]
        print(f"  Problem: {sample.get('problem', 'N/A')[:100]}...")
        print(f"  Solution: {sample.get('solution', 'N/A')[:100]}...")
        print(f"  Level: {sample.get('level', 'N/A')}")
        print(f"  Type: {sample.get('type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed with 'qwedsacf/competition_math': {e}")
        
        try:
            print("\nAttempt 2: Trying 'lighteval/MATH'...")
            dataset = load_dataset("lighteval/MATH")
            dataset.save_to_disk(str(data_dir))
            print(f"✓ MATH dataset downloaded and saved to {data_dir}")
            
            for split in dataset.keys():
                print(f"  {split}: {len(dataset[split])} samples")
            
            return True
            
        except Exception as e2:
            print(f"✗ Failed with 'lighteval/MATH': {e2}")
            
            try:
                print("\nAttempt 3: Trying 'hendrycks/competition_math'...")
                dataset = load_dataset("hendrycks/competition_math")
                dataset.save_to_disk(str(data_dir))
                print(f"✓ MATH dataset downloaded and saved to {data_dir}")
                
                for split in dataset.keys():
                    print(f"  {split}: {len(dataset[split])} samples")
                
                return True
                
            except Exception as e3:
                print(f"✗ Failed with 'hendrycks/competition_math': {e3}")
                print("\n⚠ Could not download MATH dataset from Hugging Face")
                print("The dataset might not be publicly available or requires authentication.")
                return False

if __name__ == "__main__":
    success = download_math()
    if success:
        print("\n" + "=" * 80)
        print("MATH dataset download complete!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("MATH dataset download failed.")
        print("=" * 80)

