import os
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    print("Downloading MATH dataset...")
    print("=" * 60)
    
    # Try multiple sources
    sources = [
        ("https://people.eecs.berkeley.edu/~hendrycks/MATH.tar", "MATH.tar"),
        ("https://github.com/hendrycks/math/releases/download/v1.0/MATH.tar", "MATH.tar")
    ]
    
    data_dir = Path("../data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for url, filename in sources:
        tar_path = data_dir / filename
        
        try:
            print(f"\nTrying to download from: {url}")
            download_file(url, tar_path)
            
            if tar_path.stat().st_size > 1000:  # Check if file is not empty
                print(f"\n✓ Downloaded successfully!")
                
                print(f"\nExtracting {filename}...")
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=data_dir)
                
                print("✓ Extraction complete!")
                
                # Verify extraction
                math_dir = data_dir / "MATH"
                if math_dir.exists():
                    train_dir = math_dir / "train"
                    test_dir = math_dir / "test"
                    
                    if train_dir.exists() and test_dir.exists():
                        train_topics = list(train_dir.iterdir())
                        test_topics = list(test_dir.iterdir())
                        
                        print(f"\n✓ MATH dataset structure verified:")
                        print(f"  Train topics: {len([d for d in train_topics if d.is_dir()])}")
                        print(f"  Test topics: {len([d for d in test_topics if d.is_dir()])}")
                        
                        # Count total problems
                        train_count = sum(len(list(topic.glob('*.json'))) for topic in train_topics if topic.is_dir())
                        test_count = sum(len(list(topic.glob('*.json'))) for topic in test_topics if topic.is_dir())
                        
                        print(f"  Train problems: {train_count}")
                        print(f"  Test problems: {test_count}")
                        
                        return True
                    else:
                        print("✗ Train or test directories not found")
                else:
                    print("✗ MATH directory not created")
            else:
                print(f"✗ Download failed or file is empty")
                tar_path.unlink()
                
        except Exception as e:
            print(f"✗ Error: {e}")
            if tar_path.exists():
                tar_path.unlink()
            continue
    
    print("\n✗ Could not download MATH dataset from any source")
    print("Please manually download from: https://github.com/hendrycks/math")
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nAlternative: Clone the GitHub repository:")
        print("  git clone https://github.com/hendrycks/math")

