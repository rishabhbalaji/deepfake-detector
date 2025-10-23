# scripts/check_data.py
import os
from pathlib import Path

# Navigate to the project's root directory
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "datasets/frames"

splits = ['train', 'val']
classes = ['real', 'fake'] # Assuming these are your folder names

print("--- DeepSight Data Integrity Check ---")

for split in splits:
    print(f"\nChecking Split: [{split}]")
    total_images = 0
    
    # Check if split directory exists
    split_path = DATA_PATH / split
    if not split_path.exists():
        print(f"  ERROR: Directory not found: {split_path}")
        continue
        
    for cls in classes:
        class_path = split_path / cls
        
        # Check if class directory exists
        if not class_path.exists():
            print(f"  WARNING: Class directory not found: {class_path}")
            continue
            
        count = len(os.listdir(class_path))
        print(f"  Class [{cls}]: {count} images")
        total_images += count
        
    print(f"  ---------------------")
    print(f"  Total {split}: {total_images} images")

print("\n--- Check Complete ---")