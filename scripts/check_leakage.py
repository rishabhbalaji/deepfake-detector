# scripts/check_leakage_v2.py
import os
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "datasets/frames"

def get_video_id(filename: str) -> str | None:
    """
    Extracts the video ID (e.g., '000' from '000_10.jpg').
    Returns None if the filename doesn't match the pattern or is a hidden file.
    """
    try:
        # Ignore hidden files
        if filename.startswith('.'):
            return None
            
        # This is the correct logic for your filenames:
        # '000_10.jpg' -> split by '_' -> ['000', '10.jpg'] -> take [0] -> '000'
        video_id = filename.split('_')[0]
        
        return video_id
        
    except IndexError:
        # This will catch any file that doesn't have an underscore
        print(f"\n[Warning] Skipping malformed filename: {filename}")
        return None


def check_for_leakage():
    print("--- DeepSight Leakage Check (v2) ---")
    
    train_video_ids = set()
    val_video_ids = set()
    
    # 1. Collect all video IDs from the TRAINING set
    print("Scanning TRAIN split...")
    train_path = DATA_PATH / "train"
    for class_folder in ["real", "fake"]:
        folder_path = train_path / class_folder
        if not folder_path.exists():
            continue
            
        for filename in tqdm(os.listdir(folder_path), desc=f"Train ({class_folder})"):
            video_id = get_video_id(filename)
            if video_id:  # Only add if the ID is valid
                train_video_ids.add(video_id)
            
    print(f"Found {len(train_video_ids)} unique video IDs in TRAIN set.")

    # 2. Collect all video IDs from the VALIDATION set
    print("\nScanning VALIDATION split...")
    val_path = DATA_PATH / "val"
    for class_folder in ["real", "fake"]:
        folder_path = val_path / class_folder
        if not folder_path.exists():
            continue
            
        for filename in tqdm(os.listdir(folder_path), desc=f"Val ({class_folder})"):
            video_id = get_video_id(filename)
            if video_id:  # Only add if the ID is valid
                val_video_ids.add(video_id)
            
    print(f"Found {len(val_video_ids)} unique video IDs in VAL set.")

    # 3. Check for intersection (the leak)
    print("\n--- Checking for Leaks ---")
    leaked_ids = train_video_ids.intersection(val_video_ids)
    
    if not leaked_ids:
        print("✅ SUCCESS: No data leakage found.")
        print("Your train and validation sets are clean.")
    else:
        print(f"🚨 CRITICAL LEAKAGE DETECTED: {len(leaked_ids)} video IDs are in BOTH train and val!")
        print("This invalidates your model's performance.")
        print("\nLeaked IDs (first 10):")
        for i, vid_id in enumerate(leaked_ids):
            if i >= 10:
                print(f"... and {len(leaked_ids) - 10} more.")
                break
            print(f"  - {vid_id}")

if __name__ == "__main__":
    check_for_leakage()