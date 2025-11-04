# scripts/check_integrity_v4.py
import os
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# HARDCODED project path based on your logs.
# This removes all ambiguity.
PROJECT_ROOT = Path("/home/rbk/deepsight-forensics") 

# --- DO NOT EDIT BELOW THIS LINE ---

DATA_PATH = PROJECT_ROOT / "datasets/frames"
SPLITS = ['train', 'val']
CLASSES = ['real', 'fake'] # Assuming these are your folder names

def get_video_id(filename: str) -> str | None:
    """Extracts the video ID (e.g., '000' from '000_10.jpg')."""
    if filename.startswith('.'):
        return None
    try:
        video_id = filename.split('_')[0]
        return video_id
    except IndexError:
        return None

def run_integrity_check():
    print("--- DeepSight Data Integrity Check (v4) ---")
    print(f"Using Project Root: {PROJECT_ROOT}")
    print(f"Checking for Data at: {DATA_PATH}")
    
    all_video_ids = {} # This will store sets of IDs for 'train' and 'val'

    # --- Part 1: File Counting ---
    print("\n[Phase 1] Checking file counts and paths...")
    
    for split in SPLITS:
        print(f"\nScanning Split: [{split}]")
        all_video_ids[split] = set()
        total_images = 0
        
        for cls in CLASSES:
            folder_path = DATA_PATH / split / cls
            
            if not folder_path.exists():
                print(f"  🚨 ERROR: Cannot find folder: {folder_path}")
                continue
            
            try:
                filenames = [f for f in os.listdir(folder_path) if not f.startswith('.')]
                count = len(filenames)
                print(f"  Class [{cls}]: Found {count} images in {folder_path}")
                total_images += count
                
                # --- Part 2: Leakage ID Extraction ---
                if count == 0:
                    print("  [Warning] Folder is empty.")
                    continue
                
                # We'll show a progress bar for this part
                for f_name in tqdm(filenames, desc=f"  Scanning {cls} IDs", leave=False):
                    video_id = get_video_id(f_name)
                    if video_id:
                        all_video_ids[split].add(video_id)
                
            except Exception as e:
                print(f"  🚨 CRITICAL ERROR reading folder {folder_path}: {e}")
        
        print(f"  ---------------------")
        print(f"  Total {split}: {total_images} images")
        print(f"  Unique Video IDs found in {split}: {len(all_video_ids[split])}")


    # --- Part 3: Final Leakage Analysis ---
    print("\n[Phase 2] Checking for Leaks...")
    
    if 'train' not in all_video_ids or 'val' not in all_video_ids or len(all_video_ids['train']) == 0:
        print("🚨 ERROR: Could not complete check. Train/val sets were empty or missing.")
        return

    leaked_ids = all_video_ids['train'].intersection(all_video_ids['val'])
    
    if not leaked_ids:
        print("\n✅ SUCCESS: No data leakage found.")
        print("Your train and validation sets are clean.")
    else:
        print(f"\n🚨 CRITICAL LEAKAGE DETECTED: {len(leaked_ids)} video IDs are in BOTH train and val!")
        print("This invalidates your model's 0.0635 performance.")
        print("\nLeaked IDs (first 10):")
        for i, vid_id in enumerate(leaked_ids):
            if i >= 10:
                print(f"... and {len(leaked_ids) - 10} more.")
                break
            print(f"  - {vid_id}")

if __name__ == "__main__":
    run_integrity_check()