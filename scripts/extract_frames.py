# scripts/extract_frames.py

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random

# --- Configuration ---
FRAMES_PER_VIDEO = 20
ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_REAL_DIR = ROOT_DIR / "datasets/original_sequences/youtube/c23/videos"
SOURCE_FAKE_DIR = ROOT_DIR / "datasets/manipulated_sequences/Deepfakes/c23/videos"
DESTINATION_DIR = ROOT_DIR / "datasets/frames"

# --- Main Extraction Logic ---
def extract_frames_from_dir(source_dir: Path, dest_dir: Path, video_list: list):
    """Extracts frames from a list of videos from a source directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for video_filename_str in tqdm(video_list, desc=f"Extracting to {dest_dir.name}"):
        video_path = source_dir / video_filename_str
        
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < FRAMES_PER_VIDEO or FRAMES_PER_VIDEO == 0:
            cap.release()
            continue

        frame_indices = sorted(random.sample(range(total_frames), FRAMES_PER_VIDEO))
        
        frame_idx_ptr = 0
        frame_count = 0
        
        while cap.isOpened() and frame_idx_ptr < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == frame_indices[frame_idx_ptr]:
                frame_name = f"{video_path.stem}_{frame_idx_ptr}.jpg"
                save_path = dest_dir / frame_name
                cv2.imwrite(str(save_path), frame)
                frame_idx_ptr += 1
                
            frame_count += 1
            
        cap.release()


if __name__ == "__main__":
    print("Starting frame extraction process...")
    
    real_videos = sorted(os.listdir(SOURCE_REAL_DIR))
    fake_videos = sorted(os.listdir(SOURCE_FAKE_DIR))
    
    # We will use an 80/20 split for train/validation
    real_split_idx = int(len(real_videos) * 0.8)
    fake_split_idx = int(len(fake_videos) * 0.8)

    # Process REAL videos
    print("\n--- Processing REAL videos ---")
    extract_frames_from_dir(SOURCE_REAL_DIR, DESTINATION_DIR / "train/real", real_videos[:real_split_idx])
    extract_frames_from_dir(SOURCE_REAL_DIR, DESTINATION_DIR / "val/real", real_videos[real_split_idx:])

    # Process FAKE videos
    print("\n--- Processing FAKE videos ---")
    extract_frames_from_dir(SOURCE_FAKE_DIR, DESTINATION_DIR / "train/fake", fake_videos[:fake_split_idx])
    extract_frames_from_dir(SOURCE_FAKE_DIR, DESTINATION_DIR / "val/fake", fake_videos[fake_split_idx:])
            
    print("\nFrame extraction complete! ✨")