# scripts/extract_frames.py

import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Number of frames to extract from each video
FRAMES_PER_VIDEO = 20

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Source and destination paths
SOURCE_REAL = ROOT_DIR / "datasets/original_sequences/youtube/c23/videos"
SOURCE_FAKE = ROOT_DIR / "datasets/manipulated_sequences/Deepfakes/c23/videos"
DESTINATION_DIR = ROOT_DIR / "datasets/frames"

# --- Main Extraction Logic ---
def extract_frames(video_path: Path, destination_folder: Path, video_filename: str):
    """Extracts a fixed number of frames from a single video file."""
    if not video_path.exists():
        # print(f"Warning: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure we can extract frames and avoid division by zero
    if total_frames < FRAMES_PER_VIDEO or FRAMES_PER_VIDEO == 0:
        return

    # Calculate which frames to extract to get an even distribution
    frame_indices = [int(i) for i in (total_frames / FRAMES_PER_VIDEO * n for n in range(FRAMES_PER_VIDEO))]
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_indices:
            frame_name = f"{video_filename.stem}_{extracted_count}.jpg"
            save_path = destination_folder / frame_name
            cv2.imwrite(str(save_path), frame)
            extracted_count += 1
            
        frame_count += 1
        
    cap.release()


if __name__ == "__main__":
    print("Starting frame extraction process...")
    
    video_files = sorted(os.listdir(SOURCE_REAL))
    
    # We will use an 80/20 split for train/validation
    split_index = int(len(video_files) * 0.8)
    train_videos = video_files[:split_index]
    val_videos = video_files[split_index:]
    
    # Define sets for processing
    sets = {
        "train": train_videos,
        "val": val_videos
    }

    # Process all videos
    for set_name, video_list in sets.items():
        print(f"\nProcessing '{set_name}' set...")
        
        # Create destination folders
        dest_real = DESTINATION_DIR / set_name / "real"
        dest_fake = DESTINATION_DIR / set_name / "fake"
        dest_real.mkdir(parents=True, exist_ok=True)
        dest_fake.mkdir(parents=True, exist_ok=True)
        
        for video_filename_str in tqdm(video_list, desc=f"Extracting {set_name} frames"):
            video_filename = Path(video_filename_str)
            
            # Extract from REAL video
            real_video_path = SOURCE_REAL / video_filename
            extract_frames(real_video_path, dest_real, video_filename)
            
            # Extract from FAKE video
            fake_video_path = SOURCE_FAKE / video_filename
            extract_frames(fake_video_path, dest_fake, video_filename)
            
    print("\nFrame extraction complete! ✨")