import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# ===== CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Corrected paths (use consistent format)
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")  # Changed from "raw_vid/Users/..."
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")     # Removed leading slash
SRC_DIR = os.path.join(PROJECT_ROOT, "src")                        # Simplified path

# Create directories if they don't exist
os.makedirs(os.path.join(PROCESSED_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "metadata"), exist_ok=True)
os.makedirs(SRC_DIR, exist_ok=True)

# ===== 1. ORGANIZE FILES ===== 
def organize_files():
    """Move all Python files to src/ and validate structure."""
    print("🔧 Organizing files...")
    
    for item in os.listdir(PROJECT_ROOT):
        if item.endswith(".py") and item != os.path.basename(__file__):
            src = os.path.join(PROJECT_ROOT, item)
            dst = os.path.join(SRC_DIR, item)
            shutil.move(src, dst)
            print(f"Moved: {item} → src/{item}")

    # Validate raw_videos structure
    for category in ["anomaly", "normal"]:
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, category), exist_ok=True)

# ===== 2. PROCESS VIDEOS =====
def extract_features(video_path):
    """Extract motion features from a video."""
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    motion_scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            motion = cv2.absdiff(gray, prev_frame)
            motion_scores.append(np.mean(motion))
        prev_frame = gray
    
    cap.release()
    return {
        "mean_motion": np.mean(motion_scores),
        "max_motion": np.max(motion_scores),
        "std_motion": np.std(motion_scores),
    }

def process_videos():
    """Process all videos and save features/frames."""
    print("🎥 Processing videos...")
    
    metadata = []
    
    for category in ["anomaly", "normal"]:
        category_dir = os.path.join(RAW_VIDEOS_DIR, category)
        label = 1 if category == "anomaly" else 0
        
        # Debug: Print the directory being scanned
        print(f"\nChecking directory: {category_dir}")
        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue
            
        video_files = [f for f in os.listdir(category_dir) 
                      if f.lower().endswith((".mp4", ".avi", ".mov"))]
        
        if not video_files:
            print(f"⚠️ No videos found in {category_dir}")
            continue
            
        for video_name in tqdm(video_files, desc=f"Processing {category}"):
            video_path = os.path.join(category_dir, video_name)
            
            # Debug: Verify video path
            if not os.path.exists(video_path):
                print(f"❌ Video not found: {video_path}")
                continue
                
            features = extract_features(video_path)
            features_path = os.path.join(PROCESSED_DIR, "features", f"{os.path.splitext(video_name)[0]}.npy")
            np.save(features_path, features)
            
            metadata.append({
                "video_name": video_name,
                "category": category,
                "label": label,
                "features_path": features_path,
                "mean_motion": features["mean_motion"],
            })
    
    if metadata:
        import pandas as pd
        df = pd.DataFrame(metadata)
        csv_path = os.path.join(PROCESSED_DIR, "metadata", "video_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved metadata to {csv_path}")
    else:
        print("❌ No videos were processed!")

# ===== MAIN =====
if __name__ == "__main__":
    organize_files()
    process_videos()