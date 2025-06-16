
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_VIDEOS_DIR = r"/Users/Video anomaly/data/raw_videos"
PROCESSED_DIR = r"/Users/Video anomaly/data/processed"
os.makedirs(os.path.join(PROCESSED_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "metadata"), exist_ok=True)

# ========== CORE FUNCTIONS ==========
def extract_features(video_path):
    """Calculate motion features (mean, max, std of frame differences)"""
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

def extract_frames(video_path, output_dir, resize=(224, 224)):
    """Save video frames as JPEG images"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, resize)
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{frame_count:04d}.jpg"), 
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]  # 90% quality to save space
        )
        frame_count += 1
    
    cap.release()
    return frame_count

# ========== MAIN PROCESS ==========
def process_videos():
    metadata = []
    
    for category in ["anomaly", "normal"]:
        category_dir = os.path.join(RAW_VIDEOS_DIR, category)
        if not os.path.exists(category_dir):
            print(f"⚠️ Skipping missing directory: {category_dir}")
            continue
            
        for video_name in tqdm(os.listdir(category_dir), desc=f"Processing {category}"):
            if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
                continue
                
            video_path = os.path.join(category_dir, video_name)
            video_id = os.path.splitext(video_name)[0]
            
            # 1. Extract motion features
            features = extract_features(video_path)
            np.save(
                os.path.join(PROCESSED_DIR, "features", f"{video_id}.npy"), 
                features
            )
            
            # 2. Extract frames
            frames_dir = os.path.join(PROCESSED_DIR, "frames", video_id)
            frame_count = extract_frames(video_path, frames_dir)
            
            # Update metadata
            metadata.append({
                "video_id": video_id,
                "category": category,
                "label": 1 if category == "anomaly" else 0,
                "features_path": f"features/{video_id}.npy",
                "frames_dir": f"frames/{video_id}",
                "frame_count": frame_count,
                **features  # Unpacks mean_motion, max_motion, std_motion
            })
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(
        os.path.join(PROCESSED_DIR, "metadata", "video_metadata.csv"),
        index=False
    )
    print(f"✅ Processing complete! Check {PROCESSED_DIR} for results.")

if __name__ == "__main__":
    process_videos()