import pandas as pd
import numpy as np
import cv2
import joblib
from pathlib import Path

# Directories
PROJECT_ROOT = Path("/Users/Video anomaly")
MODELS_DIR = PROJECT_ROOT / "models"

# Load model
model = joblib.load(MODELS_DIR / "rf_model.pkl")
print("✅ Model loaded")

# Feature extraction functions (same as before)
def extract_features(video_path):
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

def calc_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    flow_magnitudes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(np.mean(magnitude))
        prev_frame = gray
    cap.release()
    return {
        'mean_flow': np.mean(flow_magnitudes) if flow_magnitudes else 0,
        'max_flow': np.max(flow_magnitudes) if flow_magnitudes else 0
    }

# Test a new video
new_video_path = "/Users/Video anomaly/data/raw_videos/normal/Normal_Videos_010_x264.mp4"  # Replace with a real path
basic_features = extract_features(new_video_path)
flow_features = calc_optical_flow(new_video_path)
features = {**basic_features, **flow_features}

# Predict
feature_columns = ['mean_motion', 'max_motion', 'std_motion', 'mean_flow', 'max_flow']
X_new = pd.DataFrame([features], columns=feature_columns)
prediction = model.predict(X_new)[0]
print(f"Prediction for {new_video_path}: {'Anomaly' if prediction == 1 else 'Normal'}")