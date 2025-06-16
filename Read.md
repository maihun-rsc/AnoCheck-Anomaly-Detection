
---

# Video Anomaly Detection

This project detects anomalies in videos (e.g., abuse vs. normal behavior) using motion and optical flow features extracted with OpenCV, followed by training a Random Forest classifier with Scikit-learn. It includes feature extraction, model training, prediction, and evaluation with visualizations.

## Project Structure
```
video-anomaly-detection/
├── data/
│   ├── raw_videos/              # Raw video files
│   │   ├── anomaly/             # Anomaly videos (e.g., Abuse027_x264.mp4)
│   │   └── normal/              # Normal videos (e.g., Normal_Videos_010_x264.mp4)
│   ├── processed/               # Processed data
│   │   ├── features/            # .npy files with basic motion features
│   │   ├── frames/              # Extracted frames (if used)
│   │   └── metadata/            # Metadata files
│   │       ├── video_metadata.csv  # Video info from initial processing
│   │       └── enhanced_features.csv  # Final features for training
├── Sample/                      # Sample videos shot from phone for testing
│   ├── sample_anomaly_1.mp4     # Example anomaly video
│   ├── sample_normal_1.mp4      # Example normal video
│   └── ...                      # Add more as needed
├── src/                         # Source code
│   ├── organize_and_process.py  # Step 1: Initial feature extraction
│   ├── feature_engineering.py   # Step 2: Enhance features with optical flow
│   ├── train_model.py           # Step 3: Train the model
│   ├── detect_anomalies.py      # Step 4: Predict on new videos
│   └── evaluation.ipynb         # Step 5: Visualize results
├── models/                      # Trained model
│   └── rf_model.pkl             # Random Forest model
└── README.md                    # This file
```

## Prerequisites
- **Python 3.8+**
- Install dependencies:
  ```bash
  pip install numpy pandas opencv-python scikit-learn joblib matplotlib seaborn tqdm imblearn jupyter
  ```
## To download all the libraries used in the project
```bash
  pip install -r requirements.txt
  ```

## Execution Steps
Follow these steps to run the project from scratch or test with sample videos.

### 1. Organize and Process Videos
Extracts basic motion features from raw videos and generates metadata.
- **File**: `src/organize_and_process.py`
- **Input**: Videos in `data/raw_videos/anomaly/` and `data/raw_videos/normal/`
- **Output**: `.npy` files in `data/processed/features/` and `video_metadata.csv` in `data/processed/metadata/`
- **Run**:
  ```bash
  python src/organize_and_process.py
  ```

### 2. Feature Engineering
Enhances features with optical flow and saves them for training.
- **File**: `src/feature_engineering.py`
- **Input**: `.npy` files and raw videos
- **Output**: `enhanced_features.csv` in `data/processed/metadata/`
- **Run**:
  ```bash
  python src/feature_engineering.py
  ```
- **Note**: Takes ~1 hour on a MacBook Air M3 for 199 videos (can be optimized with multiprocessing).

### 3. Train the Model
Trains a Random Forest classifier on the enhanced features.
- **File**: `src/train_model.py`
- **Input**: `enhanced_features.csv`
- **Output**: `rf_model.pkl` in `models/`
- **Run**:
  ```bash
  python src/train_model.py
  ```
- **Output Example**:
  ```
  Classification Report:
                precision    recall  f1-score   support
           0       0.88      0.85      0.87        34
           1       0.29      0.33      0.31         6
  ```

### 4. Predict on New Videos
Uses the trained model to classify new videos, including samples from `Sample/`.
- **File**: `src/detect_anomalies.py`
- **Input**: `rf_model.pkl` and a video file (e.g., from `Sample/`)
- **Edit**: Update `new_video_path` in the script, e.g.:
  ```python
  new_video_path = "/Users/Video anomaly/Sample/sample_anomaly_1.mp4"
  ```
- **Run**:
  ```bash
  python src/detect_anomalies.py
  ```
- **Example Output**:
  ```
  Prediction for /Users/Video anomaly/Sample/sample_anomaly_1.mp4: Anomaly
  Prediction for /Users/Video anomaly/Sample/sample_normal_1.mp4: Normal
  ```
- **Note**: Successfully predicts phone-shot sample videos!

### 5. Evaluate Results
Visualizes model performance with plots (confusion matrix, ROC curve, etc.).
- **File**: `src/evaluation.ipynb`
- **Input**: `rf_model.pkl` and `enhanced_features.csv`
- **Run**:
  ```bash
  jupyter notebook src/evaluation.ipynb
  ```
  - Execute all cells in the notebook.
- **Output**: Plots like confusion matrix heatmap, ROC curve, feature importance.

## Sample Videos
The `Sample/` folder contains custom videos shot from a phone, tested with `detect_anomalies.py`. Examples:
- `sample_anomaly_1.mp4`: Predicted as “Anomaly.”
- `sample_normal_1.mp4`: Predicted as “Normal.”
These show the model generalizes to real-world data beyond the original dataset.

## Notes
- **Dataset**: Original data has 150 normal and 49 anomaly videos.
- **Performance**: Current model has 78% accuracy but struggles with anomaly recall (0.33). Tweak `class_weight='balanced'` in `train_model.py` for better anomaly detection.
- **Optimization**: Feature engineering can be sped up with multiprocessing or frame skipping—see code comments for details.



---


