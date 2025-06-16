import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

# Directories
PROJECT_ROOT = Path("/Users/Video anomaly")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Load enhanced features
input_path = PROCESSED_DIR / "metadata" / "enhanced_features.csv"
df = pd.read_csv(input_path)
print("🔍 Loaded enhanced features with columns:", df.columns.tolist())

# Features and labels
feature_columns = ['mean_motion', 'max_motion', 'std_motion', 'mean_flow', 'max_flow']
X = df[feature_columns]
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete")

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
model_path = MODELS_DIR / "rf_model.pkl"
joblib.dump(model, model_path)
print(f"💾 Model saved to {model_path}")