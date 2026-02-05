import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

HUMAN_DIR = r"E:\guvi buildathon\dataset_processed\human"
AI_DIR = r"E:\guvi buildathon\dataset_processed\ai"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc, centroid, zcr, rms])

X = []
y = []

print("🔍 Extracting HUMAN features...")
for file in os.listdir(HUMAN_DIR):
    if file.endswith(".wav"):
        X.append(extract_features(os.path.join(HUMAN_DIR, file)))
        y.append(0)  # HUMAN

print("🔍 Extracting AI features...")
for file in os.listdir(AI_DIR):
    if file.endswith(".wav"):
        X.append(extract_features(os.path.join(AI_DIR, file)))
        y.append(1)  # AI_GENERATED

X = np.array(X)
y = np.array(y)

print("🧠 Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "voice_detector.pkl")

print("✅ Model trained and saved as voice_detector.pkl")
