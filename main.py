from fastapi import FastAPI, HTTPException, Request
import base64
import io
import os
import joblib
import numpy as np
import librosa

app = FastAPI(title="AI Generated Voice Detection API")

model = joblib.load("voice_detector.pkl")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")

def extract_features_from_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    y, _ = librosa.effects.trim(y)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc, centroid, zcr, rms]).reshape(1, -1)

@app.post("/detect-voice")
def detect_voice(payload: dict, request: Request):
    # 🔐 Read Authorization header directly (GUVI-safe)
    auth_header = request.headers.get("authorization")

    if auth_header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if "audio_base64" not in payload:
        raise HTTPException(status_code=400, detail="audio_base64 missing")

    try:
        audio_bytes = base64.b64decode(payload["audio_base64"])
        features = extract_features_from_bytes(audio_bytes)

        probs = model.predict_proba(features)[0]
        pred = int(np.argmax(probs))

        return {
            "classification": "AI_GENERATED" if pred == 1 else "HUMAN",
            "confidence": float(probs[pred])
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio input")
