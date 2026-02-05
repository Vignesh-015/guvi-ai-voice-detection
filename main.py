from fastapi import FastAPI, Header, HTTPException
import base64
import os
import joblib
import numpy as np
import librosa
import tempfile

app = FastAPI(title="AI Generated Voice Detection API")

model = joblib.load("voice_detector.pkl")

API_KEY = os.getenv("API_KEY", "guvi_secret_key")

def extract_features_from_file(path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc, centroid, zcr, rms]).reshape(1, -1)

@app.post("/detect-voice")
def detect_voice(
    payload: dict,
    x_api_key: str = Header(None, alias="x-api-key")
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        audio_base64 = payload["audio_base64"]
        audio_bytes = base64.b64decode(audio_base64)

        # ✅ WRITE TO TEMP WAV FILE
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        features = extract_features_from_file(temp_path)
        os.remove(temp_path)

        probs = model.predict_proba(features)[0]
        pred = int(np.argmax(probs))

        return {
            "classification": "AI_GENERATED" if pred == 1 else "HUMAN",
            "confidence": float(probs[pred])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid audio input")
