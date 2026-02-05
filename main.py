from fastapi import FastAPI, Header, HTTPException
import base64, io, os, re
import joblib
import numpy as np
import librosa

app = FastAPI(title="AI Generated Voice Detection API")

model = joblib.load("voice_detector.pkl")

API_KEY = os.getenv("API_KEY", "guvi_secret_key")

def decode_audio_from_payload(payload: dict) -> bytes:
    """
    Handles GUVI + CMD payload formats safely
    """
    b64 = None

    if "audio_base64" in payload:
        b64 = payload["audio_base64"]
    elif "audio_base64_format" in payload:
        b64 = payload["audio_base64_format"]
    else:
        raise ValueError("Base64 audio not found")

    # remove spaces, newlines, tabs
    b64 = re.sub(r"\s+", "", b64)

    return base64.b64decode(b64)

def extract_features(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
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
        audio_bytes = decode_audio_from_payload(payload)
        features = extract_features(audio_bytes)

        probs = model.predict_proba(features)[0]
        pred = int(np.argmax(probs))

        return {
            "classification": "AI_GENERATED" if pred == 1 else "HUMAN",
            "confidence": float(probs[pred])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid audio input")
