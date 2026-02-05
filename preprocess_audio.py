import os
import librosa
import soundfile as sf

INPUT_DIRS = {
    "human": r"E:\guvi buildathon\dataset\human",
    "ai": r"E:\guvi buildathon\dataset\ai"
}

OUTPUT_BASE = r"E:\guvi buildathon\dataset_processed"

TARGET_SR = 16000
MAX_DURATION = 5  # seconds

os.makedirs(OUTPUT_BASE, exist_ok=True)

for label, in_dir in INPUT_DIRS.items():
    out_dir = os.path.join(OUTPUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔄 Processing {label} audio...")

    for file in os.listdir(in_dir):
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            in_path = os.path.join(in_dir, file)
            out_path = os.path.join(out_dir, file.rsplit(".", 1)[0] + ".wav")

            try:
                y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)
                y, _ = librosa.effects.trim(y)
                y = y[:TARGET_SR * MAX_DURATION]

                sf.write(out_path, y, TARGET_SR)
            except Exception as e:
                print(f"❌ Skipped {file}: {e}")

print("✅ Audio preprocessing complete (saved as WAV)")
