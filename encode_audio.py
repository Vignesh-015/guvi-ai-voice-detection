import base64

AUDIO_FILE = r"E:\guvi buildathon\dataset_processed\ai\1.wav"
OUTPUT_FILE = "audio_base64.txt"

with open(AUDIO_FILE, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

with open(OUTPUT_FILE, "w") as f:
    f.write(encoded)

print("✅ Base64 saved to audio_base64.txt")
