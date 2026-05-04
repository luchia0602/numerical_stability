import json
import librosa
import numpy as np
import os

with open("C:/o/speech_project/data/processed/metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)
segments = []

for i, d in enumerate(data):
    y, sr = librosa.load(d["wav"], sr=16000)
    start = int(d["start"] * sr)
    end = int(d["end"] * sr)
    segment = y[start:end]
    if len(segment) > 200:  # getting rid of short segments (probably noise)
        segments.append({
           "speaker": d["speaker"],
           "word": d["word"],
           "audio": segment.tolist()
       })
    if i % 500 == 0:
        print(f"Processed {i}/{len(data)}")
print("Total segments:", len(segments))
os.makedirs("C:/o/speech_project/data/processed", exist_ok=True)
np.save("C:/o/speech_project/data/processed/segments.npy", segments)