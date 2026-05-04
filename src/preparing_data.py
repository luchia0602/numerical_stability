import os
import json
from textgrid import TextGrid

root = r"C:/o/speech_project/data/raw/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only"
all_data = []
for speaker in os.listdir(root):
    speaker_dir = os.path.join(root, speaker)
    if not os.path.isdir(speaker_dir):
        continue
    for file in os.listdir(speaker_dir):
        if file.endswith(".TextGrid"):
            tg_path = os.path.join(speaker_dir, file)
            wav_path = tg_path.replace(".TextGrid", ".wav")
            if not os.path.exists(wav_path):
                continue  # skip if wav missing
            tg = TextGrid.fromFile(tg_path)
            tier = tg.getFirst("words")
            for interval in tier:
                word = interval.mark.strip()
                if word:
                    all_data.append({
                        "speaker": speaker,
                        "word": word,
                        "start": interval.minTime,
                        "end": interval.maxTime,
                        "wav": wav_path
                    })
print(f"{len(all_data)} word instances in total")
output_path = "C:/o/speech_project/data/processed/metadata.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)