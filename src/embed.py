import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
segments = np.load("C:\\o\\speech_project\\data\\processed\\segments.npy", allow_pickle=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()
embeddings = []

for i, s in enumerate(segments):
    audio = np.array(s["audio"], dtype=np.float32)
    inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append({
            "speaker": s["speaker"],
            "word": s["word"],
            "embedding": emb
        })
np.save("C:\\o\\speech_project\\data\\processed\\embeddings.npy", embeddings)