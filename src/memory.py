import numpy as np
import os
import time

data = np.load("C:/o/speech_project/data/processed/embeddings.npy", allow_pickle=True)

def save_version(name, convert):
    new_data = []

    for d in data:
        emb = convert(d["embedding"])
        new_data.append({
            "speaker": d["speaker"],
            "word": d["word"],
            "embedding": emb
        })
    path = f"C:/o/speech_project/data/processed/embeddings_{name}.npy"
    np.save(path, new_data)
    size = os.path.getsize(path) / (1024 * 1024)  # MB
    print(f"{name} size: {size:.2f} MB")

def f64(x):
    return x.astype(np.float64)

def f32(x):
    return x.astype(np.float32)

def f16(x):
    return x.astype(np.float16)

def int8(x):
    x_min = x.min()
    x_max = x.max()
    x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
    return (x_scaled * 255).astype(np.uint8)

save_version("float64", f64)
save_version("float32", f32)
save_version("float16", f16)
save_version("int8", int8)