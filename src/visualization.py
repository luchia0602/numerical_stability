import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict

data = np.load("C:/o/speech_project/data/processed/embeddings.npy", allow_pickle=True)
by_word = defaultdict(list)
for d in data:
    by_word[d["word"]].append(d)

def to_float64(x):
    return x.astype(np.float64)

def to_float32(x):
    return x.astype(np.float32)

def to_float16(x):
    return x.astype(np.float16)

def to_int8(x):
    x_min = x.min()
    x_max = x.max()
    x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
    x_q = (x_scaled * 255).astype(np.uint8)
    return x_q.astype(np.float32)

precisions = {
    "float64": to_float64,
    "float32": to_float32,
    "float16": to_float16,
    "int8": to_int8
}

results = {}

for name, convert in precisions.items():
    intra = []
    inter = []
    for word, items in by_word.items():
        if len(items) < 2:
            continue
        embeddings = np.array([convert(d["embedding"]) for d in items])
        speakers = [d["speaker"] for d in items]
        D = cosine_distances(embeddings)
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if speakers[i] == speakers[j]:
                    intra.append(D[i, j])
                else:
                    inter.append(D[i, j])
    results[name] = (intra, inter)

for name, (intra, inter) in results.items():
    plt.figure()
    plt.hist(intra, bins=50, alpha=0.6, label="Intra")
    plt.hist(inter, bins=50, alpha=0.6, label="Inter")
    plt.title(f"Distance distribution ({name})")
    plt.xlabel("Cosine distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"C:/o/speech_project/data/processed/plot_{name}.png")
    plt.close()

precision_names = list(precisions.keys())
intra_means = [np.mean(results[name][0]) for name in precision_names]
inter_means = [np.mean(results[name][1]) for name in precision_names]

plt.figure(figsize=(8, 5))
plt.plot(precision_names, intra_means, marker='o', linestyle='-', linewidth=2, label='Intra-speaker Mean')
plt.plot(precision_names, inter_means, marker='s', linestyle='-', linewidth=2, label='Inter-speaker Mean')

plt.title("Intra vs. Inter-Speaker Distances Across Precision Levels")
plt.ylabel("Average Cosine Distance")
plt.xlabel("Precision Format")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("C:/o/speech_project/data/processed/precision_comparison_plot.png")
plt.close()