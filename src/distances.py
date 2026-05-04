import numpy as np
import time
from scipy.stats import spearmanr
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
    print("Testing: ", name)
    intra = []
    inter = []
    
    start_time = time.perf_counter() 
    
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
                    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    intra_mean = np.mean(intra)
    inter_mean = np.mean(inter)
    ratio = intra_mean / inter_mean
    
    results[name] = {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "ratio": ratio,
        "time_seconds": execution_time,
        "intra_list": intra,  
        "inter_list": inter   
    }
    
    print(f"Time taken for {name}: {execution_time:.4f} seconds")
    print("Intra:", intra_mean)
    print("Inter:", inter_mean)
    print("Ratio:", ratio)
    print("-" * 30)

float64_all_distances = results["float64"]["intra_list"] + results["float64"]["inter_list"]
int8_all_distances = results["int8"]["intra_list"] + results["int8"]["inter_list"]
correlation, p_value = spearmanr(float64_all_distances, int8_all_distances)
print(f"Spearman Rank Correlation (Float64 vs Int8): {correlation:.4f}")