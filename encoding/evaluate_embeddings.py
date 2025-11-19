import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "encoding/encoded_output_final/aggregated_doc_embeddings_fixed.pth"
NUM_SAMPLES = 5000   # for visualization; can increase if you have enough VRAM
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Load data
# ---------------------------
print(f"[INFO] Loading embeddings from: {DATA_PATH}")
data = torch.load(DATA_PATH, map_location="cpu")

if not isinstance(data, list):
    raise TypeError("Expected a list of dicts with 'embeddings' and 'court_type_idx'")

embeddings = torch.stack([torch.tensor(d["embeddings"], dtype=torch.float32) for d in data])
labels = torch.tensor([int(d["court_type_idx"]) for d in data], dtype=torch.long)
ids = [d.get("case_id", f"sample_{i}") for i, d in enumerate(data)]

print(f"[INFO] Loaded {len(embeddings)} samples, dim={embeddings.shape[1]}")

# ---------------------------
# Basic stats
# ---------------------------
print("\n=== Basic Embedding Stats ===")
print("Shape:", embeddings.shape)
print("Mean (first 5 dims):", embeddings[:, :5].mean(dim=0))
print("Std (first 5 dims):", embeddings[:, :5].std(dim=0))
print("NaN count:", torch.isnan(embeddings).sum().item())
print("Zero vectors:", (embeddings.abs().sum(dim=1) == 0).sum().item())

# Class distribution
unique, counts = torch.unique(labels, return_counts=True)
print("\n=== Court Type Counts ===")
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f"Court {u}: {c} samples")

# ---------------------------
# Show a few random samples
# ---------------------------
print("\n=== Random Samples (5) ===")
for i in random.sample(range(len(embeddings)), 5):
    print(f"ID: {ids[i]}, Label: {labels[i].item()}, Embedding[0:10]: {embeddings[i][:10].tolist()}")

# ---------------------------
# PCA + t-SNE Visualization
# ---------------------------
n = min(NUM_SAMPLES, len(embeddings))
idx = np.random.choice(len(embeddings), n, replace=False)
sample_emb = embeddings[idx].numpy()
sample_labels = labels[idx].numpy()

print("\n[INFO] Running PCA → t-SNE (this might take ~1–3 min)...")
pca = PCA(n_components=50, random_state=SEED)
reduced = pca.fit_transform(sample_emb)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=SEED, verbose=1)
tsne_result = tsne.fit_transform(reduced)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(8, 6))
palette = sns.color_palette("husl", len(np.unique(sample_labels)))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=sample_labels, palette=palette, s=15, alpha=0.7)
plt.title("t-SNE of Court Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(title="Court Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
