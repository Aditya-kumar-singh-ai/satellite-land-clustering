
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights


# Configuration
SEED = 42
SAMPLE_SIZE = 500
N_CLUSTERS = 15

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load the same reproducible sample
print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
images = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
y_true = np.asarray(images["label"])

print(f"Images loaded: {len(images)}")


# 2. Extract ResNet18 features
print("Loading pretrained ResNet18...")
weights = ResNet18_Weights.DEFAULT
cnn_model = models.resnet18(weights=weights)
cnn_model.fc = torch.nn.Identity()
cnn_model.eval()

preprocess = weights.transforms()
features = []

print("Extracting CNN features...")
with torch.inference_mode():
    for i, sample in enumerate(images):
        image = sample["image"].convert("RGB")
        tensor = preprocess(image).unsqueeze(0)
        embedding = cnn_model(tensor)
        features.append(embedding.squeeze(0).numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{SAMPLE_SIZE}")

X = np.asarray(features, dtype=np.float32)
X_scaled = StandardScaler().fit_transform(X)


# 3. Compare clustering algorithms
algorithms = [
    (
        "K-Means",
        KMeans(
            n_clusters=N_CLUSTERS,
            random_state=SEED,
            n_init=10
        )
    ),
    (
        "Agglomerative",
        AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            linkage="ward"
        )
    )
]

results = []

for name, algorithm in algorithms:
    print(f"\nRunning {name}...")
    start = time.perf_counter()
    cluster_ids = algorithm.fit_predict(X_scaled)
    elapsed = time.perf_counter() - start

    silhouette = silhouette_score(X_scaled, cluster_ids)
    ari = adjusted_rand_score(y_true, cluster_ids)

    results.append({
        "Algorithm": name,
        "K": N_CLUSTERS,
        "Features": X.shape[1],
        "Images": len(images),
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": elapsed
    })

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {elapsed:.3f} seconds")


# 4. Save comparison table
results_df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "cnn_algorithm_comparison.csv"
results_df.to_csv(csv_path, index=False)

print("\n" + "=" * 60)
print("CNN FEATURE CLUSTERING ALGORITHM COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))


# 5. Plot metrics
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(
    results_df["Algorithm"],
    results_df["Silhouette Score"]
)
axes[0].set_title("Silhouette Score")
axes[0].set_ylabel("Score")

axes[1].bar(
    results_df["Algorithm"],
    results_df["ARI"]
)
axes[1].set_title("Adjusted Rand Index (ARI)")
axes[1].set_ylabel("Score")

plt.suptitle("Clustering Algorithms on ResNet18 Features (K=15)")
plt.tight_layout()

plot_path = RESULTS_DIR / "cnn_algorithm_comparison.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nCSV saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")
print("\nExperiment completed!")
