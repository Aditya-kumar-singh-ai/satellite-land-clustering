
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights


# Configuration
SEED = 42
SAMPLE_SIZE = 500
K_VALUES = [5, 8, 10, 12, 15, 20]

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load the same sample used in previous experiments
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

# Standardize once, consistently for every K
X_scaled = StandardScaler().fit_transform(X)


# 3. Evaluate each K
results = []

for k in K_VALUES:
    print(f"\nTesting K = {k}")
    start = time.perf_counter()

    model = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10
    )
    cluster_ids = model.fit_predict(X_scaled)

    clustering_seconds = time.perf_counter() - start
    silhouette = silhouette_score(X_scaled, cluster_ids)
    ari = adjusted_rand_score(y_true, cluster_ids)

    results.append({
        "K": k,
        "Clusters Found": len(np.unique(cluster_ids)),
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": clustering_seconds
    })

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {clustering_seconds:.3f} seconds")


# 4. Save the results
results_df = pd.DataFrame(results)
csv_path = RESULTS_DIR / "cnn_kmeans_tuning.csv"
results_df.to_csv(csv_path, index=False)

# 5. Plot Silhouette Score and ARI
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    results_df["K"],
    results_df["Silhouette Score"],
    marker="o"
)
axes[0].set_title("CNN Features: Silhouette Score by K")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_xticks(K_VALUES)
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    results_df["K"],
    results_df["ARI"],
    marker="o"
)
axes[1].set_title("CNN Features: ARI by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Adjusted Rand Index")
axes[1].set_xticks(K_VALUES)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plot_path = RESULTS_DIR / "cnn_kmeans_tuning.png"
plt.savefig(plot_path, dpi=200)
plt.close()

# 6. Report the best K for each metric
best_silhouette = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]
best_ari = results_df.loc[results_df["ARI"].idxmax()]

print("\n" + "=" * 55)
print("K-MEANS TUNING RESULTS — RESNET18 FEATURES")
print("=" * 55)
print(results_df.round(4).to_string(index=False))

print(
    f"\nBest K by Silhouette Score: "
    f"{int(best_silhouette['K'])} "
    f"({best_silhouette['Silhouette Score']:.4f})"
)
print(
    f"Best K by ARI: "
    f"{int(best_ari['K'])} "
    f"({best_ari['ARI']:.4f})"
)

print(f"\nCSV saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")
print("\nExperiment completed!")
