
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from feature_cache import load_or_extract_features


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SAMPLE_SIZE = 10000
N_CLUSTERS = 12
SILHOUETTE_SAMPLE_SIZE = 500

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD CACHED CNN FEATURES
# ============================================================

print("=" * 60)
print("CNN CLUSTERING ALGORITHM COMPARISON")
print("=" * 60)

print("\nLoading CNN features from cache...")

X, y_true = load_or_extract_features(
    sample_size=SAMPLE_SIZE,
    seed=SEED
)

print("Feature shape:", X.shape)
print("Images loaded:", len(y_true))

if len(X) != SAMPLE_SIZE:
    raise ValueError("Unexpected number of images.")

if len(X) != len(y_true):
    raise ValueError("Feature and label counts do not match.")


# ============================================================
# 2. STANDARDIZE FEATURES
# ============================================================

print("\nStandardizing CNN features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 3. DEFINE CLUSTERING ALGORITHMS
# ============================================================

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


# ============================================================
# 4. EVALUATE ALGORITHMS
# ============================================================

results = []

for name, algorithm in algorithms:

    print("\n" + "-" * 45)
    print(f"Running {name}...")

    start = time.perf_counter()

    cluster_ids = algorithm.fit_predict(X_scaled)

    elapsed = time.perf_counter() - start

    silhouette = silhouette_score(
        X_scaled,
        cluster_ids,
        sample_size=SILHOUETTE_SAMPLE_SIZE,
        random_state=SEED
    )

    ari = adjusted_rand_score(
        y_true,
        cluster_ids
    )

    clusters_found = len(set(cluster_ids))

    results.append({
        "Algorithm": name,
        "K": N_CLUSTERS,
        "Clusters Found": clusters_found,
        "Features": X.shape[1],
        "Images": len(y_true),
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": elapsed
    })

    print(f"Clusters found: {clusters_found}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {elapsed:.3f} seconds")


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "cnn_algorithm_comparison_10000.csv"

results_df.to_csv(csv_path, index=False)

print("\nCNN FEATURE CLUSTERING ALGORITHM COMPARISON")
print(results_df.round(4).to_string(index=False))


# ============================================================
# 6. PLOT COMPARISON
# ============================================================

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

plt.suptitle(
    "Clustering Algorithms on ResNet18 Features "
    "(10,000 Images, K=12)"
)

plt.tight_layout()

plot_path = RESULTS_DIR / "cnn_algorithm_comparison_10000.png"

plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nCSV saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED!")
print("=" * 60)
