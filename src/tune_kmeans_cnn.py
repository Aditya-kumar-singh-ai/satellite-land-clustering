
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from feature_cache import load_or_extract_features


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SAMPLE_SIZE = 500
K_VALUES = [5, 8, 10, 12, 15, 20]

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD CACHED CNN FEATURES
# ============================================================

print("=" * 60)
print("CNN FEATURE-BASED K-MEANS TUNING")
print("=" * 60)

print("\nLoading CNN features from cache...")

X, y_true = load_or_extract_features(
    sample_size=SAMPLE_SIZE,
    seed=SEED
)

print(f"Images loaded: {len(y_true)}")
print(f"CNN feature shape: {X.shape}")

if len(X) != SAMPLE_SIZE:
    raise ValueError(
        f"Expected {SAMPLE_SIZE} images, but received {len(X)}."
    )

if len(X) != len(y_true):
    raise ValueError("Feature and label counts do not match.")

print("\nStandardizing CNN features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 2. EVALUATE DIFFERENT VALUES OF K
# ============================================================

results = []

for k in K_VALUES:
    print("\n" + "-" * 45)
    print(f"Testing K = {k}")

    start = time.perf_counter()

    model = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10
    )

    cluster_ids = model.fit_predict(X_scaled)

    clustering_seconds = time.perf_counter() - start

    silhouette = silhouette_score(
        X_scaled,
        cluster_ids
    )

    ari = adjusted_rand_score(
        y_true,
        cluster_ids
    )

    clusters_found = len(np.unique(cluster_ids))

    results.append({
        "K": k,
        "Clusters Found": clusters_found,
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": clustering_seconds
    })

    print(f"Clusters found: {clusters_found}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {clustering_seconds:.3f} seconds")


# ============================================================
# 3. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "cnn_kmeans_tuning.csv"
results_df.to_csv(csv_path, index=False)

print(f"\nResults saved to: {csv_path}")


# ============================================================
# 4. PLOT SILHOUETTE SCORE AND ARI
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Silhouette Score plot
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

# Adjusted Rand Index plot
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
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Plot saved to: {plot_path}")


# ============================================================
# 5. REPORT THE BEST K FOR EACH METRIC
# ============================================================

best_silhouette = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]

best_ari = results_df.loc[
    results_df["ARI"].idxmax()
]

print("\n" + "=" * 60)
print("K-MEANS TUNING RESULTS — RESNET18 FEATURES")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print(
    f"\nBest K by Silhouette Score: "
    f"{int(best_silhouette['K'])} "
    f"(Score: {best_silhouette['Silhouette Score']:.4f})"
)

print(
    f"Best K by ARI: "
    f"{int(best_ari['K'])} "
    f"(ARI: {best_ari['ARI']:.4f})"
)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED!")
print("=" * 60)

print(f"CSV:  {csv_path}")
print(f"Plot: {plot_path}")
