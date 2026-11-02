
from pathlib import Path
import time

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from feature_cache import load_or_extract_features


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 10000
N_CLUSTERS = 12

SAMPLE_SEED = 42

MODEL_SEEDS = [21, 42, 63, 84, 105]

SILHOUETTE_SAMPLE_SIZE = 500

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ONE FIXED DATASET
# ============================================================

print("=" * 60)
print("K-MEANS INITIALIZATION STABILITY EXPERIMENT")
print("=" * 60)

print("\nLoading fixed 10,000-image dataset...")

X, y_true = load_or_extract_features(
    sample_size=SAMPLE_SIZE,
    seed=SAMPLE_SEED
)

print("Feature shape:", X.shape)
print("Number of images:", len(y_true))

# Standardize once so every run uses identical input features
print("\nStandardizing features...")

X_scaled = StandardScaler().fit_transform(X)


# ============================================================
# RUN K-MEANS WITH DIFFERENT INITIALIZATION SEEDS
# ============================================================

all_results = []

for model_seed in MODEL_SEEDS:

    print("\n" + "=" * 60)
    print(f"K-Means random state: {model_seed}")
    print("=" * 60)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=model_seed,
        n_init=10
    )

    start = time.perf_counter()

    cluster_ids = kmeans.fit_predict(X_scaled)

    elapsed = time.perf_counter() - start

    # Silhouette score estimated using 500 points
    silhouette = silhouette_score(
        X_scaled,
        cluster_ids,
        sample_size=SILHOUETTE_SAMPLE_SIZE,
        random_state=42
    )

    # ARI against actual land-type labels
    ari = adjusted_rand_score(
        y_true,
        cluster_ids
    )

    all_results.append({
        "Model Seed": model_seed,
        "Images": len(y_true),
        "K": N_CLUSTERS,
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": elapsed
    })

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {elapsed:.3f} seconds")


# ============================================================
# SAVE INDIVIDUAL RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

csv_path = (
    RESULTS_DIR / "cnn_initialization_stability_10000_k12.csv"
)

results_df.to_csv(csv_path, index=False)


# ============================================================
# CALCULATE SUMMARY STATISTICS
# ============================================================

metrics = [
    "Silhouette Score",
    "ARI",
    "Clustering Seconds"
]

summary = results_df[metrics].agg(["mean", "std"]).T

summary.columns = ["Mean", "Standard Deviation"]

summary_path = (
    RESULTS_DIR /
    "cnn_initialization_stability_summary_10000_k12.csv"
)

summary.to_csv(summary_path)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("INITIALIZATION STABILITY RESULTS")
print("=" * 60)

print("\nIndividual results:")
print(results_df.round(4).to_string(index=False))

print("\nMean and standard deviation:")
print(summary.round(4).to_string())

print("\nSaved files:")
print(csv_path)
print(summary_path)

print("\nInitialization stability experiment completed!")
