
import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
K = 10

FEATURE_PATH = (
    "data/cache/eurosat_resnet18_n5000_seed42_features.npy"
)

LABEL_PATH = (
    "data/cache/eurosat_resnet18_n5000_seed42_labels.npy"
)

PCA_PATH = "results/cnn_features_pca_95.npy"

OUTPUT_PATH = "results/pca_clustering_comparison.csv"


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("K-MEANS: ORIGINAL CNN FEATURES VS PCA FEATURES")
print("=" * 60)

print("\nLoading original CNN features...")

X_original = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)

print("Original feature shape:", X_original.shape)
print("Labels shape:", y.shape)


# =========================================================
# STANDARDIZE ORIGINAL FEATURES
# =========================================================

print("\nStandardizing original CNN features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_original)

print("Standardized shape:", X_scaled.shape)


# =========================================================
# LOAD PCA REDUCED FEATURES
# =========================================================

print("\nLoading PCA-reduced features...")

X_pca = np.load(PCA_PATH)

print("PCA feature shape:", X_pca.shape)


# =========================================================
# FUNCTION TO EVALUATE K-MEANS
# =========================================================

def evaluate_kmeans(X, name):

    print("\n" + "-" * 50)
    print("Experiment:", name)
    print("Feature shape:", X.shape)

    start_time = time.time()

    model = KMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=10
    )

    clusters = model.fit_predict(X)

    execution_time = time.time() - start_time

    print("K-Means completed.")

    print("Calculating Silhouette Score...")

    silhouette = silhouette_score(
        X,
        clusters,
        sample_size=2000,
        random_state=SEED
    )

    ari = adjusted_rand_score(
        y,
        clusters
    )

    print("Silhouette Score:", round(silhouette, 4))
    print("Adjusted Rand Index:", round(ari, 4))
    print("Execution Time:", round(execution_time, 2), "seconds")

    return {
        "Experiment": name,
        "Features": X.shape[1],
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Execution Time": execution_time
    }


# =========================================================
# EXPERIMENT 1: ORIGINAL CNN FEATURES
# =========================================================

result_original = evaluate_kmeans(
    X_scaled,
    "Original CNN Features"
)


# =========================================================
# EXPERIMENT 2: PCA REDUCED FEATURES
# =========================================================

result_pca = evaluate_kmeans(
    X_pca,
    "PCA Reduced Features"
)


# =========================================================
# COMPARE RESULTS
# =========================================================

results = pd.DataFrame([
    result_original,
    result_pca
])

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:", OUTPUT_PATH)

print("\nExperiment completed successfully!")
