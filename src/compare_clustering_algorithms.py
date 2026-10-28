
from pathlib import Path
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 2000
RANDOM_STATE = 42

print("Loading EuroSAT RGB dataset...")

dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
dataset = dataset.shuffle(seed=RANDOM_STATE).select(
    range(min(SAMPLE_SIZE, len(dataset)))
)

# --------------------------------------------------
# 2. Extract RGB colour-statistics features
# --------------------------------------------------

print("Extracting image features...")

features = []

for item in dataset:
    image = np.asarray(item["image"].convert("RGB"), dtype=np.float32) / 255.0

    mean_rgb = image.mean(axis=(0, 1))
    std_rgb = image.std(axis=(0, 1))

    features.append(np.concatenate([mean_rgb, std_rgb]))

X = np.asarray(features)

# Standardize features so they are on comparable scales.
X_scaled = StandardScaler().fit_transform(X)

# Reference labels are used ONLY for evaluation, not for clustering.
y_true = np.asarray(dataset["label"])

print(f"Images evaluated: {len(X_scaled)}")
print(f"Feature dimensions: {X_scaled.shape[1]}")

# --------------------------------------------------
# 3. Define the clustering algorithms
# --------------------------------------------------

algorithms = {
    "K-Means": KMeans(
        n_clusters=10,
        random_state=RANDOM_STATE,
        n_init=10
    ),
    "Agglomerative Clustering": AgglomerativeClustering(
        n_clusters=10,
        linkage="ward"
    ),
    "DBSCAN": DBSCAN(
        eps=0.5,
        min_samples=10
    )
}

# --------------------------------------------------
# 4. Run and evaluate each algorithm
# --------------------------------------------------

results = []

for name, model in algorithms.items():
    print(f"\nRunning {name}...")

    start_time = time.perf_counter()
    cluster_labels = model.fit_predict(X_scaled)
    runtime = time.perf_counter() - start_time

    # DBSCAN assigns -1 to noise points.
    noise_count = int(np.sum(cluster_labels == -1))
    non_noise_mask = cluster_labels != -1
    non_noise_labels = cluster_labels[non_noise_mask]
    unique_clusters = np.unique(non_noise_labels)
    cluster_count = len(unique_clusters)

    # Silhouette Score is undefined if fewer than two clusters
    # or if every non-noise sample belongs to its own cluster.
    if 2 <= cluster_count < np.sum(non_noise_mask):
        silhouette = silhouette_score(
            X_scaled[non_noise_mask],
            non_noise_labels
        )
    else:
        silhouette = np.nan

    # ARI uses the reference labels only after clustering.
    ari = adjusted_rand_score(y_true, cluster_labels)

    results.append({
        "Algorithm": name,
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Number of Clusters": cluster_count,
        "Noise Points": noise_count,
        "Runtime (seconds)": round(runtime, 3)
    })

    print(f"Clusters: {cluster_count}")
    print(f"Noise points: {noise_count}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Runtime: {runtime:.3f} seconds")

# --------------------------------------------------
# 5. Save and display the comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = RESULTS_DIR / "clustering_algorithm_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("CLUSTERING ALGORITHM COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))
print(f"\nResults saved to: {output_path}")

# Select the winner by Silhouette Score, if valid scores exist.
valid_results = results_df.dropna(subset=["Silhouette Score"])

if not valid_results.empty:
    winner = valid_results.loc[
        valid_results["Silhouette Score"].idxmax()
    ]
    print(
        "\nBest algorithm by Silhouette Score: "
        f"{winner['Algorithm']}"
    )
else:
    print(
        "\nNo valid Silhouette Scores were produced. "
        "Review the clustering parameters."
    )
