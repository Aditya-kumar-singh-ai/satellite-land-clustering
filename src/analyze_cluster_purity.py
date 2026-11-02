
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from feature_cache import load_or_extract_features


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SAMPLE_SIZE = 10000
N_CLUSTERS = 12

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD CACHED FEATURES AND LABELS
# ============================================================

print("=" * 60)
print("CNN CLUSTER PURITY ANALYSIS")
print("=" * 60)

print("\nLoading cached CNN features...")

X, y_true = load_or_extract_features(
    sample_size=SAMPLE_SIZE,
    seed=SEED
)

print("Feature shape:", X.shape)
print("Images:", len(y_true))

if len(X) != SAMPLE_SIZE:
    raise ValueError("Unexpected feature count.")

if len(X) != len(y_true):
    raise ValueError("Feature and label counts do not match.")


# Get actual land-type names from dataset metadata.
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
label_names = dataset.features["label"].names


# ============================================================
# 2. STANDARDIZE AND CLUSTER
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Running K-Means with K={N_CLUSTERS}...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10
)

cluster_ids = kmeans.fit_predict(X_scaled)

print("Clustering completed.")


# ============================================================
# 3. CLUSTER VS ACTUAL LAND TYPE
# ============================================================

table = pd.crosstab(
    pd.Series(cluster_ids, name="Cluster"),
    pd.Series(
        [label_names[int(label)] for label in y_true],
        name="Actual Land Type"
    )
)

print("\nCluster counts:")
print(table)


# ============================================================
# 4. CALCULATE CLUSTER PURITY
# ============================================================

row_percentages = table.div(
    table.sum(axis=1),
    axis=0
) * 100

summary = pd.DataFrame({
    "Cluster": row_percentages.index,
    "Images": table.sum(axis=1).values,
    "Dominant Land Type": row_percentages.idxmax(axis=1).values,
    "Dominant Type Percentage": row_percentages.max(axis=1).values
})

# Overall weighted cluster purity.
overall_purity = (
    table.max(axis=1).sum() / table.values.sum()
) * 100

print("\nCluster purity summary:")
print(summary.round(2).to_string(index=False))

print(
    f"\nOverall weighted cluster purity: "
    f"{overall_purity:.2f}%"
)


# ============================================================
# 5. SAVE RESULTS TO CSV
# ============================================================

counts_path = (
    RESULTS_DIR / "cnn_cluster_landtype_counts_10000_k12.csv"
)

percent_path = (
    RESULTS_DIR / "cnn_cluster_landtype_percentages_10000_k12.csv"
)

summary_path = (
    RESULTS_DIR / "cnn_cluster_purity_summary_10000_k12.csv"
)

table.to_csv(counts_path)
row_percentages.to_csv(percent_path)
summary.to_csv(summary_path)


# ============================================================
# 6. GENERATE HEATMAP
# ============================================================

plt.figure(figsize=(15, 8))

plt.imshow(
    row_percentages.values,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=100
)

plt.colorbar(label="Percentage")

plt.xticks(
    range(len(row_percentages.columns)),
    row_percentages.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(row_percentages.index)),
    row_percentages.index
)

plt.title(
    "ResNet18 + K-Means: Cluster Composition "
    "(10,000 Images, K=12)"
)

plt.xlabel("Actual Land Type")
plt.ylabel("Cluster")

plt.tight_layout()

heatmap_path = (
    RESULTS_DIR / "cnn_cluster_purity_heatmap_10000_k12.png"
)

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER PURITY ANALYSIS COMPLETED")
print("=" * 60)

print(f"Images: {SAMPLE_SIZE}")
print(f"Clusters: {N_CLUSTERS}")
print(f"Overall weighted purity: {overall_purity:.2f}%")

print("\nSaved files:")
print(counts_path)
print(percent_path)
print(summary_path)
print(heatmap_path)

print("\nExperiment completed successfully!")
