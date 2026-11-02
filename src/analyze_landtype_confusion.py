
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from feature_cache import load_or_extract_features


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 10000
SEED = 42
K = 12

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD FEATURES AND LABELS
# ============================================================

print("=" * 60)
print("LAND-TYPE CONFUSION ANALYSIS")
print("=" * 60)

print("\nLoading cached CNN features...")

X, actual_labels = load_or_extract_features(
    sample_size=SAMPLE_SIZE,
    seed=SEED
)

if len(X) != SAMPLE_SIZE:
    raise ValueError("Unexpected feature count.")

if len(X) != len(actual_labels):
    raise ValueError("Feature and label counts do not match.")

dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
label_names = dataset.features["label"].names

print("Feature shape:", X.shape)
print("Images:", len(actual_labels))


# ============================================================
# 2. STANDARDIZE AND CLUSTER
# ============================================================

print("\nStandardizing features...")

X_scaled = StandardScaler().fit_transform(X)

print(f"Running K-Means with K={K}...")

kmeans = KMeans(
    n_clusters=K,
    random_state=SEED,
    n_init=10
)

predicted_clusters = kmeans.fit_predict(X_scaled)

print("Clustering completed.")


# ============================================================
# 3. CREATE LAND-TYPE VS CLUSTER TABLE
# ============================================================

df = pd.DataFrame({
    "Actual Land Type": [
        label_names[int(label)] for label in actual_labels
    ],
    "Cluster": predicted_clusters
})

counts = pd.crosstab(
    df["Actual Land Type"],
    df["Cluster"]
).reindex(label_names, fill_value=0)

print("\nLand-type distribution across clusters:")
print(counts.to_string())


# ============================================================
# 4. NORMALIZE EACH LAND TYPE
# ============================================================

percentages = counts.div(
    counts.sum(axis=1),
    axis=0
) * 100

print("\nLand-type distribution percentages:")
print(percentages.round(2).to_string())


# ============================================================
# 5. CREATE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Actual Land Type": counts.index,
    "Images": counts.sum(axis=1).values,
    "Most Common Cluster": counts.idxmax(axis=1).values,
    "Images in Most Common Cluster": counts.max(axis=1).values
})

summary["Percentage in Most Common Cluster"] = (
    summary["Images in Most Common Cluster"]
    / summary["Images"] * 100
).round(2)

print("\nLand-type confusion summary:")
print(summary.to_string(index=False))


# ============================================================
# 6. SAVE RESULTS
# ============================================================

counts_path = (
    RESULTS_DIR / "cnn_landtype_confusion_counts_10000_k12.csv"
)

percentages_path = (
    RESULTS_DIR / "cnn_landtype_confusion_percentages_10000_k12.csv"
)

summary_path = (
    RESULTS_DIR / "cnn_landtype_confusion_summary_10000_k12.csv"
)

counts.to_csv(counts_path)
percentages.to_csv(percentages_path)
summary.to_csv(summary_path)


# ============================================================
# 7. GENERATE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(13, 7))

image = ax.imshow(
    percentages.values,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=100
)

fig.colorbar(image, ax=ax, label="Percentage")

ax.set_xticks(range(len(percentages.columns)))
ax.set_xticklabels(percentages.columns)

ax.set_yticks(range(len(percentages.index)))
ax.set_yticklabels(percentages.index)

ax.set_xlabel("Predicted Cluster")
ax.set_ylabel("Actual Land Type")

ax.set_title(
    "Land-Type Distribution Across Clusters "
    "(10,000 Images, K=12)"
)

plt.tight_layout()

heatmap_path = (
    RESULTS_DIR / "cnn_landtype_confusion_heatmap_10000_k12.png"
)

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("LAND-TYPE CONFUSION ANALYSIS COMPLETED")
print("=" * 60)

print(f"Images: {SAMPLE_SIZE}")
print(f"Clusters: {K}")

print("\nSaved files:")
print(counts_path)
print(percentages_path)
print(summary_path)
print(heatmap_path)

print("\nExperiment completed successfully!")
