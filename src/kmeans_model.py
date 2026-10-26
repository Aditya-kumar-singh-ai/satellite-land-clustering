
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from experiment_logger import save_result


# ==========================================
# 1. Load the EuroSAT dataset
# ==========================================
print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Shuffle the dataset so our sample includes varied land types
images = dataset.shuffle(seed=42).select(range(500))

# Store the actual land-type labels
labels = np.array(images["label"])

print("Images loaded:", len(images))
print("Unique land types:", len(np.unique(labels)))


# ==========================================
# 2. Convert images into numerical features
# ==========================================
print("\nExtracting image features...")

features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.float32)
    features.append(image.flatten())

X = np.array(features)

# Scale pixel values from 0-255 to 0-1
X = X / 255.0

print("Original feature matrix shape:", X.shape)


# ==========================================
# 3. Reduce dimensions using PCA
# ==========================================
print("\nStarting PCA...")

pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X)

variance_retained = pca.explained_variance_ratio_.sum()

print("PCA completed.")
print("Reduced feature matrix shape:", X_reduced.shape)
print("PCA variance retained:", round(variance_retained, 3))


# ==========================================
# 4. Cluster images using K-Means
# ==========================================
print("\nStarting K-Means...")

model = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_reduced)

print("K-Means completed.")
print("Images clustered:", len(clusters))
print("Number of clusters:", len(np.unique(clusters)))


# ==========================================
# 5. Evaluate the clustering
# ==========================================
print("\nEvaluating clustering...")

score = silhouette_score(X_reduced, clusters)
ari = adjusted_rand_score(labels, clusters)


# Save experiment results
save_result(
    experiment="PCA + K-Means",
    features=X_reduced.shape[1],
    clusters=len(np.unique(clusters)),
    silhouette=score,
    ari=ari
)


print("Silhouette Score:", round(score, 4))
print("Adjusted Rand Index:", round(ari, 4))


# ==========================================
# 6. Compare clusters with actual land types
# ==========================================
results = pd.DataFrame({
    "cluster": clusters,
    "land_type": [
        images.features["label"].int2str(int(label))
        for label in labels
    ]
})

table = pd.crosstab(
    results["cluster"],
    results["land_type"],
    rownames=["Cluster"],
    colnames=["Actual Land Type"]
)

print("\nCluster vs Actual Land-Type Table:")
print(table.to_string())

# Visualize clusters against actual land types
fig, ax = plt.subplots(figsize=(14, 7))

heatmap = ax.imshow(table.values, aspect="auto", cmap="Blues")

ax.set_xticks(range(len(table.columns)))
ax.set_xticklabels(
    table.columns,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(table.index)))
ax.set_yticklabels(
    [f"Cluster {i}" for i in table.index]
)

ax.set_title("K-Means Clusters vs Actual Land Types")
ax.set_xlabel("Actual Land Type")
ax.set_ylabel("K-Means Cluster")

fig.colorbar(heatmap, ax=ax, label="Number of Images")

# Display counts inside the cells
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        count = table.iloc[i, j]
        if count > 0:
            ax.text(
                j, i, str(count),
                ha="center",
                va="center",
                fontsize=7
            )

plt.tight_layout()
plt.savefig(
    "results/kmeans_cluster_landtype_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("Land-type heatmap saved.")


# ==========================================
# 7. Summarize cluster quality
# ==========================================
dominant_type = table.idxmax(axis=1)
dominant_count = table.max(axis=1)
cluster_size = table.sum(axis=1)

summary = pd.DataFrame({
    "Dominant land type": dominant_type,
    "Matching images": dominant_count,
    "Total images": cluster_size,
    "Dominance (%)": (
        dominant_count / cluster_size * 100
    ).round(2)
})

print("\nCluster Quality Summary:")
print(summary.to_string())


# ==========================================
# 8. Visualization: Cluster-size distribution
# ==========================================

from pathlib import Path

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Count how many images belong to each cluster
cluster_sizes = (
    pd.Series(clusters)
    .value_counts()
    .reindex(range(10), fill_value=0)
)

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    cluster_sizes.index,
    cluster_sizes.values
)

ax.set_title(
    "Number of Satellite Images per Cluster",
    fontsize=14
)
ax.set_xlabel("Cluster ID")
ax.set_ylabel("Number of Images")
ax.set_xticks(range(10))
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars, padding=3)

fig.tight_layout()

chart_path = results_dir / "kmeans_cluster_sizes.png"

fig.savefig(
    chart_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\nCluster-size chart saved to: {chart_path}")



# ==========================================
# 8. Visualize five images from each cluster
# ==========================================

# 8. Visualize five images from each cluster
print("\nPreparing cluster visualization...")

fig, axes = plt.subplots(10, 5, figsize=(14, 20))

fig.suptitle(
    "Satellite Images Grouped by K-Means",
    fontsize=16,
    fontweight="bold"
)

for cluster_id in range(10):
    image_indices = np.where(clusters == cluster_id)[0]
    selected_indices = image_indices[:5]

    for j in range(5):
        ax = axes[cluster_id, j]
        ax.axis("off")

        if j < len(selected_indices):
            image_index = selected_indices[j]
            ax.imshow(images[image_index]["image"])

            true_label = labels[image_index]
            label_name = images.features["label"].int2str(
                int(true_label)
            )

            ax.set_title(label_name, fontsize=8, pad=4)

        if j == 0:
            ax.text(
                -0.12, 0.5,
                f"Cluster {cluster_id}",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                clip_on=False
            )

plt.subplots_adjust(
    left=0.13,
    right=0.99,
    top=0.95,
    bottom=0.02,
    hspace=0.35,
    wspace=0.08
)

plot_path = "results/kmeans_cluster_visualization.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Cluster visualization saved to: {plot_path}")
