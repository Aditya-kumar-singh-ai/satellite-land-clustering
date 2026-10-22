
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


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
# 8. Visualize five images from each cluster
# ==========================================
print("\nPreparing cluster visualization...")

fig, axes = plt.subplots(10, 5, figsize=(14, 22))

for cluster_id in range(10):

    # Find indices of images belonging to this cluster
    image_indices = np.where(clusters == cluster_id)[0]

    # Select up to five images
    selected_indices = image_indices[:5]

    print(
        f"Cluster {cluster_id}: "
        f"{len(selected_indices)} images displayed"
    )

    for j in range(5):
        ax = axes[cluster_id, j]

        if j < len(selected_indices):
            image_index = selected_indices[j]
            image = images[image_index]["image"]

            # Get the true label of this image
            true_label = labels[image_index]
            label_name = images.features["label"].int2str(
                int(true_label)
            )

            ax.imshow(image)
            ax.axis("off")

            # Show the cluster title on the first image
            if j == 0:
                ax.set_title(
                    f"Cluster {cluster_id}\n{label_name}"
                )

        else:
            ax.axis("off")

plt.tight_layout()
plt.show()

print("\nProject run completed successfully!")

