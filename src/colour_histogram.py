
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from experiment_logger import save_result

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Use the same 500 images for a fair comparison
images = dataset.shuffle(seed=42).select(range(500))
labels = np.array(images["label"])

label_names = images.features["label"].names
land_types = [label_names[int(label)] for label in labels]

# Extract 8-bin RGB histograms: 24 features per image
features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.uint8)

    channel_histograms = []

    for channel in range(3):
        histogram, _ = np.histogram(
            image[:, :, channel],
            bins=8,
            range=(0, 256)
        )

        # Normalize each channel histogram
        histogram = histogram / histogram.sum()
        channel_histograms.extend(histogram)

    features.append(channel_histograms)

X = np.array(features, dtype=np.float32)

print("Histogram feature shape:", X.shape)

# K-Means clustering
model = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X)

# Evaluation
silhouette = silhouette_score(X, clusters)
ari = adjusted_rand_score(labels, clusters)
# Save experiment results
save_result(
    experiment="RGB Histogram",
    features=X.shape[1],
    clusters=len(set(clusters)),
    silhouette=silhouette,
    ari=ari
)

print("\n--- RGB Histogram Results ---")
print("Images clustered:", len(clusters))
print("Number of clusters:", len(set(clusters)))
print("Silhouette Score:", round(silhouette, 4))
print("Adjusted Rand Index:", round(ari, 4))

# Compare clusters against actual land types
comparison = pd.crosstab(
    clusters,
    land_types,
    rownames=["Cluster"],
    colnames=["Actual Land Type"]
)

print("\nCluster vs Actual Land Type:")
print(comparison.to_string())

# Display five images from each cluster
fig, axes = plt.subplots(10, 5, figsize=(12, 20))

for cluster_id in range(10):
    indices = np.where(clusters == cluster_id)[0]

    for j in range(5):
        ax = axes[cluster_id, j]

        if j < len(indices):
            index = indices[j]
            ax.imshow(images[index]["image"])

            if j == 0:
                ax.set_title(f"Cluster {cluster_id}")

        ax.axis("off")

plt.tight_layout()
plt.show()

print("\nRGB histogram experiment completed!")
