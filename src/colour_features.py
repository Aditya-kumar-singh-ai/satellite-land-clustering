
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

# Select 500 shuffled images and get their actual labels
images = dataset.shuffle(seed=42).select(range(500))
labels = np.array(images["label"])

label_names = images.features["label"].names
land_types = [label_names[int(label)] for label in labels]

print("Images loaded:", len(images))

# Extract six colour features from every image
features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.float32)

    mean_rgb = image.mean(axis=(0, 1))
    std_rgb = image.std(axis=(0, 1))

    image_features = np.concatenate([mean_rgb, std_rgb])
    features.append(image_features)

X = np.array(features) / 255.0

print("Feature matrix shape:", X.shape)
print("First image's six features:", X[0])

# Cluster images using K-Means
model = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X)

# Evaluate the clustering
score = silhouette_score(X, clusters)
ari = adjusted_rand_score(labels, clusters)

# Save experiment results
save_result(
    experiment="Mean RGB + Std RGB",
    features=X.shape[1],
    clusters=len(set(clusters)),
    silhouette=score,
    ari=ari
)

print("\n--- Clustering Results ---")
print("Images clustered:", len(clusters))
print("Number of clusters:", len(set(clusters)))
print("Colour-feature Silhouette Score:", round(score, 4))
print("Adjusted Rand Index:", round(ari, 4))

# Compare clusters with actual land types
comparison = pd.crosstab(
    clusters,
    land_types,
    rownames=["Cluster"],
    colnames=["Actual Land Type"]
)

print("\nCluster vs Actual Land Type:")
print(comparison.to_string())

# Display five sample images from each cluster
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

print("\nColour-feature experiment completed!")


