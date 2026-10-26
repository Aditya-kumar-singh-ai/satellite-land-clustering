
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from experiment_logger import save_result

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Use the same shuffled sample as our previous experiment
images = dataset.shuffle(seed=42).select(range(500))
labels = np.array(images["label"])

label_names = images.features["label"].names
land_types = [label_names[int(label)] for label in labels]

# Extract six colour features per image
features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.float32)

    mean_rgb = image.mean(axis=(0, 1))
    std_rgb = image.std(axis=(0, 1))

    features.append(np.concatenate([mean_rgb, std_rgb]))

X = np.array(features)

print("Original feature shape:", X.shape)

# Standardize each feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled feature shape:", X_scaled.shape)

# Apply K-Means
model = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, clusters)
ari = adjusted_rand_score(labels, clusters)

# Save experiment results
save_result(
    experiment="Scaled Colour Features",
    features=X_scaled.shape[1],
    clusters=len(set(clusters)),
    silhouette=silhouette,
    ari=ari
)

print("\n--- Scaled Colour-Feature Results ---")
print("Images clustered:", len(clusters))
print("Number of clusters:", len(set(clusters)))
print("Silhouette Score:", round(silhouette, 4))
print("Adjusted Rand Index:", round(ari, 4))

# Compare predicted clusters with actual labels
comparison = pd.crosstab(
    clusters,
    land_types,
    rownames=["Cluster"],
    colnames=["Actual Land Type"]
)

print("\nCluster vs Actual Land Type:")
print(comparison.to_string())

# Visualize five images per cluster
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

print("\nScaled colour-feature experiment completed!")
