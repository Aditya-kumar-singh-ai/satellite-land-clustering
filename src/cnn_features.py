
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

images = dataset.shuffle(seed=42).select(range(500))
labels = np.array(images["label"])

label_names = images.features["label"].names
land_types = [label_names[int(label)] for label in labels]

# Load pretrained ResNet18
print("Loading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove the final classification layer
model.fc = torch.nn.Identity()
model.eval()

# Use CPU
device = torch.device("cpu")
model = model.to(device)

# Preprocess images as required by the pretrained model
preprocess = weights.transforms()

features = []

print("Extracting CNN features...")

with torch.no_grad():
    for i, sample in enumerate(images):
        image = sample["image"].convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        embedding = model(tensor)
        features.append(embedding.squeeze(0).numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/500 images")

X = np.array(features, dtype=np.float32)

print("CNN feature shape:", X.shape)

# Standardize the extracted features
X_scaled = StandardScaler().fit_transform(X)

# Cluster images into 10 groups
print("Running K-Means...")

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering
silhouette = silhouette_score(X_scaled, clusters)
ari = adjusted_rand_score(labels, clusters)

from experiment_logger import save_result

save_result(
    experiment="Pretrained ResNet18",
    features=X.shape[1],
    clusters=len(set(clusters)),
    silhouette=silhouette,
    ari=ari
)

print("\n--- CNN Feature Clustering Results ---")
print("Images clustered:", len(clusters))
print("Number of clusters:", len(set(clusters)))
print("Silhouette Score:", round(silhouette, 4))
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

print("\nCNN feature experiment completed!")

