
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights


# Paths and configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 500
N_CLUSTERS = 10

# Load the same sample used in the previous experiment
print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
images = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

labels = np.asarray(images["label"])
label_names = images.features["label"].names

# Load pretrained ResNet18
print("Loading pretrained ResNet18...")
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

preprocess = weights.transforms()
features = []

print("Extracting CNN features...")
with torch.inference_mode():
    for i, sample in enumerate(images):
        image = sample["image"].convert("RGB")
        tensor = preprocess(image).unsqueeze(0)

        embedding = model(tensor)
        features.append(embedding.squeeze(0).numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{SAMPLE_SIZE}")

X = np.asarray(features, dtype=np.float32)

# Standardize and cluster the features
print("Standardizing features and running K-Means...")
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10
)
clusters = kmeans.fit_predict(X_scaled)

# Project the 512 features into 2 dimensions for visualization
print("Running PCA...")
pca = PCA(n_components=2, random_state=SEED)
X_2d = pca.fit_transform(X_scaled)

print(
    "PCA variance retained in 2D:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# Plot 1: predicted clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=clusters,
    cmap="tab10",
    s=24,
    alpha=0.8
)
plt.title("ResNet18 Features — K-Means Clusters (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Predicted Cluster")
plt.tight_layout()

cluster_plot = RESULTS_DIR / "cnn_clusters_pca.png"
plt.savefig(cluster_plot, dpi=200)
plt.close()

# Plot 2: actual land types
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=24,
    alpha=0.8
)
plt.title("ResNet18 Features — Actual EuroSAT Land Types (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

colorbar = plt.colorbar(scatter, ticks=range(len(label_names)))
colorbar.ax.set_yticklabels(label_names)

plt.tight_layout()

label_plot = RESULTS_DIR / "cnn_actual_labels_pca.png"
plt.savefig(label_plot, dpi=200)
plt.close()

print("\nSaved plots:")
print(cluster_plot)
print(label_plot)
print("\nPCA visualization completed!")

