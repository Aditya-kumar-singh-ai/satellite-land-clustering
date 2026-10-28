
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from torchvision.models import ResNet18_Weights


# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 500
N_CLUSTERS = 10

# Load dataset and reproduce the same sample
print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
images = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

y_true = np.asarray(images["label"])
label_names = images.features["label"].names

# Extract pretrained ResNet18 features
print("Loading ResNet18...")
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
X_scaled = StandardScaler().fit_transform(X)

# Cluster
print("Running K-Means...")
model_kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10
)
cluster_ids = model_kmeans.fit_predict(X_scaled)

ari = adjusted_rand_score(y_true, cluster_ids)
print(f"ARI: {ari:.4f}")

# Cluster versus actual land type
table = pd.crosstab(
    pd.Series(cluster_ids, name="Cluster"),
    pd.Series(
        [label_names[int(label)] for label in y_true],
        name="Actual Land Type"
    )
)

# Save raw counts
counts_path = RESULTS_DIR / "cnn_cluster_landtype_counts.csv"
table.to_csv(counts_path)

# Calculate percentage composition of each cluster
row_percentages = table.div(table.sum(axis=1), axis=0) * 100

percent_path = RESULTS_DIR / "cnn_cluster_landtype_percentages.csv"
row_percentages.to_csv(percent_path)

# Draw heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    row_percentages,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100
)
plt.title("ResNet18 + K-Means: Land-Type Composition of Each Cluster")
plt.xlabel("Actual Land Type")
plt.ylabel("Predicted Cluster")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

heatmap_path = RESULTS_DIR / "cnn_cluster_purity_heatmap.png"
plt.savefig(heatmap_path, dpi=200)
plt.close()

# Summarize the dominant class and purity for each cluster
summary = pd.DataFrame({
    "Cluster": row_percentages.index,
    "Images": table.sum(axis=1).values,
    "Dominant Land Type": row_percentages.idxmax(axis=1).values,
    "Dominant Type Percentage": row_percentages.max(axis=1).values
})

summary_path = RESULTS_DIR / "cnn_cluster_purity_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nCluster composition summary:")
print(summary.round(2).to_string(index=False))

print("\nSaved:")
print(counts_path)
print(percent_path)
print(heatmap_path)
print(summary_path)
print("\nCluster analysis completed!")
