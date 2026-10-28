
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights

SAMPLE_SIZE = 500
SEED = 42
K = 15
BATCH_SIZE = 32

LABEL_NAMES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "SeaLake",
]

device = torch.device("cpu")

print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
sample = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = weights.transforms()

images = []
actual_labels = []

for item in sample:
    images.append(transform(item["image"].convert("RGB")))
    actual_labels.append(item["label"])

images = torch.stack(images)
actual_labels = np.asarray(actual_labels)

features = []

print("Extracting CNN features...")
with torch.no_grad():
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE].to(device)
        features.append(model(batch).cpu().numpy())
        print(f"Processed {min(start + BATCH_SIZE, len(images))}/{len(images)}")

features = np.vstack(features)
features = StandardScaler().fit_transform(features)

print(f"Running K-Means with K={K}...")
predicted_clusters = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
).fit_predict(features)

print("ARI:", round(adjusted_rand_score(actual_labels, predicted_clusters), 4))

# Count actual land types in every cluster.
df = pd.DataFrame({
    "Actual Land Type": [LABEL_NAMES[i] for i in actual_labels],
    "Cluster": predicted_clusters
})

counts = pd.crosstab(
    df["Actual Land Type"],
    df["Cluster"]
).reindex(LABEL_NAMES, fill_value=0)

counts.to_csv("results/cnn_landtype_confusion_counts.csv")

# Normalize each land-type row to percentages.
percentages = counts.div(counts.sum(axis=1), axis=0) * 100
percentages.to_csv("results/cnn_landtype_confusion_percentages.csv")

# Find the most common cluster for each actual land type.
summary = pd.DataFrame({
    "Actual Land Type": counts.index,
    "Images": counts.sum(axis=1).values,
    "Most Common Cluster": counts.idxmax(axis=1).values,
    "Images in Most Common Cluster": counts.max(axis=1).values,
})

summary["Percentage in Most Common Cluster"] = (
    summary["Images in Most Common Cluster"] / summary["Images"] * 100
).round(2)

summary.to_csv("results/cnn_landtype_confusion_summary.csv", index=False)

print("\nLand-type distribution across clusters (counts):")
print(counts.to_string())

print("\nLand-type summary:")
print(summary.to_string(index=False))

print("\nSaved:")
print("results/cnn_landtype_confusion_counts.csv")
print("results/cnn_landtype_confusion_percentages.csv")
print("results/cnn_landtype_confusion_summary.csv")
