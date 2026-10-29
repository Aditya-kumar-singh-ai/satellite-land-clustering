
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights

# Configuration
SAMPLE_SIZE = 500
SEED = 42
K = 15
BATCH_SIZE = 32
OUTPUT_DIR = "results/difficult_landtypes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

TARGET_TYPES = ["Highway", "Permanent Crop"]

device = torch.device("cpu")

# 1. Load the dataset
print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
sample = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

# 2. Load pretrained CNN
print("Loading pretrained ResNet18...")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval().to(device)
transform = weights.transforms()

# 3. Prepare images and actual labels
images = []
actual_labels = []

for item in sample:
    images.append(transform(item["image"].convert("RGB")))
    actual_labels.append(item["label"])

images_tensor = torch.stack(images)
actual_labels = np.array(actual_labels)

# 4. Extract CNN features in batches
features = []

print("Extracting CNN features...")
with torch.no_grad():
    for start in range(0, len(images_tensor), BATCH_SIZE):
        batch = images_tensor[start:start + BATCH_SIZE].to(device)
        features.append(model(batch).cpu().numpy())
        print(f"Processed {min(start + BATCH_SIZE, len(images_tensor))}/{len(images_tensor)}")

features = np.vstack(features)

# 5. Scale features and cluster
print("Running K-Means with K=15...")
features_scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)
cluster_ids = kmeans.fit_predict(features_scaled)

# 6. Build a table of image metadata
records = []

for i, item in enumerate(sample):
    actual_type = LABEL_NAMES[int(actual_labels[i])]

    records.append({
        "Image Index": i,
        "Actual Land Type": actual_type,
        "Cluster": int(cluster_ids[i]),
    })

metadata = pd.DataFrame(records)

# 7. Save image examples for Highway and Permanent Crop
print("\nSaving example images...")

for land_type in TARGET_TYPES:
    type_dir = os.path.join(OUTPUT_DIR, land_type.replace(" ", "_"))
    os.makedirs(type_dir, exist_ok=True)

    subset = metadata[metadata["Actual Land Type"] == land_type]

    if subset.empty:
        print(f"No images found for {land_type} in this sample.")
        continue

    for _, row in subset.head(12).iterrows():
        idx = int(row["Image Index"])
        image = sample[idx]["image"].convert("RGB")

        filename = (
            f"image_{idx:03d}_cluster_{int(row['Cluster'])}.png"
        )
        image.save(os.path.join(type_dir, filename))

    print(f"{land_type}: saved {min(12, len(subset))} images to {type_dir}")

metadata.to_csv(
    os.path.join(OUTPUT_DIR, "image_cluster_metadata.csv"),
    index=False
)

print("\nDone! Open the results/difficult_landtypes folder.")
