
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights

SAMPLE_SIZE = 500
SEED = 42
K = 15
BATCH_SIZE = 32
TARGET_CLUSTERS = [0, 1, 4, 6]

OUTPUT_DIR = "results/mixed_cluster_images"
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

print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
sample = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

print("Loading pretrained ResNet18...")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

transform = weights.transforms()

images = []
actual_labels = []

for item in sample:
    images.append(transform(item["image"].convert("RGB")))
    actual_labels.append(item["label"])

images_tensor = torch.stack(images)
actual_labels = np.array(actual_labels)

print("Extracting CNN features...")
features = []

with torch.no_grad():
    for start in range(0, SAMPLE_SIZE, BATCH_SIZE):
        batch = images_tensor[start:start + BATCH_SIZE]
        features.append(model(batch).numpy())
        print(f"Processed {min(start + BATCH_SIZE, SAMPLE_SIZE)}/{SAMPLE_SIZE}")

features = np.vstack(features)
features_scaled = StandardScaler().fit_transform(features)

print("Running K-Means...")
cluster_ids = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
).fit_predict(features_scaled)

# Create a contact sheet for each selected cluster.
for cluster_id in TARGET_CLUSTERS:
    indices = np.where(cluster_ids == cluster_id)[0]

    # Show Highway and Permanent Crop images first.
    target_indices = [
        i for i in indices
        if LABEL_NAMES[int(actual_labels[i])]
        in ["Highway", "Permanent Crop"]
    ]

    if not target_indices:
        print(f"No target images found in cluster {cluster_id}")
        continue

    max_images = min(12, len(target_indices))
    selected = target_indices[:max_images]

    cols = 4
    rows = int(np.ceil(len(selected) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for position, image_index in enumerate(selected):
        image = sample[image_index]["image"].convert("RGB")
        label = LABEL_NAMES[int(actual_labels[image_index])]

        axes[position].imshow(image)
        axes[position].set_title(
            f"{label}\nCluster {cluster_id} | Image {image_index}",
            fontsize=9
        )
        axes[position].axis("off")

    fig.suptitle(
        f"Cluster {cluster_id}: Highway and Permanent Crop examples",
        fontsize=14
    )
    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, f"cluster_{cluster_id}_contact_sheet.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

# Save assignments for further analysis.
pd.DataFrame({
    "Image Index": range(SAMPLE_SIZE),
    "Actual Land Type": [LABEL_NAMES[int(label)] for label in actual_labels],
    "Cluster": cluster_ids
}).to_csv(
    os.path.join(OUTPUT_DIR, "cluster_assignments.csv"),
    index=False
)

print("\nDone! Contact sheets saved in:", OUTPUT_DIR)
