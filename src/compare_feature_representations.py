
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from torchvision.models import ResNet18_Weights


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
SAMPLE_SIZE = 500
N_CLUSTERS = 10

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading EuroSAT RGB dataset...")

dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
images = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
labels = np.asarray(images["label"])

print(f"Images loaded: {len(images)}")
print(f"Number of actual land types: {len(np.unique(labels))}")


# --------------------------------------------------
# 2. Extract RGB statistics
# --------------------------------------------------

print("\nExtracting RGB statistics...")
start = time.perf_counter()

rgb_features = []

for sample in images:
    image = np.asarray(sample["image"].convert("RGB"), dtype=np.float32)
    image /= 255.0

    means = image.mean(axis=(0, 1))
    stds = image.std(axis=(0, 1))

    rgb_features.append(np.concatenate([means, stds]))

rgb_features = np.asarray(rgb_features, dtype=np.float32)

rgb_extraction_time = time.perf_counter() - start


# --------------------------------------------------
# 3. Extract RGB histograms
# --------------------------------------------------

print("Extracting RGB histograms...")
start = time.perf_counter()

histogram_features = []

for sample in images:
    image = np.asarray(sample["image"].convert("RGB"), dtype=np.uint8)
    channel_features = []

    for channel in range(3):
        histogram, _ = np.histogram(
            image[:, :, channel],
            bins=8,
            range=(0, 256)
        )
        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum()
        channel_features.extend(histogram)

    histogram_features.append(channel_features)

histogram_features = np.asarray(
    histogram_features, dtype=np.float32
)

histogram_extraction_time = time.perf_counter() - start


# --------------------------------------------------
# 4. Extract pretrained ResNet18 features
# --------------------------------------------------

print("Loading pretrained ResNet18...")
start = time.perf_counter()

weights = ResNet18_Weights.DEFAULT
cnn_model = models.resnet18(weights=weights)
cnn_model.fc = torch.nn.Identity()
cnn_model.eval()

device = torch.device("cpu")
cnn_model = cnn_model.to(device)
preprocess = weights.transforms()

cnn_features = []

with torch.inference_mode():
    for i, sample in enumerate(images):
        image = sample["image"].convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        embedding = cnn_model(tensor)
        cnn_features.append(embedding.squeeze(0).cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"CNN processed {i + 1}/{SAMPLE_SIZE}")

cnn_features = np.asarray(cnn_features, dtype=np.float32)
cnn_extraction_time = time.perf_counter() - start


# --------------------------------------------------
# 5. Compare feature representations
# --------------------------------------------------

feature_sets = [
    ("RGB Mean + Std", rgb_features, rgb_extraction_time),
    ("RGB Histogram", histogram_features, histogram_extraction_time),
    ("ResNet18 CNN", cnn_features, cnn_extraction_time),
]

results = []

for feature_name, X, extraction_time in feature_sets:
    print(f"\nEvaluating: {feature_name}")

    # Standardize each representation for K-Means.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start = time.perf_counter()

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=SEED,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)
    clustering_time = time.perf_counter() - start

    silhouette = silhouette_score(X_scaled, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    results.append({
        "Feature Representation": feature_name,
        "Feature Count": X.shape[1],
        "Images": len(images),
        "Clusters": len(np.unique(cluster_labels)),
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Feature Extraction Seconds": extraction_time,
        "Clustering Seconds": clustering_time,
    })

    print("Feature count:", X.shape[1])
    print("Silhouette Score:", round(silhouette, 4))
    print("ARI:", round(ari, 4))
    print("Feature extraction time:", round(extraction_time, 2), "seconds")
    print("Clustering time:", round(clustering_time, 2), "seconds")


# --------------------------------------------------
# 6. Save and display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = RESULTS_DIR / "feature_representation_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("FEATURE REPRESENTATION COMPARISON")
print("=" * 65)
print(results_df.round(4).to_string(index=False))
print(f"\nResults saved to: {output_path}")
print("\nExperiment completed!")
