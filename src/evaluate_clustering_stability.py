
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torchvision.models as models

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights


# Configuration
SAMPLE_SIZE = 500
N_CLUSTERS = 15
SAMPLE_SEEDS = [21, 42, 63, 84, 105]

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset once
print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
label_names = dataset.features["label"].names

# Load pretrained ResNet18 once
print("Loading pretrained ResNet18...")
weights = ResNet18_Weights.DEFAULT
cnn_model = models.resnet18(weights=weights)
cnn_model.fc = torch.nn.Identity()
cnn_model.eval()
preprocess = weights.transforms()


def extract_features(images):
    features = []

    with torch.inference_mode():
        for i, sample in enumerate(images):
            image = sample["image"].convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            embedding = cnn_model(tensor)
            features.append(embedding.squeeze(0).numpy())

            if (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(images)} images")

    return np.asarray(features, dtype=np.float32)


# Run each sample independently
all_results = []

for seed in SAMPLE_SEEDS:
    print(f"\n{'=' * 50}")
    print(f"Sample seed: {seed}")
    print("=" * 50)

    images = dataset.shuffle(seed=seed).select(range(SAMPLE_SIZE))
    y_true = np.asarray(images["label"])

    print("Extracting CNN features...")
    X = extract_features(images)

    X_scaled = StandardScaler().fit_transform(X)

    print(f"Running K-Means with K={N_CLUSTERS}...")
    start = time.perf_counter()

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    clustering_seconds = time.perf_counter() - start

    silhouette = silhouette_score(X_scaled, cluster_ids)
    ari = adjusted_rand_score(y_true, cluster_ids)

    all_results.append({
        "Sample Seed": seed,
        "Images": len(images),
        "K": N_CLUSTERS,
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Clustering Seconds": clustering_seconds
    })

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Clustering time: {clustering_seconds:.3f} seconds")


# Save individual results
results_df = pd.DataFrame(all_results)
csv_path = RESULTS_DIR / "cnn_clustering_stability.csv"
results_df.to_csv(csv_path, index=False)

# Calculate summary statistics
metrics = ["Silhouette Score", "ARI", "Clustering Seconds"]
summary = results_df[metrics].agg(["mean", "std"]).T
summary.columns = ["Mean", "Standard Deviation"]

summary_path = RESULTS_DIR / "cnn_clustering_stability_summary.csv"
summary.to_csv(summary_path)

print("\n" + "=" * 60)
print("STABILITY RESULTS")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

print("\nMean and standard deviation across five samples:")
print(summary.round(4).to_string())

print(f"\nIndividual results saved to: {csv_path}")
print(f"Summary saved to: {summary_path}")
print("\nStability experiment completed!")
