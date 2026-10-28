
import time
import numpy as np
import pandas as pd

from datasets import load_dataset
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

SAMPLE_SIZE = 500
SEEDS = [21, 42, 63, 84, 105]
K_VALUES = [10, 15]
BATCH_SIZE = 32

device = torch.device("cpu")

print("Loading EuroSAT RGB dataset...")
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = weights.transforms()
results = []

for seed in SEEDS:
    print(f"\nProcessing sample seed: {seed}")
    sample = dataset.shuffle(seed=seed).select(range(SAMPLE_SIZE))

    images = []
    labels = []

    for item in sample:
        img = item["image"].convert("RGB")
        images.append(transform(img))
        labels.append(item["label"])

    x = torch.stack(images)
    y = np.array(labels)

    features = []

    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = x[start:start + BATCH_SIZE].to(device)
            output = model(batch)
            features.append(output.cpu().numpy())

    features = np.vstack(features)
    features = StandardScaler().fit_transform(features)

    for k in K_VALUES:
        print(f"  Testing K={k}...")
        start_time = time.perf_counter()

        model_k = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        cluster_labels = model_k.fit_predict(features)
        elapsed = time.perf_counter() - start_time

        silhouette = silhouette_score(features, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)

        results.append({
            "Sample Seed": seed,
            "Images": SAMPLE_SIZE,
            "K": k,
            "Silhouette Score": silhouette,
            "ARI": ari,
            "Clustering Seconds": elapsed
        })

        print(
            f"    Silhouette={silhouette:.4f}, "
            f"ARI={ari:.4f}, Time={elapsed:.3f}s"
        )

df = pd.DataFrame(results)
df.to_csv("results/cnn_k_stability_comparison.csv", index=False)

summary = df.groupby("K").agg(
    Mean_Silhouette=("Silhouette Score", "mean"),
    Std_Silhouette=("Silhouette Score", "std"),
    Mean_ARI=("ARI", "mean"),
    Std_ARI=("ARI", "std"),
    Mean_Clustering_Seconds=("Clustering Seconds", "mean")
).reset_index()

summary.to_csv(
    "results/cnn_k_stability_summary.csv",
    index=False
)

print("\nRESULTS FOR EACH SAMPLE")
print(df.round(4).to_string(index=False))

print("\nSUMMARY BY K")
print(summary.round(4).to_string(index=False))

print("\nSaved results to results/cnn_k_stability_comparison.csv")
print("Saved summary to results/cnn_k_stability_summary.csv")
