
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Keep the same sample for a fair comparison
images = dataset.shuffle(seed=42).select(range(500))
labels = np.array(images["label"])

# Extract six colour features per image
features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.float32)

    mean_rgb = image.mean(axis=(0, 1))
    std_rgb = image.std(axis=(0, 1))

    features.append(np.concatenate([mean_rgb, std_rgb]))

X = np.array(features)

# Scale the features
X_scaled = StandardScaler().fit_transform(X)

# Test different numbers of clusters
results = []

for k in range(2, 16):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(labels, clusters)

    results.append({
        "K": k,
        "Silhouette Score": round(silhouette, 4),
        "ARI": round(ari, 4)
    })

results_df = pd.DataFrame(results)

print("\n--- K-Means Cluster Count Comparison ---")
print(results_df.to_string(index=False))

best_silhouette = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]

best_ari = results_df.loc[
    results_df["ARI"].idxmax()
]

print("\nBest K by Silhouette Score:", int(best_silhouette["K"]))
print("Best Silhouette Score:", best_silhouette["Silhouette Score"])

print("\nBest K by ARI:", int(best_ari["K"]))
print("Best ARI:", best_ari["ARI"])

print("\nK-Means tuning experiment completed!")
