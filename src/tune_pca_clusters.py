
import os
import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# Configuration
SEED = 42
K_VALUES = range(2, 11)

FEATURE_PATH = "results/cnn_features_pca_95.npy"
LABEL_PATH = "data/cache/eurosat_resnet18_n5000_seed42_labels.npy"

OUTPUT_PATH = "results/pca_kmeans_tuning.csv"


# Load data
X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)

print("PCA feature shape:", X.shape)
print("Labels shape:", y.shape)


results = []

for k in K_VALUES:

    print("\nTesting K =", k)

    start = time.time()

    model = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10
    )

    clusters = model.fit_predict(X)

    execution_time = time.time() - start

    silhouette = silhouette_score(
        X,
        clusters,
        sample_size=2000,
        random_state=SEED
    )

    ari = adjusted_rand_score(y, clusters)

    results.append({
        "K": k,
        "Silhouette Score": silhouette,
        "ARI": ari,
        "Execution Time": execution_time
    })

    print("Silhouette:", round(silhouette, 4))
    print("ARI:", round(ari, 4))
    print("Time:", round(execution_time, 2))


# Save results
df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("CLUSTER TUNING RESULTS")
print("=" * 50)

print(df.to_string(index=False))

print("\nBest K by Silhouette:")
print(df.loc[df["Silhouette Score"].idxmax()])

print("\nBest K by ARI:")
print(df.loc[df["ARI"].idxmax()])

print("\nSaved:", OUTPUT_PATH)
