
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
K = 10

FEATURE_PATH = "results/cnn_features_pca_95.npy"

LABEL_PATH = (
    "data/cache/eurosat_resnet18_n5000_seed42_labels.npy"
)

OUTPUT_DIR = "results"


# =====================================================
# EURO SAT CLASS NAMES
# =====================================================

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# =====================================================
# LOAD DATA
# =====================================================

print("Loading PCA features and labels...")

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# =====================================================
# APPLY K-MEANS
# =====================================================

print("\nApplying K-Means with K =", K)

model = KMeans(
    n_clusters=K,
    random_state=SEED,
    n_init=10
)

clusters = model.fit_predict(X)

ari = adjusted_rand_score(y, clusters)

print("ARI:", round(ari, 4))


# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame({
    "Actual Class": y,
    "Cluster": clusters
})

df["Actual Class Name"] = df["Actual Class"].map(
    lambda x: class_names[int(x)]
)


# =====================================================
# CLUSTER COMPOSITION TABLE
# =====================================================

print("\nCluster composition table:")

composition = pd.crosstab(
    df["Cluster"],
    df["Actual Class Name"]
)

composition = composition.reindex(
    columns=class_names,
    fill_value=0
)

print(composition)


# =====================================================
# SAVE TABLE
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

composition.to_csv(
    f"{OUTPUT_DIR}/cluster_composition.csv"
)

print("\nComposition saved.")


# =====================================================
# HEATMAP
# =====================================================

plt.figure(figsize=(14, 8))

sns.heatmap(
    composition,
    annot=True,
    fmt="d",
    cmap="YlGnBu"
)

plt.title("K-Means Cluster Composition")
plt.xlabel("Actual EuroSAT Land Type")
plt.ylabel("K-Means Cluster")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/cluster_composition_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# CLUSTER PURITY
# =====================================================

print("\nCluster purity:")

purity_values = []

for cluster_id in range(K):

    cluster_data = df[df["Cluster"] == cluster_id]

    counts = cluster_data["Actual Class"].value_counts()

    majority_count = counts.max()

    total_count = len(cluster_data)

    purity = majority_count / total_count

    purity_values.append(purity)

    majority_class = class_names[int(counts.index[0])]

    print(
        f"Cluster {cluster_id}: "
        f"Majority class = {majority_class}, "
        f"Purity = {purity:.4f}"
    )


overall_purity = (
    sum(
        len(df[df["Cluster"] == cluster_id])
        * purity_values[cluster_id]
        for cluster_id in range(K)
    )
    / len(df)
)

print("\nOverall cluster purity:", round(overall_purity, 4))


# =====================================================
# SAVE PURITY RESULTS
# =====================================================

purity_df = pd.DataFrame({
    "Cluster": range(K),
    "Purity": purity_values
})

purity_df.to_csv(
    f"{OUTPUT_DIR}/cluster_purity.csv",
    index=False
)

print("\nCluster interpretation completed!")
