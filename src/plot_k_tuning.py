
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/pca_kmeans_tuning.csv")

# Plot 1: Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(
    df["K"],
    df["Silhouette Score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.xticks(df["K"])
plt.grid(True)

plt.savefig(
    "results/silhouette_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Plot 2: Adjusted Rand Index
plt.figure(figsize=(8, 5))
plt.plot(
    df["K"],
    df["ARI"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Adjusted Rand Index")
plt.title("ARI vs Number of Clusters")
plt.xticks(df["K"])
plt.grid(True)

plt.savefig(
    "results/ari_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graphs saved successfully!")
