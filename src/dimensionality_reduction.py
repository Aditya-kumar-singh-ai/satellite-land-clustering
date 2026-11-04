
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42
SAMPLE_SIZE = 5000

ROOT_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = ROOT_DIR / "data" / "cache"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = (
    CACHE_DIR
    / f"eurosat_resnet18_n{SAMPLE_SIZE}_seed{SEED}_features.npy"
)

LABEL_PATH = (
    CACHE_DIR
    / f"eurosat_resnet18_n{SAMPLE_SIZE}_seed{SEED}_labels.npy"
)


# ==================================================
# 2. LOAD EXISTING CNN FEATURES
# ==================================================

print("=" * 60)
print("DIMENSIONALITY REDUCTION")
print("=" * 60)

print("\nLoading saved CNN features...")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found: {FEATURE_PATH}"
    )

if not LABEL_PATH.exists():
    raise FileNotFoundError(
        f"Label file not found: {LABEL_PATH}"
    )

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

if len(X) != len(y):
    raise ValueError(
        "Feature and label counts do not match."
    )

print("Images:", len(X))
print("Original features:", X.shape[1])


# ==================================================
# 3. STANDARDIZE FEATURES
# ==================================================

print("\nStandardizing CNN features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled feature shape:", X_scaled.shape)


# ==================================================
# 4. PCA ANALYSIS
# ==================================================

print("\nRunning PCA...")

start = time.perf_counter()

pca = PCA(
    n_components=0.95,
    random_state=SEED
)

X_pca = pca.fit_transform(X_scaled)

pca_time = time.perf_counter() - start

print("PCA completed.")
print("Original dimensions:", X_scaled.shape[1])
print("Reduced dimensions:", X_pca.shape[1])

print(
    "Explained variance retained:",
    round(
        pca.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

print(
    "PCA execution time:",
    round(pca_time, 2),
    "seconds"
)


# ==================================================
# 5. SAVE PCA EXPLAINED VARIANCE
# ==================================================

variance_df = pd.DataFrame({
    "Principal Component": np.arange(
        1,
        len(pca.explained_variance_ratio_) + 1
    ),
    "Explained Variance Ratio":
        pca.explained_variance_ratio_,
    "Cumulative Explained Variance":
        np.cumsum(pca.explained_variance_ratio_)
})

variance_path = (
    RESULTS_DIR / "pca_explained_variance.csv"
)

variance_df.to_csv(
    variance_path,
    index=False
)

print("\nPCA variance results saved.")


# ==================================================
# 6. PCA EXPLAINED VARIANCE PLOT
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    variance_df["Principal Component"],
    variance_df["Cumulative Explained Variance"],
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title("PCA Cumulative Explained Variance")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_explained_variance.png",
    dpi=300
)

plt.close()


# ==================================================
# 7. PCA 2D VISUALIZATION
# ==================================================

print("\nGenerating PCA 2D visualization...")

pca_2d = PCA(
    n_components=2,
    random_state=SEED
)

X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=y,
    cmap="tab10",
    s=8,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("EuroSAT CNN Features - PCA 2D")

plt.colorbar(
    scatter,
    label="Actual Land-Type Label"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_2d_visualization.png",
    dpi=300
)

plt.close()

print("PCA visualization saved.")


# ==================================================
# 8. t-SNE ANALYSIS
# ==================================================

print("\nRunning t-SNE...")

start = time.perf_counter()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=SEED
)

X_tsne = tsne.fit_transform(X_scaled)

tsne_time = time.perf_counter() - start

print("t-SNE completed.")

print(
    "t-SNE execution time:",
    round(tsne_time, 2),
    "seconds"
)


# ==================================================
# 9. t-SNE 2D VISUALIZATION
# ==================================================

print("\nGenerating t-SNE visualization...")

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="tab10",
    s=8,
    alpha=0.7
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.title("EuroSAT CNN Features - t-SNE 2D")

plt.colorbar(
    scatter,
    label="Actual Land-Type Label"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "tsne_2d_visualization.png",
    dpi=300
)

plt.close()

print("t-SNE visualization saved.")


# ==================================================
# 10. SAVE REDUCED FEATURES
# ==================================================

np.save(
    RESULTS_DIR / "cnn_features_pca_95.npy",
    X_pca
)

np.save(
    RESULTS_DIR / "cnn_features_pca_2d.npy",
    X_pca_2d
)

np.save(
    RESULTS_DIR / "cnn_features_tsne_2d.npy",
    X_tsne
)


# ==================================================
# 11. SAVE EXPERIMENT SUMMARY
# ==================================================

summary = pd.DataFrame([
    {
        "Method": "PCA",
        "Original Features": X_scaled.shape[1],
        "Reduced Features": X_pca.shape[1],
        "Explained Variance (%)":
            pca.explained_variance_ratio_.sum() * 100,
        "Execution Time (seconds)": pca_time
    },
    {
        "Method": "PCA 2D",
        "Original Features": X_scaled.shape[1],
        "Reduced Features": 2,
        "Explained Variance (%)":
            pca_2d.explained_variance_ratio_.sum() * 100,
        "Execution Time (seconds)": np.nan
    },
    {
        "Method": "t-SNE",
        "Original Features": X_scaled.shape[1],
        "Reduced Features": 2,
        "Explained Variance (%)": np.nan,
        "Execution Time (seconds)": tsne_time
    }
])

summary_path = (
    RESULTS_DIR / "dimensionality_reduction_comparison.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 60)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("=" * 60)

print(summary.round(4).to_string(index=False))

print("\nFiles saved inside:", RESULTS_DIR)

print("\nExperiment completed successfully!")
