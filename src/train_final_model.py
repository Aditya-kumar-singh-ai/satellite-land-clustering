
from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 10000
SAMPLE_SEED = 42
N_CLUSTERS = 12
MODEL_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = ROOT_DIR / "data" / "cache"
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CACHED CNN FEATURES
# ============================================================

print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)

cache_prefix = (
    f"eurosat_resnet18_n{SAMPLE_SIZE}_seed{SAMPLE_SEED}"
)

features_path = CACHE_DIR / f"{cache_prefix}_features.npy"
labels_path = CACHE_DIR / f"{cache_prefix}_labels.npy"

if not features_path.exists() or not labels_path.exists():
    raise FileNotFoundError(
        "Required CNN feature cache not found."
    )

print("\nLoading CNN features...")

X = np.load(features_path)
y_true = np.load(labels_path)

print("Feature shape:", X.shape)
print("Label shape:", y_true.shape)

if X.shape != (SAMPLE_SIZE, 512):
    raise ValueError("Unexpected CNN feature shape.")

if len(y_true) != SAMPLE_SIZE:
    raise ValueError("Unexpected label count.")


# ============================================================
# FIT STANDARD SCALER
# ============================================================

print("\nFitting StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# TRAIN FINAL K-MEANS MODEL
# ============================================================

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=MODEL_SEED,
    n_init=10
)

cluster_ids = kmeans.fit_predict(X_scaled)

print("K-Means training completed.")


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating final model...")

ari = adjusted_rand_score(y_true, cluster_ids)

silhouette = silhouette_score(
    X_scaled,
    cluster_ids,
    sample_size=500,
    random_state=MODEL_SEED
)

print("ARI:", round(ari, 4))
print("Silhouette Score:", round(silhouette, 4))


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

print("\nSaving model artifacts...")

scaler_path = MODELS_DIR / "scaler.joblib"
kmeans_path = MODELS_DIR / "kmeans.joblib"
metadata_path = MODELS_DIR / "model_metadata.json"

joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, kmeans_path)

metadata = {
    "dataset": "EuroSAT RGB",
    "training_images": SAMPLE_SIZE,
    "sampling_seed": SAMPLE_SEED,
    "feature_extractor": "ResNet18 pretrained",
    "feature_dimensions": 512,
    "clustering_algorithm": "KMeans",
    "n_clusters": N_CLUSTERS,
    "model_seed": MODEL_SEED,
    "n_init": 10,
    "ari": float(ari),
    "silhouette_score": float(silhouette)
}

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nSaved files:")
print(scaler_path)
print(kmeans_path)
print(metadata_path)

print("\nFinal model configuration:")
print(json.dumps(metadata, indent=4))
