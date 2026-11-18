
import io
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

SCALER_PATH = MODEL_DIR / "scaler.joblib"
KMEANS_PATH = MODEL_DIR / "kmeans.joblib"


# Load trained clustering assets once
scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)


# Load pretrained ResNet18 feature extractor
weights = ResNet18_Weights.DEFAULT

extractor = resnet18(weights=weights)
extractor.fc = torch.nn.Identity()
extractor.eval()

transform = weights.transforms()


def predict_image(image_bytes: bytes) -> int:
    """
    Extract ResNet18 features from an uploaded image,
    scale them, and predict its K-Means cluster.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        features = extractor(image_tensor)

    features = features.cpu().numpy()
    scaled_features = scaler.transform(features)

    cluster_id = int(kmeans.predict(scaled_features)[0])

    return cluster_id
