
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from torchvision.models import resnet18, ResNet18_Weights


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_or_extract_features(
    sample_size=500,
    seed=42,
    batch_size=32,
):
    # Each sample configuration gets separate cache files.
    cache_prefix = f"eurosat_resnet18_n{sample_size}_seed{seed}"

    features_file = CACHE_DIR / f"{cache_prefix}_features.npy"
    labels_file = CACHE_DIR / f"{cache_prefix}_labels.npy"

    # Reuse existing valid cache files.
    if features_file.exists() and labels_file.exists():
        print(f"Loading cached features: {cache_prefix}")

        features = np.load(features_file)
        labels = np.load(labels_file)

        if (
            features.shape == (sample_size, 512)
            and labels.shape == (sample_size,)
        ):
            print("Cached features loaded:", features.shape)
            return features, labels

        print("Cache files have unexpected shapes. Re-extracting...")

    print("Loading EuroSAT dataset...")
    dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")

    if sample_size > len(dataset):
        raise ValueError(
            f"Requested {sample_size} images, "
            f"but dataset contains only {len(dataset)}."
        )

    sample = dataset.shuffle(seed=seed).select(range(sample_size))
    labels = np.asarray(sample["label"])

    # Load pretrained ResNet18.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = weights.transforms()

    feature_batches = []

    print(f"Extracting CNN features from {sample_size} images...")

    # Transform and extract features batch by batch.
    # This avoids storing all transformed images in RAM.
    with torch.inference_mode():
        for start in range(0, sample_size, batch_size):

            end = min(start + batch_size, sample_size)

            batch_images = [
                transform(sample[i]["image"].convert("RGB"))
                for i in range(start, end)
            ]

            image_tensor = torch.stack(batch_images)

            output = model(image_tensor)

            feature_batches.append(output.cpu().numpy())

            print(f"Processed {end}/{sample_size}")

    features = np.vstack(feature_batches).astype(np.float32)

    # Save extracted features and labels.
    np.save(features_file, features)
    np.save(labels_file, labels)

    print("Features saved:", features_file)
    print("Labels saved:", labels_file)

    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)

    return features, labels


if __name__ == "__main__":
    X, y = load_or_extract_features(sample_size=10000, seed=42)

    print("\nFeature extraction completed.")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
