
from pathlib import Path
from io import BytesIO
import argparse
import zipfile

import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image, UnidentifiedImageError
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png"
}

BATCH_SIZE = 32

DEVICE = torch.device("cpu")


# ============================================================
# LOAD TRAINED MODEL ARTIFACTS
# ============================================================

def load_trained_models():

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    if not KMEANS_PATH.exists():
        raise FileNotFoundError(
            f"K-Means model not found: {KMEANS_PATH}"
        )

    print("Loading saved StandardScaler...")
    scaler = joblib.load(SCALER_PATH)

    print("Loading saved K-Means model...")
    kmeans = joblib.load(KMEANS_PATH)

    print("Loading pretrained ResNet18...")

    weights = ResNet18_Weights.DEFAULT

    feature_extractor = resnet18(weights=weights)

    # Remove classification layer to obtain 512 features.
    feature_extractor.fc = torch.nn.Identity()

    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    transform = weights.transforms()

    return scaler, kmeans, feature_extractor, transform


# ============================================================
# COLLECT INPUT IMAGES
# ============================================================

def collect_images(input_paths):

    image_sources = []

    for input_path in input_paths:

        path = Path(input_path)

        if not path.exists():
            print(f"Warning: Input does not exist: {path}")
            continue

        # Individual image
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:

            image_sources.append({
                "filename": path.name,
                "source": path
            })

        # ZIP archive
        elif path.is_file() and path.suffix.lower() == ".zip":

            print(f"Reading ZIP archive: {path.name}")

            try:
                with zipfile.ZipFile(path, "r") as archive:

                    for member in archive.infolist():

                        if member.is_dir():
                            continue

                        member_path = Path(member.filename)

                        if member_path.suffix.lower() not in SUPPORTED_FORMATS:
                            continue

                        # Ignore hidden/system files.
                        if "__MACOSX" in member.filename.split("/"):
                            continue

                        image_sources.append({
                            "filename": member.filename,
                            "source": (path, member.filename)
                        })

            except zipfile.BadZipFile:
                print(f"Warning: Invalid ZIP file: {path}")

        # Folder containing images
        elif path.is_dir():

            for image_path in path.rglob("*"):

                if image_path.is_file() and (
                    image_path.suffix.lower() in SUPPORTED_FORMATS
                ):
                    image_sources.append({
                        "filename": image_path.relative_to(path).as_posix(),
                        "source": image_path
                    })

        else:
            print(f"Unsupported input: {path}")

    return image_sources


# ============================================================
# LOAD IMAGE SAFELY
# ============================================================

def load_image(source):

    if isinstance(source, tuple):

        zip_path, member_name = source

        with zipfile.ZipFile(zip_path, "r") as archive:

            image_bytes = archive.read(member_name)

        image = Image.open(BytesIO(image_bytes))

    else:

        image = Image.open(source)

    image.load()

    return image.convert("RGB")


# ============================================================
# EXTRACT RESNET18 FEATURES
# ============================================================

def extract_features(
    image_sources,
    feature_extractor,
    transform
):

    valid_features = []
    valid_filenames = []
    failed_results = []

    print("\nExtracting ResNet18 features...")

    for start in range(0, len(image_sources), BATCH_SIZE):

        batch_sources = image_sources[
            start:start + BATCH_SIZE
        ]

        batch_tensors = []
        batch_filenames = []

        for item in batch_sources:

            try:

                image = load_image(item["source"])

                tensor = transform(image)

                batch_tensors.append(tensor)
                batch_filenames.append(item["filename"])

            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
                zipfile.BadZipFile,
                KeyError
            ) as error:

                failed_results.append({
                    "Filename": item["filename"],
                    "Cluster ID": None,
                    "Status": f"Failed: {error}"
                })

                print(
                    f"Could not read {item['filename']}: {error}"
                )

        if not batch_tensors:
            continue

        image_tensor = torch.stack(batch_tensors).to(DEVICE)

        with torch.inference_mode():

            embeddings = feature_extractor(image_tensor)

        features = embeddings.cpu().numpy().astype(np.float32)

        valid_features.append(features)
        valid_filenames.extend(batch_filenames)

        print(
            f"Processed {min(start + BATCH_SIZE, len(image_sources))}"
            f"/{len(image_sources)} input files"
        )

    if not valid_features:

        return (
            np.empty((0, 512), dtype=np.float32),
            [],
            failed_results
        )

    X = np.vstack(valid_features)

    return X, valid_filenames, failed_results


# ============================================================
# PREDICT CLUSTERS
# ============================================================

def predict_batch(input_paths, output_csv=None):

    print("=" * 60)
    print("SATELLITE IMAGE BATCH INFERENCE")
    print("=" * 60)

    # Load saved models
    scaler, kmeans, feature_extractor, transform = (
        load_trained_models()
    )

    # Collect images
    image_sources = collect_images(input_paths)

    print("\nTotal input images:", len(image_sources))

    if not image_sources:
        raise ValueError(
            "No supported images found. Use JPG, JPEG, PNG or ZIP."
        )

    # Extract CNN features
    X, filenames, failed_results = extract_features(
        image_sources,
        feature_extractor,
        transform
    )

    results = []

    # Standardize and predict
    if len(X) > 0:

        print("\nApplying saved StandardScaler...")

        X_scaled = scaler.transform(X)

        print("Predicting clusters using saved K-Means...")

        cluster_ids = kmeans.predict(X_scaled)

        for filename, cluster_id in zip(filenames, cluster_ids):

            results.append({
                "Filename": filename,
                "Cluster ID": int(cluster_id),
                "Status": "Success"
            })

    results.extend(failed_results)

    results_df = pd.DataFrame(results)

    # Save CSV
    if output_csv is None:

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = RESULTS_DIR / "batch_predictions.csv"

    else:

        output_path = Path(output_csv)

        if not output_path.is_absolute():
            output_path = ROOT_DIR / output_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    results_df.to_csv(output_path, index=False)

    # Final summary
    successful = sum(
        result["Status"] == "Success"
        for result in results
    )

    failed = len(results) - successful

    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)

    print("Total images:", len(results))
    print("Successful predictions:", successful)
    print("Failed images:", failed)

    if successful:

        print("\nCluster distribution:")

        distribution = (
            results_df[
                results_df["Status"] == "Success"
            ]["Cluster ID"]
            .value_counts()
            .sort_index()
        )

        print(distribution.to_string())

    print("\nResults saved to:")
    print(output_path)

    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))

    return results_df


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Cluster new satellite images using saved models."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image paths, ZIP archives or folders."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path."
    )

    args = parser.parse_args()

    predict_batch(
        input_paths=args.inputs,
        output_csv=args.output
    )
