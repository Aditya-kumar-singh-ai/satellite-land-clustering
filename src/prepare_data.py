
from datasets import load_dataset
import numpy as np

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Start with 500 images to test our pipeline
images = dataset.select(range(500))

features = []

for sample in images:
    image = np.array(sample["image"], dtype=np.float32)

    # Convert the image into a one-dimensional array
    image_features = image.flatten()

    features.append(image_features)

X = np.array(features)

print("Feature matrix shape:", X.shape)
print("Number of images:", X.shape[0])
print("Features per image:", X.shape[1])
