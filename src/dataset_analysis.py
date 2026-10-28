
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset

# Project folders
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading EuroSAT RGB dataset...")

# Load the dataset
dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")

print("\nDataset Analysis")
print("-" * 40)

# Total number of images
total_images = len(dataset)
print(f"Total images: {total_images}")

# Inspect image dimensions
first_image = dataset[0]["image"]
print(f"Image dimensions: {first_image.size}")

# Find the available dataset columns
print(f"Dataset columns: {dataset.column_names}")

# Find the land-cover class column
possible_label_columns = ["label", "land_type", "class", "category"]
label_column = next(
    (column for column in possible_label_columns
     if column in dataset.column_names),
    None
)

if label_column is None:
    raise ValueError(
        "Could not find a label column. "
        "Check the dataset columns printed above."
    )

# Count images in each class

# Count images in each class
class_counts = {}

label_feature = dataset.features["label"]
label_names = getattr(label_feature, "names", None)

for item in dataset:
    label = item["label"]

    if label_names is not None:
        label = label_names[label]

    label = str(label)
    class_counts[label] = class_counts.get(label, 0) + 1

print("\nImages per class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

# Plot the class distribution
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("EuroSAT RGB: Images per Land-Cover Class")
plt.xlabel("Land-Cover Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

chart_path = RESULTS_DIR / "dataset_class_distribution.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"\nChart saved to: {chart_path}")
print("Dataset analysis completed!")
