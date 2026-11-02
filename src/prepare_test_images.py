
from pathlib import Path
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT_DIR / "test_images"

TEST_DIR.mkdir(parents=True, exist_ok=True)

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

# Reproduce the exact shuffle used during model training.
shuffled = dataset.shuffle(seed=42)

# Select five images outside the 10,000-image training sample.
test_images = shuffled.select(range(10000, 10005))

label_names = dataset.features["label"].names

print("\nSaving held-out test images...")

for index, sample in enumerate(test_images):

    image = sample["image"].convert("RGB")

    label = label_names[int(sample["label"])]

    filename = f"test_{index + 1}_{label}.jpg"

    image.save(TEST_DIR / filename)

    print("Saved:", filename)

print("\nTest dataset created successfully!")
print("Total images:", len(test_images))
print("Location:", TEST_DIR)
