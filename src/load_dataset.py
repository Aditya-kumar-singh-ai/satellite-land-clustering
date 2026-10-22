
from datasets import load_dataset

print("Loading EuroSAT dataset...")

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

print("Dataset loaded successfully!")
print("Number of images:", len(dataset))
print("Available columns:", dataset.column_names)
print("First sample:", dataset[0])
