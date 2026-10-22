

from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset(
    "blanchon/EuroSAT_RGB",
    split="train"
)

sample = dataset[0]

print("Image size:", sample["image"].size)
print("Numeric label:", sample["label"])
print("Filename:", sample["filename"])

plt.imshow(sample["image"])
plt.title(f"Filename: {sample['filename']}")
plt.axis("off")
plt.show()
