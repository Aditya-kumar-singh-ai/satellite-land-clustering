
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root and results directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Results collected from completed experiments
results = [
    {
        "Experiment": "PCA + K-Means",
        "Features": 50,
        "Clusters (K)": 10,
        "Silhouette Score": 0.1699,
        "ARI": 0.1221,
        "Preprocessing": "Pixel values / 255 + PCA"
    },
    {
        "Experiment": "Mean RGB + Std RGB",
        "Features": 6,
        "Clusters (K)": 10,
        "Silhouette Score": 0.3414,
        "ARI": 0.1672,
        "Preprocessing": "Colour statistics / 255"
    },
    {
        "Experiment": "Scaled Colour Features",
        "Features": 6,
        "Clusters (K)": 10,
        "Silhouette Score": 0.3310,
        "ARI": 0.1861,
        "Preprocessing": "StandardScaler"
    },
    {
        "Experiment": "RGB Histogram",
        "Features": 24,
        "Clusters (K)": 10,
        "Silhouette Score": 0.3198,
        "ARI": 0.1138,
        "Preprocessing": "Normalized 8-bin RGB histograms"
    },
    {
        "Experiment": "Pretrained ResNet18",
        "Features": 512,
        "Clusters (K)": 10,
        "Silhouette Score": 0.0645,
        "ARI": 0.4293,
        "Preprocessing": "ResNet18 pretrained transforms + StandardScaler"
    }
]

df = pd.DataFrame(results)

# Display the comparison table
print("\n--- Model Comparison ---")
print(df.to_string(index=False))

# Save the table
csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV saved to: {csv_path}")

# Plot the two metrics separately because their scales differ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(df["Experiment"], df["Silhouette Score"])
axes[0].set_title("Silhouette Score by Experiment")
axes[0].set_ylabel("Silhouette Score")
axes[0].tick_params(axis="x", rotation=60)

axes[1].bar(df["Experiment"], df["ARI"])
axes[1].set_title("Adjusted Rand Index by Experiment")
axes[1].set_ylabel("ARI")
axes[1].tick_params(axis="x", rotation=60)

plt.tight_layout()

plot_path = RESULTS_DIR / "model_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Comparison graph saved to: {plot_path}")
print("\nModel comparison completed!")

