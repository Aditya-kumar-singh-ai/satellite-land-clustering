

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root and results directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load actual experiment results
input_path = RESULTS_DIR / "experiment_results.csv"

if not input_path.exists():
    raise FileNotFoundError(
        "Experiment results not found. Run the experiments first."
    )

all_results = pd.read_csv(input_path)

# Select the five main experiments at K=10
main_experiments = [
    "PCA + K-Means",
    "Mean RGB + Std RGB",
    "Scaled Colour Features",
    "RGB Histogram",
    "Pretrained ResNet18"
]

df = all_results[
    all_results["Experiment"].isin(main_experiments)
].copy()

# Keep the latest result for each main experiment
df = df.drop_duplicates(
    subset=["Experiment"],
    keep="last"
)

# Add preprocessing descriptions for the report
preprocessing = {
    "PCA + K-Means": "Pixel values / 255 + PCA",
    "Mean RGB + Std RGB": "Colour statistics / 255",
    "Scaled Colour Features": "StandardScaler",
    "RGB Histogram": "Normalized 8-bin RGB histograms",
    "Pretrained ResNet18": "ResNet18 transforms + StandardScaler"
}

df["Preprocessing"] = df["Experiment"].map(preprocessing)

# Rename the cluster column for the report
df = df.rename(columns={"Clusters": "Clusters (K)"})

# Display the comparison table
print("\n--- Model Comparison ---")

if df.empty:
    print("No main experiment results found.")
    print("Check experiment names in experiment_results.csv.")
    raise SystemExit(1)

print(df.to_string(index=False))

# Save the comparison table
csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV saved to: {csv_path}")

# Plot the metrics separately because their scales differ
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

