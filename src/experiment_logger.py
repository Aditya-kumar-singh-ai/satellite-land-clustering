
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "experiment_results.csv"

COLUMNS = [
    "Experiment",
    "Features",
    "Clusters",
    "Silhouette Score",
    "ARI"
]


def save_result(experiment, features, clusters, silhouette, ari):
    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "Experiment": experiment,
            "Features": features,
            "Clusters": clusters,
            "Silhouette Score": round(float(silhouette), 4),
            "ARI": round(float(ari), 4)
        })

    print(f"Experiment results saved to: {CSV_PATH}")
