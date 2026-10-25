# Satellite Land-Type Clustering

An unsupervised machine learning project exploring satellite image clustering using the EuroSAT RGB dataset. This project compares pixel-based features, colour statistics, RGB histograms, and features extracted using a pretrained ResNet18 convolutional neural network (CNN).

## Project Overview

Satellite images contain patterns related to forests, crops, water bodies, roads, and buildings. This project investigates how different image representations affect clustering quality. K-Means clustering is used without training on the land-use labels; the labels are used only to evaluate the resulting clusters.

## Objectives

- Explore and preprocess satellite images.
- Apply dimensionality reduction using PCA.
- Compare colour statistics, RGB histograms, and pretrained CNN features.
- Evaluate clustering with the Silhouette Score and Adjusted Rand Index (ARI).
- Investigate the effect of changing the number of clusters.
- Record experiment results and create comparison visualizations.

## Dataset

The project uses the EuroSAT RGB dataset through the Hugging Face `datasets` library. For the current experiments, a shuffled sample of 500 images is used. The dataset covers 10 land-use and land-cover categories.

## Technologies Used

- Python, NumPy, and Pandas
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- PyTorch and TorchVision
- Pretrained ResNet18 image features
- Git and GitHub

## Experiments and Results

The following recorded experiments use 500 images and K-Means with 10 clusters.

| Feature representation | Features per image | Silhouette Score | ARI |
|---|---:|---:|---:|
| PCA + K-Means | 50 | 0.1699 | 0.1221 |
| Mean RGB + RGB standard deviation | 6 | 0.3414 | 0.1672 |
| Scaled colour features | 6 | 0.3310 | 0.1861 |
| RGB histograms | 24 | 0.3198 | 0.1138 |
| Pretrained ResNet18 | 512 | 0.0645 | 0.4293 |

Among these experiments, ResNet18 features achieved the highest ARI (0.4293), while the six-feature mean-and-standard-deviation colour representation achieved the highest Silhouette Score (0.3414). These metrics measure different properties: Silhouette Score measures cluster separation in feature space, while ARI measures agreement between clusters and the dataset's true labels. Results are preliminary and should not be interpreted as full-dataset performance.

## K-Means Tuning

The number of clusters was tested from K = 2 through K = 15 using scaled colour features.

- Best Silhouette Score in the tested range: **0.4780 at K = 2**.
- Best ARI in the tested range: **0.1927 at K = 15**.

The metrics favour different cluster counts, so the choice of K depends on the goal of the experiment.

## Project Structure

```text
satellite-land-clustering/
├── src/
│   ├── load_dataset.py
│   ├── explore_dataset.py
│   ├── prepare_data.py
│   ├── kmeans_model.py
│   ├── colour_features.py
│   ├── colour_features_scale.py
│   ├── colour_histogram.py
│   ├── kmeans_tuning.py
│   ├── cnn_features.py
│   ├── compare_models.py
│   ├── experiment_logger.py
│   └── test_logger.py
├── results/
│   ├── kmeans_tuning.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── experiment_results.csv
├── requirement.txt
└── README.md
```

This tree reflects the project files discussed during development. If you later add, remove, or rename a file, update this section to match the actual folder contents.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aditya-kumar-singh-ai/satellite-land-clustering.git
cd satellite-land-clustering
```

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirement.txt
```

The dependency file should include the libraries required by the scripts you plan to run. The dataset-loading scripts need `datasets`; the CNN experiment needs `torch` and `torchvision`.

## Run the Experiments

Run commands from the repository root in an activated environment:

```powershell
python src/kmeans_model.py
python src/colour_features.py
python src/colour_features_scale.py
python src/colour_histogram.py
python src/kmeans_tuning.py
python src/cnn_features.py
python src/compare_models.py
```

The first CNN run may download pretrained model weights. Dataset and model downloads require internet access.

## Evaluation Metrics

- **Silhouette Score:** Measures how well-separated clusters are in feature space; higher is generally better.
- **Adjusted Rand Index (ARI):** Measures agreement between clusters and actual labels, adjusted for chance; higher is generally better.

ARI is used only for evaluation. K-Means does not use the true land-use labels to form clusters.

## Limitations

- Experiments use a 500-image sample rather than the full dataset.
- Results may vary with the sample, feature scaling, preprocessing, and clustering parameters.
- ResNet18 was pretrained on a general image dataset, not specifically optimized for EuroSAT.
- Colour features and independent RGB histograms lose some spatial information.
- The clustering results are exploratory and are not a replacement for a validated land-cover classification system.

## Future Improvements

- Evaluate promising approaches on larger samples and the full dataset.
- Test alternative pretrained feature extractors.
- Investigate dimensionality reduction on CNN embeddings.
- Improve cluster visualizations and reproducible experiment summaries.

## Author

**Aditya Kumar Singh**

GitHub: https://github.com/Aditya-kumar-singh-ai

LinkedIn: https://www.linkedin.com/in/aditya-kumar-b19019338/
