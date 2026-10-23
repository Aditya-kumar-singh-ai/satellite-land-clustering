# Satellite Land-Type Clustering

An unsupervised machine learning project that explores satellite image clustering using the EuroSAT RGB dataset. The project compares traditional pixel and colour-based features with features extracted using a pretrained ResNet18 convolutional neural network (CNN).

## Project Overview

Satellite images contain useful patterns related to forests, crops, water bodies, roads and buildings. This project investigates how different image representations affect clustering quality.

Rather than training a classifier to predict labels directly, the project uses **K-Means clustering** and evaluates how the resulting clusters align with the dataset's actual land-use categories.

## Objectives

- Explore and preprocess satellite images.
- Apply dimensionality reduction using PCA.
- Compare colour statistics, RGB histograms and pretrained CNN features.
- Evaluate clustering using the Silhouette Score and Adjusted Rand Index (ARI).
- Investigate the effect of changing the number of clusters.
- Visualize clusters and document experimental results.

## Dataset

**EuroSAT RGB** — loaded using the Hugging Face `datasets` library.

The dataset contains satellite images across 10 land-use and land-cover categories, including Annual Crop, Forest, Highway, Residential Buildings, River and SeaLake.

The experiments described here use a shuffled sample of 500 images for development and comparison.

## Technologies Used

- Python
- NumPy and Pandas
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- PyTorch and TorchVision
- ResNet18 pretrained image features
- Git and GitHub

## Experiments and Results

The following results were obtained from the project's recorded experiments using 500 images and 10 clusters, except for the separate K-means tuning experiment.

| Feature representation | Features per image | Silhouette Score | ARI |
|---|---:|---:|---:|
| PCA + K-Means | 50 | 0.1699 | 0.1221 |
| Mean RGB + RGB standard deviation | 6 | 0.3414 | 0.1672 |
| Scaled colour features | 6 | 0.3310 | 0.1861 |
| RGB histograms | 24 | 0.3198 | 0.1138 |
| Pretrained ResNet18 | 512 | 0.0645 | 0.4293 |

**Key finding:** The pretrained ResNet18 features achieved the highest Adjusted Rand Index among these experiments, at 0.4293. The six-feature colour representation achieved the highest Silhouette Score, at 0.3414.

These metrics measure different properties. The Silhouette Score evaluates cluster separation, while ARI measures agreement between the clustering and actual labels. A lower Silhouette Score does not necessarily mean that a model aligns less well with the land-use categories.

The results are preliminary and based on a 500-image sample; they should not be interpreted as full-dataset performance.

## K-Means Tuning

The number of clusters was tested from K = 2 through K = 15 using scaled colour features.

- Best Silhouette Score in the tested range: **0.4780 at K = 2**.
- Best ARI in the tested range: **0.1927 at K = 15**.

The two metrics favour different cluster counts. The choice of K should depend on the goal of the experiment, not on one metric alone.

## Project Structure

```text
satellite-land-clustering/
├── src/
│   ├── load_dataset.py
│   ├── explore_dataset.py
│   ├── prepare_data.py
│   ├── kmeans_model.py
│   ├── colour_features.py
│   ├── colour_features_scaled.py
│   ├── colour_histogram.py
│   ├── kmeans_tuning.py
│   ├── cnn_features.py
│   └── compare_models.py
├── results/
│   ├── kmeans_tuning.png
│   ├── model_comparison.csv
│   └── model_comparison.png
├── main.py
└── requirement.txt
```

This structure describes the current intended organization; filenames may vary as the project develops.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aditya-kumar-singh-ai/satellite-land-clustering.git
cd satellite-land-clustering
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the project dependency file is named `requirement.txt`, run:

```powershell
python -m pip install -r requirement.txt
```

Make sure it includes the libraries required by the experiment you intend to run, including PyTorch and TorchVision for the CNN experiment.

## Run the Experiments

Run commands from the project root.

```powershell
python src/kmeans_model.py
python src/colour_features.py
python src/colour_features_scaled.py
python src/colour_histogram.py
python src/kmeans_tuning.py
python src/cnn_features.py
python src/compare_models.py
```

The scripts load the dataset and may download pretrained model weights on the first run. Internet access is required for these downloads.

## Evaluation Metrics

- **Silhouette Score:** Measures how well-separated clusters are in the feature space; higher is generally better.
- **Adjusted Rand Index (ARI):** Measures agreement between predicted clusters and actual labels, adjusted for chance; higher is generally better.

ARI is used for evaluation only. The K-Means algorithm itself does not use the actual land-type labels to form clusters.

## Limitations

- Experiments use a sample of 500 images rather than the full dataset.
- Results may vary with the sample, feature scaling, preprocessing and clustering parameters.
- ResNet18 was pretrained on a general image dataset rather than specifically optimized for EuroSAT.
- Colour features and independent RGB histograms lose some spatial information.
- Clustering results are exploratory and are not a replacement for a validated land-cover classification system.

## Future Improvements

- Evaluate the strongest approaches on larger samples and the full dataset.
- Test alternative pretrained feature extractors.
- Investigate PCA and other dimensionality-reduction methods on CNN embeddings.
- Save experiment configurations and results automatically.
- Improve cluster visualizations and add a reproducible experiment summary.

## Author

**Aditya Kumar Singh**

GitHub: https://github.com/Aditya-kumar-singh-ai

LinkedIn: https://www.linkedin.com/in/aditya-kumar-b19019338/
