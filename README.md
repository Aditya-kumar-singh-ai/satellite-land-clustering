# Satellite Land-Type Clustering

An unsupervised machine learning project that groups satellite images by visual similarity using CNN feature extraction and K-Means clustering. The project uses the EuroSAT RGB dataset and a pretrained ResNet18 model to extract image features.

## Project Overview

Satellite images contain visual patterns associated with forests, crops, rivers, residential areas, and other land-cover types.

This project explores how pretrained CNN features can help group satellite images into clusters without using land-use labels during model training.

It also includes a batch prediction pipeline and an interactive Streamlit dashboard that allows users to upload new satellite images and receive cluster assignments.

## Objectives

- Explore and preprocess satellite images.
- Extract 512-dimensional image features using pretrained ResNet18.
- Apply K-Means clustering to CNN features.
- Compare clustering algorithms and different values of K.
- Evaluate clustering using Adjusted Rand Index (ARI) and Silhouette Score.
- Examine cluster purity, sampling robustness, and initialization stability.
- Build a reusable model for batch prediction on new images.
- Develop an interactive Streamlit dashboard.

## Dataset

- **Dataset:** EuroSAT RGB
- **Source:** Hugging Face `blanchon/EuroSAT_RGB`
- **Image size:** 64 × 64 pixels
- **Number of land-use categories:** 10
- **Training sample used for the final model:** 10,000 images
- **Sampling seed:** 42

The dataset labels are used for evaluation, not as training targets for the clustering algorithm.

## Technologies Used

- Python 3.11
- NumPy and Pandas
- Matplotlib
- Scikit-learn
- PyTorch and TorchVision
- Hugging Face Datasets
- Streamlit
- Joblib
- Git and GitHub

## Feature Extraction and Clustering

A pretrained ResNet18 model is used as a feature extractor. Its final classification layer is replaced with an identity layer, producing a 512-dimensional feature vector for each image.

The extracted features are standardized using `StandardScaler` and passed to K-Means.

### Final Model Configuration

| Parameter | Value |
|---|---|
| Feature extractor | Pretrained ResNet18 |
| Feature dimensions | 512 |
| Training images | 10,000 |
| Clustering algorithm | K-Means |
| Number of clusters (K) | 12 |
| Random state | 42 |
| Initializations (`n_init`) | 10 |

## Final Model Results

| Evaluation metric | Result |
|---|---:|
| Adjusted Rand Index (ARI) | 0.5349 |
| Silhouette Score | 0.0408 |
| Weighted cluster purity | 74.45% |

**Interpretation:**

- ARI measures agreement between the predicted clusters and the reference dataset labels.
- Silhouette Score measures how well-separated the clusters are in feature space.
- Weighted cluster purity describes the proportion of images belonging to the dominant reference class within their assigned clusters.

The ARI indicates meaningful agreement with the reference labels. However, the relatively low Silhouette Score suggests that the clusters are not strongly separated. Cluster purity is an evaluation measure, not classification accuracy.

## Model Comparison

Using 10,000 images and K-Means with K = 12 as the selected configuration:

| Algorithm | Silhouette Score | ARI |
|---|---:|---:|
| K-Means | 0.0408 | 0.5349 |
| Agglomerative Clustering (Ward) | 0.0241 | 0.4855 |

K-Means achieved a higher ARI and Silhouette Score in this comparison.

## K-Means Tuning

Different cluster counts were evaluated using CNN features.

| Number of clusters | Silhouette Score | ARI |
|---:|---:|---:|
| 5 | 0.0574 | 0.3584 |
| 8 | 0.0342 | 0.4632 |
| 10 | 0.0399 | 0.4702 |
| 12 | 0.0408 | 0.5349 |
| 15 | 0.0444 | 0.5137 |
| 20 | 0.0476 | 0.4647 |

K = 12 was selected because it achieved the highest ARI among the tested configurations. The best Silhouette Score occurred at K = 5, demonstrating that the two evaluation metrics favour different cluster counts.

## Stability Evaluation

The selected model was evaluated across different sampling seeds and K-Means initialization seeds.

- Sampling robustness: mean ARI of approximately 0.5343.
- Initialization stability: mean ARI of approximately 0.5350.

These results indicate that the measured ARI remained relatively consistent across the tested seeds.

## Batch Prediction

The project supports predictions on new satellite images without retraining the clustering model.

Supported inputs:

