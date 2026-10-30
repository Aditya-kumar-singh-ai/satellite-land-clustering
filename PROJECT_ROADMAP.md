# Satellite Land-Type Clustering — Project Roadmap

Repository: https://github.com/Aditya-kumar-singh-ai/satellite-land-clustering

## Project goal

Build an end-to-end unsupervised learning application that explores satellite-image clustering and lets users analyse new images through an interactive Streamlit interface.

## Current progress

**Estimated completion: approximately 65%.** This is a planning estimate based on the work completed so far, not an automated audit of every repository file.

### Completed

- Loaded and explored the EuroSAT RGB dataset.
- Built RGB mean/std and RGB histogram feature representations.
- Extracted 512-dimensional image features using pretrained ResNet18.
- Implemented K-Means clustering and PCA experiments.
- Compared feature representations using Silhouette Score and Adjusted Rand Index (ARI).
- Tuned K-Means over K = 5, 8, 10, 12, 15 and 20.
- Compared K-Means and Agglomerative clustering on CNN features.
- Conducted cluster purity, land-type distribution and stability analyses.
- Generated cluster contact sheets and difficult-land-type image examples.
- Added CNN feature caching for the 500-image, seed-42 experiment.
- Built a Streamlit dashboard with overview, feature comparison, algorithm comparison, K-Means tuning, cluster explorer, evaluation and project information pages.

### Baseline experiment

The initial experiments use a reproducible sample of 500 images from the EuroSAT RGB training split.

Selected results:
- RGB mean + standard deviation: Silhouette Score 0.3309; ARI 0.1861.
- RGB histogram: Silhouette Score 0.2702; ARI 0.1161.
- ResNet18 CNN features, K=10: Silhouette Score 0.0645; ARI 0.4293.
- ResNet18 CNN features, K=15: ARI 0.4757 in the K-Means tuning experiment.

These are results from different experiments/configurations. Do not present them as though they were all measured under the same K value or evaluation setup.

## Remaining roadmap

### Phase 1 — Scale up the experiments

- [ ] Preserve the existing 500-image results as the baseline.
- [ ] Run a separate 5,000-image experiment.
- [ ] Evaluate the full 16,200-image EuroSAT RGB training split if compute and memory allow.
- [ ] Keep sample size, random seed, feature configuration and K value recorded with each result.
- [ ] Avoid mixing results from different sample sizes or configurations in the same comparison.

### Phase 2 — Finalize the inference pipeline

- [ ] Select and document the final feature extractor and clustering configuration using evaluation evidence.
- [ ] Fit the feature scaler on the training features and save it.
- [ ] Fit K-Means on the selected training features and save the fitted model.
- [ ] Save the exact ResNet18 preprocessing configuration and model weights/version information.
- [ ] Ensure training and inference use identical image preprocessing and feature scaling.
- [ ] Test loading the saved artifacts in a clean run.

### Phase 3 — Single-image cluster assignment

- [ ] Add satellite image upload to Streamlit.
- [ ] Validate uploaded file type and image readability.
- [ ] Apply the saved ResNet18 preprocessing and feature extraction.
- [ ] Apply the saved scaler.
- [ ] Assign the new image to an existing cluster using the fitted K-Means model's predict() method.
- [ ] Display the uploaded image, assigned cluster and representative examples.
- [ ] Explain that a cluster ID is not itself a verified land-type classification.

### Phase 4 — Batch-image clustering

- [ ] Allow users to upload multiple satellite images.
- [ ] Extract features using the same saved ResNet18 pipeline.
- [ ] Offer assignment to the existing trained clusters.
- [ ] Optionally provide a separate mode to cluster the uploaded batch itself.
- [ ] Display cluster counts, image previews and downloadable assignments.

### Phase 5 — Streamlit integration and testing

- [ ] Add an Upload & Cluster New Images page.
- [ ] Test single-image and batch workflows with images not used to fit the clustering model.
- [ ] Handle invalid files, large uploads and missing model artifacts gracefully.
- [ ] Verify results are reproducible and preprocessing is consistent.
- [ ] Keep experiment-analysis pages distinct from new-image inference features.

### Phase 6 — Documentation and final delivery

- [ ] Update the README with project objective, dataset, methodology and results.
- [ ] Add dashboard screenshots and instructions to run the application.
- [ ] Document limitations and explain Silhouette Score versus ARI.
- [ ] Add dependency/setup instructions.
- [ ] Review .gitignore and avoid committing virtual environments or unnecessary large files.
- [ ] Test the project from a fresh environment.
- [ ] Prepare a concise project demonstration and interview explanation.

## Important technical notes

- The EuroSAT RGB training split contains 16,200 images; the current baseline scripts intentionally select 500.
- Feature caching must use distinct cache files for different sample sizes and seeds. If the extractor, weights or preprocessing changes, the cache identity should also change.
- K-Means is an unsupervised clustering algorithm. Its cluster IDs do not inherently correspond to semantic land-type names.
- ARI uses reference labels for evaluation; those labels must not be used to fit the unsupervised clustering model.
- Assigning a new image to an existing cluster requires a fitted scaler and fitted K-Means model. Do not refit either on each uploaded image.
- Batch clustering of a new upload is a different workflow from assigning images to the already-trained clusters; label the two modes clearly.
- Results from different sample sizes, random seeds, feature representations or K values must be identified separately.

## Current next action

Start Phase 1 by preserving the 500-image baseline and preparing a separate 5,000-image experiment. Do not overwrite the existing baseline outputs.
