from pathlib import Path
from io import BytesIO
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import resnet18, ResNet18_Weights

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Satellite Land-Type Clustering",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1120; color: #e5edf8; }
    [data-testid="stSidebar"] {
        background-color: #111a2d;
        border-right: 1px solid #26344b;
    }
    h1 { color: #f4f8ff; font-weight: 750; }
    h2, h3 { color: #dceaff; }
    div[data-testid="stMetric"] {
        background-color: #121d30;
        border: 1px solid #2a3a52;
        padding: 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] { color: #7dd3fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
        return data.loc[:, ~data.columns.str.startswith("Unnamed:")]
    except Exception:
        return pd.DataFrame()


def show_table(data, title):
    st.subheader(title)
    if data.empty:
        st.info("No data available for this table.")
    else:
        st.dataframe(data, hide_index=True, use_container_width=True)


def show_metric_chart(data, category_column, metric_column, title):
    if data.empty or category_column not in data.columns or metric_column not in data.columns:
        st.info(f"{title}: result file or required columns are unavailable.")
        return
    chart = data[[category_column, metric_column]].copy()
    chart[metric_column] = pd.to_numeric(chart[metric_column], errors="coerce")
    chart = chart.dropna().set_index(category_column)
    if not chart.empty:
        st.subheader(title)
        st.bar_chart(chart)


def show_image(title, filename, description=""):
    path = RESULTS_DIR / filename
    if path.exists():
        st.subheader(title)
        if description:
            st.caption(description)
        st.image(str(path), use_container_width=True)


@st.cache_resource
def load_inference_assets():
    scaler_path = MODELS_DIR / "scaler.joblib"
    kmeans_path = MODELS_DIR / "kmeans.joblib"
    if not scaler_path.exists() or not kmeans_path.exists():
        raise FileNotFoundError(
            "Saved model files are missing. Expected models/scaler.joblib "
            "and models/kmeans.joblib."
        )

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)

    weights = ResNet18_Weights.DEFAULT
    extractor = resnet18(weights=weights)
    extractor.fc = torch.nn.Identity()
    extractor.eval().to(DEVICE)
    return scaler, kmeans, extractor, weights.transforms()


def image_from_bytes(raw):
    image = Image.open(BytesIO(raw))
    image.load()
    return image.convert("RGB")


def collect_uploaded_images(uploaded_files):
    """Return (filename, bytes) pairs from image uploads and ZIP uploads."""
    items = []
    failures = []

    for uploaded in uploaded_files:
        name = uploaded.name
        suffix = Path(name).suffix.lower()

        if suffix in SUPPORTED_FORMATS:
            items.append((Path(name).name, uploaded.getvalue()))
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(BytesIO(uploaded.getvalue())) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        member_path = Path(member.filename)
                        if member_path.suffix.lower() not in SUPPORTED_FORMATS:
                            continue
                        if "__MACOSX" in member.filename.split("/"):
                            continue
                        # Avoid extracting files to disk; read supported members in memory.
                        if member.file_size > 20 * 1024 * 1024:
                            failures.append((member.filename, "File exceeds 20 MB limit"))
                            continue
                        items.append((member.filename, archive.read(member)))
            except (zipfile.BadZipFile, OSError) as exc:
                failures.append((name, f"Invalid ZIP: {exc}"))
        else:
            failures.append((name, "Unsupported format. Use JPG, JPEG, PNG or ZIP."))

    return items, failures


def predict_uploaded_images(uploaded_files):
    items, failures = collect_uploaded_images(uploaded_files)
    if not items and not failures:
        raise ValueError("No supported images found in the uploaded files.")

    scaler, kmeans, extractor, transform = load_inference_assets()
    records = []
    previews = []

    progress = st.progress(0, text="Preparing images...")
    total = len(items)

    for start in range(0, total, BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        tensors = []
        valid_names = []
        valid_images = []

        for filename, raw in batch:
            try:
                image = image_from_bytes(raw)
                tensors.append(transform(image))
                valid_names.append(filename)
                valid_images.append(image.copy())
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                records.append({
                    "Filename": filename,
                    "Cluster ID": None,
                    "Status": f"Failed: {exc}",
                })

        if tensors:
            batch_tensor = torch.stack(tensors).to(DEVICE)
            with torch.inference_mode():
                embeddings = extractor(batch_tensor).cpu().numpy().astype(np.float32)

            scaled = scaler.transform(embeddings)
            cluster_ids = kmeans.predict(scaled)

            for filename, image, cluster_id in zip(valid_names, valid_images, cluster_ids):
                records.append({
                    "Filename": filename,
                    "Cluster ID": int(cluster_id),
                    "Status": "Success",
                })
                previews.append((filename, image, int(cluster_id)))

        progress.progress(
            min(start + len(batch), total) / max(total, 1),
            text=f"Processed {min(start + len(batch), total)} of {total} images",
        )

    for filename, reason in failures:
        records.append({"Filename": filename, "Cluster ID": None, "Status": f"Failed: {reason}"})

    progress.empty()
    return pd.DataFrame(records), previews


# Load current experiment result files. Missing older files are handled gracefully.
cnn_algorithm_df = load_csv("cnn_algorithm_comparison_10000.csv")
tuning_df = load_csv("cnn_kmeans_tuning_10000.csv")
purity_df = load_csv("cnn_cluster_purity_summary_10000_k12.csv")
landtype_summary_df = load_csv("cnn_landtype_confusion_summary_10000_k12.csv")
cluster_counts_df = load_csv("cnn_cluster_landtype_counts_10000_k12.csv")
cluster_percentages_df = load_csv("cnn_cluster_landtype_percentages_10000_k12.csv")
sampling_df = load_csv("cnn_sampling_robustness_summary_10000_k12.csv")
initialization_df = load_csv("cnn_initialization_stability_summary_10000_k12.csv")

pages = [
    "Overview",
    "Feature Comparison",
    "Algorithm Comparison",
    "K-Means Tuning",
    "Cluster Explorer",
    "Evaluation & Insights",
    "Dimensionality Reduction",
    "Batch Prediction",
    "About Project",
]
page = st.sidebar.radio("Navigate to", pages)
st.sidebar.title("🛰️ Satellite Intelligence")
st.sidebar.caption("EuroSAT RGB · Unsupervised Land-Cover Clustering")
st.sidebar.divider()
st.sidebar.caption("Training sample: 10,000 images")
st.sidebar.caption("Feature extractor: ResNet18 (512-D)")
st.sidebar.caption("Algorithm: K-Means · K=12")
st.sidebar.caption("Inference: JPG / JPEG / PNG / ZIP")

if page == "Overview":
    st.title("🛰️ Satellite Land-Type Clustering")
    st.markdown(
        """
        An unsupervised machine-learning project that groups satellite images
        according to visual features extracted by a pretrained ResNet18 network.

        **Dataset:** EuroSAT RGB · **Training sample:** 10,000 images
        **Feature representation:** 512-dimensional ResNet18 embeddings
        **Clustering:** StandardScaler + K-Means (K=12)
        """
    )
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images used", "10,000")
    c2.metric("Land-type categories", "10")
    c3.metric("CNN features", "512")
    c4.metric("Clusters", "12")
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Adjusted Rand Index", "0.5349")
    c2.metric("Silhouette Score", "0.0408")
    st.caption(
        "ARI measures agreement with reference labels; it is not classification accuracy. "
        "The low silhouette score indicates limited separation in the feature space."
    )
    st.info(
        "The final model was selected using ARI. Cluster IDs are arbitrary group identifiers, "
        "not automatically verified land-type labels."
    )

elif page == "Feature Comparison":
    st.title("📊 Feature Representation Comparison")
    st.write(
        "Compare colour statistics, RGB histograms, and pretrained ResNet18 "
        "embeddings for satellite image clustering."
    )

    feature_df = load_csv("feature_representation_comparison.csv")

    # Use the recorded historical experiment results if the CSV is absent.
    if feature_df.empty:
        feature_df = pd.DataFrame({
            "Feature Representation": [
                "RGB Mean + Std",
                "RGB Histogram",
                "ResNet18 CNN",
            ],
            "Feature Count": [6, 24, 512],
            "Images": [500, 500, 500],
            "Clusters": [10, 10, 10],
            "Silhouette Score": [0.3309496641, 0.2701827288, 0.0644616112],
            "ARI": [0.1861337451, 0.1161421225, 0.4293159518],
            "Feature Extraction Seconds": [
                0.4219018,
                0.5722620,
                15.9277325,
            ],
            "Clustering Seconds": [
                0.1316515,
                0.0613199,
                0.2161330,
            ],
        })
        st.info(
            "Showing the recorded 500-image experiment results because "
            "results/feature_representation_comparison.csv was not found."
        )

    st.subheader("Experiment Results")
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

    st.divider()
    show_metric_chart(
        feature_df,
        "Feature Representation",
        "Silhouette Score",
        "Silhouette Score Comparison",
    )
    show_metric_chart(
        feature_df,
        "Feature Representation",
        "ARI",
        "Adjusted Rand Index (ARI) Comparison",
    )
    show_metric_chart(
        feature_df,
        "Feature Representation",
        "Feature Extraction Seconds",
        "Feature Extraction Time (seconds)",
    )
    show_metric_chart(
        feature_df,
        "Feature Representation",
        "Feature Count",
        "Number of Features",
    )

    st.divider()
    st.subheader("Key Finding")
    st.markdown(
        """
        - **RGB Mean + Std:** 6 features; ARI 0.1861.
        - **RGB Histogram:** 24 features; ARI 0.1161.
        - **ResNet18 CNN:** 512 features; ARI 0.4293.
        """
    )
    st.success(
        "ResNet18 embeddings achieved the highest ARI among these historical "
        "feature representation experiments."
    )
    st.caption(
        "These experiments used 500 images and 10 clusters. The final deployed "
        "model uses 10,000 images, 512-dimensional ResNet18 embeddings, and "
        "12 clusters. These historical results are not directly comparable "
        "to the final model's evaluation metrics."
    )

elif page == "Algorithm Comparison":
    st.title("🔬 Clustering Algorithm Comparison")
    show_table(cnn_algorithm_df, "CNN Feature Comparison · 10,000 Images")
    if not cnn_algorithm_df.empty:
        show_metric_chart(cnn_algorithm_df, "Algorithm", "ARI", "Adjusted Rand Index")
        show_metric_chart(cnn_algorithm_df, "Algorithm", "Silhouette Score", "Silhouette Score")
    else:
        st.write(
            "**Recorded results:** K-Means — ARI 0.5349, silhouette 0.0408; "
            "Agglomerative Ward — ARI 0.4855, silhouette 0.0241."
        )
    st.caption("K-Means was selected based on the higher ARI in this comparison.")

elif page == "K-Means Tuning":
    st.title("⚙️ K-Means Hyperparameter Tuning")
    show_table(tuning_df, "K-Means Tuning · 10,000 Images")
    if not tuning_df.empty:
        show_metric_chart(tuning_df, "K", "ARI", "ARI by Number of Clusters")
        show_metric_chart(tuning_df, "K", "Silhouette Score", "Silhouette Score by K")
    else:
        st.write(pd.DataFrame({
            "K": [5, 8, 10, 12, 15, 20],
            "Silhouette Score": [0.0574, 0.0342, 0.0399, 0.0408, 0.0444, 0.0476],
            "ARI": [0.3584, 0.4632, 0.4702, 0.5349, 0.5137, 0.4647],
        }),)

    st.info(
        "K=12 was selected because it achieved the highest ARI in the tested configurations. "
        "K=5 achieved the highest silhouette score, so the two metrics do not select the same K."
    )

elif page == "Cluster Explorer":
    st.title("🛰️ Cluster Explorer")
    st.write(
        "The model assigns each image to one of 12 numerical clusters. "
        "These IDs should not be interpreted as land-type names without a validated mapping."
    )
    show_table(cluster_counts_df, "Cluster vs. Actual Land-Type Counts")
    show_table(cluster_percentages_df, "Cluster vs. Actual Land-Type Percentages")
    show_image(
        "Cluster Purity Heatmap",
        "cnn_cluster_purity_heatmap_10000_k12.png",
        "Reference land-type composition of each discovered cluster.",
    )
    show_image(
        "Land-Type Confusion Heatmap",
        "cnn_landtype_confusion_heatmap_10000_k12.png",
        "Shows how reference land types are distributed across clusters.",
    )

elif page == "Evaluation & Insights":
    st.title("📈 Evaluation & Insights")
    st.subheader("Final model")
    st.write(
        "- Training sample: 10,000 EuroSAT RGB images\n"
        "- Feature extractor: pretrained ResNet18, 512 features\n"
        "- Preprocessing: StandardScaler\n"
        "- Clustering: K-Means, K=12, random_state=42, n_init=10\n"
        "- ARI: 0.5349\n"
        "- Silhouette score: 0.0408 (computed on a 500-point sample)"
    )
    st.divider()
    show_table(purity_df, "Cluster Purity Summary · K=12")
    show_table(landtype_summary_df, "Land-Type Confusion Summary · K=12")
    show_table(sampling_df, "Sampling Robustness")
    show_table(initialization_df, "K-Means Initialization Stability")
    st.caption(
        "Weighted cluster purity is 74.45%; purity is not classification accuracy. "
        "Sampling robustness and initialization stability measure consistency under the tested seeds."
    )
    show_image("Cluster Purity Heatmap", "cnn_cluster_purity_heatmap_10000_k12.png")
    show_image("Land-Type Confusion Heatmap", "cnn_landtype_confusion_heatmap_10000_k12.png")


elif page == "Dimensionality Reduction":
    st.title("📉 Dimensionality Reduction")
    st.write(
        "Explore PCA and t-SNE visualizations of ResNet18 satellite image "
        "features and compare clustering results after PCA."
    )

    st.subheader("PCA Results")

    pca_df = load_csv("pca_explained_variance.csv")
    show_table(pca_df, "PCA Explained Variance")

    show_image(
        "PCA 2D Visualization",
        "pca_2d_visualization.png",
        "Projection of satellite image features into two principal components."
    )

    show_image(
        "PCA Explained Variance",
        "pca_explained_variance.png",
        "Variance retained by the principal components."
    )

    st.divider()
    st.subheader("t-SNE Results")

    show_image(
        "t-SNE 2D Visualization",
        "tsne_2d_visualization.png",
        "Two-dimensional visualization of local similarity in CNN feature space."
    )

 
    st.divider()
    st.subheader("PCA Clustering Comparison")

    pca_comparison_df = load_csv("pca_clustering_comparison.csv")

    show_table(
        pca_comparison_df,
        "Original CNN Features vs PCA Features"
    )

    if not pca_comparison_df.empty:
        show_metric_chart(
            pca_comparison_df,
            "Experiment",
            "ARI",
            "Adjusted Rand Index Comparison"
        )

        show_metric_chart(
            pca_comparison_df,
            "Experiment",
            "Silhouette Score",
            "Silhouette Score Comparison"
        )


    st.divider()
    st.subheader("PCA K-Means Tuning")

    pca_tuning_df = load_csv("pca_kmeans_tuning.csv")
    show_table(pca_tuning_df, "PCA K-Means Tuning Results")

    show_image(
        "Silhouette Score vs K",
        "silhouette_vs_k.png",
        "Silhouette scores for different numbers of clusters."
    )

    st.divider()
    st.subheader("Cluster Interpretation")

    composition_df = load_csv("cluster_composition.csv")
    purity_df = load_csv("cluster_purity.csv")

    show_table(composition_df, "Cluster Composition")
    show_table(purity_df, "Cluster Purity")

    show_image(
        "Cluster Composition Heatmap",
        "cluster_composition_heatmap.png",
        "Distribution of reference land types across discovered clusters."
    )


elif page == "Batch Prediction":

    st.title("🛰️ Batch Satellite Image Prediction")
    st.write(
        "Upload one or more satellite images, or a ZIP archive containing images. "
        "The dashboard uses the saved ResNet18 feature extractor, StandardScaler and K-Means model."
    )
    st.warning(
        "Predictions are cluster assignments, not supervised land-type classifications. "
        "A cluster number does not directly mean Forest, River, Highway, etc."
    )
    uploaded_files = st.file_uploader(
        "Upload JPG, JPEG, PNG images or ZIP archives",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        help="ZIP files are read in memory. Images larger than 20 MB inside a ZIP are skipped.",
    )
    if uploaded_files and st.button("Run clustering", type="primary"):
        try:
            with st.spinner("Loading model and predicting clusters..."):
                result_df, previews = predict_uploaded_images(uploaded_files)
            st.session_state["batch_result_df"] = result_df
            st.session_state["batch_previews"] = previews
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result_df = st.session_state.get("batch_result_df")
    previews = st.session_state.get("batch_previews", [])
    if result_df is not None:
        successful = result_df[result_df["Status"] == "Success"]
        failed = len(result_df) - len(successful)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total files", len(result_df))
        c2.metric("Successful predictions", len(successful))
        c3.metric("Failed files", failed)
        st.subheader("Prediction results")
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="satellite_cluster_predictions.csv",
            mime="text/csv",
        )
        if not successful.empty:
            distribution = successful["Cluster ID"].value_counts().sort_index()
            st.subheader("Cluster distribution")
            st.bar_chart(distribution)
            st.subheader("Images grouped by predicted cluster")
            cluster_ids = sorted(successful["Cluster ID"].dropna().astype(int).unique())
            for cluster_id in cluster_ids:
                cluster_items = [item for item in previews if item[2] == cluster_id]
                with st.expander(f"Cluster {cluster_id} · {len(cluster_items)} image(s)"):
                    if cluster_items:
                        cols = st.columns(min(3, len(cluster_items)))
                        for index, (filename, image, _) in enumerate(cluster_items):
                            with cols[index % len(cols)]:
                                st.image(image, caption=filename, use_container_width=True)

elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ### Objective
        Explore how satellite images can be grouped by visual similarity using
        unsupervised machine learning.

        ### Dataset
        EuroSAT RGB satellite imagery. The final experiment used a reproducible
        sample of 10,000 images from the training split.

        ### Final pipeline
        1. Resize/normalize images using the pretrained ResNet18 weight transforms.
        2. Extract 512-dimensional embeddings.
        3. Standardize embeddings using the fitted StandardScaler.
        4. Assign clusters using the saved K-Means model (K=12).

        ### Technologies
        Python, NumPy, Pandas, scikit-learn, PyTorch, Torchvision, Pillow, Joblib and Streamlit.

        ### Limitations
        - K-Means does not use land-type labels when learning clusters.
        - Cluster IDs are arbitrary and may contain multiple land types.
        - The silhouette score is low, indicating limited cluster separation.
        - The pretrained ResNet18 was not fine-tuned specifically for EuroSAT.
        - The model is an experimental clustering system, not a validated land-cover classifier.
        """
    )
    st.caption("Satellite Land-Type Clustering · Python + PyTorch + Streamlit")
