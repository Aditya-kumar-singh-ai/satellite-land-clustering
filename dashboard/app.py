
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

st.set_page_config(
    page_title="Satellite Land-Type Clustering",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5edf8;
    }

    [data-testid="stSidebar"] {
        background-color: #111a2d;
        border-right: 1px solid #26344b;
    }

    h1 {
        color: #f4f8ff;
        font-weight: 750;
    }

    h2, h3 {
        color: #dceaff;
    }

    div[data-testid="stMetric"] {
        background-color: #121d30;
        border: 1px solid #2a3a52;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #7dd3fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================


@st.cache_data
def load_csv(filename):
    path = RESULTS_DIR / filename

    if path.exists():
        try:
            data = pd.read_csv(path)

            data = data.loc[
                :,
                ~data.columns.str.startswith("Unnamed:")
            ]

            return data

        except Exception as error:
            st.warning(f"Could not read {filename}: {error}")

    return pd.DataFrame()


def show_table(data, title):
    st.subheader(title)

    if data.empty:
        st.info("No data available.")
        return

    st.dataframe(
        data,
        hide_index=True,
        use_container_width=True
    )


def show_metric(data, column, label):
    if data.empty or column not in data.columns:
        st.metric(label, "N/A")
        return

    values = pd.to_numeric(data[column], errors="coerce").dropna()

    if values.empty:
        st.metric(label, "N/A")
    else:
        st.metric(label, f"{values.max():.4f}")


def show_image(title, filename, description=""):
    image_path = RESULTS_DIR / filename

    if image_path.exists():
        st.subheader(title)

        if description:
            st.caption(description)

        st.image(
            str(image_path),
            use_container_width=True
        )



def show_folder_images(folder_name, title):
    folder = RESULTS_DIR / folder_name

    if not folder.exists():
        st.warning(f"Folder not found: {folder}")
        return

    image_files = []

    for extension in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(folder.rglob(extension))

    image_files = sorted(image_files)

    if not image_files:
        st.warning(f"No images found inside: {folder}")
        return

    st.subheader(title)

    st.caption(f"Total images found: {len(image_files)}")

    columns = st.columns(2)

    for index, image_path in enumerate(image_files):

        with columns[index % 2]:
            st.image(
                str(image_path),
                caption=image_path.relative_to(folder).as_posix(),
                use_container_width=True
            )


def show_metric_chart(data, category_column, metric_column, title):
    if (
        data.empty
        or category_column not in data.columns
        or metric_column not in data.columns
    ):
        return

    chart_data = data[
        [category_column, metric_column]
    ].copy()

    chart_data[metric_column] = pd.to_numeric(
        chart_data[metric_column],
        errors="coerce"
    )

    chart_data = chart_data.dropna()

    if chart_data.empty:
        return

    chart_data = chart_data.set_index(category_column)

    st.subheader(title)
    st.bar_chart(chart_data)


# ============================================================
# 4. LOAD EXPERIMENT RESULTS
# ============================================================

model_df = load_csv("model_comparison.csv")

feature_df = load_csv(
    "feature_representation_comparison.csv"
)

algorithm_df = load_csv(
    "clustering_algorithm_comparison.csv"
)

cnn_algorithm_df = load_csv(
    "cnn_algorithm_comparison.csv"
)

tuning_df = load_csv(
    "cnn_kmeans_tuning.csv"
)

purity_df = load_csv(
    "cnn_cluster_purity_summary.csv"
)

landtype_summary_df = load_csv(
    "cnn_landtype_confusion_summary.csv"
)

cluster_counts_df = load_csv(
    "cnn_cluster_landtype_counts.csv"
)

cluster_percentages_df = load_csv(
    "cnn_cluster_landtype_percentages.csv"
)

stability_df = load_csv(
    "cnn_clustering_stability_summary.csv"
)

k_stability_df = load_csv(
    "cnn_k_stability_summary.csv"
)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛰️ Satellite Intelligence")
st.sidebar.caption("Land-Cover Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Feature Comparison",
        "Algorithm Comparison",
        "K-Means Tuning",
        "Cluster Explorer",
        "Evaluation & Insights",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Dataset: EuroSAT RGB")
st.sidebar.caption("Images: 500")
st.sidebar.caption("Feature extraction: ResNet18")
st.sidebar.caption("Main algorithm: K-Means")


# ============================================================
# 6. OVERVIEW
# ============================================================

if page == "Overview":

    st.title("🛰️ Satellite Land-Type Clustering")

    st.markdown(
        """
        An unsupervised machine-learning project that explores
        how satellite images can be grouped according to their
        visual characteristics.

        **Dataset:** EuroSAT RGB

        **Sample:** 500 satellite images

        **Techniques:** RGB features, histograms, ResNet18,
        K-Means and Agglomerative Clustering.
        """
    )

    st.divider()

    st.header("Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Images Analysed", "500")

    with col2:
        st.metric("Land-Type Categories", "10")

    with col3:
        st.metric("CNN Features", "512")

    st.divider()

    st.header("Feature Representation Performance")

    if not feature_df.empty:

        col1, col2 = st.columns(2)

        with col1:
            show_metric(
                feature_df,
                "Silhouette Score",
                "Best Silhouette Score"
            )

        with col2:
            show_metric(
                feature_df,
                "ARI",
                "Best ARI"
            )

        show_table(
            feature_df,
            "Feature Representation Results"
        )

        show_metric_chart(
            feature_df,
            "Feature Representation",
            "ARI",
            "ARI by Feature Representation"
        )

    else:
        st.warning("Feature comparison results are unavailable.")

    st.divider()

    st.info(
        "The highest ARI and highest Silhouette Score may come "
        "from different feature representations. They measure "
        "different aspects of clustering quality."
    )


# ============================================================
# 7. FEATURE COMPARISON
# ============================================================

elif page == "Feature Comparison":

    st.title("📊 Feature Representation Comparison")

    st.write(
        "Compare three ways of representing satellite images "
        "before applying K-Means clustering."
    )

    if feature_df.empty:

        st.warning(
            "Run src/compare_feature_representations.py "
            "to generate the results."
        )

    else:

        show_table(
            feature_df,
            "Complete Feature Comparison"
        )

        st.divider()

        st.header("Silhouette Score Comparison")

        show_metric_chart(
            feature_df,
            "Feature Representation",
            "Silhouette Score",
            "Cluster Separation"
        )

        st.header("Adjusted Rand Index Comparison")

        show_metric_chart(
            feature_df,
            "Feature Representation",
            "ARI",
            "Agreement with Actual Land-Type Labels"
        )

        st.divider()

        st.header("Feature Extraction Time")

        show_metric_chart(
            feature_df,
            "Feature Representation",
            "Feature Extraction Seconds",
            "Feature Extraction Time"
        )

        st.header("Clustering Time")

        show_metric_chart(
            feature_df,
            "Feature Representation",
            "Clustering Seconds",
            "Clustering Execution Time"
        )

        st.markdown(
            """
            ### Interpretation

            - RGB Mean + Std represents basic colour statistics.
            - RGB Histogram represents the distribution of colours.
            - ResNet18 extracts deep visual representations
              using a pretrained convolutional neural network.

            CNN features can capture richer visual information,
            but their Silhouette Score may be lower.
            """
        )


# ============================================================
# 8. ALGORITHM COMPARISON
# ============================================================

elif page == "Algorithm Comparison":

    st.title("🔬 Clustering Algorithm Comparison")

    st.write(
        "Compare clustering algorithms and understand "
        "their performance on satellite image features."
    )

    st.header("CNN Feature Algorithm Comparison")

    if not cnn_algorithm_df.empty:

        show_table(
            cnn_algorithm_df,
            "K-Means vs Agglomerative"
        )

        show_metric_chart(
            cnn_algorithm_df,
            "Algorithm",
            "Silhouette Score",
            "Silhouette Score Comparison"
        )

        show_metric_chart(
            cnn_algorithm_df,
            "Algorithm",
            "ARI",
            "Adjusted Rand Index Comparison"
        )

        show_metric_chart(
            cnn_algorithm_df,
            "Algorithm",
            "Clustering Seconds",
            "Clustering Runtime"
        )

    else:
        st.warning(
            "CNN algorithm comparison results are unavailable."
        )

    st.divider()

    st.header("Other Clustering Algorithm Experiments")

    if not algorithm_df.empty:

        show_table(
            algorithm_df,
            "General Algorithm Comparison"
        )

        show_metric_chart(
            algorithm_df,
            "Algorithm",
            "Silhouette Score",
            "Silhouette Score"
        )

        show_metric_chart(
            algorithm_df,
            "Algorithm",
            "ARI",
            "Adjusted Rand Index"
        )

    else:
        st.info("No additional algorithm results available.")

    st.caption(
        "Compare algorithms using the same feature representation "
        "and evaluation conditions whenever possible."
    )


# ============================================================
# 9. K-MEANS TUNING
# ============================================================

elif page == "K-Means Tuning":

    st.title("⚙️ K-Means Hyperparameter Tuning")

    st.write(
        "Evaluate how changing the number of clusters affects "
        "CNN-based K-Means clustering."
    )

    if tuning_df.empty:

        st.warning(
            "Run src/tune_kmeans_cnn.py to generate tuning results."
        )

    else:

        show_table(
            tuning_df,
            "K-Means Tuning Results"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            show_metric(
                tuning_df,
                "Silhouette Score",
                "Best Silhouette Score"
            )

            if "Silhouette Score" in tuning_df.columns:
                valid = tuning_df.dropna(
                    subset=["Silhouette Score"]
                )

                if not valid.empty:
                    best = valid.loc[
                        valid["Silhouette Score"].idxmax()
                    ]

                    st.caption(
                        f"Best K: {int(best['K'])}"
                    )

        with col2:

            show_metric(
                tuning_df,
                "ARI",
                "Best Adjusted Rand Index"
            )

            if "ARI" in tuning_df.columns:
                valid = tuning_df.dropna(subset=["ARI"])

                if not valid.empty:
                    best = valid.loc[
                        valid["ARI"].idxmax()
                    ]

                    st.caption(
                        f"Best K: {int(best['K'])}"
                    )

        st.divider()

        show_metric_chart(
            tuning_df,
            "K",
            "Silhouette Score",
            "Silhouette Score by Number of Clusters"
        )

        show_metric_chart(
            tuning_df,
            "K",
            "ARI",
            "ARI by Number of Clusters"
        )

        show_metric_chart(
            tuning_df,
            "K",
            "Clustering Seconds",
            "Clustering Runtime by K"
        )

        st.info(
            "A higher ARI indicates stronger agreement with the "
            "reference labels. A higher Silhouette Score indicates "
            "better separation in feature space."
        )



elif page == "Cluster Explorer":

    st.title("🛰️ Satellite Cluster Explorer")

    st.write(
        "Explore satellite images grouped by CNN-based K-Means clustering."
    )

    folder = RESULTS_DIR / "mixed_cluster_images"

    st.header("Mixed Cluster Contact Sheets")

    contact_sheets = [
        folder / "cluster_0_contact_sheet.png",
        folder / "cluster_1_contact_sheet.png",
        folder / "cluster_4_contact_sheet.png",
        folder / "cluster_6_contact_sheet.png",
    ]

    columns = st.columns(2)

    for index, image_path in enumerate(contact_sheets):

        with columns[index % 2]:

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption=image_path.stem.replace("_", " "),
                    use_container_width=True
                )

            else:
                st.warning(f"Image not found: {image_path.name}")

    st.divider()

    st.header("Difficult Land-Type Examples")

    difficult_folder = RESULTS_DIR / "difficult_landtypes"

    if difficult_folder.exists():

        images = []

        for extension in ["*.png", "*.jpg", "*.jpeg"]:
            images.extend(difficult_folder.rglob(extension))

        images = sorted(images)

        if images:

            columns = st.columns(2)

            for index, image_path in enumerate(images):

                with columns[index % 2]:

                    st.image(
                        str(image_path),
                        caption=image_path.relative_to(
                            difficult_folder
                        ).as_posix(),
                        use_container_width=True
                    )

        else:
            st.info("No difficult land-type images found.")

    else:
        st.warning("Difficult land-type folder not found.")

    st.info(
        "Cluster IDs represent groups discovered by the algorithm. "
        "They are not automatically equivalent to actual land types."
    )


# ============================================================
# 11. EVALUATION AND INSIGHTS
# ============================================================

elif page == "Evaluation & Insights":

    st.title("📈 Evaluation & Insights")

    st.markdown(
        """
        ### Evaluation Metrics

        **Silhouette Score**

        Measures how well-separated clusters are in the
        selected feature space.

        **Adjusted Rand Index (ARI)**

        Measures agreement between cluster assignments
        and known reference labels, adjusted for chance.

        **Cluster Purity**

        Measures the proportion of the dominant actual
        land-type category within a cluster.
        """
    )

    st.divider()
    st.header("Cluster Purity Analysis — K=10")

    show_table(
        purity_df,
        "CNN Cluster Purity Summary"
    )

    st.divider()

    st.header("Land-Type Confusion Analysis — K=15")

    show_table(
        landtype_summary_df,
        "Land-Type Confusion Summary"
    )

    with st.expander("View Cluster vs Land-Type Counts"):

        show_table(
            cluster_counts_df,
            "Cluster Land-Type Counts"
        )

    with st.expander("View Cluster vs Land-Type Percentages"):

        show_table(
            cluster_percentages_df,
            "Cluster Land-Type Percentages"
        )

    st.divider()

    st.header("Clustering Stability — K=15")

    show_table(
        stability_df,
        "CNN Clustering Stability Summary"
    )

    show_table(
        k_stability_df,
        "K Stability Comparison"
    )

    st.divider()

    st.header("Saved Evaluation Visualizations")

    show_image(
        "Cluster vs Actual Land Type",
        "kmeans_cluster_landtype_heatmap.png"
    )

    show_image(
        "Images per Cluster",
        "kmeans_cluster_sizes.png"
    )

    show_image(
        "CNN K-Means Tuning",
        "cnn_kmeans_tuning.png"
    )

    show_image(
        "Feature Representation Comparison",
        "model_comparison.png"
    )

    st.caption(
        "Results depend on feature extraction, preprocessing, "
        "sample selection and clustering configuration."
    )


# ============================================================
# 12. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### Objective

        Explore whether satellite images can be grouped
        according to their visual similarity using
        unsupervised machine learning.

        ### Dataset

        EuroSAT RGB satellite imagery.

        ### Feature Representations

        - RGB colour statistics
        - RGB histograms
        - Pretrained ResNet18 CNN features

        ### Machine Learning Techniques

        - Principal Component Analysis (PCA)
        - K-Means clustering
        - Agglomerative clustering
        - Hyperparameter tuning
        - Clustering stability analysis

        ### Evaluation Metrics

        - Silhouette Score
        - Adjusted Rand Index (ARI)
        - Cluster purity

        ### Limitations

        - Current experiments use a sample of 500 images.
        - K-Means does not use actual land-type labels
          when forming clusters.
        - A cluster can contain multiple land types.
        - Pretrained CNN features are not specifically
          trained on the EuroSAT dataset.

        ### Technologies

        - Python
        - NumPy
        - Pandas
        - Scikit-learn
        - PyTorch
        - Torchvision
        - Hugging Face Datasets
        - Matplotlib
        - Streamlit
        """
    )

    st.info(
        "This project investigates unsupervised grouping "
        "of satellite imagery. It is an experimental research "
        "project, not a production land-classification system."
    )

    st.caption(
        "Satellite Land-Type Clustering | Built with Python"
    )
