
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

st.set_page_config(
    page_title="Satellite Land-Type Clustering",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
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
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Navigation
# --------------------------------------------------
st.sidebar.title("🛰️ Satellite Intelligence")
st.sidebar.caption("Land-Cover Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Model Comparison",
        "Algorithm Comparison",
        "Cluster Explorer",
        "Evaluation & Insights",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset: EuroSAT RGB")
st.sidebar.caption("Algorithm: K-Means")

# --------------------------------------------------
# Load results
# --------------------------------------------------
csv_path = RESULTS_DIR / "model_comparison.csv"

df = pd.DataFrame()

if csv_path.exists():
    df = pd.read_csv(csv_path)

    for column in ["Silhouette Score", "ARI"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

# Load clustering algorithm comparison results
algorithm_csv_path = (
    RESULTS_DIR / "clustering_algorithm_comparison.csv"
)

algorithm_df = pd.DataFrame()

if algorithm_csv_path.exists():
    algorithm_df = pd.read_csv(algorithm_csv_path)

    numeric_columns = [
        "Silhouette Score",
        "ARI",
        "Number of Clusters",
        "Noise Points",
        "Runtime (seconds)",
    ]

    for column in numeric_columns:
        if column in algorithm_df.columns:
            algorithm_df[column] = pd.to_numeric(
                algorithm_df[column], errors="coerce"
            )
# --------------------------------------------------
# Reusable functions
# --------------------------------------------------
def show_image(title, filename, description):
    image_path = RESULTS_DIR / filename

    if image_path.exists():
        st.subheader(title)
        st.caption(description)
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(f"Visualization not found: {filename}")


def show_best_metric(column, label):
    if column not in df.columns:
        st.metric(label, "N/A")
        return

    valid = df.dropna(subset=[column])

    if valid.empty:
        st.metric(label, "N/A")
        return

    best = valid.loc[valid[column].idxmax()]
    st.metric(label, f"{best[column]:.4f}")

    if "Experiment" in best.index:
        st.caption(f"Method: {best['Experiment']}")


def show_comparison_table():
    if df.empty:
        st.warning(
            "No model results are available. "
            "Check results/model_comparison.csv."
        )
    else:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )


# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.title("🛰️ Satellite Land-Type Clustering")

    st.markdown(
        """
        Explore how different machine-learning feature
        representations group satellite images using
        **K-Means clustering**.

        **Dataset:** EuroSAT RGB · **Sample:** 500 images ·
        **Approach:** Unsupervised learning
        """
    )

    st.divider()

    if df.empty:
        st.warning(
            "Model comparison results were not found or are empty. "
            "Check results/model_comparison.csv."
        )
    else:
        st.header("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            show_best_metric(
                "Silhouette Score",
                "Best Silhouette Score",
            )

        with col2:
            show_best_metric(
                "ARI",
                "Best Adjusted Rand Index (ARI)",
            )

        st.subheader("Experiment Comparison")
        show_comparison_table()

        if "Experiment" in df.columns:
            if "Silhouette Score" in df.columns:
                st.subheader("Silhouette Score by Method")
                chart = df.set_index("Experiment")[
                    ["Silhouette Score"]
                ]
                st.bar_chart(chart)

            if "ARI" in df.columns:
                st.subheader("Adjusted Rand Index by Method")
                chart = df.set_index("Experiment")[["ARI"]]
                st.bar_chart(chart)

    show_image(
        "Model Comparison",
        "model_comparison.png",
        "Compare cluster separation and agreement with true labels.",
    )

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    st.write(
        "Compare the feature representations used in your "
        "clustering experiments."
    )

    if df.empty:
        st.warning("Model comparison data is unavailable.")
    else:
        show_comparison_table()

        available_metrics = [
            metric for metric in
            ["Silhouette Score", "ARI"]
            if metric in df.columns
        ]

        if available_metrics and "Experiment" in df.columns:
            selected_metric = st.selectbox(
                "Choose evaluation metric",
                available_metrics,
            )

            chart_df = df[
                ["Experiment", selected_metric]
            ].dropna().sort_values(
                selected_metric, ascending=False
            )

            st.subheader(f"{selected_metric} by Experiment")
            st.bar_chart(
                chart_df.set_index("Experiment")
            )

    show_image(
        "Saved Comparison Chart",
        "model_comparison.png",
        "Previously generated experiment comparison.",
    )


# --------------------------------------------------
# Algorithm Comparison
# --------------------------------------------------
elif page == "Algorithm Comparison":
    st.title("🔬 Clustering Algorithm Comparison")

    st.write(
        "Compare K-Means, Agglomerative Clustering, "
        "and DBSCAN using the same image features."
    )

    if algorithm_df.empty:
        st.warning(
            "Results not found. Run "
            "python src/compare_clustering_algorithms.py"
        )
    else:
        st.subheader("Comparison Results")

        st.dataframe(
            algorithm_df,
            hide_index=True,
            use_container_width=True,
        )

        valid = algorithm_df.dropna(
            subset=["Silhouette Score"]
        )

        if not valid.empty:
            winner = valid.loc[
                valid["Silhouette Score"].idxmax()
            ]

            st.success(
                f"Provisional winner by Silhouette Score: "
                f"{winner['Algorithm']} "
                f"({winner['Silhouette Score']:.4f})"
            )

        st.subheader("Silhouette Score")
        st.bar_chart(
            algorithm_df.set_index("Algorithm")[
                ["Silhouette Score"]
            ]
        )

        st.subheader("Adjusted Rand Index (ARI)")
        st.bar_chart(
            algorithm_df.set_index("Algorithm")[["ARI"]]
        )

        st.caption(
            "These are initial results. DBSCAN's parameters "
            "and the evaluation procedure should be reviewed "
            "before final model selection."
        )


# --------------------------------------------------
# Cluster Explorer
# --------------------------------------------------
elif page == "Cluster Explorer":
    st.title("🛰️ Cluster Explorer")

    st.write(
        "Explore the satellite image groups produced by K-Means."
    )

    show_image(
        "Satellite Images by Cluster",
        "kmeans_cluster_visualization.png",
        "Examples of satellite images grouped by K-Means.",
    )

    st.info(
        "Cluster IDs are group identifiers, not guaranteed "
        "land-type labels. Interpret them using reference data."
    )

# --------------------------------------------------
# Evaluation and Insights
# --------------------------------------------------
elif page == "Evaluation & Insights":
    st.title("📈 Evaluation & Insights")

    st.markdown(
        """
        **Silhouette Score** measures how well-separated
        the clusters are in the chosen feature space.

        **Adjusted Rand Index (ARI)** measures agreement between
        cluster assignments and known reference labels, adjusted
        for agreement expected by chance.
        """
    )

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            show_best_metric(
                "Silhouette Score",
                "Best Silhouette Score",
            )

        with col2:
            show_best_metric("ARI", "Best ARI")

    show_image(
        "Cluster vs Actual Land Type",
        "kmeans_cluster_landtype_heatmap.png",
        "Compare actual land-use categories across clusters.",
    )

    show_image(
        "Images per Cluster",
        "kmeans_cluster_sizes.png",
        "Compare the number of images assigned to each cluster.",
    )

    st.caption(
        "Evaluation results depend on preprocessing, "
        "features, sample selection, and the number of clusters."
    )

# --------------------------------------------------
# About Project
# --------------------------------------------------
elif page == "About Project":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### Objective
        Explore whether satellite images can be grouped by
        visual similarity using unsupervised learning.

        ### Dataset
        EuroSAT RGB satellite imagery.

        ### Techniques
        - PCA-reduced pixel values
        - RGB colour statistics
        - RGB histograms
        - Pretrained ResNet18 features
        - K-Means clustering

        ### Evaluation
        - Silhouette Score
        - Adjusted Rand Index (ARI)

        ### Limitations
        K-Means does not use true land-type labels when
        forming clusters. A single cluster may contain
        multiple land types.
        """
    )

    st.info(
        "The current experiments use a sample of 500 images. "
        "Results should not be treated as full-dataset performance."
    )

    st.caption(
        "Built with Python, Pandas, scikit-learn, and Streamlit."
    )
