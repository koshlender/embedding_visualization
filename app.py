import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# Initial teaching dataset: semantically related sentences should cluster in embedding space.
INITIAL_SENTENCES = [
    ("The cat sits on the mat", "Animals"),
    ("A kitten is on the rug", "Animals"),
    ("The dog lies on the floor", "Animals"),
    ("It is raining today", "Weather"),
    ("Heavy rain is falling", "Weather"),
    ("The weather is wet and cold", "Weather"),
    ("I love eating pizza", "Food"),
    ("Pizza tastes delicious", "Food"),
    ("I enjoy Italian food", "Food"),
    ("The car engine is broken", "Outlier"),
]

CLUSTER_COLORS = {
    "Animals": "#1f77b4",
    "Weather": "#2ca02c",
    "Food": "#ff7f0e",
    "Outlier": "#7f7f7f",
}

INFERRED_COLOR = "#d62728"
HIGHLIGHT_COLOR = "#17becf"


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load the real embedding model once and reuse it across reruns."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def initialize_embeddings(_model: SentenceTransformer) -> dict:
    """
    Encode the initial corpus into high-dimensional embeddings.

    Educational note:
    - Each sentence becomes a 384-dimensional numeric vector.
    - High dimensionality allows the model to encode many semantic factors simultaneously.
    """
    sentences = [text for text, _ in INITIAL_SENTENCES]
    clusters = [cluster for _, cluster in INITIAL_SENTENCES]

    embeddings = _model.encode(sentences, convert_to_numpy=True)
    return {
        "sentences": sentences,
        "clusters": clusters,
        "embeddings": embeddings,
    }


@st.cache_resource(show_spinner=False)
def fit_pca(embeddings: np.ndarray) -> PCA:
    """
    Fit PCA once on original embeddings.

    Educational note:
    - PCA finds orthogonal directions (principal components) with highest variance.
    - Geometrically, this is a rotation + projection from 384D down to 3D.
    - We do NOT refit for new points; we only transform them into the same learned 3D space.
    """
    pca = PCA(n_components=3, random_state=42)
    pca.fit(embeddings)
    return pca


def project_embeddings(pca: PCA, embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings into the fixed PCA space."""
    return pca.transform(embeddings)


def compute_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity manually with NumPy.

    Educational note:
    - Cosine compares angle between vectors, not raw magnitude.
    - In embedding spaces, direction often captures semantic meaning better than length.
    """
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Prevent division-by-zero in degenerate cases.
    denom = np.clip(query_norm * matrix_norms, a_min=1e-12, a_max=None)
    return np.dot(matrix, query) / denom


def compute_euclidean_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance manually with NumPy.

    Educational note:
    - Euclidean uses straight-line distance.
    - In high-dimensional embedding spaces, Euclidean can be less robust than cosine
      because vector magnitudes can vary for reasons unrelated to semantic similarity.
    """
    return np.linalg.norm(matrix - query, axis=1)


def add_new_sentence(
    inferred_sentences: list,
    sentence: str,
    model: SentenceTransformer,
    pca: PCA,
) -> dict:
    """Embed a new sentence, project with existing PCA, and append to inferred list."""
    embedding = model.encode([sentence], convert_to_numpy=True)[0]
    projection = project_embeddings(pca, embedding.reshape(1, -1))[0]

    record = {
        "sentence": sentence,
        "embedding": embedding,
        "projection": projection,
        "cluster": "Inferred",
    }
    inferred_sentences.append(record)
    return record


def _score_against_all(
    query_embedding: np.ndarray,
    all_sentences: list,
    all_embeddings: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    """Return sorted similarity/distance table for a query sentence."""
    if metric == "Cosine similarity":
        scores = compute_cosine_similarity(query_embedding, all_embeddings)
        df = pd.DataFrame({"Sentence": all_sentences, "Cosine Similarity": scores})
        return df.sort_values("Cosine Similarity", ascending=False).reset_index(drop=True)

    distances = compute_euclidean_distance(query_embedding, all_embeddings)
    df = pd.DataFrame({"Sentence": all_sentences, "Euclidean Distance": distances})
    return df.sort_values("Euclidean Distance", ascending=True).reset_index(drop=True)


def build_plot(
    original_sentences: list,
    original_clusters: list,
    original_points_3d: np.ndarray,
    inferred_records: list,
    selected_sentence: str,
    metric: str,
    show_lines: bool,
    highlight_top2: bool,
    all_sentences: list,
    all_embeddings: np.ndarray,
    all_points_3d: np.ndarray,
) -> go.Figure:
    """Construct interactive 3D plot with optional similarity highlighting."""
    fig = go.Figure()

    # Plot original points by semantic cluster.
    unique_clusters = sorted(set(original_clusters))
    for cluster in unique_clusters:
        idx = [i for i, c in enumerate(original_clusters) if c == cluster]
        points = original_points_3d[idx]
        hover = [original_sentences[i] for i in idx]

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=cluster,
                marker=dict(size=7, color=CLUSTER_COLORS.get(cluster, "#444"), opacity=0.9),
                text=hover,
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    # Plot inferred points in a distinct color.
    if inferred_records:
        inferred_points = np.vstack([r["projection"] for r in inferred_records])
        inferred_text = [r["sentence"] for r in inferred_records]
        fig.add_trace(
            go.Scatter3d(
                x=inferred_points[:, 0],
                y=inferred_points[:, 1],
                z=inferred_points[:, 2],
                mode="markers",
                name="Inferred",
                marker=dict(size=8, color=INFERRED_COLOR, symbol="diamond", opacity=0.95),
                text=inferred_text,
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    # Similarity overlays for the currently selected sentence.
    if selected_sentence in all_sentences:
        selected_idx = all_sentences.index(selected_sentence)
        query_point = all_points_3d[selected_idx]
        query_embedding = all_embeddings[selected_idx]

        if metric == "Cosine similarity":
            values = compute_cosine_similarity(query_embedding, all_embeddings)
            # Exclude the selected sentence itself.
            ranked = np.argsort(values)[::-1]
            ranked = [i for i in ranked if i != selected_idx]
        else:
            values = compute_euclidean_distance(query_embedding, all_embeddings)
            ranked = np.argsort(values)
            ranked = [i for i in ranked if i != selected_idx]

        top2 = ranked[:2]

        # Mark selected sentence.
        fig.add_trace(
            go.Scatter3d(
                x=[query_point[0]],
                y=[query_point[1]],
                z=[query_point[2]],
                mode="markers",
                name="Selected",
                marker=dict(size=11, color="#111111", symbol="circle-open"),
                text=[selected_sentence],
                hovertemplate="<b>Selected:</b> %{text}<extra></extra>",
            )
        )

        if highlight_top2 and top2:
            neighbor_points = all_points_3d[top2]
            neighbor_text = [all_sentences[i] for i in top2]
            fig.add_trace(
                go.Scatter3d(
                    x=neighbor_points[:, 0],
                    y=neighbor_points[:, 1],
                    z=neighbor_points[:, 2],
                    mode="markers",
                    name="Top 2 Matches",
                    marker=dict(size=10, color=HIGHLIGHT_COLOR, symbol="cross"),
                    text=neighbor_text,
                    hovertemplate="<b>Match:</b> %{text}<extra></extra>",
                )
            )

        if show_lines:
            targets = top2 if highlight_top2 else ranked[:2]
            for i in targets:
                p = all_points_3d[i]
                if metric == "Cosine similarity":
                    label = f"cos={values[i]:.3f}"
                else:
                    label = f"dist={values[i]:.3f}"

                fig.add_trace(
                    go.Scatter3d(
                        x=[query_point[0], p[0]],
                        y=[query_point[1], p[1]],
                        z=[query_point[2], p[2]],
                        mode="lines",
                        name="Similarity Link",
                        line=dict(color="#555555", width=4),
                        hovertemplate=f"{label}<extra></extra>",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title="3D Semantic Embedding Space (PCA Projection)",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="#f8fafc",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, b=0, t=40),
        height=680,
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="Embedding Lab: Real Inference + PCA", layout="wide")

    st.title("Interactive Embedding Lab: Real Model, PCA, and Similarity")
    st.caption(
        "Model: all-MiniLM-L6-v2 (384D) | PCA: 384D -> 3D | Live inference + similarity analysis"
    )

    model = load_model()
    base = initialize_embeddings(model)
    pca = fit_pca(base["embeddings"])

    if "inferred_records" not in st.session_state:
        st.session_state.inferred_records = []

    with st.sidebar:
        st.subheader("Controls")
        metric = st.radio("Metric", ["Cosine similarity", "Euclidean distance"], index=0)
        show_lines = st.toggle("Show connecting lines to most similar points", value=True)
        highlight_top2 = st.toggle("Highlight top 2 similar sentences", value=True)

        if st.button("Reset Inferred Sentences", width="stretch"):
            st.session_state.inferred_records = []

        st.markdown("---")
        st.markdown(
            "**Educational Notes**\n"
            "- Embeddings are dense vectors representing meaning.\n"
            "- Similar meanings often cluster because vector directions become similar.\n"
            "- PCA projects high-dimensional structure into 3D for human visualization."
        )

    col_plot, col_panels = st.columns([2.1, 1], gap="large")

    # Merge original + inferred for analysis panels.
    inferred = st.session_state.inferred_records
    inferred_sentences = [r["sentence"] for r in inferred]

    if inferred:
        inferred_embeddings = np.vstack([r["embedding"] for r in inferred])
        inferred_points_3d = np.vstack([r["projection"] for r in inferred])
    else:
        inferred_embeddings = np.empty((0, base["embeddings"].shape[1]))
        inferred_points_3d = np.empty((0, 3))

    all_sentences = base["sentences"] + inferred_sentences
    all_embeddings = np.vstack([base["embeddings"], inferred_embeddings])
    all_points_3d = np.vstack([project_embeddings(pca, base["embeddings"]), inferred_points_3d])

    with col_panels:
        st.subheader("Inference Panel")
        new_sentence = st.text_area(
            "Enter a sentence for live inference",
            value="",
            placeholder="Type a new sentence and click Generate Embedding...",
            height=90,
        )

        if st.button("Generate Embedding", type="primary", width="stretch"):
            text = new_sentence.strip()
            if not text:
                st.warning("Please enter a sentence.")
            else:
                add_new_sentence(st.session_state.inferred_records, text, model, pca)
                st.success("New sentence embedded, projected, and added to the plot.")
                # Rebuild merged arrays after adding.
                inferred = st.session_state.inferred_records
                inferred_sentences = [r["sentence"] for r in inferred]
                inferred_embeddings = np.vstack([r["embedding"] for r in inferred])
                inferred_points_3d = np.vstack([r["projection"] for r in inferred])
                all_sentences = base["sentences"] + inferred_sentences
                all_embeddings = np.vstack([base["embeddings"], inferred_embeddings])
                all_points_3d = np.vstack([project_embeddings(pca, base["embeddings"]), inferred_points_3d])

        target_sentence = st.selectbox(
            "Select sentence for similarity analysis",
            options=all_sentences,
            index=max(len(all_sentences) - 1, 0),
        )

        selected_embedding = all_embeddings[all_sentences.index(target_sentence)]
        scored_df = _score_against_all(selected_embedding, all_sentences, all_embeddings, metric)

        st.subheader("Similarity Panel")
        st.dataframe(scored_df, width="stretch", hide_index=True)

        st.markdown("### Why cosine often works well")
        st.write(
            "Cosine similarity focuses on angular alignment between vectors, which usually reflects "
            "semantic relatedness better than absolute distance in high-dimensional embedding spaces."
        )

    with col_plot:
        base_points_3d = project_embeddings(pca, base["embeddings"])
        fig = build_plot(
            original_sentences=base["sentences"],
            original_clusters=base["clusters"],
            original_points_3d=base_points_3d,
            inferred_records=st.session_state.inferred_records,
            selected_sentence=target_sentence,
            metric=metric,
            show_lines=show_lines,
            highlight_top2=highlight_top2,
            all_sentences=all_sentences,
            all_embeddings=all_embeddings,
            all_points_3d=all_points_3d,
        )
        st.plotly_chart(fig, width="stretch")

    explained_variance = pca.explained_variance_ratio_.sum() * 100
    st.info(
        f"PCA explanatory power: first 3 principal components retain ~{explained_variance:.2f}% "
        "of variance from the original 384D embeddings."
    )


if __name__ == "__main__":
    main()
