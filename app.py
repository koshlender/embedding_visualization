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


def cosine_to_angle_degrees(cosine_values: np.ndarray) -> np.ndarray:
    """Convert cosine similarity values into angles in degrees."""
    clipped = np.clip(cosine_values, -1.0, 1.0)
    return np.degrees(np.arccos(clipped))


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


def compute_pair_metrics(
    sentence_a: str,
    sentence_b: str,
    all_sentences: list,
    all_embeddings: np.ndarray,
    all_points_3d: np.ndarray,
) -> dict:
    """Compute high-dimensional and projected 3D metrics for a chosen sentence pair."""
    idx_a = all_sentences.index(sentence_a)
    idx_b = all_sentences.index(sentence_b)

    embedding_a = all_embeddings[idx_a]
    embedding_b = all_embeddings[idx_b]
    point_a = all_points_3d[idx_a]
    point_b = all_points_3d[idx_b]

    cosine_value = float(compute_cosine_similarity(embedding_a, embedding_b.reshape(1, -1))[0])
    euclidean_value = float(
        compute_euclidean_distance(embedding_a, embedding_b.reshape(1, -1))[0]
    )
    angle_value = float(cosine_to_angle_degrees(np.array([cosine_value]))[0])

    point_a_norm = np.linalg.norm(point_a)
    point_b_norm = np.linalg.norm(point_b)
    if point_a_norm < 1e-12 or point_b_norm < 1e-12:
        projected_cosine = 0.0
    else:
        projected_cosine = float(
            np.clip(np.dot(point_a / point_a_norm, point_b / point_b_norm), -1.0, 1.0)
        )

    return {
        "sentence_a": sentence_a,
        "sentence_b": sentence_b,
        "cosine_similarity": cosine_value,
        "angle_degrees": angle_value,
        "euclidean_distance": euclidean_value,
        "projected_angle_degrees": float(np.degrees(np.arccos(projected_cosine))),
        "point_a": point_a,
        "point_b": point_b,
    }


def _add_line_between_points(
    fig: go.Figure,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    color: str = "#555555",
) -> None:
    """Add a straight 3D segment between two points."""
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            name="Distance Link",
            line=dict(color=color, width=4),
            hovertemplate=f"{label}<extra></extra>",
            showlegend=False,
        )
    )


def _add_cosine_angle_wedge(
    fig: go.Figure,
    query_point: np.ndarray,
    target_point: np.ndarray,
    target_sentence: str,
    show_query_vector: bool,
) -> None:
    """Draw two rays from the origin and an arc between them to show the angle."""
    query_norm = np.linalg.norm(query_point)
    target_norm = np.linalg.norm(target_point)
    if query_norm < 1e-12 or target_norm < 1e-12:
        return

    query_unit = query_point / query_norm
    target_unit = target_point / target_norm
    orthogonal = target_unit - np.dot(target_unit, query_unit) * query_unit
    orthogonal_norm = np.linalg.norm(orthogonal)
    if orthogonal_norm < 1e-12:
        return
    orthogonal_unit = orthogonal / orthogonal_norm

    projected_cosine = float(np.clip(np.dot(query_unit, target_unit), -1.0, 1.0))
    angle_radians = np.arccos(projected_cosine)
    angle_degrees = np.degrees(angle_radians)
    arc_radius = min(query_norm, target_norm) * 0.32
    arc_steps = np.linspace(0.0, angle_radians, 40)
    arc_points = np.array(
        [
            arc_radius * (np.cos(step) * query_unit + np.sin(step) * orthogonal_unit)
            for step in arc_steps
        ]
    )

    origin = np.zeros(3)
    if show_query_vector:
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], query_point[0]],
                y=[origin[1], query_point[1]],
                z=[origin[2], query_point[2]],
                mode="lines",
                name="Selected Vector",
                line=dict(color="#111111", width=6),
                hovertemplate="Selected vector from origin<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers",
                name="Origin",
                marker=dict(size=5, color="#111111"),
                hovertemplate="Origin<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], target_point[0]],
            y=[origin[1], target_point[1]],
            z=[origin[2], target_point[2]],
            mode="lines",
            name="Comparison Vector",
            line=dict(color=HIGHLIGHT_COLOR, width=5),
            hovertemplate=f"{target_sentence}<br>Vector from origin<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=arc_points[:, 0],
            y=arc_points[:, 1],
            z=arc_points[:, 2],
            mode="lines",
            name="Angle Wedge",
            line=dict(color="#ef6c00", width=8),
            hovertemplate=(
                f"{target_sentence}<br>"
                f"projected angle={angle_degrees:.2f} deg<br>"
                f"projected cos={projected_cosine:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )


def build_plot(
    original_sentences: list,
    original_clusters: list,
    original_points_3d: np.ndarray,
    inferred_records: list,
    sentence_a: str,
    sentence_b: str,
    metric: str,
    all_sentences: list,
    all_points_3d: np.ndarray,
) -> go.Figure:
    """Construct interactive 3D plot for a chosen sentence pair."""
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

    if sentence_a in all_sentences and sentence_b in all_sentences:
        point_a = all_points_3d[all_sentences.index(sentence_a)]
        point_b = all_points_3d[all_sentences.index(sentence_b)]

        fig.add_trace(
            go.Scatter3d(
                x=[point_a[0]],
                y=[point_a[1]],
                z=[point_a[2]],
                mode="markers",
                name="Sentence A",
                marker=dict(size=11, color="#111111", symbol="circle-open"),
                text=[sentence_a],
                hovertemplate="<b>Sentence A:</b> %{text}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[point_b[0]],
                y=[point_b[1]],
                z=[point_b[2]],
                mode="markers",
                name="Sentence B",
                marker=dict(size=10, color=HIGHLIGHT_COLOR, symbol="x"),
                text=[sentence_b],
                hovertemplate="<b>Sentence B:</b> %{text}<extra></extra>",
            )
        )

        if metric == "Cosine similarity":
            _add_cosine_angle_wedge(
                fig=fig,
                query_point=point_a,
                target_point=point_b,
                target_sentence=sentence_b,
                show_query_vector=True,
            )
        else:
            _add_line_between_points(
                fig=fig,
                start=point_a,
                end=point_b,
                label=f"{sentence_a} vs {sentence_b}",
                color="#1f77b4",
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="#f8fafc",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, b=0, t=70),
        height=680,
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="Interactive Embedding Lab", layout="wide")

    st.title("Interactive Embedding Lab")
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

        if st.button("Generate Embedding", type="primary", use_container_width=True):
            text = new_sentence.strip()
            if not text:
                st.warning("Please enter a sentence.")
            else:
                add_new_sentence(st.session_state.inferred_records, text, model, pca)
                st.session_state.sentence_a = text
                st.success("New sentence embedded, projected, and added to the plot.")
                # Rebuild merged arrays after adding.
                inferred = st.session_state.inferred_records
                inferred_sentences = [r["sentence"] for r in inferred]
                inferred_embeddings = np.vstack([r["embedding"] for r in inferred])
                inferred_points_3d = np.vstack([r["projection"] for r in inferred])
                all_sentences = base["sentences"] + inferred_sentences
                all_embeddings = np.vstack([base["embeddings"], inferred_embeddings])
                all_points_3d = np.vstack([project_embeddings(pca, base["embeddings"]), inferred_points_3d])

        if st.session_state.get("sentence_a") not in all_sentences:
            st.session_state.sentence_a = all_sentences[-1]

        sentence_a = st.selectbox(
            "Choose sentence A",
            options=all_sentences,
            index=all_sentences.index(st.session_state.sentence_a),
            key="sentence_a",
        )
        sentence_b_options = [sentence for sentence in all_sentences if sentence != sentence_a]

        if st.session_state.get("sentence_b") not in sentence_b_options:
            st.session_state.sentence_b = sentence_b_options[0]

        sentence_b = st.selectbox(
            "Choose sentence B",
            options=sentence_b_options,
            index=sentence_b_options.index(st.session_state.sentence_b),
            key="sentence_b",
        )

        pair_metrics = compute_pair_metrics(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            all_sentences=all_sentences,
            all_embeddings=all_embeddings,
            all_points_3d=all_points_3d,
        )
        pair_df = pd.DataFrame(
            [
                {
                    "Sentence A": pair_metrics["sentence_a"],
                    "Sentence B": pair_metrics["sentence_b"],
                    "Cosine Similarity": pair_metrics["cosine_similarity"],
                    "Angle (degrees)": pair_metrics["angle_degrees"],
                    "Euclidean Distance": pair_metrics["euclidean_distance"],
                    "Projected 3D Angle": pair_metrics["projected_angle_degrees"],
                }
            ]
        )

        st.subheader("Similarity Panel")
        st.caption(f"Visual emphasis: {metric}")
        st.dataframe(pair_df, use_container_width=True, hide_index=True)
        st.write(
            f"`{sentence_a}` vs `{sentence_b}` gives cosine "
            f"`{pair_metrics['cosine_similarity']:.3f}`, angle "
            f"`{pair_metrics['angle_degrees']:.2f} deg`, and distance "
            f"`{pair_metrics['euclidean_distance']:.3f}`."
        )
        st.write(
            f"The 3D plot currently shows a projected angle of "
            f"`{pair_metrics['projected_angle_degrees']:.2f} deg` after PCA."
        )

        st.markdown("### How to read the comparison")
        st.write(
            "Cosine similarity focuses on angular alignment between vectors, while Euclidean distance "
            "measures straight-line separation. Picking two explicit sentences makes that difference "
        )

    with col_plot:
        base_points_3d = project_embeddings(pca, base["embeddings"])
        fig = build_plot(
            original_sentences=base["sentences"],
            original_clusters=base["clusters"],
            original_points_3d=base_points_3d,
            inferred_records=st.session_state.inferred_records,
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            metric=metric,
            all_sentences=all_sentences,
            all_points_3d=all_points_3d,
        )
        st.plotly_chart(fig, use_container_width=True)

    explained_variance = pca.explained_variance_ratio_.sum() * 100
    st.info(
        f"PCA explanatory power: first 3 principal components retain ~{explained_variance:.2f}% "
        "of variance from the original 384D embeddings."
    )


if __name__ == "__main__":
    main()
