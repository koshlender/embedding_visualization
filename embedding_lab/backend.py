"""Backend services for embeddings, projections, and pairwise metrics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from embedding_lab.data import INITIAL_SENTENCES
from embedding_lab.types import BaseEmbeddings, CombinedEmbeddings, InferredRecord, PairMetrics


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load the embedding model once for the process."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_base_embeddings() -> BaseEmbeddings:
    """Encode the teaching dataset once and reuse it."""
    model = load_model()
    sentences = [text for text, _ in INITIAL_SENTENCES]
    clusters = [cluster for _, cluster in INITIAL_SENTENCES]
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return BaseEmbeddings(sentences=sentences, clusters=clusters, embeddings=embeddings)


@lru_cache(maxsize=1)
def load_pca() -> PCA:
    """Fit PCA once on the base dataset."""
    pca = PCA(n_components=3, random_state=42)
    pca.fit(load_base_embeddings().embeddings)
    return pca


def project_embeddings(pca: PCA, embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings into the fixed PCA space."""
    return pca.transform(embeddings)


def compute_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity manually with NumPy."""
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = np.clip(query_norm * matrix_norms, a_min=1e-12, a_max=None)
    return np.dot(matrix, query) / denom


def cosine_to_angle_degrees(cosine_values: np.ndarray) -> np.ndarray:
    """Convert cosine similarity values into angles in degrees."""
    clipped = np.clip(cosine_values, -1.0, 1.0)
    return np.degrees(np.arccos(clipped))


def compute_euclidean_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance manually with NumPy."""
    return np.linalg.norm(matrix - query, axis=1)


def create_inferred_record(sentence: str, model: SentenceTransformer, pca: PCA) -> InferredRecord:
    """Embed a new sentence and project it into the learned PCA space."""
    embedding = model.encode([sentence], convert_to_numpy=True)[0]
    projection = project_embeddings(pca, embedding.reshape(1, -1))[0]
    return InferredRecord(sentence=sentence, embedding=embedding, projection=projection)


def merge_embeddings(base: BaseEmbeddings, inferred_records: list[InferredRecord], pca: PCA) -> CombinedEmbeddings:
    """Combine base and inferred sentences into one analysis state."""
    base_points = project_embeddings(pca, base.embeddings)
    if inferred_records:
        inferred_embeddings = np.vstack([record.embedding for record in inferred_records])
        inferred_points = np.vstack([record.projection for record in inferred_records])
    else:
        inferred_embeddings = np.empty((0, base.embeddings.shape[1]))
        inferred_points = np.empty((0, 3))

    return CombinedEmbeddings(
        sentences=base.sentences + [record.sentence for record in inferred_records],
        embeddings=np.vstack([base.embeddings, inferred_embeddings]),
        points_3d=np.vstack([base_points, inferred_points]),
    )


def compute_pair_metrics(
    sentence_a: str,
    sentence_b: str,
    combined: CombinedEmbeddings,
) -> PairMetrics:
    """Compute high-dimensional and projected metrics for a chosen sentence pair."""
    idx_a = combined.sentences.index(sentence_a)
    idx_b = combined.sentences.index(sentence_b)

    embedding_a = combined.embeddings[idx_a]
    embedding_b = combined.embeddings[idx_b]
    point_a = combined.points_3d[idx_a]
    point_b = combined.points_3d[idx_b]

    cosine_value = float(compute_cosine_similarity(embedding_a, embedding_b.reshape(1, -1))[0])
    euclidean_value = float(compute_euclidean_distance(embedding_a, embedding_b.reshape(1, -1))[0])
    angle_value = float(cosine_to_angle_degrees(np.array([cosine_value]))[0])

    point_a_norm = np.linalg.norm(point_a)
    point_b_norm = np.linalg.norm(point_b)
    if point_a_norm < 1e-12 or point_b_norm < 1e-12:
        projected_cosine = 0.0
    else:
        projected_cosine = float(
            np.clip(np.dot(point_a / point_a_norm, point_b / point_b_norm), -1.0, 1.0)
        )

    return PairMetrics(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        cosine_similarity=cosine_value,
        angle_degrees=angle_value,
        euclidean_distance=euclidean_value,
        projected_angle_degrees=float(np.degrees(np.arccos(projected_cosine))),
        point_a=point_a,
        point_b=point_b,
    )


def build_pair_dataframe(metrics: PairMetrics) -> pd.DataFrame:
    """Convert pair metrics into a single-row dataframe for display."""
    return pd.DataFrame(
        [
            {
                "Sentence A": metrics.sentence_a,
                "Sentence B": metrics.sentence_b,
                "Cosine Similarity": metrics.cosine_similarity,
                "Angle (degrees)": metrics.angle_degrees,
                "Euclidean Distance": metrics.euclidean_distance,
                "Projected 3D Angle": metrics.projected_angle_degrees,
            }
        ]
    )
