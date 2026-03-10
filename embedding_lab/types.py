"""Dataclasses shared across the app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaseEmbeddings:
    sentences: list[str]
    clusters: list[str]
    embeddings: np.ndarray


@dataclass
class InferredRecord:
    sentence: str
    embedding: np.ndarray
    projection: np.ndarray
    cluster: str = "Inferred"


@dataclass
class CombinedEmbeddings:
    sentences: list[str]
    embeddings: np.ndarray
    points_3d: np.ndarray


@dataclass
class PairMetrics:
    sentence_a: str
    sentence_b: str
    cosine_similarity: float
    angle_degrees: float
    euclidean_distance: float
    projected_angle_degrees: float
    point_a: np.ndarray
    point_b: np.ndarray
