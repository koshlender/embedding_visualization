"""Plotly figure construction for the embedding lab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from embedding_lab.data import CLUSTER_COLORS, HIGHLIGHT_COLOR, INFERRED_COLOR
from embedding_lab.types import BaseEmbeddings, CombinedEmbeddings, InferredRecord


def _add_line_between_points(
    fig: go.Figure,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    color: str = "#555555",
) -> None:
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
    base: BaseEmbeddings,
    inferred_records: list[InferredRecord],
    combined: CombinedEmbeddings,
    sentence_a: str,
    sentence_b: str,
    metric: str,
) -> go.Figure:
    """Construct the 3D plot for a chosen sentence pair."""
    fig = go.Figure()
    base_point_count = len(base.sentences)
    base_points_3d = combined.points_3d[:base_point_count]

    unique_clusters = sorted(set(base.clusters))
    for cluster in unique_clusters:
        idx = [i for i, cluster_name in enumerate(base.clusters) if cluster_name == cluster]
        points = base_points_3d[idx]
        hover = [base.sentences[i] for i in idx]
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

    if inferred_records:
        inferred_points = np.vstack([record.projection for record in inferred_records])
        inferred_text = [record.sentence for record in inferred_records]
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

    if sentence_a in combined.sentences and sentence_b in combined.sentences:
        point_a = combined.points_3d[combined.sentences.index(sentence_a)]
        point_b = combined.points_3d[combined.sentences.index(sentence_b)]

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
