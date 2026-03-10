"""Frontend helpers for Streamlit state and text rendering."""

from __future__ import annotations

import streamlit as st


def ensure_session_state() -> None:
    """Initialize session state keys used by the app."""
    if "inferred_records" not in st.session_state:
        st.session_state.inferred_records = []
    if "sentence_a" not in st.session_state:
        st.session_state.sentence_a = None
    if "sentence_b" not in st.session_state:
        st.session_state.sentence_b = None


def sync_sentence_selectors(all_sentences: list[str]) -> tuple[str, str]:
    """Keep sentence selectors stable across reruns and new inference."""
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
    return sentence_a, sentence_b


def render_educational_notes() -> None:
    """Render the sidebar notes."""
    st.markdown("---")
    st.markdown(
        "**Educational Notes**\n"
        "- Embeddings are dense vectors representing meaning.\n"
        "- Similar meanings often cluster because vector directions become similar.\n"
        "- PCA projects high-dimensional structure into 3D for human visualization."
    )


def render_pair_explanation(metric: str, sentence_a: str, sentence_b: str, cosine: float, angle: float, distance: float, projected_angle: float) -> None:
    """Render the pair-metric explanation block."""
    st.subheader("Similarity Panel")
    st.caption(f"Visual emphasis: {metric}")
    st.write(
        f"`{sentence_a}` vs `{sentence_b}` gives cosine "
        f"`{cosine:.3f}`, angle `{angle:.2f} deg`, and distance `{distance:.3f}`."
    )
    st.write(
        f"The 3D plot currently shows a projected angle of `{projected_angle:.2f} deg` after PCA."
    )
    st.markdown("### How to read the comparison")
    st.write(
        "Cosine similarity focuses on angular alignment between vectors, while Euclidean distance "
        "measures straight-line separation."
    )
