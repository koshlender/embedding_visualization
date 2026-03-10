import streamlit as st

from embedding_lab.backend import (
    build_pair_dataframe,
    compute_pair_metrics,
    create_inferred_record,
    load_base_embeddings,
    load_model,
    load_pca,
    merge_embeddings,
)
from embedding_lab.data import METRIC_OPTIONS
from embedding_lab.plotting import build_plot
from embedding_lab.ui import ensure_session_state, render_educational_notes, render_pair_explanation, sync_sentence_selectors


def main() -> None:
    st.set_page_config(page_title="Interactive Embedding Lab", layout="wide")

    st.title("Interactive Embedding Lab")
    st.caption(
        "Model: all-MiniLM-L6-v2 (384D) | PCA: 384D -> 3D | Live inference + similarity analysis"
    )

    ensure_session_state()
    model = load_model()
    base = load_base_embeddings()
    pca = load_pca()

    with st.sidebar:
        st.subheader("Controls")
        metric = st.radio("Metric", METRIC_OPTIONS, index=0)
        render_educational_notes()

    col_plot, col_panels = st.columns([2.1, 1], gap="large")

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
                st.session_state.inferred_records.append(create_inferred_record(text, model, pca))
                st.session_state.sentence_a = text
                st.success("New sentence embedded, projected, and added to the plot.")

        combined = merge_embeddings(base, st.session_state.inferred_records, pca)
        sentence_a, sentence_b = sync_sentence_selectors(combined.sentences)
        pair_metrics = compute_pair_metrics(sentence_a, sentence_b, combined)
        pair_df = build_pair_dataframe(pair_metrics)

        st.dataframe(pair_df, use_container_width=True, hide_index=True)
        render_pair_explanation(
            metric=metric,
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            cosine=pair_metrics.cosine_similarity,
            angle=pair_metrics.angle_degrees,
            distance=pair_metrics.euclidean_distance,
            projected_angle=pair_metrics.projected_angle_degrees,
        )

    with col_plot:
        fig = build_plot(
            base=base,
            inferred_records=st.session_state.inferred_records,
            combined=combined,
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            metric=metric,
        )
        st.plotly_chart(fig, use_container_width=True)

    explained_variance = load_pca().explained_variance_ratio_.sum() * 100
    st.info(
        f"PCA explanatory power: first 3 principal components retain ~{explained_variance:.2f}% "
        "of variance from the original 384D embeddings."
    )


if __name__ == "__main__":
    main()
