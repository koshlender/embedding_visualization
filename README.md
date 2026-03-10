# embedding_visualization

This app helps students visualize how sentence embeddings behave in a 3D PCA projection.

## Structure

- `app.py`: Streamlit frontend entrypoint.
- `embedding_lab/backend.py`: model loading, embedding generation, PCA projection, and metric computation.
- `embedding_lab/plotting.py`: Plotly figure construction for the 3D vector-space view.
- `embedding_lab/ui.py`: Streamlit session-state and presentation helpers.
- `embedding_lab/data.py`: static dataset and shared constants.
- `embedding_lab/types.py`: shared dataclasses used across frontend and backend modules.

## Run

```bash
streamlit run app.py
```
