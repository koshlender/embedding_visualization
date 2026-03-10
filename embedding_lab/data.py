"""Static datasets and UI constants for the embedding lab."""

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
METRIC_OPTIONS = ["Cosine similarity", "Euclidean distance"]
