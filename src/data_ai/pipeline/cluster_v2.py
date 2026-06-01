import os
from pathlib import Path

# Force CPU fallback for MPS issues on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        # Force CPU to avoid MPS float64 issues on Apple Silicon
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedding_model


def cluster_documents(
    documents: list[tuple[Path, str]],
    min_topic_size: int = 10,
) -> tuple[dict[int, list[Path]], BERTopic]:
    """
    Cluster documents using BERTopic.

    Args:
        documents: List of (file_path, text) tuples
        min_topic_size: Minimum documents per topic

    Returns:
        Tuple of:
        - Dict mapping topic_id to list of file paths
        - The fitted BERTopic model (for keyword extraction)
    """
    if not documents:
        return {}, None

    paths = [doc[0] for doc in documents]
    texts = [doc[1] for doc in documents]

    # Need at least min_topic_size documents to cluster
    if len(documents) < min_topic_size:
        # Too few documents - return all as outliers
        return {-1: paths}, None

    embedding_model = _get_embedding_model()

    try:
        # Pre-compute embeddings as float32 numpy array to avoid MPS issues
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # Configure UMAP with explicit settings
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        # Configure HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            prediction_data=True,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=False,
        )

        # Fit with pre-computed float32 embeddings
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

        # Group paths by topic
        result: dict[int, list[Path]] = {}
        for path, topic_id in zip(paths, topics):
            if topic_id not in result:
                result[topic_id] = []
            result[topic_id].append(path)

        return result, topic_model
    except Exception as e:
        # Clustering failed - return all as outliers
        print(f"Clustering error: {e}")
        return {-1: paths}, None


def get_topic_keywords(
    model: BERTopic,
    topic_id: int,
    top_n: int = 5,
) -> list[str]:
    """
    Get top keywords for a topic.

    Args:
        model: Fitted BERTopic model
        topic_id: Topic ID
        top_n: Number of keywords to return

    Returns:
        List of keywords
    """
    if topic_id == -1:
        return ["sonstiges"]

    topic_words = model.get_topic(topic_id)
    if not topic_words:
        return []

    return [word for word, _ in topic_words[:top_n]]
