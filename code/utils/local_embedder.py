"""
Local embedding (no XInference dependency).
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class LocalEmbedder:
    """Local BGE embedding."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("[ok] model loaded")

    def embed(
            self,
            texts: Union[str, List[str]],
            batch_size: int = 32,
            show_progress: bool = False
    ) -> List[List[float]]:
        """Encode text."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()


_local_embedder = None


def get_local_embedder():
    """Return the singleton embedder."""
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = LocalEmbedder()
    return _local_embedder
