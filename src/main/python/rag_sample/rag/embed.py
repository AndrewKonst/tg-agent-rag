import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL

# The SentenceTransformer is loaded lazily and shared process-wide: it is
# ~90 MB of weights and used to be instantiated twice (here and in query.py).
_model = None


def get_model():
    """Return the shared embedding model, loading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks):
    """Generate embeddings for all chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query into a (1, dim) float32 array."""
    return np.asarray(get_model().encode([text]), dtype="float32")


if __name__ == "__main__":
    # Test embedding
    test_chunks = [
        {"text": "This is a test chunk.", "source": "test.txt", "chunk_id": 0}
    ]
    embeddings = embed_chunks(test_chunks)
    print(f"Embedding shape: {embeddings.shape}")
