"""Dense vector retrieval over the FAISS index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VECTOR_TOP_K
from rag import store
from rag.embed import embed_query


def vector_search(query: str, top_k: int = VECTOR_TOP_K):
    """Return ``[(chunk_index, similarity)]`` best-first for a query.

    Similarity is inner product on L2-normalised vectors, i.e. cosine, so
    larger is better and the range is [-1, 1].
    """
    index = store.get_faiss_index()
    chunks = store.get_chunks()
    if index is None or not chunks:
        return []

    import faiss

    q_emb = embed_query(query)
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, min(top_k, index.ntotal))

    # FAISS pads with -1 when it finds fewer neighbours than requested;
    # chunks[-1] would silently return the last chunk of the corpus.
    return [
        (int(idx), float(score))
        for idx, score in zip(ids[0], scores[0])
        if idx >= 0
    ]
