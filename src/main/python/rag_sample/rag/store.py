"""Lazy, thread-safe access to the three artefacts retrieval needs.

The original code loaded the FAISS index (and built it, if missing) as an
import side effect of ``rag.query``, which meant importing a module could
spend a minute embedding documents. Loading is lazy here instead, and guarded
by a lock because the vector and full-text retrievers run in separate worker
threads and would otherwise race to load on the first query.
"""

import pickle
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_PATH, FAISS_INDEX_PATH, FTS_DB_PATH
from rag.fts import FtsIndex, build_fts_index

_lock = threading.RLock()
_index = None
_chunks = []
_fts = None
_loaded = False


def src_dir() -> Path:
    return Path(__file__).parent.parent


def paths():
    """Absolute paths of (faiss index, chunks pickle, fts database)."""
    base = src_dir()
    return (
        base / FAISS_INDEX_PATH,
        base / CHUNKS_PATH,
        base / FTS_DB_PATH,
    )


def _load_locked() -> bool:
    """Load whatever exists on disk. Caller must hold the lock."""
    global _index, _chunks, _fts, _loaded

    import faiss

    index_path, chunks_path, fts_path = paths()
    if not (index_path.exists() and chunks_path.exists()):
        return False

    try:
        _index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            _chunks = pickle.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Error loading existing index: {e}")
        _index, _chunks = None, []
        return False

    _fts = _open_or_rebuild_fts(fts_path, _chunks)
    _loaded = True
    return True


def _open_or_rebuild_fts(fts_path: Path, chunks):
    """Open the FTS index, rebuilding it if absent or out of sync.

    Rebuilding costs a fraction of a second because it needs no embeddings, so
    a stale or missing full-text index is self-healing rather than fatal.
    """
    try:
        if fts_path.exists():
            fts = FtsIndex(fts_path)
            if fts.size == len(chunks):
                return fts
            fts.close()
            print(
                f"🔁 Full-text index out of sync "
                f"({fts.size} rows vs {len(chunks)} chunks). Rebuilding..."
            )
        else:
            print("🔁 Full-text index missing. Building...")
        build_fts_index(chunks, fts_path)
        return FtsIndex(fts_path)
    except Exception as e:
        # Losing FTS degrades the system to vector-only, which is still a
        # working RAG. It must never be the reason a query fails.
        print(f"⚠️  Warning: full-text index unavailable ({e}). "
              f"Falling back to vector-only search.")
        return None


def ensure_loaded(build_if_missing: bool = True) -> bool:
    """Make the indexes available, building them from documents if needed."""
    global _loaded

    with _lock:
        if _loaded and _index is not None and _chunks:
            return True

        if _load_locked():
            return True

        if not build_if_missing:
            return False

        print("📦 Index not found. Building index from documents...")
        try:
            from rag.build_index import build_index
            build_index()
        except Exception as e:
            print(f"❌ Error building index: {e}")
            import traceback
            traceback.print_exc()
            return False

        if _load_locked():
            print("✅ Index built and loaded successfully")
            return True

        from config import DOCUMENTS_DIR
        print("❌ Failed to build index. No documents found or error occurred.")
        print(f"   Check that documents exist in: {src_dir() / DOCUMENTS_DIR}")
        return False


def get_chunks():
    ensure_loaded()
    return _chunks


def get_faiss_index():
    ensure_loaded()
    return _index


def get_fts_index():
    ensure_loaded()
    return _fts


def reset():
    """Drop the cached handles, so the next call reloads from disk."""
    global _index, _chunks, _fts, _loaded
    with _lock:
        if _fts is not None:
            _fts.close()
        _index, _chunks, _fts, _loaded = None, [], None, False
