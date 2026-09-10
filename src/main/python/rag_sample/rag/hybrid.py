"""Hybrid retrieval: query expansion, two retrievers in parallel, then fusion.

    query
      │
      ├─ LLM keyword generation (rag.keywords)
      │
      ├──────────────┬───────────────┐
      ▼              ▼               │  submitted to the same pool, so the
    FAISS          SQLite FTS5       │  two searches overlap in wall-clock
    (rag.vector)   (rag.fts)         │  time instead of running back to back
      └──────────────┴───────────────┘
                     ▼
              RRF / weighted fusion (rag.fusion)
                     ▼
                  top-K chunks

Threads rather than asyncio: both retrievers spend essentially all of their
time inside C extensions (FAISS and SQLite) that release the GIL, so a thread
pool gives real concurrency here, and the public API stays synchronous - which
keeps it a drop-in replacement for the original ``retrieve()``. An asyncio
entry point is provided for callers that already have an event loop.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ENABLE_HYBRID,
    ENABLE_QUERY_EXPANSION,
    FTS_TOP_K,
    FTS_WEIGHT,
    FUSION_MODE,
    RETRIEVAL_TIMEOUT,
    RRF_K,
    TOP_K,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
)
from rag import store
from rag.embed import get_model
from rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rag.keywords import ExpandedQuery, expand_query
from rag.vector import vector_search

VECTOR = "vector"
FTS = "fts"

# Two long-lived workers: one per retriever. Created once, because paying
# thread-creation cost on every query would eat the latency the parallelism is
# supposed to save.
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever")


@dataclass
class ScoredChunk:
    """A retrieved chunk with the provenance needed to explain the ranking."""

    index: int
    chunk: dict
    score: float
    ranks: dict = field(default_factory=dict)

    @property
    def retrievers(self):
        return sorted(self.ranks)

    @property
    def source(self) -> str:
        return self.chunk.get("source", "")


@dataclass
class HybridResult:
    """Everything one hybrid retrieval produced, including diagnostics."""

    chunks: list = field(default_factory=list)
    ranked: list = field(default_factory=list)
    expansion: ExpandedQuery = None
    timings: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def explain(self) -> str:
        parts = [
            f"expansion={self.expansion.source}" if self.expansion else "",
            f"vector={self.counts.get(VECTOR, 0)}",
            f"fts={self.counts.get(FTS, 0)}",
            f"fused={len(self.ranked)}",
            " ".join(f"{k}={v:.0f}ms" for k, v in self.timings.items()),
        ]
        return "  ".join(p for p in parts if p)


def _run_fts(expansion: ExpandedQuery, top_k: int):
    fts = store.get_fts_index()
    if fts is None:
        return []
    return fts.search(expansion.search_terms(), top_k=top_k)


def _timed(fn, *args, **kwargs):
    """Run ``fn``, returning ``(result, error, elapsed_ms)`` and never raising.

    One retriever failing has to leave the other one's results intact: a
    corrupt FTS database should cost recall, not the answer.
    """
    started = time.perf_counter()
    try:
        return fn(*args, **kwargs), None, (time.perf_counter() - started) * 1000
    except Exception as e:
        return [], f"{type(e).__name__}: {e}", (time.perf_counter() - started) * 1000


def hybrid_retrieve(
    query: str,
    top_k: int = TOP_K,
    use_expansion: bool = None,
    use_fts: bool = None,
    use_vector: bool = True,
    expand_vector: bool = None,
    fusion_mode: str = None,
    parallel: bool = True,
) -> HybridResult:
    """Retrieve the top-K chunks for a query using both retrievers.

    Args:
        query: the user's question, verbatim.
        top_k: how many chunks to return after fusion.
        use_expansion: override ``ENABLE_QUERY_EXPANSION``.
        use_fts: override ``ENABLE_HYBRID`` (False = vector-only baseline).
        use_vector: include dense retrieval (False = lexical-only, used by the
            benchmark to measure each retriever on its own).
        expand_vector: override ``EXPAND_VECTOR_QUERY``, i.e. whether the
            generated keywords are appended to the dense query as well.
        fusion_mode: ``"rrf"`` or ``"weighted"``; overrides ``FUSION_MODE``.
        parallel: run the retrievers concurrently. Setting it False runs them
            sequentially and is used by the benchmark to measure what the
            concurrency actually buys.
    """
    use_expansion = ENABLE_QUERY_EXPANSION if use_expansion is None else use_expansion
    use_fts = ENABLE_HYBRID if use_fts is None else use_fts
    fusion_mode = fusion_mode or FUSION_MODE

    timings, counts, errors = {}, {}, {}
    started = time.perf_counter()

    # Loading is done here, before the workers start, so that both threads do
    # not race into a cold index and duplicate the load.
    if not store.ensure_loaded():
        return HybridResult(expansion=ExpandedQuery(original=query))
    chunks = store.get_chunks()

    # The embedding model is loaded lazily and takes seconds on a cold start.
    # Left inside the worker, that cost lands inside RETRIEVAL_TIMEOUT and
    # trips it on the very first query, so it is paid for out here where the
    # timeout does not apply and it is not attributed to search time.
    if use_vector:
        get_model()

    # Step 1: keyword generation. Sequential by necessity - the retrievers
    # need its output - and the single largest cost in the pipeline, which is
    # why it is cached and why it is optional.
    expansion_started = time.perf_counter()
    expansion = expand_query(query, enabled=use_expansion)
    timings["expansion"] = (time.perf_counter() - expansion_started) * 1000
    if expansion.error:
        errors["expansion"] = expansion.error

    # Step 2: the two retrievers.
    vector_args = (
        vector_search, expansion.vector_query(expand_vector), VECTOR_TOP_K
    )
    fts_args = (_run_fts, expansion, FTS_TOP_K)

    retrieval_started = time.perf_counter()
    vector_hits, fts_hits = [], []
    vector_error, fts_error = None, None

    if use_vector and use_fts and parallel:
        # The only branch that matters in production: both searches in flight
        # at once, so the wall-clock cost is max(vector, fts) rather than the
        # sum, and adding lexical search costs no latency.
        vector_future = _pool.submit(_timed, *vector_args)
        fts_future = _pool.submit(_timed, *fts_args)
        vector_hits, vector_error, timings[VECTOR] = vector_future.result(
            timeout=RETRIEVAL_TIMEOUT
        )
        fts_hits, fts_error, timings[FTS] = fts_future.result(
            timeout=RETRIEVAL_TIMEOUT
        )
    else:
        if use_vector:
            vector_hits, vector_error, timings[VECTOR] = _timed(*vector_args)
        if use_fts:
            fts_hits, fts_error, timings[FTS] = _timed(*fts_args)
    timings["retrieval"] = (time.perf_counter() - retrieval_started) * 1000

    if vector_error:
        errors[VECTOR] = vector_error
    if fts_error:
        errors[FTS] = fts_error
    counts[VECTOR] = len(vector_hits)
    counts[FTS] = len(fts_hits)

    # Step 3: fusion. Chunk indices are the fusion key, which deduplicates
    # across retrievers for free.
    fusion_started = time.perf_counter()
    weights = {VECTOR: VECTOR_WEIGHT, FTS: FTS_WEIGHT}
    if fusion_mode == "weighted":
        fused = weighted_score_fusion(
            {VECTOR: vector_hits, FTS: fts_hits}, weights=weights
        )
    else:
        fused = reciprocal_rank_fusion(
            {
                VECTOR: [idx for idx, _ in vector_hits],
                FTS: [idx for idx, _ in fts_hits],
            },
            k=RRF_K,
            weights=weights,
        )
    timings["fusion"] = (time.perf_counter() - fusion_started) * 1000

    ranked = [
        ScoredChunk(index=idx, chunk=chunks[idx], score=score, ranks=ranks)
        for idx, score, ranks in fused[:top_k]
        if 0 <= idx < len(chunks)
    ]
    timings["total"] = (time.perf_counter() - started) * 1000

    return HybridResult(
        chunks=[sc.chunk for sc in ranked],
        ranked=ranked,
        expansion=expansion,
        timings=timings,
        counts=counts,
        errors=errors,
    )


async def hybrid_retrieve_async(query: str, top_k: int = TOP_K, **kwargs):
    """asyncio entry point.

    The retrievers are blocking C calls, so the idiomatic way to overlap them
    from a coroutine is ``gather`` over ``to_thread`` - the same threads, just
    awaited instead of joined.
    """
    import asyncio

    return await asyncio.to_thread(hybrid_retrieve, query, top_k, **kwargs)


async def gather_retrievers_async(expansion: ExpandedQuery):
    """Explicit ``asyncio.gather`` over the two retrievers.

    Kept separate from ``hybrid_retrieve_async`` because it shows the fan-out
    directly rather than delegating to the thread-pool version.
    """
    import asyncio

    vector_hits, fts_hits = await asyncio.gather(
        asyncio.to_thread(vector_search, expansion.vector_query(), VECTOR_TOP_K),
        asyncio.to_thread(_run_fts, expansion, FTS_TOP_K),
    )
    return vector_hits, fts_hits
