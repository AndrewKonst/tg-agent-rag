"""Rank fusion for combining several retrievers into one ranking.

Kept free of I/O and of any knowledge of chunks, FAISS or SQLite: fusion is
the part of a hybrid retriever that is easiest to get subtly wrong and easiest
to test in isolation, so it lives on its own.
"""


def reciprocal_rank_fusion(ranked_lists, k: int = 60, weights=None):
    """Reciprocal Rank Fusion of several best-first ranked lists.

        RRF(d) = Σ_m  w_m / (k + rank_m(d))

    with 1-based ranks. Only ranks are used, never scores, which is the whole
    point: cosine similarity and BM25 live on incomparable scales, and trying
    to normalise them across queries is where weighted schemes get flaky.
    The constant ``k`` flattens the head of each list, so a document both
    retrievers rank reasonably beats one that a single retriever ranks first.

    Args:
        ranked_lists: mapping of retriever name -> sequence of document keys,
            best first. Keys must be hashable and comparable across lists.
        k: RRF damping constant.
        weights: optional mapping of retriever name -> weight (default 1.0).

    Returns:
        ``[(key, score, {retriever: rank})]`` sorted by score descending, ties
        broken by key for determinism.
    """
    weights = weights or {}
    scores = {}
    ranks = {}

    for name, keys in ranked_lists.items():
        weight = float(weights.get(name, 1.0))
        for position, key in enumerate(keys, start=1):
            key_ranks = ranks.setdefault(key, {})
            if name in key_ranks:
                # A retriever returning the same key twice must not be able to
                # inflate that key's score; the better rank wins.
                continue
            key_ranks[name] = position
            scores[key] = scores.get(key, 0.0) + weight / (k + position)

    return sorted(
        ((key, score, ranks[key]) for key, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _minmax_normalise(scored):
    """Scale scores into [0, 1] per retriever.

    A flat list (every score identical, which happens on a tiny corpus where
    BM25 IDF collapses) maps to 1.0 rather than dividing by zero.
    """
    if not scored:
        return {}
    values = [score for _, score in scored]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {key: 1.0 for key, _ in scored}
    return {key: (score - lo) / span for key, score in scored}


def weighted_score_fusion(scored_lists, weights=None):
    """Weighted sum of min-max normalised scores.

    The alternative to RRF, offered because it keeps score magnitude (a
    document at cosine 0.9 does outrank one at 0.4, information RRF discards).
    Its weakness is that normalisation is per query: a query where everything
    is a poor match still produces a 1.0, so absolute confidence is lost.

    Args:
        scored_lists: mapping of retriever name -> sequence of
            ``(key, score)``, larger score meaning more relevant.
        weights: optional mapping of retriever name -> weight (default 1.0).

    Returns:
        ``[(key, score, {retriever: rank})]`` sorted by score descending.
    """
    weights = weights or {}
    scores = {}
    ranks = {}

    for name, scored in scored_lists.items():
        weight = float(weights.get(name, 1.0))
        normalised = _minmax_normalise(scored)
        for position, (key, _) in enumerate(scored, start=1):
            ranks.setdefault(key, {}).setdefault(name, position)
            scores[key] = scores.get(key, 0.0) + weight * normalised[key]

    return sorted(
        ((key, score, ranks[key]) for key, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
