"""Retrieval benchmark: vector-only baseline vs each hybrid configuration.

    python -m bench.run_bench                 # full run, printed as tables
    python -m bench.run_bench --out BENCH.md  # also write a markdown report
    python -m bench.run_bench --repeats 5     # latency repeats

Metrics, all at K = TOP_K and all at document level:

  Hit@K     share of questions with at least one chunk from a gold document
  Recall@K  share of the gold documents that were surfaced, averaged
  MRR@K     mean reciprocal rank of the first gold chunk

Latency is reported for the retrieval stage alone and for the whole pipeline,
because query expansion costs an LLM call and would otherwise hide what the
retrievers themselves do.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.questions import ALL, CATEGORIES
from config import TOP_K
from rag import store
from rag.hybrid import FTS, VECTOR, hybrid_retrieve
from rag.keywords import clear_cache

# Each mode is a set of overrides for hybrid_retrieve. "vector-only" with
# expansion off reproduces the original system exactly and is the baseline
# every other row is measured against.
MODES = {
    "vector-only (baseline)": dict(use_vector=True, use_fts=False, use_expansion=False),
    "fts-only": dict(use_vector=False, use_fts=True, use_expansion=False),
    "hybrid RRF": dict(use_vector=True, use_fts=True, use_expansion=False),
    "hybrid RRF + expansion": dict(use_vector=True, use_fts=True, use_expansion=True),
    "hybrid RRF + exp (vec too)": dict(
        use_vector=True, use_fts=True, use_expansion=True, expand_vector=True
    ),
    "hybrid weighted + expansion": dict(
        use_vector=True, use_fts=True, use_expansion=True, fusion_mode="weighted"
    ),
}


def score_one(result, gold, k=TOP_K):
    """Hit, recall and reciprocal rank for a single retrieval."""
    sources = [c.get("source", "") for c in result.chunks[:k]]
    gold_set = set(gold)

    hit = 1.0 if gold_set & set(sources) else 0.0
    recall = len(gold_set & set(sources)) / len(gold_set) if gold_set else 0.0

    rr = 0.0
    for position, source in enumerate(sources, start=1):
        if source in gold_set:
            rr = 1.0 / position
            break

    return hit, recall, rr


def fts_only_contribution(result):
    """How many of the returned chunks the vector retriever never found.

    This is the number that justifies the second retriever: if it is zero,
    lexical search is dead weight.
    """
    return sum(
        1 for sc in result.ranked
        if FTS in sc.ranks and VECTOR not in sc.ranks
    )


def run_mode(name, overrides, questions, k=TOP_K):
    # Keywords are cached process-wide, which would hand every mode after the
    # first a free ride on the previous mode's LLM calls and make the latency
    # column meaningless. Each mode pays for its own expansion.
    clear_cache()
    rows = []
    for item in questions:
        started = time.perf_counter()
        result = hybrid_retrieve(item["query"], top_k=k, **overrides)
        elapsed = (time.perf_counter() - started) * 1000

        hit, recall, rr = score_one(result, item["gold"], k)
        rows.append({
            "query": item["query"],
            "category": item["category"],
            "hit": hit,
            "recall": recall,
            "rr": rr,
            "total_ms": elapsed,
            "retrieval_ms": result.timings.get("retrieval", 0.0),
            "expansion_ms": result.timings.get("expansion", 0.0),
            "expansion_source": result.expansion.source if result.expansion else "-",
            "fts_only": fts_only_contribution(result),
            "vector_ms": result.timings.get(VECTOR, 0.0),
            "fts_ms": result.timings.get(FTS, 0.0),
        })
    return {"name": name, "rows": rows}


def aggregate(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "hit": sum(r["hit"] for r in rows) / len(rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
        "mrr": sum(r["rr"] for r in rows) / len(rows),
        "retrieval_p50": statistics.median(r["retrieval_ms"] for r in rows),
        "total_p50": statistics.median(r["total_ms"] for r in rows),
        "fts_only": sum(r["fts_only"] for r in rows) / len(rows),
    }


def _table(header, widths, rows):
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(out)


def format_quality(results, k):
    header = ["mode", f"Hit@{k}", f"Recall@{k}", f"MRR@{k}", "FTS-only hits"]
    widths = [29, 8, 10, 8, 13]
    rows = []
    for res in results:
        agg = aggregate(res["rows"])
        # "chunks only the lexical retriever found" is only meaningful when
        # both retrievers ran.
        both = MODES.get(res["name"], {})
        contribution = (
            f"{agg['fts_only']:.2f}"
            if both.get("use_vector", True) and both.get("use_fts", True)
            else "-"
        )
        rows.append([
            res["name"],
            f"{agg['hit']:.3f}",
            f"{agg['recall']:.3f}",
            f"{agg['mrr']:.3f}",
            contribution,
        ])
    return _table(header, widths, rows)


def format_by_category(results, k):
    names = list(CATEGORIES)
    header = ["mode"] + [f"{c} Recall@{k}" for c in names]
    widths = [29] + [18] * len(names)
    rows = []
    for res in results:
        row = [res["name"]]
        for category in names:
            subset = [r for r in res["rows"] if r["category"] == category]
            agg = aggregate(subset)
            row.append(f"{agg['recall']:.3f}")
        rows.append(row)
    return _table(header, widths, rows)


def format_latency(results):
    """Latency broken down per stage.

    ``retrieval`` is wall-clock for the two retrievers together. Comparing it
    with the ``vector`` and ``fts`` columns is what shows the parallelism:
    under overlap it tracks max(vector, fts), not their sum.
    """
    header = ["mode", "vector", "fts", "retrieval", "expansion", "pipeline"]
    widths = [29, 10, 9, 10, 10, 10]
    rows = []
    for res in results:
        def med(field):
            return statistics.median(r[field] for r in res["rows"])

        rows.append([
            res["name"],
            f"{med('vector_ms'):.2f} ms",
            f"{med('fts_ms'):.2f} ms",
            f"{med('retrieval_ms'):.2f} ms",
            f"{med('expansion_ms'):.0f} ms",
            f"{med('total_ms'):.0f} ms",
        ])
    return _table(header, widths, rows)


def measure_parallelism(questions, repeats):
    """Compare parallel and sequential retrieval on identical work.

    Expansion is warmed into the cache first, so this measures the retrievers
    and nothing else.
    """
    for item in questions:
        hybrid_retrieve(item["query"], use_expansion=True)

    samples = {True: [], False: []}
    components = {True: [], False: []}
    for _ in range(repeats):
        for parallel in (True, False):
            for item in questions:
                result = hybrid_retrieve(
                    item["query"], use_expansion=True, parallel=parallel
                )
                samples[parallel].append(result.timings.get("retrieval", 0.0))
                components[parallel].append((
                    result.timings.get(VECTOR, 0.0),
                    result.timings.get(FTS, 0.0),
                ))

    vector_only = []
    for _ in range(repeats):
        for item in questions:
            result = hybrid_retrieve(
                item["query"], use_expansion=True, use_fts=False
            )
            vector_only.append(result.timings.get("retrieval", 0.0))

    def stats(values):
        values = sorted(values)
        p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
        return [
            f"{statistics.median(values):.2f} ms",
            f"{statistics.fmean(values):.2f} ms",
            f"{p95:.2f} ms",
        ]

    header = ["retrieval mode", "p50", "mean", "p95", "vs sum of parts"]
    widths = [26, 10, 10, 10, 16]

    def overlap(parallel):
        """Wall-clock as a fraction of the two retrievers added together.

        1.00 means no overlap at all, 0.50 means the two searches cost what
        the slower one alone costs.
        """
        pairs = components[parallel]
        total = statistics.fmean(samples[parallel])
        summed = statistics.fmean(v + f for v, f in pairs)
        return f"{total / summed:.2f}x" if summed else "-"

    rows = [
        ["vector only (no FTS)"] + stats(vector_only) + ["-"],
        ["vector + FTS sequential"] + stats(samples[False]) + [overlap(False)],
        ["vector + FTS parallel"] + stats(samples[True]) + [overlap(True)],
    ]
    return _table(header, widths, rows)


def measure_overlap_controlled(delay_ms=60, repeats=5):
    """Controlled proof that the thread pool actually overlaps the searches.

    On this corpus the lexical search costs a fraction of a millisecond, so
    the real measurement above cannot show overlap - there is nothing to
    overlap with. Here each retriever is padded with a known delay standing in
    for a larger index. Sequential must cost ~2x the delay and parallel ~1x;
    anything else means the fan-out is not doing what it claims.
    """
    from rag import hybrid

    delay = delay_ms / 1000.0
    real_vector, real_fts = hybrid.vector_search, hybrid._run_fts

    def slow_vector(*args, **kwargs):
        time.sleep(delay)
        return real_vector(*args, **kwargs)

    def slow_fts(*args, **kwargs):
        time.sleep(delay)
        return real_fts(*args, **kwargs)

    query = "what does error E-5031 mean?"
    hybrid_retrieve(query, use_expansion=True)  # prime the keyword cache

    hybrid.vector_search, hybrid._run_fts = slow_vector, slow_fts
    try:
        measured = {}
        for parallel in (False, True):
            values = [
                hybrid_retrieve(
                    query, use_expansion=True, parallel=parallel
                ).timings["retrieval"]
                for _ in range(repeats)
            ]
            measured[parallel] = statistics.median(values)
    finally:
        hybrid.vector_search, hybrid._run_fts = real_vector, real_fts

    header = ["retrieval mode", f"with {delay_ms} ms per retriever", "expected"]
    widths = [26, 30, 22]
    rows = [
        ["sequential", f"{measured[False]:.1f} ms", f"~{2 * delay_ms} ms (sum)"],
        ["parallel", f"{measured[True]:.1f} ms", f"~{delay_ms} ms (max)"],
    ]
    return _table(header, widths, rows)


def expansion_report(results):
    """How often the LLM produced usable keywords versus falling back."""
    for res in results:
        if "expansion" not in res["name"]:
            continue
        counts = {}
        for row in res["rows"]:
            counts[row["expansion_source"]] = counts.get(row["expansion_source"], 0) + 1
        total = sum(counts.values())
        lines = [f"Query expansion outcomes over {total} questions:"]
        for source, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {source:22} {count:3}  ({count / total:.0%})")
        return "\n".join(lines)
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write a markdown report to this path")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeats for the latency comparison")
    parser.add_argument("--k", type=int, default=TOP_K, help="top-K to score")
    args = parser.parse_args()

    if not store.ensure_loaded():
        print("❌ No index. Run: python main.py build-index")
        return 1

    # Warm up: the first query pays for loading the embedding model, which
    # would otherwise be charged to whichever mode happens to run first.
    print("🔥 Warming up...")
    hybrid_retrieve("warm up the embedding model", use_expansion=False)
    clear_cache()

    results = []
    for name, overrides in MODES.items():
        print(f"▶ {name}")
        results.append(run_mode(name, overrides, ALL, k=args.k))

    print(f"\n📊 Retrieval quality ({len(ALL)} questions, K={args.k})\n")
    quality = format_quality(results, args.k)
    print(quality)

    print("\n📊 Recall by question type\n")
    by_category = format_by_category(results, args.k)
    print(by_category)

    print("\n⏱  Latency\n")
    latency = format_latency(results)
    print(latency)

    print(f"\n⏱  Parallel vs sequential retrieval ({args.repeats} repeats,"
          f" expansion cached)\n")
    parallelism = measure_parallelism(ALL, args.repeats)
    print(parallelism)

    print("\n⏱  Controlled overlap check (synthetic delay per retriever)\n")
    controlled = measure_overlap_controlled()
    print(controlled)

    print()
    expansion = expansion_report(results)
    print(expansion)

    if args.out:
        report = "\n\n".join([
            "# Retrieval benchmark",
            f"Corpus: {len(store.get_chunks())} chunks, "
            f"{len(ALL)} questions, K={args.k}.",
            f"## Retrieval quality\n\n```\n{quality}\n```",
            f"## Recall by question type\n\n```\n{by_category}\n```",
            f"## Latency\n\n```\n{latency}\n```",
            f"## Parallel vs sequential retrieval\n\n```\n{parallelism}\n```",
            f"## Controlled overlap check\n\n```\n{controlled}\n```",
            f"## Query expansion\n\n```\n{expansion}\n```",
        ])
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\n💾 Report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
