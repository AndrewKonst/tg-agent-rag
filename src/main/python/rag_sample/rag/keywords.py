"""LLM query expansion: turn a question into search keywords.

A 0.6B model is not reliable enough to be trusted with a format. It will
occasionally answer the question instead of extracting from it, wrap the
output in a ``<think>`` block, emit JSON when asked for a comma list, or
return prose. Every one of those cases is handled here, and anything that
survives none of the parsers falls back to terms taken from the query itself
- so expansion can degrade, but it cannot break retrieval.
"""

import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EXPAND_VECTOR_QUERY,
    KEYWORD_MODEL,
    KEYWORD_NUM_PREDICT,
    KEYWORD_TEMPERATURE,
    KEYWORD_THINK,
    KEYWORD_TIMEOUT,
    MAX_KEYWORDS,
    OLLAMA_URL,
)
from rag.fts import salient_terms

# Deliberately minimal, and deliberately without a worked example. Three
# prompt shapes were measured on the 30 benchmark questions; the terser the
# prompt, the fewer questions the 0.6B model derailed on. Priming the answer
# position with a trailing "Keywords:" is what keeps it from writing prose.
_PROMPT = """Keywords: {query}
Answer with {n} comma separated search keywords only.
Keywords:"""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

# A keyword is a phrase, not a sentence. These give away that the model
# answered the question rather than extracting terms from it.
_PROSE_MARKERS = ("  ", ". ", "? ", "! ", ":", "\t")

# Qwen3's thinking-control token, which Ollama injects into the prompt when
# reasoning is disabled. A 0.6B model sometimes echoes it back as if it were
# an answer ("/no_think", "/no_check", "/no_time_back").
_CONTROL_ECHO = re.compile(r"^no[_\s-]?(?:think|check)", re.IGNORECASE)

_SOURCE_LLM = "llm"
_SOURCE_HEURISTIC = "fallback:heuristic"
_SOURCE_ORIGINAL = "fallback:original"


@dataclass
class ExpandedQuery:
    """The original question plus whatever expansion managed to produce."""

    original: str
    keywords: list = field(default_factory=list)
    source: str = _SOURCE_ORIGINAL
    latency_ms: float = 0.0
    error: str = ""

    @property
    def used_llm(self) -> bool:
        return self.source == _SOURCE_LLM

    def search_terms(self):
        """Terms for the lexical retriever: keywords plus the query's own."""
        return list(self.keywords) + salient_terms(self.original)

    def vector_query(self, expand: bool = None) -> str:
        """Text for the dense retriever.

        Defaults to the question as written: the embedding model is good at
        exactly the part of the query that keywords throw away. Set
        EXPAND_VECTOR_QUERY to append keywords and measure it - the benchmark
        reports both.
        """
        expand = EXPAND_VECTOR_QUERY if expand is None else expand
        if not expand or not self.keywords:
            return self.original
        return f"{self.original} {' '.join(self.keywords)}"


def _strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by Qwen3-family thinking models."""
    text = _THINK_BLOCK.sub(" ", text)
    return _UNCLOSED_THINK.sub(" ", text)


def _clean_term(term: str) -> str:
    term = _LIST_PREFIX.sub("", term.strip())
    term = term.strip().strip('"\'`').strip()
    # Trailing punctuation is noise; internal hyphens and dots are not
    # (E-5031, OPS-RB-003, config.py).
    return term.strip(" \t\n.,;:!?")


def _looks_like_keyword(term: str) -> bool:
    if not term or len(term) > 60:
        return False
    if _CONTROL_ECHO.match(term):
        return False
    if len(term.split()) > 5:
        return False
    if any(marker in term for marker in _PROSE_MARKERS):
        return False
    return bool(re.search(r"[0-9A-Za-zÀ-￿]", term))


def _from_json(text: str):
    """Parse the JSON shapes a small model reaches for unprompted."""
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return []
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return []

    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []

    if isinstance(data, dict):
        # {"keywords": [...]} / {"queries": [...]} / {"a": "x", "b": "y"}
        for key in ("keywords", "queries", "terms", "keyword", "query"):
            if key in data:
                data = data[key]
                break
        else:
            data = list(data.values())

    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return []

    return [str(item) for item in data if isinstance(item, (str, int, float))]


def _from_delimited(text: str):
    """Split on any delimiter the model plausibly reaches for.

    Asked for commas it also produces newlines, semicolons, slashes and pipes
    ("/number of days /price per night /hotel in London" is verbatim output).
    """
    return [part for part in re.split(r"[,;/|\n]+", text) if part.strip()]


def _tokens(text: str):
    return {
        t.lower()
        for t in re.findall(r"[0-9A-Za-zÀ-￿][\w\-./]*", text or "")
        if len(t) > 2 or any(c.isdigit() for c in t)
    }


def is_grounded(keywords, query: str) -> bool:
    """Sanity-check that the keywords actually came from *this* question.

    Guards against an observed failure of the 0.6B model: when the prompt
    carries a worked example, the model parrots that example's keywords
    verbatim and ignores the question. The output parses cleanly, so only a
    grounding check catches it - and injecting confidently wrong keywords into
    the search is worse than having none. Requiring a single shared token is
    loose enough to leave real synonym expansion alone.

    It does have a cost: a keyword set made purely of synonyms ("time off"
    for a question about vacation) would be rejected too. Measured on the 30
    benchmark questions that never happens - the model always echoes at least
    one token of the question - while the check does reject three real garbage
    outputs, so it stays.
    """
    if not keywords:
        return False
    query_tokens = _tokens(query)
    if not query_tokens:
        return True
    return any(_tokens(kw) & query_tokens for kw in keywords)


def parse_keywords(raw: str, max_keywords: int = MAX_KEYWORDS):
    """Extract a keyword list from a raw model response.

    Pure function with no I/O, which is what makes the fragile part of this
    feature straightforward to unit-test against real malformed outputs.
    """
    text = _strip_think(raw or "")
    text = _FENCE.sub("", text).strip()
    if not text:
        return []

    candidates = _from_json(text) or _from_delimited(text)

    keywords = []
    seen = set()
    for candidate in candidates:
        term = _clean_term(candidate)
        if not _looks_like_keyword(term):
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(term)
        if len(keywords) >= max_keywords:
            break

    return keywords


def _ask_llm_for_keywords(query: str, timeout: float) -> str:
    payload = {
        "model": KEYWORD_MODEL,
        "prompt": _PROMPT.format(n="3-5", query=query),
        "stream": False,
        "options": {
            "temperature": KEYWORD_TEMPERATURE,
            "num_predict": KEYWORD_NUM_PREDICT,
        },
    }
    if not KEYWORD_THINK:
        # Only meaningful for thinking models; Ollama ignores it otherwise.
        payload["think"] = False

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "")


_cache = {}
_cache_lock = threading.Lock()


def expand_query(
    query: str,
    enabled: bool = True,
    timeout: float = KEYWORD_TIMEOUT,
    use_cache: bool = True,
) -> ExpandedQuery:
    """Expand a question into keywords, never raising.

    Degradation ladder: model output -> parsed keywords; unparseable output or
    an unreachable model -> keywords derived from the query itself; nothing
    usable there either -> the original query alone.
    """
    query = (query or "").strip()
    if not query:
        return ExpandedQuery(original=query, source=_SOURCE_ORIGINAL)

    if not enabled:
        return ExpandedQuery(original=query, source=_SOURCE_ORIGINAL)

    if use_cache:
        with _cache_lock:
            cached = _cache.get(query)
        if cached is not None:
            return cached

    started = time.perf_counter()
    keywords, source, error = [], _SOURCE_ORIGINAL, ""

    try:
        raw = _ask_llm_for_keywords(query, timeout)
        parsed = parse_keywords(raw)
        if parsed and is_grounded(parsed, query):
            keywords, source = parsed, _SOURCE_LLM
        elif parsed:
            error = f"ungrounded model output: {parsed}"
        else:
            error = f"unparseable model output: {raw.strip()[:120]!r}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    if not keywords:
        # Fallback: the query's own content words. Worse than good keywords,
        # strictly better than nothing, and it keeps the lexical retriever fed.
        keywords = salient_terms(query)[:MAX_KEYWORDS]
        if keywords:
            source = _SOURCE_HEURISTIC

    result = ExpandedQuery(
        original=query,
        keywords=keywords,
        source=source,
        latency_ms=(time.perf_counter() - started) * 1000,
        error=error,
    )

    if use_cache:
        with _cache_lock:
            _cache[query] = result
    return result


def clear_cache():
    with _cache_lock:
        _cache.clear()


if __name__ == "__main__":
    for q in [
        "How do I request vacation days?",
        "What does error E-5031 mean during deployment?",
        "Which form do I use to claim a per diem?",
    ]:
        exp = expand_query(q)
        print(f"\n❓ {q}")
        print(f"   keywords: {exp.keywords}")
        print(f"   source:   {exp.source}  ({exp.latency_ms:.0f} ms)")
        if exp.error:
            print(f"   note:     {exp.error}")
