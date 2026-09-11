"""SQLite FTS5 full-text index over the same chunks the vector index holds.

Why SQLite FTS5 rather than an in-process BM25 library: the index persists on
disk next to index.faiss (no rebuild on every start), it needs no dependency
beyond the standard library, and - the reason that actually matters here - the
query runs inside SQLite's C code with the GIL released, so running it in a
worker thread alongside the FAISS search gives real parallelism instead of two
Python functions taking turns.
"""

import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FTS_TOP_K

# bm25() weights per column. The source path is a genuinely strong signal
# ("vacation policy" -> docs/vacation-policy.md), but a filename must not be
# able to outrank a chunk whose body actually answers the question.
_COL_WEIGHT_TEXT = 1.0
_COL_WEIGHT_SOURCE = 0.4

# A term is only useful to FTS5 if it contains something the tokenizer will
# keep. Anything else ("---", "?!") would produce an empty phrase and a syntax
# error, so it gets dropped before the query is assembled.
_HAS_TOKEN = re.compile(r"[0-9A-Za-zÀ-￿]")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "so", "that", "the", "their", "there",
    "these", "this", "to", "was", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}


def _escape_phrase(term: str) -> str:
    """Wrap a term as an FTS5 phrase, neutralising the query syntax.

    Everything a user types - hyphens, quotes, colons, ``*``, ``NEAR``, ``OR``
    - is syntax to FTS5's MATCH parser and raises OperationalError if passed
    through raw. Wrapping the term in double quotes turns it into a literal
    phrase; internal quotes are escaped by doubling them.

    A hyphenated identifier keeps working with the default unicode61
    tokenizer: ``"E-5031"`` becomes the phrase ``e 5031``, which matches the
    document exactly, while the bare token ``5031`` still matches on its own.
    """
    return '"' + term.replace('"', '""') + '"'


def build_match_query(terms) -> str:
    """Build an FTS5 MATCH expression as an OR of literal phrases."""
    phrases = []
    seen = set()
    for term in terms:
        term = (term or "").strip()
        if not term or not _HAS_TOKEN.search(term):
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(_escape_phrase(term))
    return " OR ".join(phrases)


def salient_terms(query: str):
    """Content-bearing tokens of a raw query, for use as FTS terms.

    Tokens that look like identifiers (digits, hyphens, ALL-CAPS) are kept
    whatever their length: ``E-5031`` and ``HRP-204`` are exactly the queries
    where lexical search beats embeddings, so they must survive filtering.
    """
    out = []
    for token in re.findall(r"[0-9A-Za-zÀ-￿][\w\-./]*", query):
        looks_like_id = any(c.isdigit() for c in token) or "-" in token or (
            token.isupper() and len(token) > 1
        )
        if looks_like_id:
            out.append(token)
        elif len(token) > 2 and token.lower() not in _STOPWORDS:
            out.append(token)
    return out


class FtsIndex:
    """Read-only handle on the FTS5 chunk index."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        # check_same_thread=False: the connection is read-only and shared with
        # the retriever worker thread. Reads are serialised by SQLite itself.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT count(*) FROM chunks_fts").fetchone()
        return row[0] if row else 0

    def search(self, terms, top_k: int = FTS_TOP_K):
        """Return ``[(chunk_index, score)]`` best-first for the given terms.

        ``score`` is ``-bm25()``, so larger is better (SQLite returns bm25 as
        a negative number where more negative means more relevant).
        """
        match = build_match_query(terms)
        if not match:
            return []

        sql = (
            "SELECT rowid, -bm25(chunks_fts, ?, ?) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY score DESC LIMIT ?"
        )
        try:
            rows = self._conn.execute(
                sql, (_COL_WEIGHT_TEXT, _COL_WEIGHT_SOURCE, match, top_k)
            ).fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH expression must degrade to "no lexical hits",
            # never take the whole retrieval down: the vector side still has
            # an answer for the user.
            return []

        # rowid is 1-based, chunk indices are 0-based.
        return [(int(rowid) - 1, float(score)) for rowid, score in rows]

    def close(self):
        self._conn.close()


def build_fts_index(chunks, db_path):
    """(Re)build the FTS5 index from scratch for the given chunks."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "  text, source,"
            "  tokenize = 'unicode61 remove_diacritics 2'"
            ")"
        )
        conn.executemany(
            "INSERT INTO chunks_fts(rowid, text, source) VALUES (?, ?, ?)",
            [
                (i + 1, c["text"], c.get("source", ""))
                for i, c in enumerate(chunks)
            ],
        )
        # Merge the incremental b-trees into one, so queries do not pay for
        # the write pattern of the build.
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        conn.commit()
    finally:
        conn.close()

    return len(chunks)


if __name__ == "__main__":
    demo = [
        {"text": "Rollback with lrctl rollback --env prod. Error E-5031 means "
                 "the image tag was not found.", "source": "docs/runbook.md"},
        {"text": "Vacation is requested with form HRP-204 in the People "
                 "Portal.", "source": "docs/vacation-policy.md"},
    ]
    path = Path("/tmp/_fts_demo.db")
    build_fts_index(demo, path)
    idx = FtsIndex(path)
    for q in ["E-5031", "HRP-204", "how do I take a holiday?", '"broken( query']:
        print(f"{q!r:32} -> {idx.search(salient_terms(q), 2)}")
    idx.close()
    path.unlink()
