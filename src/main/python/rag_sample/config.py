# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration.
# Reduced from 700/100: at 700 tokens most policy documents collapsed into a
# single chunk, so retrieval could not distinguish sections of a document, and
# TOP_K=5 handed a 0.6B model ~3500 tokens of context. Smaller chunks give
# both retrievers something to actually rank.
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# SQLite FTS5 full-text index (relative to src directory)
FTS_DB_PATH = "fts.db"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:0.6b"

# RAG retrieval configuration
TOP_K = 5

# ─────────────────────────── Hybrid retrieval ───────────────────────────

# Master switches. With both off the pipeline behaves exactly like the
# original vector-only RAG, which is what the benchmark uses as a baseline.
ENABLE_HYBRID = True
ENABLE_QUERY_EXPANSION = True

# Candidates pulled from each retriever *before* fusion. Deliberately larger
# than TOP_K: fusion can only rerank what it was given, so a wider candidate
# pool is what actually buys recall.
VECTOR_TOP_K = 15
FTS_TOP_K = 15

# Fusion strategy: "rrf" (Reciprocal Rank Fusion) or "weighted" (min-max
# normalised weighted sum of the raw scores).
FUSION_MODE = "rrf"

# RRF constant. 60 is the value from the original Cormack et al. paper and
# damps the influence of the top ranks just enough to let a document that
# both retrievers like beat one that a single retriever loves.
RRF_K = 60

# Whether generated keywords are appended to the dense retriever's query too.
# Off by default: embeddings already handle the semantics of the original
# wording, and appending bare keywords shifts the query vector without adding
# meaning. Keywords are what the lexical retriever needs, not the dense one.
EXPAND_VECTOR_QUERY = False

# Per-retriever weights, used by both fusion modes.
VECTOR_WEIGHT = 1.0
FTS_WEIGHT = 1.0

# Hard ceiling on how long we wait for the two retrievers, in seconds.
RETRIEVAL_TIMEOUT = 20.0

# ───────────────────────── Query expansion (LLM) ─────────────────────────

# Keyword generation runs on the same small local model. temperature=0 keeps
# a 0.6B model from inventing terms that are not in the corpus.
KEYWORD_MODEL = OLLAMA_MODEL
KEYWORD_TEMPERATURE = 0.0
# Qwen3 is a thinking model and Ollama reports the <think> block in a separate
# field, so with reasoning enabled a truncated generation returns an empty
# response - and on 30 benchmark questions the 0.6B model ran away past any
# reasonable budget on 14 of them. Measured on the same questions:
#   reasoning on,  zero-shot prompt: 47% usable keywords, 1078 ms average
#   reasoning off, zero-shot prompt: 80% usable keywords,   96 ms average
# So reasoning is off. (Reasoning off with a *few-shot* prompt is the one
# combination to avoid: the model parrots the example's keywords verbatim and
# ignores the question, which is why the prompt below carries no example and
# why is_grounded() exists.)
KEYWORD_THINK = False
KEYWORD_NUM_PREDICT = 64
KEYWORD_TIMEOUT = 10.0
MAX_KEYWORDS = 6
