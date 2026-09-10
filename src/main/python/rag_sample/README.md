# Donor RAG implementation

This directory contains the copied RAG sample from:

```text
/Users/konst/Developer/rag-sample/src
```

It is kept as a donor implementation for the Telegram RAG homework.

## What can be reused

- `rag/ingest.py`:
  - `.txt` and `.md` text extraction;
  - `.pdf` extraction through `pypdf`;
  - `.docx` extraction through `python-docx`.

- `rag/chunk.py`:
  - token-based chunking through `tiktoken`;
  - `CHUNK_SIZE = 220`;
  - `CHUNK_OVERLAP = 40`.

- `rag/embed.py`:
  - local embeddings through `sentence-transformers`;
  - model: `all-MiniLM-L6-v2`.

- `rag/fts.py`, `rag/hybrid.py`, `rag/fusion.py`, `rag/keywords.py`:
  - optional bonus ideas for hybrid search and result fusion.

- `bench/`:
  - useful starting point for the required RAG evaluation dataset.

## What must be changed for the homework

The copied sample currently uses:

```text
FAISS + pickle + shared docs directory
```

The homework requires:

```text
SQLite + sqlite-vec + per-user document isolation
```

So the storage/search part must be adapted or rewritten.

Do not keep the sample's global index as the final implementation, because it does
not isolate Telegram users:

```text
User A document
User B document
```

must never be searched together unless the `user_id` matches the current Telegram
chat/user.

## Target integration shape

The Kotlin application should depend on a small RAG interface:

```text
index document
search documents
list documents
delete document
```

Telegram file upload and commands should call that service directly.

The AI agent should access search through a tool:

```text
search_documents(query)
```

That tool must always pass the current Telegram `chatId`/`userId`, so retrieval is
limited to the current user's documents.
