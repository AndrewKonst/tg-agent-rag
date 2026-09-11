import sys
from pathlib import Path

import requests

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_URL, TOP_K
from rag import store
from rag.hybrid import HybridResult, hybrid_retrieve


def retrieve(query: str):
    """Retrieve relevant chunks for a query.

    Signature and return type are unchanged from the vector-only version, so
    every existing caller keeps working; the ranking underneath is now the
    fused result of dense and lexical retrieval.
    """
    return hybrid_retrieve(query, top_k=TOP_K).chunks


def retrieve_hybrid(query: str, top_k: int = TOP_K, **kwargs) -> HybridResult:
    """Retrieve with the full diagnostic detail: keywords, ranks, timings."""
    return hybrid_retrieve(query, top_k=top_k, **kwargs)


def ensure_index() -> bool:
    """Load the indexes, building them from documents if they are missing."""
    return store.ensure_loaded()


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
