import json
import ollama
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from rag.query import retrieve_hybrid, build_prompt, ask_llm
from mcp.client import MCPClient
from config import OLLAMA_MODEL

class CompanyKBAssistant:
    """Company Knowledge Base Assistant combining RAG and MCP."""
    
    def __init__(self):
        self.llm_client = ollama.Client()
        self.mcp = None
        self._init_mcp()
        
    def _init_mcp(self):
        """Initialize MCP client."""
        try:
            import sys
            from pathlib import Path
            python_cmd = sys.executable
            # Get absolute path to MCP server
            mcp_path = Path(__file__).parent / "mcp" / "server.py"
            self.mcp = MCPClient([python_cmd, str(mcp_path)])
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}")
            self.mcp = None
    
    def _llm_decide_mcp_usage(self, query: str, contexts):
        """Ask LLM if MCP tools are needed based on query and retrieved contexts."""
        if not self.mcp:
            return None, None
        
        # Build context summary for LLM decision
        context_summary = ""
        if contexts:
            context_summary = f"Retrieved {len(contexts)} relevant chunks from knowledge base:\n"
            for i, ctx in enumerate(contexts[:3], 1):  # Show first 3 chunks
                context_summary += f"{i}. From {ctx['source']}: {ctx['text'][:200]}...\n"
        else:
            context_summary = "No relevant chunks found in knowledge base.\n"
        
        decision_prompt = f"""You are helping answer a question using a knowledge base system with RAG (retrieval) and MCP tools.

User question: {query}

{context_summary}

Available MCP tools:
1. read_document(file_path: str) - Read a specific document file (use when you need full document content)
2. list_documents() - List all available documents (use when user asks "what documents exist" or "list all docs")
3. search_documents(query: str) - Search for documents by name (use when user asks to find a specific document)

Decision rules:
- If the retrieved chunks fully answer the question, set use_mcp to false
- If chunks are empty or insufficient, consider using MCP tools
- If user explicitly asks to read/list/search documents, use the appropriate tool
- If you need the full content of a specific document mentioned in chunks, use read_document
- When in doubt, prefer not using MCP (chunks are usually sufficient)

Respond ONLY with valid JSON, no other text:
{{"use_mcp": true/false, "tool": "tool_name_or_null", "args": {{"arg_name": "value"}}}}

Examples:
{{"use_mcp": false, "tool": null, "args": {{}}}}
{{"use_mcp": true, "tool": "read_document", "args": {{"file_path": "docs/vacation-policy.md"}}}}
{{"use_mcp": true, "tool": "list_documents", "args": {{}}}}
{{"use_mcp": true, "tool": "search_documents", "args": {{"query": "vacation"}}}}

Your JSON response:"""

        try:
            response = self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that decides when to use tools. Always respond with valid JSON only."},
                    {"role": "user", "content": decision_prompt}
                ]
            )
            
            response_text = response["message"]["content"].strip()
            
            # Clean up JSON if wrapped in markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            decision = json.loads(response_text)
            
            if decision.get("use_mcp", False):
                tool_name = decision.get("tool")
                tool_args = decision.get("args", {})
                return tool_name, tool_args
            
            return None, None
            
        except Exception as e:
            # If LLM decision fails, don't use MCP
            return None, None
    
    def _call_mcp_tool(self, tool_name: str, tool_args: dict):
        """Call an MCP tool with given name and arguments."""
        if not self.mcp:
            return None
        
        try:
            result = self.mcp.call_tool(tool_name, tool_args)
            return result.get("result", "")
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"
    
    def query(self, user_query: str, verbose=False):
        """Answer a question using hybrid RAG and optionally MCP tools."""
        # Step 1: Hybrid retrieval - LLM keyword generation, then vector and
        # full-text search in parallel, fused with RRF.
        retrieval = retrieve_hybrid(user_query)
        contexts = retrieval.chunks

        if verbose:
            self._print_retrieval(retrieval)
        
        # Step 2: Ask LLM if MCP tools are needed
        mcp_result = None
        mcp_tool_used = None
        tool_name, tool_args = self._llm_decide_mcp_usage(user_query, contexts)
        
        if tool_name:
            if verbose:
                print(f"🔧 LLM decided to use MCP tool: {tool_name} with args: {tool_args}")
            mcp_result = self._call_mcp_tool(tool_name, tool_args)
            mcp_tool_used = tool_name
            if verbose and mcp_result:
                print(f"✅ MCP tool returned result (length: {len(mcp_result)} chars)")
        
        # Step 3: Build prompt with RAG context
        prompt = build_prompt(user_query, contexts)
        
        # Step 4: Add MCP result if available
        if mcp_result:
            prompt += f"\n\n<additional_info_from_mcp_tool>\n{mcp_result}\n</additional_info_from_mcp_tool>\n"
        
        # Step 5: Generate answer
        answer = ask_llm(prompt)
        
        # Step 6: Prepare response with sources
        sources = [c["source"] for c in contexts] if contexts else []
        
        return {
            "answer": answer,
            "sources": sources,
            "mcp_used": mcp_result is not None,
            "mcp_tool": mcp_tool_used,
            "retrieval": retrieval,
        }

    @staticmethod
    def _print_retrieval(retrieval):
        """Show what each retriever contributed and what it cost.

        Worth printing because the interesting failure of a hybrid retriever
        is silent: one side returning nothing still yields a plausible answer,
        just a worse one.
        """
        expansion = retrieval.expansion
        keywords = ", ".join(expansion.keywords) if expansion.keywords else "-"
        print(f"🔑 Keywords [{expansion.source}]: {keywords}")
        if expansion.error:
            print(f"   ↳ expansion fell back: {expansion.error}")

        counts = retrieval.counts
        timings = retrieval.timings
        print(
            f"📚 Retrieved {len(retrieval.chunks)} chunks "
            f"(vector {counts.get('vector', 0)}, fts {counts.get('fts', 0)}, "
            f"fused in {timings.get('retrieval', 0):.1f} ms, "
            f"total {timings.get('total', 0):.0f} ms)"
        )
        for scored in retrieval.ranked:
            found_by = "+".join(scored.retrievers) or "-"
            print(
                f"   {scored.score:.5f}  [{found_by}]  "
                f"{scored.source}#{scored.chunk.get('chunk_id', '?')}"
            )
        for stage, error in retrieval.errors.items():
            if stage != "expansion":
                print(f"   ⚠️  {stage} retriever failed: {error}")
    
    def close(self):
        """Clean up resources."""
        if self.mcp:
            self.mcp.close()


if __name__ == "__main__":
    assistant = CompanyKBAssistant()
    
    print("🤖 Company Knowledge Base Assistant")
    print("Type 'exit' or 'quit' to stop\n")
    
    try:
        while True:
            query = input("❓ Question: ")
            if query.lower() in {"exit", "quit"}:
                break
            
            print("\n" + "─" * 60)
            result = assistant.query(query, verbose=True)
            
            print("\n🤖 Answer:\n")

            console = Console(force_terminal=True)
            console.print(Markdown(result["answer"]))
            
            if result["sources"]:
                print("\n📚 Sources:")
                seen_sources = set()
                for src in result["sources"]:
                    if src not in seen_sources:
                        print(f"  • {src}")
                        seen_sources.add(src)
            
            if result["mcp_used"]:
                print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")
            
            print("─" * 60 + "\n")
    
    finally:
        assistant.close()
