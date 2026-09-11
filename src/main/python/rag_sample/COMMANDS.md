# Setup and Run Commands

Run these commands in order to set up and use the Company Knowledge Base Assistant.

## Prerequisites

1. **Install Python 3.10+** (if not already installed)
2. **Install Ollama** and pull the model:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   
   # Start the server and pull the model
   ollama serve &
   ollama pull qwen3:0.6b
   ```

## Setup Steps

### 1. Navigate to the src directory
```bash
cd src
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create documents directory
```bash
mkdir -p docs
```

### 5. Add company documentation
Add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`) to the `docs/` directory:
```bash
# Example: Copy some sample documents
# cp /path/to/company/docs/* docs/
```

### 6. Update configuration (optional)
Edit `config.py` if needed:
- Set `DOCUMENTS_DIR` to your documents path (default: `./docs`)
- Change `OLLAMA_MODEL` if using a different model
- Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `TOP_K` as needed
- Toggle `ENABLE_HYBRID` / `ENABLE_QUERY_EXPANSION`, or switch `FUSION_MODE`
  between `"rrf"` and `"weighted"`

### 7. Build the indexes (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python main.py build-index
```

Or directly:
```bash
python -m rag.build_index
```

This will:
- Load all documents from the `docs/` directory
- Chunk them into smaller pieces
- Generate embeddings
- Build the FAISS vector index and the SQLite FTS5 full-text index
- Save `index.faiss`, `chunks.pkl` and `fts.db`

## Usage

### Interactive CLI Mode

Run the assistant interactively:
```bash
python main.py
```

Then ask questions like:
- "What is our vacation policy?"
- "How do I request time off?"
- "What are the company values?"

Type `exit` or `quit` to stop.

## Updating the Knowledge Base

When you add new documents or update existing ones:

1. Add/update files in the `docs/` directory
2. Rebuild the index:
   ```bash
   python main.py build-index
   ```

## Troubleshooting

### "Index not found" error
- The index will be built automatically on first use
- Or manually run `python main.py build-index`

### "No documents found"
- Check that `docs/` directory exists and contains files
- Verify `DOCUMENTS_DIR` in `config.py` is correct
- Ensure files have supported extensions (`.txt`, `.md`, `.pdf`, `.docx`)

### Ollama connection errors
- Make sure Ollama is running: `ollama list`
- Verify the model is installed: `ollama pull qwen3:0.6b`
- Check `OLLAMA_URL` in `config.py` (default: `http://localhost:11434/api/generate`)

### MCP client errors
- MCP tools are optional - the assistant will work without them
- If MCP fails, RAG will still function

### Import errors
- Make sure you're in the `src/` directory
- Verify virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Benchmark and Tests

Compare the vector-only baseline against every hybrid configuration:
```bash
python -m bench.run_bench --out ../BENCHMARK.md --repeats 5
```

Run the unit tests (fusion arithmetic, FTS5 escaping, keyword parsing and
fallbacks — no Ollama required):
```bash
python -m pytest tests/ -q
```

## Quick Start Summary

```bash
# 1. Setup
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare documents
mkdir -p docs
# Add your company documentation files to docs/

# 3. Build index
python main.py build-index

# 4. Run
python main.py

# 5. Optional: measure it
python -m bench.run_bench
python -m pytest tests/ -q
```
