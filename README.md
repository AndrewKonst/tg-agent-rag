# tg-bot-ai

A Telegram bot that is a genuine agent: it decides what to do, runs real commands in
a sandbox to do it, and follows written instructions — skills — for tasks it has been
taught.

The agentic loop is written here rather than taken from a framework. Koog contributes
one thing: turning a conversation and a set of tool schemas into a provider's wire
format, and parsing the reply back. Deciding what to do with that reply — running the
tools, feeding the results back, knowing when to stop — is [`AgentLoop`](src/main/kotlin/harness/AgentLoop.kt),
about 150 lines.

## What it does

- **Agentic loop** — calls the model repeatedly until it produces an answer, running
  whatever tools it asks for along the way, with hard limits on both steps and retries.
- **`exec`** — a universal shell tool, so anything with a command-line interface or a
  REST API is reachable. It runs inside a Docker container, and only for chats on an
  allowlist.
- **Skills** — Markdown instruction sheets the agent reads on demand: how to drive a
  particular CLI, or the steps of a routine.
- **Memory** — a chat is one long conversation, stored in SQLite so it survives a
  restart. `/new` starts a fresh one.
- **Token observability** — every run is measured: tokens in and out, what the
  reasoning cost, how much of each prompt repeats the previous turn, which tool
  produced the most text, and what it would all cost on a hosted model. `/stats` and
  `/trace` read it back.
- **Document RAG** — send the bot a `.txt`, `.md`, `.pdf` or `.docx` file and ask
  about it. Chunks are embedded with `all-MiniLM-L6-v2`, retrieved through sqlite-vec,
  and the agent answers from them and cites the filename. A document belongs to the
  chat that uploaded it and to no other.

## Stack

| Purpose | Library |
| --- | --- |
| Language / runtime | Kotlin 2.3.21 on JVM 21 |
| Telegram | [tgbotapi](https://github.com/InsanusMokrassar/TelegramBotAPI) 36.1.0 |
| LLM plumbing | [Koog](https://github.com/JetBrains/koog) 1.2.0 (prompt executor only) |
| Concurrency | kotlinx-coroutines 1.11.0 |
| HTTP | Ktor Client 3.5.2 |
| History | SQLite via sqlite-jdbc 3.53.4.0 |
| RAG parsing | PDFBox for PDF, Apache POI for DOCX |
| Embeddings | `all-MiniLM-L6-v2` (`all-minilm`) via Ollama, 384-d |
| Vector search | SQLite + sqlite-vec |
| Sandbox | Docker (alpine 3.22, ~13 MB) |
| Build | Gradle 9.7.1 (Kotlin DSL) |

## Requirements

- **JDK 21+** — the only hard requirement. Gradle arrives via the wrapper.
- A **Telegram bot token** from [@BotFather](https://t.me/BotFather).
- An **LLM**: an API key for a hosted provider, or a local [Ollama](https://ollama.com).
- **Docker**, if you want the `exec` tool. Without it the bot runs fine; the shell is
  simply unavailable.

## Quick start

```bash
cp .env.example .env     # then fill in TELEGRAM_BOT_TOKEN and the model settings
./gradlew run
```

Send the bot `/whoami`, put the number it replies with into `OWNER_CHAT_IDS` in
`.env`, and restart. That is what enables the shell — for you and nobody else.

## Configuration

Everything comes from environment variables, falling back to `.env`. Real environment
variables always win, so a stray `.env` on a server cannot override a deployment.

### Required

| Variable | Description |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `LLM_MODEL` | Model id, e.g. `gpt-4o-mini` or `qwen3:14b` |
| `LLM_API_KEY` | API key for the provider. Not needed for Ollama |

### The model

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_BASE_URL` | provider default | Point at a gateway: Azure, OpenRouter, LiteLLM, vLLM |
| `LLM_TIMEOUT_MS` | `30000` | Deadline for a whole run, every step together |
| `LLM_CALL_TIMEOUT_MS` | `60000` | Deadline for one call to the model |
| `LLM_MAX_ATTEMPTS` | `3` | Retries after a transient failure, counted separately from steps |
| `SYSTEM_PROMPT` | built-in | Instructions given to the agent |
| `AGENT_MAX_STEPS` | `8` | How many times the model may be asked for one message |

### Memory

| Variable | Default | Description |
| --- | --- | --- |
| `CONVERSATION_STORE` | `sqlite` | `sqlite` \| `memory` |
| `CONVERSATION_DB_PATH` | `data/conversations.db` | Where history is kept |
| `CONVERSATION_MAX_CHARS` | `12000` | How much history is replayed into each run |

### Shell access

| Variable | Default | Description |
| --- | --- | --- |
| `OWNER_CHAT_IDS` | *(empty)* | Chats allowed to run commands. Empty means nobody, and `exec` stays off |
| `EXEC_SANDBOX` | `docker` | `docker` \| `host` \| `off` |
| `EXEC_TIMEOUT_MS` | `30000` | Hard limit on one command |
| `EXEC_OUTPUT_LIMIT` | `4000` | How much output reaches the model, in characters |
| `EXEC_IMAGE` | `tg-agent-sandbox:1` | Built from `sandbox/Dockerfile` on first use |
| `EXEC_CONTAINER` | `tg-agent-sandbox` | Container name |
| `EXEC_WORKDIR` | `data/workspace` | Working directory for `EXEC_SANDBOX=host` |
| `PROJECT_GIT_DIR` | `.git` | Repository mounted read-only at `/project/.git` |
| `SKILLS_DIR` | `skills` | Directory of Markdown skill files |

### Document RAG

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_DB_PATH` | `data/rag.db` | SQLite database for documents, chunks, and vectors |
| `SQLITE_VEC_EXTENSION_PATH` | *(unset)* | Absolute path to sqlite-vec `vec0.dylib`, `.so`, or `.dll` |
| `RAG_MAX_DOCUMENT_BYTES` | `20971520` | Largest document accepted for indexing (20 MB) |
| `EMBEDDING_PROVIDER` | `ollama` | Where embeddings come from: `ollama` or `hashing` |
| `EMBEDDING_MODEL` | `all-minilm` | `all-MiniLM-L6-v2`, as Ollama names it |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Embedding server, for `EMBEDDING_PROVIDER=ollama` |
| `EMBEDDING_DIMENSION` | `384` | Vector size the store is built for |

If `SQLITE_VEC_EXTENSION_PATH` is unset, the app keeps a slower Kotlin fallback for
local development. The homework implementation uses sqlite-vec when the extension
path is configured.

The sqlite-vec extension is a single loadable binary. This repository ships the
macOS arm64 build at `vendor/sqlite-vec/vec0.dylib`; for any other platform, take the
matching one from the [sqlite-vec releases](https://github.com/asg017/sqlite-vec/releases)
and point `SQLITE_VEC_EXTENSION_PATH` at it:

```bash
SQLITE_VEC_EXTENSION_PATH=$PWD/vendor/sqlite-vec/vec0.dylib
```

The embedding model has to be pulled once:

```bash
ollama pull all-minilm
```

Uploading a document is the normal way to index one: send a `.txt`, `.md`, `.pdf` or
`.docx` file to the bot and it replies when the file is searchable.

Commands:

- `/documents` lists the current user's saved documents.
- `/delete <filename>` removes a document, its chunks, and its vectors.
- `/save_docs` indexes the bundled fixtures from `data/test-documents` (debug).
- `/search_docs <query>` searches saved documents directly, without the LLM (debug).

In normal chat, the agent has a `search_documents(query)` tool and uses it when a
question is about uploaded documents.

### Observability

| Variable | Default | Description |
| --- | --- | --- |
| `OBSERVABILITY_ENABLED` | `true` | Measure and store every run |
| `OBSERVABILITY_DB_PATH` | `data/observability.db` | SQLite file for traces |
| `COST_REFERENCE_MODEL` | `gpt-4o-mini` | Whose price list turns tokens into dollars |

- `/stats` — the dashboard: totals, averages, tool costs, where context goes.
- `/trace [run]` — one run turn by turn, newest by default.

```bash
./gradlew benchmark --args="baseline"    # run the task suite, record it as "baseline"
./gradlew benchmark --args="optimized"   # the same suite after the optimisations
./gradlew benchmark --args="report"      # before/after, with the target judged
./gradlew benchmark --args="export"      # write reports/ so the numbers survive
```

Traces live in `data/observability.db`, which is git-ignored and is wiped by the next
run — so a measurement that matters is exported into `reports/`, which is committed:

| File | What it holds |
| --- | --- |
| `reports/baseline-dashboard.txt` | The dashboard as it stood before the optimisations |
| `reports/baseline-runs.csv` | One row per task: tokens, turns, cost, context split |
| `reports/baseline-calls.csv` | One row per LLM call and per tool call |
| `reports/optimized-dashboard.txt` | The same, after the optimisations |
| `reports/comparison.txt` | Before/after, with the target judged |

The audit that reads those numbers — where the tokens actually go, and what was done
about it — is in [TOKEN_AUDIT.md](TOKEN_AUDIT.md). The short version: 54% fewer tokens
per task at the same success rate, and answers that arrive in 2.8 seconds instead of
15.3.

## Architecture

```
Telegram
    │
    ▼
TelegramBot              long polling; /start /help /new /whoami
    │
    ▼
MessageHandler           one coroutine per message, whole-run timeout
    │
    ▼
HarnessAgentService      per-chat lock: load history → run → append
    │
    ▼
AgentLoop  ★             the agentic loop
    │
    ├── Llm  ─────────►  Ollama / OpenAI / Anthropic
    │
    └── tools
         ├── current_datetime
         ├── search_documents ──► SQLite + sqlite-vec ──► user's chunks
         ├── skill   ──►  skills/*.md
         └── exec    ──►  Docker container ──► the network, /project/.git
```

## Document RAG

Upload/indexing pipeline:

```
Telegram document or /save_docs
    │
    ▼
extract text (.txt/.md/.pdf/.docx)
    │
    ▼
chunk text
    │
    ▼
generate embeddings
    │
    ▼
save documents + chunks + vectors
    │
    ▼
SQLite + sqlite-vec
```

Question pipeline:

```
user question
    │
    ▼
agent calls search_documents(query)
    │
    ▼
query embedding
    │
    ▼
sqlite-vec vector search filtered by user_id
    │
    ▼
top-K chunks
    │
    ▼
LLM answer with source filename
```

### Chunking

**1000 characters, 150 characters of overlap** (`TextChunker`), cut at a sentence
boundary when one is available in the second half of the window, so a chunk rarely
ends mid-sentence.

Why those numbers: 1000 characters is roughly 150-250 words — about one section of a
document, which is the unit a question is usually about. Five such chunks still fit
comfortably in the context of a small local model alongside the conversation.

The 150-character overlap exists because a chunk boundary is arbitrary and a fact
often straddles it: "employees receive 25 paid vacation days" can be split between two
chunks, leaving neither able to answer. Overlapping by about a sentence and a half
means a fact near a boundary appears whole in one of the two neighbours.

What goes wrong at the extremes:

- **Too small** (say 200 characters): each chunk loses the context that made it
  meaningful — a number without the sentence explaining it — and one answer gets
  spread over many chunks, so top-K fills up with fragments of a single paragraph.
- **Too large** (say 5000 characters): the embedding averages several topics into one
  vector, which makes it a weak match for any specific question; retrieval also
  becomes coarse — the model receives pages of text to find one line, which costs
  context and invites it to answer from the wrong part.

### Embeddings

Chunks are embedded with `all-MiniLM-L6-v2` — the `all-minilm` model served by
Ollama — into 384-dimensional vectors, normalized to unit length so a dot product is
a cosine similarity and sqlite-vec's L2 distance ranks chunks in the same order.
Chunks are sent in batches of 32, because one HTTP round trip per chunk dominates
indexing time otherwise.

A second implementation, `HashingEmbeddingService`, sits behind the same
`EmbeddingService` interface. It is lexical rather than semantic — it can only match
words a chunk actually contains — and exists so the pipeline and its tests run with
no server and no download. It is what the bot falls back to when the embedding model
is unreachable at startup, with a warning in the log; `EMBEDDING_PROVIDER=hashing`
selects it deliberately.

Which one is in use is visible in the startup log, in `/save_docs` output, and in the
reply to an uploaded document.

Vectors from two different models share no geometry, so mixing them would make every
search quietly wrong. The store records the model and dimension it was built with,
and clears the index — loudly — when either changes. Documents then have to be
uploaded again.

### Retrieval

`search_documents(query)` embeds the query with the same model as the chunks and asks
sqlite-vec for the **5 nearest** chunks belonging to that chat.

**Distance metric: cosine similarity**, expressed as L2 distance over unit vectors.
Both chunk and query vectors are normalized to length 1, and for unit vectors L2
distance and cosine similarity rank identically — so sqlite-vec's default `vec0`
distance gives a cosine ranking with no extra work. Cosine is the right metric here
because it compares direction and ignores magnitude: a long chunk and a short question
about the same subject should still match.

**K = 5.** One or two chunks is brittle — the answer often sits in the second- or
third-best chunk, especially when a fact straddles a boundary. Beyond about five, the
context fills with text that is not about the question, which both costs a local model
its context window and gives it more chances to answer from the wrong chunk. Five
chunks of 1000 characters is roughly 5 KB of context per question, which a 14B model
handles comfortably. It is one constant, `DocumentSearchService.DEFAULT_TOP_K`.

Each result reaches the model with its provenance attached:

```
[1] Source: vacation_policy.pdf, chunk #3, score=-0.412
<chunk text>
```

followed by an instruction to answer only from these chunks and to say so when they do
not contain the answer.

### Security: one chat's documents are only its own

`user_id` is the Telegram chat id, and it is a column on `documents` and a partition
key on the `rag_vectors` virtual table. Every read is filtered by it, in SQL, in the
store itself:

```sql
WHERE rv.embedding MATCH ? AND rv.k = ? AND rv.user_id = ?
```

There is no unfiltered search path to call by mistake — `RagStore.search` takes the
`userId` as its first parameter, and `SearchDocumentsTool` is constructed per chat with
that chat's id, so the tool the model can call has no way to name a different user.
Listing and deletion are scoped the same way. `SearchDocumentsToolTest` asserts that a
second chat searching the same database gets nothing back, and
`RagDocumentPipelineTest` asserts the same at the store level.

Uploaded files are also kept per chat, under `data/uploads/<chat id>/`.

### Storage

SQLite tables:

- `documents(id, user_id, filename, file_type, created_at)`
- `chunks(id, document_id, chunk_index, text)`
- `vectors(chunk_id, dimension, embedding)`
- `rag_vectors`, a sqlite-vec `vec0` virtual table with `user_id` as partition key
- `embedding_meta(model, dimension)`, the embedding the stored vectors came from

`user_id` comes from the Telegram chat id. Search always filters by this id, so one
user's documents are not retrieved for another user.

### Failure modes

Every way indexing can fail gets its own reply, because "wrong format" and "the file
is damaged" ask different things of the person who sent the file:

| Situation | What the user is told |
| --- | --- |
| Unsupported extension | Which formats are accepted |
| File over `RAG_MAX_DOCUMENT_BYTES` | Its size and the limit |
| Damaged PDF or DOCX | That the file could not be read |
| Document with no text | That there is nothing to search |
| Embedding model not answering | To try again shortly |
| Anything else | A short apology; the details go to the log |

A failed upload leaves nothing behind: indexing writes documents, chunks and vectors
in one transaction. During `/save_docs`, a file that cannot be read is reported under
"Skipped" and the rest of the batch still gets indexed. Search, listing and deletion
each fail into a message rather than an exception, so a broken query cannot stop long
polling.

### Evaluation

The evaluation dataset lives at `data/rag-eval/questions.json`: five questions, each
with the document that has to be retrieved for it. `RagEvaluationTest` indexes the
fixtures and asserts that the expected source comes back in the top 5 for every
question — against the real embedding model when one is installed, and against the
hashing fallback otherwise.

`OllamaIntegrationTest` covers the step after retrieval: it asks an ordinary question
about the documents, with no mention of any tool, and asserts from the transcript that
the model called `search_documents` on its own and that the source document came back
in the tool result.

### Limitations

- **Retrieval is dense-only.** There is no keyword or hybrid search, so an exact
  identifier that the embedding smooths over — an order number, an error code — can be
  missed. A `chunks` FTS5 index fused with the vector results would fix that.
- **No reranker.** The top 5 chunks go to the model as they come out of sqlite-vec.
- **Chunking is character-based**, not structural: a Markdown heading or a table can be
  split across two chunks.
- **One vector space at a time.** Changing `EMBEDDING_MODEL` clears the index instead
  of re-embedding the stored chunks, so documents have to be uploaded again.
- **A document is scoped to the chat that uploaded it**, and there is no sharing
  between chats, no folders and no versioning.
- **Scanned PDFs yield nothing.** Text is extracted, never OCR'd, so an image-only PDF
  is rejected as empty.
- **Uploaded files are kept** under `data/uploads/<chat id>/` after indexing, and
  `/delete` removes a document from the index but not that copy.
- **Search is exact-k, not filtered by score**: with a small collection, the five
  nearest chunks come back even when none of them is relevant. The system prompt is
  what keeps the model from answering anyway.

### The loop

```
messages = [system prompt] + [history] + [the new message]

repeat up to AGENT_MAX_STEPS:
    reply = llm(messages, tools)          ← retried up to LLM_MAX_ATTEMPTS
    if reply has no tool calls:
        return reply                      ← done
    for each tool call:
        run it, append the result
```

Three guards keep a run finite, and they are independent on purpose:

- **`AGENT_MAX_STEPS`** bounds reasoning: how many times the model may be asked.
- **`LLM_MAX_ATTEMPTS`** bounds flakiness: retries on network failures, HTTP 429 and
  5xx, with exponential backoff and jitter. A flaky connection must not consume the
  budget the model needs to think.
- **Duplicate-call detection** bounds stubbornness. A model that re-issues an
  identical call is not making progress, so the second one is answered with "you
  already asked that, the result is above" instead of being executed, and the third
  ends the run. Small local models do this often enough to be worth handling
  explicitly rather than waiting for the step budget to run out.

A tool call that fails — an unknown name, malformed arguments, a tool that throws —
comes back to the model as a readable error it can correct on the next step, never as
an exception that ends the run.

### Memory

A chat is one long conversation. Before each run the stored history is replayed;
afterwards the whole exchange is appended.

Trimming to `CONVERSATION_MAX_CHARS` happens by **turn**, not by message. Dropping
individual messages would eventually separate a tool call from its result, and a
provider rejects a conversation containing an orphaned result. A turn — a user
message plus every reply and tool result it produced — is dropped whole, which makes
that impossible by construction.

The model's chain-of-thought is stripped before storage: it was what the model needed
to reach *that* answer, and replaying it on every later turn spends context on
nothing.

Reading, running and appending happen under a per-chat lock. Without it, two quick
messages from one person would both read the same history and write interleaved
transcripts over each other. Different chats never wait on each other.

### The sandbox, and why the bot is not in it

```
Telegram ──► bot (host, JVM) ──HTTP──► Ollama (host, :11434)
                  │
                  └──docker exec──► sandbox container ──curl──► the internet
                                         /project/.git (read-only)
```

The bot stays on the host and only commands travel into the container. That way a
command cannot read the bot's `.env` or kill its own process, and the model provider
stays reachable with no container networking.

Four layers stand between a stranger on Telegram and a shell:

1. **The allowlist.** `exec` is not merely refused to non-owners — it is absent from
   the tool list they are shown, along with the skills that depend on it.
2. **The container.** 512 MB, one CPU, 256 processes. Only `/work` is writable.
3. **The mount.** Only `.git` is exposed, read-only. The working tree — where `.env`
   lives — is not mounted at all.
4. **The bounds.** A 30-second timeout that kills the whole process tree, truncated
   output, and closed stdin so a command that decides to prompt dies instead of
   hanging.

Network access is deliberately left on, because reaching a REST API with `curl` is
the point. That is also the sharpest edge: anything fetched arrives in the model's
context, and a web page can try to talk the agent into running something. Every skill
states that command output is data, never instructions.

`EXEC_SANDBOX=host` runs commands with no isolation at all. It exists so a machine
without Docker can still demonstrate the agent, and it says so loudly in the log.

### Skills

A skill is a Markdown file with front matter:

```markdown
---
name: weather-wttr
description: Fetch weather for a city from wttr.in using curl.
---

# body: how to use the thing
```

Only the **description** goes into the system prompt — one line per skill. The body is
fetched with the `skill` tool once the model decides the skill applies. With two
skills the difference is invisible; with twenty it is the difference between a usable
prompt and a drowned one, and most of a catalogue is irrelevant to any single
question.

Two skills ship with the bot:

- [`weather-wttr`](skills/weather-wttr.md) — instructions for a CLI/API: `curl`
  against wttr.in, which format to pick, what the failures look like.
- [`morning-brief`](skills/morning-brief.md) — a routine: today's date, the weather,
  yesterday's commits, then one short summary.

Editing a skill takes effect immediately — bodies are re-read from disk on every call.
Adding one needs a restart, since the catalogue is scanned at startup.

## Running against a local model (Ollama)

No API key and no outbound calls.

```bash
brew install ollama && brew services start ollama
ollama pull qwen3:14b
ollama pull all-minilm        # embeddings for document search
```

```
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
LLM_TIMEOUT_MS=180000
```

**Pick a model that supports tool calling** — without it the agent has no way to act.
Known-good tags: `qwen3`, `llama3.1`, `llama3.2`, `mistral-nemo`. `phi` and `gemma`
will not work here.

Two caveats specific to local models. Ollama's default context window is small, and
overflowing it silently drops the beginning of a conversation rather than failing —
raise it if runs get long. And a 14B model is markedly worse at multi-step tool use
than a hosted one: it invents tool names, repeats calls and loses track of long
histories. The guards above exist largely because of that.

## Tests

```bash
./gradlew test
```

134 tests, no network required. The suites that do need something skip themselves
when it is absent: `OllamaIntegrationTest` needs Ollama with a tool-calling model,
`DockerSandboxIntegrationTest` needs Docker, and the embedding tests in
`EmbeddingServiceTest` need `ollama pull all-minilm`. To run the whole RAG side as it
ships, point the tests at sqlite-vec too:

```bash
SQLITE_VEC_EXTENSION_PATH=$PWD/vendor/sqlite-vec/vec0.dylib ./gradlew test
```

The interesting ones:

- **`AgentLoopTest`** — every guard, provoked deliberately against a scripted model:
  the step limit, duplicate calls, unknown tool names, malformed arguments, a tool
  that throws.
- **`ChatHistoryTest`** — sweeps every budget from 1 to 400 characters and asserts no
  orphaned tool result ever survives trimming.
- **`SandboxTest`** — runs real processes: timeouts, output caps, closed stdin, the
  scrubbed environment. Mocking the code that hands a model a shell would be
  self-deception.
- **`ConversationServiceTest`** — history replay, `/new`, chat isolation, and that two
  concurrent messages in one chat cannot interleave.
- **`RagDocumentPipelineTest`** — extraction of all four formats, chunking, storage,
  retrieval, per-user isolation, deletion, and each way a document can be rejected
  (unsupported, oversized, damaged, empty).
- **`EmbeddingServiceTest`** — that the real model returns unit vectors of the right
  size, ranks a paraphrase above an unrelated sentence, reports an unreachable server
  clearly, and that changing the embedding clears the index instead of mixing vectors.
- **`RagEndToEndTest`** — the whole pipeline against a scripted model: document,
  index, question, retrieval, reply with a source — and the same question from another
  chat retrieving nothing. Needs no Ollama, so it never skips.
- **`RagEvaluationTest`** — the five-question retrieval evaluation.

## Extending it

**A tool** — implement [`AgentTool`](src/main/kotlin/harness/AgentTool.kt) and add one
line to `AgentFactory.buildToolBox`:

```kotlin
class CalendarTool : AgentTool {
    override val descriptor = ToolDescriptor(
        name = "calendar_events",
        description = "Lists the user's events for a given day.",
        requiredParameters = listOf(
            ToolParameterDescriptor("date", "Date as YYYY-MM-DD", ToolParameterType.String),
        ),
        optionalParameters = emptyList(),
    )

    override suspend fun execute(args: JsonObject): String = TODO()
}
```

**A skill** — drop a `.md` file with front matter into `skills/` and restart. No code.

## Project layout

```
src/main/kotlin/
  App.kt                  entry point: config, wiring, shutdown
  harness/                the agentic loop, tool abstraction, retries
  agent/                  the AI boundary and what each chat may do
  conversation/           history: trimming, storage, per-chat locks
  sandbox/                where commands run
  tools/                  current_datetime, exec, skill, search_documents
  rag/                    extraction, chunking, embeddings, sqlite-vec store
  observability/          per-run token measurement, traces, dashboards
  benchmark/              the task suite and the before/after runner
  skills/                 the skill catalogue
  telegram/               long polling, commands, replies
  error/                  failures → short, non-technical replies
skills/                   the skill files themselves
sandbox/Dockerfile        the sandbox image
data/test-documents/      fixtures for /save_docs and the tests
data/rag-eval/            the retrieval evaluation questions
vendor/sqlite-vec/        the sqlite-vec extension binary
```
