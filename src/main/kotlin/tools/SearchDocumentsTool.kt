package tools

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import harness.AgentTool
import harness.requiredString
import kotlinx.serialization.json.JsonObject
import rag.ChunkExcerpt
import rag.DocumentSearchService
import rag.RagSearchResult

/**
 * Document search, as the agent sees it.
 *
 * What this returns is the single largest thing the agent sends to the model — the
 * audit measured it at 64% of all input tokens — so the output is deliberately
 * frugal: a few results, each cut down to the part that bears on the question, and
 * no boilerplate the system prompt already says.
 *
 * What is never trimmed is the source: a chunk without its filename cannot be cited,
 * and an answer that cannot be traced to a document is the thing this whole feature
 * exists to avoid.
 */
class SearchDocumentsTool(
    private val userId: Long,
    private val searchService: DocumentSearchService,
    /** How much of each chunk reaches the model. See [ChunkExcerpt]. */
    private val excerptChars: Int = DEFAULT_EXCERPT_CHARS,
) : AgentTool {

    override val descriptor = ToolDescriptor(
        name = "search_documents",
        // Every word here is sent on every request, whether the tool is used or not —
        // the schemas were measured at 29% of all input. But cutting it to "Searches
        // the user's uploaded documents" cost a benchmark task: asked something its
        // documents could answer, without the words "my documents" in the question,
        // the model answered from memory and invented a filename to cite. What it
        // needs is when to call this, not how to use the result — that is in the
        // system prompt, once.
        description = "Searches the user's uploaded documents. Use it for any question " +
            "their documents might answer.",
        requiredParameters = listOf(
            ToolParameterDescriptor(
                name = "query",
                description = "What to look for.",
                type = ToolParameterType.String,
            ),
        ),
        optionalParameters = emptyList(),
    )

    override suspend fun execute(args: JsonObject): String {
        val query = args.requiredString("query")
        return searchService.searchDocuments(userId, query).toToolOutput(query)
    }

    private fun List<RagSearchResult>.toToolOutput(query: String): String {
        if (isEmpty()) {
            return "No relevant chunks found in this user's saved documents."
        }

        return buildString {
            this@toToolOutput.forEachIndexed { index, result ->
                if (index > 0) appendLine()
                // The source line is what the answer cites; the excerpt is what it uses.
                appendLine("[${index + 1}] ${result.filename}, chunk #${result.chunkIndex}")
                appendLine(ChunkExcerpt.excerpt(result.text, query, excerptChars))
            }
        }.trimEnd()
    }

    private companion object {
        const val DEFAULT_EXCERPT_CHARS = 400
    }
}
