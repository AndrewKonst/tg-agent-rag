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
        description = "Searches the current user's uploaded documents. Call this when the user asks " +
            "about information that may be in their documents. Use only the returned chunks as " +
            "document evidence, and cite the source filename in the final answer.",
        requiredParameters = listOf(
            ToolParameterDescriptor(
                name = "query",
                description = "The user's question or a concise search query for the document collection.",
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
