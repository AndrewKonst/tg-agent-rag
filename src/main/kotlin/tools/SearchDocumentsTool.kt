package tools

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import harness.AgentTool
import harness.requiredString
import kotlinx.serialization.json.JsonObject
import rag.DocumentSearchService
import rag.RagSearchResult

class SearchDocumentsTool(
    private val userId: Long,
    private val searchService: DocumentSearchService,
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
        val results = searchService.searchDocuments(userId, query)
        return results.toToolOutput()
    }

    private fun List<RagSearchResult>.toToolOutput(): String {
        if (isEmpty()) {
            return "No relevant chunks found in this user's saved documents."
        }

        return buildString {
            appendLine("Relevant document chunks:")
            this@toToolOutput.forEachIndexed { index, result ->
                appendLine()
                appendLine("[${index + 1}] Source: ${result.filename}, chunk #${result.chunkIndex}, score=${"%.3f".format(result.score)}")
                appendLine(result.text)
            }
            appendLine()
            appendLine("Answer using only these chunks. If they do not contain the answer, say that the information was not found in the uploaded documents.")
        }
    }
}
