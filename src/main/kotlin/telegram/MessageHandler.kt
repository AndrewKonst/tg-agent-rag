package telegram

import agent.AgentService
import dev.inmo.tgbotapi.extensions.api.send.reply
import dev.inmo.tgbotapi.extensions.behaviour_builder.BehaviourContext
import dev.inmo.tgbotapi.types.message.abstracts.ChatContentMessage
import dev.inmo.tgbotapi.types.message.content.TextContent
import error.ErrorHandler
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import rag.DocumentIndexingService
import rag.DocumentParseException
import rag.DocumentSearchService
import rag.DocumentTooLargeException
import rag.EmbeddingException
import rag.EmptyDocumentException
import rag.IndexDocumentResult
import rag.RagSearchResult
import rag.SaveDocsResult
import rag.UnsupportedDocumentTypeException
import java.nio.file.Path
import kotlin.time.Duration

private val logger = KotlinLogging.logger {}

/**
 * Handles one text message: ask the agent, reply with the answer.
 *
 * Holds no per-request state — everything about a request lives on the stack of
 * [handle] — so a single instance is safe to share across all concurrent users.
 */
class MessageHandler(
    private val agentService: AgentService,
    private val timeout: Duration,
    private val documentIndexingService: DocumentIndexingService = DocumentIndexingService(),
    private val documentSearchService: DocumentSearchService = DocumentSearchService(),
) {

    /**
     * Processes [message] and replies. Never throws for an ordinary failure: any
     * error becomes a friendly reply, so one bad request cannot stop long polling.
     */
    suspend fun handle(context: BehaviourContext, message: ChatContentMessage<TextContent>) {
        val text = message.content.text.trim()
        if (text.isEmpty()) return

        val reply = answerOrExplain(text, message.chat.id.chatId.long)
        sendReply(context, message, reply)
    }

    /**
     * Runs the agent under [timeout] and returns either its answer or a friendly
     * explanation of what went wrong.
     *
     * This is where the resilience requirements are met, and it is deliberately free
     * of Telegram types so it can be tested on its own. Cancellation of the enclosing
     * scope (shutdown) is propagated rather than swallowed.
     */
    internal suspend fun answerOrExplain(text: String, chatId: Long): String {
        logger.info { "Request from chat=$chatId (${text.length} chars)" }

        return try {
            val answer = withTimeout(timeout) {
                agentService.ask(chatId, text)
            }
            logger.info { "Answered chat=$chatId (${answer.length} chars)" }
            answer
        } catch (e: TimeoutCancellationException) {
            // Must be caught before CancellationException — it is a subtype of it.
            ErrorHandler.toUserMessage(e, "chat=$chatId")
        } catch (e: CancellationException) {
            // The bot is shutting down, or the parent scope was cancelled. Stay cooperative.
            logger.debug { "Request for chat=$chatId cancelled" }
            throw e
        } catch (e: Throwable) {
            ErrorHandler.toUserMessage(e, "chat=$chatId")
        }
    }

    /**
     * Clears [chatId]'s history and returns the confirmation to send back.
     *
     * Failure is reported to the user rather than swallowed: silently continuing an
     * old conversation after someone asked for a new one is worse than an error.
     */
    internal suspend fun startNewChat(chatId: Long): String = try {
        agentService.reset(chatId)
        logger.info { "chat=$chatId: started a new conversation" }
        NEW_CHAT_CONFIRMATION
    } catch (e: CancellationException) {
        throw e
    } catch (e: Throwable) {
        logger.error(e) { "Failed to clear history for chat=$chatId" }
        "I could not clear the history. Please try again."
    }

    /**
     * Debug command: runs the whole indexing pipeline over the bundled fixtures.
     *
     * Uploading a document is the real path; this one exists so retrieval can be
     * tried out — and demonstrated — without four files to hand.
     */
    internal fun saveDocs(chatId: Long): String = try {
        documentIndexingService.indexTestDocuments(chatId).toTelegramMessage()
    } catch (e: EmbeddingException) {
        logger.error(e) { "Embedding model failed while indexing the test documents" }
        EMBEDDING_UNAVAILABLE
    } catch (e: Throwable) {
        logger.error(e) { "Failed to index the local test documents" }
        "I could not process the local test documents. Please check the logs."
    }

    internal fun searchDocs(chatId: Long, query: String): String = try {
        val trimmed = query.trim()
        if (trimmed.isBlank()) {
            return "Usage: /search_docs your question"
        }
        documentSearchService.searchDocuments(chatId, trimmed).toTelegramMessage(trimmed)
    } catch (e: Throwable) {
        logger.error(e) { "Failed to search local RAG documents" }
        "I could not search the saved documents. Please check the logs."
    }

    internal fun listDocuments(chatId: Long): String = try {
        val documents = documentSearchService.listDocuments(chatId)
        if (documents.isEmpty()) {
            "No documents saved yet. Send me a .txt, .md, .pdf or .docx file, or run /save_docs."
        } else {
            buildString {
                appendLine("Your documents:")
                appendLine()
                documents.forEachIndexed { index, document ->
                    appendLine("${index + 1}. ${document.filename} (${document.fileType}, ${document.chunkCount} chunks)")
                }
            }
        }
    } catch (e: Throwable) {
        logger.error(e) { "Failed to list RAG documents" }
        "I could not list your documents. Please check the logs."
    }

    internal fun deleteDocument(chatId: Long, filename: String): String = try {
        val trimmed = filename.trim()
        if (trimmed.isBlank()) {
            return "Usage: /delete filename.pdf"
        }
        if (documentSearchService.deleteDocument(chatId, trimmed)) {
            "Deleted document: $trimmed"
        } else {
            "I did not find a saved document named: $trimmed"
        }
    } catch (e: Throwable) {
        logger.error(e) { "Failed to delete RAG document" }
        "I could not delete that document. Please check the logs."
    }

    /**
     * Indexes a document a user uploaded, and says what happened.
     *
     * Each failure gets its own message: "wrong format" and "the file is damaged"
     * ask for different things from the person who sent it.
     */
    internal fun indexUploadedDocument(chatId: Long, path: Path): String = try {
        documentIndexingService.indexDocument(chatId, path).toTelegramMessage()
    } catch (e: UnsupportedDocumentTypeException) {
        logger.info { "Rejected '${path.fileName}': ${e.message}" }
        "I can only read .txt, .md, .pdf and .docx files. ${e.message}"
    } catch (e: DocumentTooLargeException) {
        logger.info { "Rejected '${path.fileName}': ${e.message}" }
        "That document is too large for me to index. ${e.message}"
    } catch (e: DocumentParseException) {
        logger.warn(e) { "Could not parse uploaded document '${path.fileName}'" }
        "I could not read that file — it looks damaged or is not really a ${path.fileName.toString().substringAfterLast('.', "document")}."
    } catch (e: EmptyDocumentException) {
        logger.info { "Rejected '${path.fileName}': ${e.message}" }
        "That document has no text in it, so there is nothing to search."
    } catch (e: EmbeddingException) {
        logger.error(e) { "Embedding model failed for '${path.fileName}'" }
        EMBEDDING_UNAVAILABLE
    } catch (e: Throwable) {
        logger.error(e) { "Failed to index uploaded document '${path.fileName}'" }
        "Something went wrong while indexing that document. Please try again."
    }

    /**
     * Sends [text] back, splitting it across messages when it exceeds Telegram's
     * per-message limit. A Telegram API failure here is logged, not rethrown —
     * there is no way left to tell the user about it.
     */
    private suspend fun sendReply(
        context: BehaviourContext,
        message: ChatContentMessage<TextContent>,
        text: String,
    ) {
        try {
            chunk(text).forEach { part -> context.reply(message, part) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logger.error(e) { "Failed to deliver reply to chat=${message.chat.id.chatId.long}" }
        }
    }

    internal companion object {
        const val EMBEDDING_UNAVAILABLE =
            "I could not turn that text into vectors — the embedding model is not answering. " +
                "Please try again in a moment."

        const val NEW_CHAT_CONFIRMATION =
            "History cleared. This is a new chat — I no longer remember what we discussed."

        /** Telegram rejects text messages longer than 4096 UTF-16 code units. */
        const val TELEGRAM_MAX_MESSAGE_LENGTH = 4096

        /**
         * Splits [text] into Telegram-sized parts, preferring to break at a paragraph
         * or line boundary so lists and code blocks stay readable.
         */
        fun chunk(text: String, limit: Int = TELEGRAM_MAX_MESSAGE_LENGTH): List<String> {
            require(limit > 0) { "limit must be positive" }
            if (text.length <= limit) return listOf(text)

            val parts = mutableListOf<String>()
            var rest = text
            while (rest.length > limit) {
                val window = rest.substring(0, limit)
                val breakAt = window.lastIndexOf("\n\n").takeIf { it > limit / 2 }
                    ?: window.lastIndexOf('\n').takeIf { it > limit / 2 }
                    ?: window.lastIndexOf(' ').takeIf { it > limit / 2 }
                    ?: limit
                parts += rest.substring(0, breakAt).trimEnd()
                rest = rest.substring(breakAt).trimStart()
            }
            if (rest.isNotEmpty()) parts += rest
            return parts
        }
    }
}

private fun SaveDocsResult.toTelegramMessage(): String = buildString {
    appendLine("Documents processed.")
    appendLine()
    appendLine("Indexed: ${indexed.size}")
    appendLine("Chunks created: $totalChunks")
    appendLine("Embeddings generated: $totalEmbeddings")
    appendLine()
    appendLine("Stored in SQLite:")
    appendLine("- documents: $storedDocuments")
    appendLine("- chunks: $storedChunks")
    appendLine("- vectors: $storedVectors")
    appendLine("- embedding model: $embeddingModel")
    appendLine("- vector search: $vectorSearchBackend")

    if (indexed.isNotEmpty()) {
        appendLine()
        appendLine("Documents:")
        indexed.forEach { result ->
            appendLine("- ${result.document.filename}: ${result.chunks.size} chunk(s)")
        }
    }

    if (skipped.isNotEmpty()) {
        appendLine()
        appendLine("Skipped:")
        skipped.forEach { appendLine("- $it") }
    }

    appendLine()
    append("Now you can ask questions about these documents.")
}

private fun List<RagSearchResult>.toTelegramMessage(query: String): String = buildString {
    appendLine("Search results")
    appendLine()
    appendLine("Query: $query")
    appendLine()

    if (this@toTelegramMessage.isEmpty()) {
        append("No chunks found. Upload a document first, or run /save_docs.")
        return@buildString
    }

    this@toTelegramMessage.take(5).forEachIndexed { index, result ->
        appendLine("${index + 1}. ${result.filename}, chunk #${result.chunkIndex}, score=${"%.3f".format(result.score)}")
        appendLine(result.text.take(500))
        appendLine()
    }
}

private fun IndexDocumentResult.toTelegramMessage(): String = buildString {
    appendLine("Document ready.")
    appendLine()
    appendLine("File: ${document.filename}")
    appendLine("Type: ${document.fileType}")
    appendLine("Chunks: ${chunks.size}")
    appendLine("Embeddings: ${embeddings.size}")
    appendLine("Embedding model: $embeddingModel")
    appendLine("Vector search: $vectorSearchBackend")
    appendLine()
    append("Now you can ask questions about this document.")
}
