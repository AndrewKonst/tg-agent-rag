package telegram

import dev.inmo.tgbotapi.extensions.api.bot.getMe
import dev.inmo.tgbotapi.extensions.api.files.downloadFile
import dev.inmo.tgbotapi.extensions.api.send.reply
import dev.inmo.tgbotapi.extensions.behaviour_builder.BehaviourContext
import dev.inmo.tgbotapi.extensions.behaviour_builder.buildBehaviourWithLongPolling
import dev.inmo.tgbotapi.extensions.behaviour_builder.triggers_handling.onCommand
import dev.inmo.tgbotapi.extensions.behaviour_builder.triggers_handling.onDocument
import dev.inmo.tgbotapi.extensions.behaviour_builder.triggers_handling.onText
import dev.inmo.tgbotapi.bot.ktor.telegramBot
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import dev.inmo.tgbotapi.types.message.abstracts.ChatMessage
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

/**
 * Owns the Telegram connection: creates the bot, starts long polling, and routes
 * updates to [MessageHandler].
 *
 * Deliberately thin — it knows nothing about LLMs, providers, or tools.
 */
class TelegramBot(
    private val token: String,
    private val messageHandler: MessageHandler,
) {

    /**
     * Starts long polling and returns the [Job] that runs it.
     *
     * Each incoming message is dispatched into its own coroutine on [requestScope],
     * so a slow LLM call for one user never delays anyone else's message. That scope
     * is expected to be backed by a `SupervisorJob`, so a failed request cannot take
     * down its siblings or the polling loop.
     */
    suspend fun start(requestScope: CoroutineScope): Job {
        val bot = telegramBot(token)

        val me = bot.getMe()
        logger.info { "Connected to Telegram as @${me.username?.withoutAt ?: me.firstName}" }

        return bot.buildBehaviourWithLongPolling(
            // A failure that escapes a handler is logged; polling continues.
            defaultExceptionsHandler = { e ->
                logger.error(e) { "Unhandled exception in Telegram update processing" }
            },
            // An idle long poll timing out is normal, not an error worth logging.
            autoSkipTimeoutExceptions = true,
            timeoutSeconds = POLL_TIMEOUT_SECONDS,
        ) {
            onCommand("start") { message ->
                reply(
                    message,
                    "Hi! Send me a message and I'll answer it with AI.\n\n" +
                        "You can also send me a .txt, .md, .pdf or .docx document and ask " +
                        "questions about it. /help lists everything.",
                )
            }

            onCommand("help") { message ->
                reply(
                    message,
                    "Just send any text message and I'll pass it to the AI and reply.\n\n" +
                        "A chat is one long conversation: I remember what we already said, " +
                        "and what my tools returned.\n\n" +
                        "Send me a .txt, .md, .pdf or .docx file and I will index it. " +
                        "After that just ask about it, and I answer from the document " +
                        "and name the file I took the answer from.\n\n" +
                        "/new — forget this conversation and start a fresh one\n" +
                        "/documents — list your saved documents\n" +
                        "/delete <filename> — delete a saved document\n" +
                        "/save_docs — index the bundled example documents\n" +
                        "/search_docs <query> — search your documents without the AI\n" +
                        "/whoami — show your chat id",
                )
            }

            onCommand("new") { message ->
                // Runs on the polling coroutine: clearing history is a single fast write,
                // and doing it here keeps it ordered against the messages around it.
                reply(message, messageHandler.startNewChat(message.chat.id.chatId.long))
            }

            onCommand("whoami") { message ->
                val chatId = message.chat.id.chatId.long
                reply(message, "Your chat id is $chatId.")
            }

            onCommand("save_docs") { message ->
                reply(message, messageHandler.saveDocs(message.chat.id.chatId.long))
            }

            onCommand("search_docs") { message ->
                val query = message.content.text.removePrefix("/search_docs")
                reply(message, messageHandler.searchDocs(message.chat.id.chatId.long, query))
            }

            onCommand("documents") { message ->
                reply(message, messageHandler.listDocuments(message.chat.id.chatId.long))
            }

            onCommand("delete") { message ->
                val filename = message.content.text.removePrefix("/delete")
                reply(message, messageHandler.deleteDocument(message.chat.id.chatId.long, filename))
            }

            onDocument { message ->
                // Downloading and indexing take seconds, so they go to the request scope
                // rather than holding up the poller.
                requestScope.launch {
                    val chatId = message.chat.id.chatId.long
                    val document = message.content.media
                    val filename = document.fileName ?: "telegram-document-${document.fileUniqueId}.bin"

                    replyOrLog(message, "Got $filename. Indexing it now...")

                    val answer = try {
                        val destination = withContext(Dispatchers.IO) {
                            // The file keeps the name the user sees, and Telegram's id
                            // disambiguates the directory instead: the filename is what
                            // /documents lists, what the agent cites and what /delete takes.
                            val directory = Files.createDirectories(
                                Path.of(UPLOADS_DIR, chatId.toString(), document.fileUniqueId.toString()),
                            )
                            val target = directory.resolve(safeUploadFilename(filename, document.fileUniqueId.toString()))
                            bot.downloadFile(document, target.toFile())
                            target
                        }
                        withContext(Dispatchers.IO) {
                            messageHandler.indexUploadedDocument(chatId, destination)
                        }
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Throwable) {
                        // Telegram refuses to serve files over 20 MB to bots, and a download
                        // can simply fail. Either way the user is owed an answer.
                        logger.error(e) { "Failed to download '$filename' for chat=$chatId" }
                        "I could not download that file from Telegram. " +
                            "Files over 20 MB cannot be sent to bots — otherwise, please try again."
                    }

                    replyOrLog(message, answer)
                }
            }

            onText { message ->
                // Skip commands; they are handled by the triggers above.
                if (message.content.text.startsWith("/")) return@onText

                requestScope.launch {
                    messageHandler.handle(this@buildBehaviourWithLongPolling, message)
                }
            }

            logger.info { "Long polling started; waiting for messages" }
        }
    }

    /**
     * Replies, and logs instead of throwing when Telegram will not take the message.
     *
     * The upload handler runs in its own coroutine, where an escaping exception would
     * only reach the scope's handler — and the user would be left waiting.
     */
    private suspend fun BehaviourContext.replyOrLog(message: ChatMessage, text: String) {
        try {
            reply(message, text)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logger.error(e) { "Failed to deliver a reply to chat=${message.chat.id.chatId.long}" }
        }
    }

    private companion object {
        /** Where uploaded documents are kept, one directory per chat. */
        const val UPLOADS_DIR = "data/uploads"

        /**
         * Long-poll window. Must stay below tgbotapi's own 30s HTTP request timeout,
         * otherwise every idle poll aborts as a timeout instead of returning cleanly.
         */
        const val POLL_TIMEOUT_SECONDS = 20
    }
}

/**
 * Reduces a name Telegram sent us to one that is safe to write inside a directory.
 *
 * Only what could leave that directory is removed — path separators, control
 * characters, leading dots — so spaces and non-Latin letters survive and the document
 * keeps the name its owner recognises.
 */
internal fun safeUploadFilename(raw: String, fallback: String): String {
    val cleaned = raw
        .replace(Regex("[\\\\/\\p{Cntrl}]"), "_")
        .trim()
        .trimStart('.')
        .trim()
        .take(MAX_UPLOAD_FILENAME_LENGTH)
    return cleaned.ifBlank { "document-$fallback" }
}

/** Long enough for any real document name, short of any filesystem's limit. */
private const val MAX_UPLOAD_FILENAME_LENGTH = 120
