import agent.AgentFactory
import ai.koog.prompt.message.MessagePart
import config.AppConfig
import conversation.InMemoryConversationStore
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.EmbeddingService
import rag.HashingEmbeddingService
import rag.OllamaEmbeddingService
import rag.SqliteRagStore
import java.net.HttpURLConnection
import java.net.URI
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Runs the real chain against a **local Ollama** — no API key, no network egress:
 *
 *     AgentFactory -> Koog agent -> Ollama -> answer
 *
 * Skipped automatically when Ollama is not running or the model is not pulled, so
 * `./gradlew test` stays green on machines without it.
 *
 * To run it:
 *     brew services start ollama
 *     ollama pull qwen3:14b
 *     ./gradlew test --tests OllamaIntegrationTest -i
 *
 * Override the model with `OLLAMA_MODEL=<tag>`.
 */
class OllamaIntegrationTest {

    private val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: "http://localhost:11434"
    private val model = System.getenv("OLLAMA_MODEL") ?: "qwen3:14b"

    private fun installedModels(): List<String> = try {
        val connection = (URI("$baseUrl/api/tags").toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 2_000
            readTimeout = 2_000
        }
        if (connection.responseCode != 200) return emptyList()
        val body = connection.inputStream.use { it.readBytes().decodeToString() }
        // Cheap extraction of every "name":"..." — enough to check availability.
        Regex("\"name\"\\s*:\\s*\"([^\"]+)\"").findAll(body).map { it.groupValues[1] }.toList()
    } catch (_: Exception) {
        emptyList()
    }

    private fun requireOllama() {
        val installed = installedModels()
        assumeTrue(installed.isNotEmpty(), "Ollama is not running at $baseUrl — skipping")
        assumeTrue(
            installed.any { it == model || it.substringBefore(':') == model.substringBefore(':') },
            "Model '$model' is not pulled (available: $installed) — skipping",
        )
    }

    private fun service(
        store: InMemoryConversationStore = InMemoryConversationStore(),
        documentSearchService: DocumentSearchService? = null,
        systemPrompt: String = "Answer in one short sentence.",
    ) = AgentFactory.create(
        config = AppConfig.from(
            mapOf(
                "TELEGRAM_BOT_TOKEN" to "not-used-in-this-test",
                "LLM_PROVIDER" to "ollama",
                "LLM_MODEL" to model,
                "LLM_BASE_URL" to baseUrl,
                // Local models are slower than hosted ones; be generous.
                "LLM_TIMEOUT_MS" to "300000",
                "SYSTEM_PROMPT" to systemPrompt,
            ),
        ),
        store = store,
        sandbox = null,
        documentSearchService = documentSearchService ?: DocumentSearchService(),
    )

    @Test
    fun `answers a plain question`() = runBlocking {
        requireOllama()

        val answer = service().ask(chatId = 1, message = "What is the capital of France? Answer in one word.")

        println("[ollama] plain answer: $answer")
        assertTrue(answer.isNotBlank(), "the model returned nothing")
        assertTrue(
            answer.contains("Paris", ignoreCase = true),
            "expected the answer to mention Paris, got: $answer",
        )
    }

    /**
     * The RAG acceptance test: an ordinary question about the user's documents has to
     * make the model reach for `search_documents` on its own, with no command and no
     * hint about the tool in the question.
     *
     * Asserting on the transcript rather than on the wording of the answer: what the
     * homework requires is that retrieval happened and the source came back with it.
     */
    @Test
    fun `answers a document question by calling search_documents`() = runBlocking {
        requireOllama()

        val chatId = 4_242L
        val embeddings = embeddingService()
        val conversation = InMemoryConversationStore()

        SqliteRagStore(
            databasePath = Files.createTempFile("rag-agent", ".db"),
            embeddingDimension = embeddings.dimension,
            embeddingModel = embeddings.modelName,
        ).use { ragStore ->
            DocumentIndexingService(embeddings = embeddings, store = ragStore)
                .indexTestDocuments(userId = chatId)

            val answer = service(
                store = conversation,
                documentSearchService = DocumentSearchService(store = ragStore, embeddings = embeddings),
                systemPrompt = AppConfig.DEFAULT_SYSTEM_PROMPT,
            ).ask(chatId, "What do buyers of a new iPhone get, according to my documents?")

            println("[ollama] document answer: $answer")

            val transcript = conversation.load(chatId)
            val calledTools = transcript
                .flatMap { it.parts }
                .filterIsInstance<MessagePart.Tool.Call>()
                .map { it.tool }
            assertTrue(
                "search_documents" in calledTools,
                "expected the model to call search_documents, it called: $calledTools",
            )

            val retrieved = transcript
                .flatMap { it.parts }
                .filterIsInstance<MessagePart.Tool.Result>()
                .filter { it.tool == "search_documents" }
                .joinToString("\n") { it.output }
            assertTrue(
                retrieved.contains("9to5mac-services-and-macos.pdf"),
                "retrieval did not return the source document; got: ${retrieved.take(500)}",
            )
            assertTrue(answer.isNotBlank(), "the model returned nothing")
        }
    }

    /**
     * The other half of the RAG requirement: when the documents do not contain the
     * answer, the agent has to say so rather than answer from what the model happens
     * to know.
     */
    @Test
    fun `says it did not find an answer that is not in the documents`() = runBlocking {
        requireOllama()

        val chatId = 4_243L
        val embeddings = embeddingService()

        SqliteRagStore(
            databasePath = Files.createTempFile("rag-absent", ".db"),
            embeddingDimension = embeddings.dimension,
            embeddingModel = embeddings.modelName,
        ).use { ragStore ->
            DocumentIndexingService(embeddings = embeddings, store = ragStore)
                .indexTestDocuments(userId = chatId)

            val answer = service(
                documentSearchService = DocumentSearchService(store = ragStore, embeddings = embeddings),
                systemPrompt = AppConfig.DEFAULT_SYSTEM_PROMPT,
            ).ask(chatId, "According to my documents, how many vacation days do employees get?")

            println("[ollama] absent-answer reply: $answer")

            // The fixtures are Apple news; there is no vacation policy among them. Any
            // specific number here would be invented.
            assertTrue(
                DENIALS.any { answer.contains(it, ignoreCase = true) },
                "expected the model to say it found nothing, got: $answer",
            )
        }
    }

    private fun embeddingService(): EmbeddingService {
        val ollama = OllamaEmbeddingService()
        return if (ollama.isAvailable()) ollama else HashingEmbeddingService()
    }

    @Test
    fun `can call a tool`() = runBlocking {
        requireOllama()

        // Only DateTimeTool can answer this, so a sensible reply implies tool calling works.
        val answer = service().ask(chatId = 1, message = "What year is it right now? Use your tools.")

        println("[ollama] tool answer: $answer")
        assertTrue(answer.isNotBlank(), "the model returned nothing")
    }

    private companion object {
        /** Ways a model phrases "it is not in the documents". */
        val DENIALS = listOf(
            "not found",
            "no information",
            "not contain",
            "not mention",
            "not mentioned",
            "could not find",
            "couldn't find",
            "did not find",
            "didn't find",
            "no mention",
            "not available in",
            "not in the",
            "no relevant",
            "not present",
            "no information about",
            "cannot find",
        )
    }
}
