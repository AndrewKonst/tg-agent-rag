import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.EmbeddingService
import rag.HashingEmbeddingService
import rag.OllamaEmbeddingService
import rag.SqliteRagStore
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The retrieval evaluation: five questions, each with the document that must be
 * retrieved for it.
 *
 * It runs against the real embedding model when one is installed, and against the
 * hashing fallback otherwise, so the same expectations hold in both setups.
 */
class RagEvaluationTest {

    @Serializable
    private data class EvalQuestion(
        val question: String,
        val expected_source: String,
    )

    @Test
    fun `retrieval returns expected source for evaluation questions`() {
        val questions = Json.decodeFromString<List<EvalQuestion>>(
            Files.readString(Path.of("data/rag-eval/questions.json")),
        )

        val embeddings = embeddingService()
        println("[rag-eval] embedding model: ${embeddings.modelName} (${embeddings.dimension}d)")

        SqliteRagStore(
            databasePath = Files.createTempFile("rag-eval", ".db"),
            embeddingDimension = embeddings.dimension,
            embeddingModel = embeddings.modelName,
        ).use { store ->
            DocumentIndexingService(embeddings = embeddings, store = store).indexTestDocuments(userId = 777)
            val search = DocumentSearchService(store = store, embeddings = embeddings, topK = 5)

            questions.forEach { item ->
                val results = search.searchDocuments(777, item.question)
                assertTrue(results.isNotEmpty(), "No results for: ${item.question}")
                assertTrue(
                    results.any { it.filename == item.expected_source },
                    "Expected ${item.expected_source} for '${item.question}', got ${results.map { it.filename }}",
                )
            }
        }

        assertEquals(5, questions.size)
    }

    private fun embeddingService(): EmbeddingService {
        val ollama = OllamaEmbeddingService()
        return if (ollama.isAvailable()) ollama else HashingEmbeddingService()
    }
}
