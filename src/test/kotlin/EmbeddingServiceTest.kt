import org.junit.jupiter.api.Assumptions.assumeTrue
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.EmbeddingException
import rag.HashingEmbeddingService
import rag.OllamaEmbeddingService
import rag.SqliteRagStore
import java.nio.file.Files
import java.sql.DriverManager
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Covers the real embedding model, and the parts of it that must hold whether or
 * not the model is installed.
 *
 * The Ollama tests skip themselves when `all-minilm` is not pulled, so
 * `./gradlew test` stays green on a machine without it:
 *
 *     ollama pull all-minilm
 */
class EmbeddingServiceTest {

    private val service = OllamaEmbeddingService()

    private fun requireModel() =
        assumeTrue(service.isAvailable(), "Ollama with '${service.modelName}' is not available — skipping")

    @Test
    fun `returns normalized vectors of the model's dimension`() {
        requireModel()

        val vectors = service.embed(listOf("Apple announced a foldable iPhone.", "Siri gets new languages."))

        assertEquals(2, vectors.size)
        assertEquals(384, service.dimension)
        vectors.forEach { vector ->
            assertEquals(384, vector.size)
            val norm = sqrt(vector.sumOf { (it * it).toDouble() })
            assertTrue(abs(norm - 1.0) < 1e-4, "expected a unit vector, got norm=$norm")
        }
    }

    @Test
    fun `scores a paraphrase above an unrelated sentence`() {
        requireModel()

        val (chunk, paraphrase, unrelated) = service.embed(
            listOf(
                "Buyers of a new iPhone get a three-month Apple One trial.",
                "New iPhone purchases include three free months of Apple One.",
                "The recipe calls for two cups of flour and a pinch of salt.",
            ),
        ).let { Triple(it[0], it[1], it[2]) }

        val similar = chunk.dot(paraphrase)
        val different = chunk.dot(unrelated)

        // This is the whole reason for a semantic model: the paraphrase shares almost
        // no words with the chunk, so a lexical embedding would rank it no better.
        assertTrue(similar > different, "paraphrase=$similar should beat unrelated=$different")
    }

    @Test
    fun `embedding an empty list makes no request`() {
        assertEquals(emptyList(), service.embed(emptyList()))
    }

    @Test
    fun `an unreachable embedding server fails with a clear message`() {
        val unreachable = OllamaEmbeddingService(baseUrl = "http://127.0.0.1:1")

        val error = assertFailsWith<EmbeddingException> { unreachable.embed(listOf("anything")) }

        assertTrue(error.message!!.contains("unreachable"), "got: ${error.message}")
    }

    @Test
    fun `a wrong configured dimension is reported instead of stored`() {
        requireModel()

        val mismatched = OllamaEmbeddingService(dimension = 512)

        val error = assertFailsWith<EmbeddingException> { mismatched.embed(listOf("anything")) }

        assertTrue(error.message!!.contains("EMBEDDING_DIMENSION"), "got: ${error.message}")
    }

    @Test
    fun `retrieval works end to end with the real model`() {
        requireModel()

        SqliteRagStore(
            databasePath = Files.createTempFile("rag-ollama", ".db"),
            embeddingDimension = service.dimension,
            embeddingModel = service.modelName,
        ).use { store ->
            DocumentIndexingService(embeddings = service, store = store).indexTestDocuments(userId = 555)

            val results = DocumentSearchService(store = store, embeddings = service, topK = 3)
                .searchDocuments(555, "Do new iPhone buyers get free Apple subscriptions?")

            assertTrue(results.isNotEmpty())
            assertEquals("9to5mac-services-and-macos.pdf", results.first().filename)
        }
    }

    @Test
    fun `changing the embedding model clears the index instead of mixing vectors`() {
        val databasePath = Files.createTempFile("rag-embedding-change", ".db")

        SqliteRagStore(
            databasePath = databasePath,
            embeddingDimension = 384,
            embeddingModel = "model-a",
        ).use { store ->
            DocumentIndexingService(embeddings = HashingEmbeddingService(), store = store)
                .indexTestDocuments(userId = 1)
            assertTrue(store.countDocuments(1) > 0)
        }

        // A different model, and a different vector size: the old rows are unusable.
        SqliteRagStore(
            databasePath = databasePath,
            embeddingDimension = 16,
            embeddingModel = "model-b",
        ).use { store ->
            assertEquals(0, store.countDocuments(1))
            assertEquals(0, store.countChunks(1))
            assertEquals(0, store.countVectors(1))

            DocumentIndexingService(embeddings = HashingEmbeddingService(dimension = 16), store = store)
                .indexTestDocuments(userId = 1)
            assertTrue(store.countVectors(1) > 0)
        }
    }

    @Test
    fun `an index from before embeddings were recorded is not trusted`() {
        val databasePath = Files.createTempFile("rag-embedding-unknown", ".db")

        SqliteRagStore(databasePath = databasePath).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 3)
            assertTrue(store.countDocuments(3) > 0)
        }

        // A database written before the store recorded which embedding produced its
        // vectors: they could be from any model, so they cannot be searched.
        DriverManager.getConnection("jdbc:sqlite:${databasePath.toAbsolutePath()}").use { connection ->
            connection.createStatement().use { it.execute("DROP TABLE embedding_meta") }
        }

        SqliteRagStore(databasePath = databasePath).use { store ->
            assertEquals(0, store.countDocuments(3))
            assertEquals(0, store.countVectors(3))
        }
    }

    @Test
    fun `reopening with the same embedding keeps the index`() {
        val databasePath = Files.createTempFile("rag-embedding-same", ".db")

        SqliteRagStore(databasePath = databasePath).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 2)
        }

        SqliteRagStore(databasePath = databasePath).use { store ->
            assertTrue(store.countDocuments(2) > 0)
            assertTrue(store.countVectors(2) > 0)
        }
    }

    private fun FloatArray.dot(other: FloatArray): Double {
        var sum = 0.0
        for (index in indices) sum += this[index] * other[index]
        return sum
    }
}
