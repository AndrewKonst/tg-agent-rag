import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import rag.ChunkExcerpt
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.SqliteRagStore
import tools.SearchDocumentsTool
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Retrieval output was 54% of everything the agent sent, so this is where the
 * optimisation had to bite. These tests pin down what may be cut and what may not.
 */
class ChunkExcerptTest {

    private val chunk = """
        Apple announced three new iPhone models this autumn, and the line-up is the
        widest it has been in years. Buyers of an eligible new iPhone receive a
        three-month Apple One trial at no cost. The offer covers the iPhone 18 Pro,
        the iPhone 18 Pro Max and the foldable iPhone Duo. Separately, the company
        confirmed that the next macOS release will ship in October alongside a set of
        smaller updates to its services.
    """.trimIndent().replace("\n", " ")

    @Test
    fun `an excerpt keeps the part the question is about`() {
        val excerpt = ChunkExcerpt.excerpt(chunk, "What do buyers of a new iPhone get?", maxChars = 200)

        assertTrue(excerpt.length <= 210, "should be near the budget, was ${excerpt.length}")
        assertContains(excerpt, "Apple One trial")
    }

    @Test
    fun `a different question moves the window`() {
        val excerpt = ChunkExcerpt.excerpt(chunk, "When does the next macOS release ship?", maxChars = 200)

        assertContains(excerpt, "macOS")
    }

    @Test
    fun `text that already fits is returned whole`() {
        val short = "Employees receive 25 paid vacation days."

        assertEquals(short, ChunkExcerpt.excerpt(short, "vacation days", maxChars = 400))
    }

    @Test
    fun `a paraphrased question still returns something usable`() {
        // Semantic search retrieves chunks that share meaning, not words; the excerpt
        // must not come back empty just because the question quotes nothing.
        val excerpt = ChunkExcerpt.excerpt(chunk, "free subscriptions for phone purchasers", maxChars = 200)

        assertTrue(excerpt.isNotBlank())
        assertTrue(excerpt.length <= 210)
    }

    @Test
    fun `sentences are never cut in half`() {
        val excerpt = ChunkExcerpt.excerpt(chunk, "Apple One trial", maxChars = 200)
            .trim('…')
            .trim()

        assertTrue(
            excerpt.last() in charArrayOf('.', '!', '?'),
            "an excerpt should end on a sentence boundary, got: …${excerpt.takeLast(40)}",
        )
    }

    @Test
    fun `the tool still names the source it took each excerpt from`() = runTest {
        SqliteRagStore(Files.createTempFile("excerpt-tool", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 500)
            val tool = SearchDocumentsTool(
                userId = 500,
                searchService = DocumentSearchService(store = store, topK = 3),
                excerptChars = 300,
            )

            val output = tool.execute(JsonObject(mapOf("query" to JsonPrimitive("Apple One trial"))))

            // Trimming may never cost the answer its provenance: the filename is what
            // the agent cites and what the whole RAG feature exists to make possible.
            assertContains(output, "9to5mac-services-and-macos.pdf")
            assertContains(output, "chunk #")
            assertTrue(output.length < 1_500, "the whole point was a smaller payload: ${output.length}")
        }
    }

    @Test
    fun `an empty result still tells the model nothing was found`() = runTest {
        SqliteRagStore(Files.createTempFile("excerpt-empty", ".db")).use { store ->
            val tool = SearchDocumentsTool(
                userId = 501,
                searchService = DocumentSearchService(store = store, topK = 3),
            )

            val output = tool.execute(JsonObject(mapOf("query" to JsonPrimitive("anything"))))

            assertContains(output, "No relevant chunks found")
        }
    }
}
