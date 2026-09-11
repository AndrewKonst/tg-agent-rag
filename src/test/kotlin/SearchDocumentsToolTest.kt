import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.SqliteRagStore
import tools.SearchDocumentsTool
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContains

class SearchDocumentsToolTest {

    @Test
    fun `returns document chunks with source attribution`() = runTest {
        SqliteRagStore(Files.createTempFile("rag-tool", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 123)
            val tool = SearchDocumentsTool(
                userId = 123,
                searchService = DocumentSearchService(store = store, topK = 2),
            )

            val output = tool.execute(JsonObject(mapOf("query" to JsonPrimitive("Apple One trial"))))

            assertContains(output, "9to5mac-services-and-macos.pdf")
            assertContains(output, "chunk #")
        }
    }

    @Test
    fun `does not leak another user's chunks`() = runTest {
        SqliteRagStore(Files.createTempFile("rag-tool-isolation", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 123)
            val tool = SearchDocumentsTool(
                userId = 456,
                searchService = DocumentSearchService(store = store, topK = 2),
            )

            val output = tool.execute(JsonObject(mapOf("query" to JsonPrimitive("Apple One trial"))))

            assertContains(output, "No relevant chunks found")
        }
    }
}
