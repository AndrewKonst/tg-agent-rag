import org.apache.poi.xwpf.usermodel.XWPFDocument
import rag.DocumentIndexingService
import rag.DocumentParseException
import rag.DocumentSearchService
import rag.DocumentTextExtractor
import rag.DocumentTooLargeException
import rag.EmptyDocumentException
import rag.HashingEmbeddingService
import rag.SqliteRagStore
import rag.TextChunker
import rag.UnsupportedDocumentTypeException
import kotlin.math.abs
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RagDocumentPipelineTest {

    private val extractor = DocumentTextExtractor()

    @Test
    fun `extracts text from txt`() {
        val path = Files.createTempFile("rag-test", ".txt")
        path.writeText("Plain text document about iPhone Duo.")

        val document = extractor.extract(path)

        assertEquals("txt", document.fileType)
        assertContains(document.text, "iPhone Duo")
    }

    @Test
    fun `extracts text from markdown`() {
        val path = Files.createTempFile("rag-test", ".md")
        path.writeText("# AirPods 5\n\nActive Noise Cancellation is standard.")

        val document = extractor.extract(path)

        assertEquals("md", document.fileType)
        assertContains(document.text, "Active Noise Cancellation")
    }

    @Test
    fun `extracts text from pdf fixture`() {
        val path = Path.of("data/test-documents/9to5mac-services-and-macos.pdf")

        val document = extractor.extract(path)

        assertEquals("pdf", document.fileType)
        assertContains(document.text, "Apple One Trial")
        assertContains(document.text, "three-month Apple One trial")
    }

    @Test
    fun `extracts text from docx`() {
        val path = Files.createTempFile("rag-test", ".docx")
        XWPFDocument().use { document ->
            document.createParagraph().createRun().setText("Siri AI launches in English first.")
            document.createParagraph().createRun().setText("Five more languages arrive in October.")
            Files.newOutputStream(path).use { document.write(it) }
        }

        val document = extractor.extract(path)

        assertEquals("docx", document.fileType)
        assertContains(document.text, "Siri AI launches")
        assertContains(document.text, "Five more languages")
    }

    @Test
    fun `an unsupported extension is rejected by name`() {
        val path = Files.createTempFile("rag-test", ".pages")
        path.writeText("Not a format we can read.")

        val error = assertFailsWith<UnsupportedDocumentTypeException> { extractor.extract(path) }

        assertContains(error.message!!, "pages")
    }

    @Test
    fun `an empty document is rejected`() {
        val path = Files.createTempFile("rag-test", ".txt")
        path.writeText("   \n\n  ")

        assertFailsWith<EmptyDocumentException> { extractor.extract(path) }
    }

    @Test
    fun `a document over the size limit is rejected before parsing`() {
        val path = Files.createTempFile("rag-test", ".txt")
        path.writeText("x".repeat(5_000))

        val error = assertFailsWith<DocumentTooLargeException> {
            DocumentTextExtractor(maxBytes = 1_000).extract(path)
        }

        assertContains(error.message!!, "limit")
    }

    @Test
    fun `a damaged pdf fails as a parse error, not a crash`() {
        val path = Files.createTempFile("rag-broken", ".pdf")
        path.writeText("%PDF-1.7 and then nothing that a PDF reader can use")

        assertFailsWith<DocumentParseException> { extractor.extract(path) }
    }

    @Test
    fun `a damaged docx fails as a parse error`() {
        val path = Files.createTempFile("rag-broken", ".docx")
        path.writeText("this is not a zip container at all")

        assertFailsWith<DocumentParseException> { extractor.extract(path) }
    }

    @Test
    fun `one unreadable file does not stop the batch`() {
        val directory = Files.createTempDirectory("rag-mixed")
        directory.resolve("good.txt").writeText("The iPhone Duo is a foldable phone.")
        directory.resolve("broken.pdf").writeText("not really a pdf")
        directory.resolve("notes.pages").writeText("unsupported format")

        SqliteRagStore(Files.createTempFile("rag-mixed", ".db")).use { store ->
            val result = DocumentIndexingService(documentsDir = directory, store = store)
                .indexTestDocuments(userId = 7)

            assertEquals(listOf("good.txt"), result.indexed.map { it.document.filename })
            assertEquals(2, result.skipped.size)
            assertTrue(result.skipped.any { it.startsWith("broken.pdf") })
            assertTrue(result.skipped.any { it.startsWith("notes.pages") })
        }
    }

    @Test
    fun `chunks extracted documents`() {
        val raw = extractor.extract(Path.of("data/test-documents/9to5mac-apple-news.md"))
        val chunks = TextChunker(chunkSize = 260, overlap = 40).chunk(raw)

        assertTrue(chunks.size > 1)
        assertTrue(chunks.all { it.documentName == raw.filename })
        assertEquals(chunks.indices.toList(), chunks.map { it.chunkIndex })
    }

    @Test
    fun `local test document service chunks all fixture formats`() {
        SqliteRagStore(Files.createTempFile("rag-test", ".db")).use { store ->
            val result = DocumentIndexingService(store = store).indexTestDocuments(userId = 42)

            assertTrue(result.skipped.isEmpty(), "unexpected skipped documents: ${result.skipped}")
            assertTrue(result.indexed.map { it.document.fileType }.containsAll(listOf("txt", "md", "pdf", "docx")))
            assertTrue(result.totalChunks >= result.indexed.size)
            assertEquals(result.totalChunks, result.totalEmbeddings)
            assertEquals(result.indexed.size, result.storedDocuments)
            assertEquals(result.totalChunks, result.storedChunks)
            assertEquals(result.totalEmbeddings, result.storedVectors)
            assertEquals(store.vectorSearchBackend, result.vectorSearchBackend)
            if (!System.getenv("SQLITE_VEC_EXTENSION_PATH").isNullOrBlank()) {
                assertEquals("sqlite-vec", store.vectorSearchBackend)
            }
        }
    }

    @Test
    fun `embedding service creates normalized fixed-size vectors`() {
        val service = HashingEmbeddingService(dimension = 32)

        val vectors = service.embed(listOf("iPhone Duo foldable phone", "AirPods 5 noise cancellation"))

        assertEquals(2, vectors.size)
        assertTrue(vectors.all { it.size == 32 })
        vectors.forEach { vector ->
            val norm = kotlin.math.sqrt(vector.sumOf { (it * it).toDouble() })
            assertTrue(abs(norm - 1.0) < 0.0001, "vector should be normalized, got norm=$norm")
        }
    }

    @Test
    fun `search returns relevant chunks for the current user`() {
        SqliteRagStore(Files.createTempFile("rag-search", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 100)
            val search = DocumentSearchService(store = store, topK = 3)

            val results = search.searchDocuments(100, "Apple One trial eligible iPhone buyers")

            assertTrue(results.isNotEmpty())
            assertEquals("9to5mac-services-and-macos.pdf", results.first().filename)
            assertContains(results.first().text, "Apple One trial")
        }
    }

    @Test
    fun `search is isolated by user id`() {
        SqliteRagStore(Files.createTempFile("rag-isolation", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 100)
            val search = DocumentSearchService(store = store, topK = 3)

            val otherUserResults = search.searchDocuments(200, "Apple One trial")

            assertTrue(otherUserResults.isEmpty())
        }
    }

    @Test
    fun `documents can be listed and deleted with chunks and vectors`() {
        SqliteRagStore(Files.createTempFile("rag-delete", ".db")).use { store ->
            DocumentIndexingService(store = store).indexTestDocuments(userId = 100)
            val search = DocumentSearchService(store = store, topK = 3)

            val before = search.listDocuments(100)
            assertTrue(before.any { it.filename == "9to5mac-services-and-macos.pdf" })

            assertTrue(search.deleteDocument(100, "9to5mac-services-and-macos.pdf"))

            val after = search.listDocuments(100)
            assertTrue(after.none { it.filename == "9to5mac-services-and-macos.pdf" })
            assertTrue(search.searchDocuments(100, "Apple One trial eligible iPhone buyers")
                .none { it.filename == "9to5mac-services-and-macos.pdf" })
            assertEquals(store.countChunks(100), store.countVectors(100))
        }
    }
}
