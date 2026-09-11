package rag

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile

/**
 * The indexing pipeline: file -> text -> chunks -> embeddings -> SQLite/sqlite-vec.
 *
 * [indexDocument] is the real path, used for every document uploaded to the bot.
 * [indexTestDocuments] runs the same pipeline over the fixtures in [documentsDir],
 * which is what `/save_docs` and the evaluation test use.
 */
class DocumentIndexingService(
    private val documentsDir: Path = Path.of("data/test-documents"),
    private val extractor: DocumentTextExtractor = DocumentTextExtractor(),
    private val chunker: TextChunker = TextChunker(),
    private val embeddings: EmbeddingService = HashingEmbeddingService(),
    private val store: RagStore = SqliteRagStore(),
) {

    /**
     * Indexes one file for [userId], replacing any earlier version of it.
     *
     * Throws for a document that cannot be indexed — [UnsupportedDocumentTypeException],
     * [DocumentTooLargeException], [DocumentParseException], [EmptyDocumentException],
     * [EmbeddingException] — so the caller can say which of them happened.
     */
    fun indexDocument(userId: Long, path: Path): IndexDocumentResult {
        val document = extractor.extract(path)
        val chunks = chunker.chunk(document)
        if (chunks.isEmpty()) throw EmptyDocumentException("File '${document.filename}' produced no chunks.")
        val vectors = embeddings.embed(chunks.map { it.text })
        val result = DocumentChunkingResult(document, chunks, vectors)
        store.saveIndexedDocuments(userId, listOf(result))
        return IndexDocumentResult(
            document = document,
            chunks = chunks,
            embeddings = vectors,
            embeddingModel = embeddings.modelName,
            vectorSearchBackend = store.vectorSearchBackend,
        )
    }

    /**
     * Indexes every fixture in [documentsDir] for [userId].
     *
     * One unreadable file must not cost the user the rest of the batch, so each
     * failure is collected into [SaveDocsResult.skipped] and the loop continues.
     */
    fun indexTestDocuments(userId: Long): SaveDocsResult {
        if (!Files.isDirectory(documentsDir)) {
            return SaveDocsResult(
                indexed = emptyList(),
                skipped = listOf("Directory not found: $documentsDir"),
            )
        }

        val indexed = mutableListOf<DocumentChunkingResult>()
        val skipped = mutableListOf<String>()

        Files.list(documentsDir).use { paths ->
            paths
                .filter { it.isRegularFile() }
                .sorted { left, right -> left.fileName.toString().compareTo(right.fileName.toString()) }
                .forEach { path ->
                    try {
                        val result = indexDocument(userId, path)
                        indexed += DocumentChunkingResult(result.document, result.chunks, result.embeddings)
                    } catch (e: UnsupportedDocumentTypeException) {
                        skipped += "${path.fileName}: ${e.message}"
                    } catch (e: DocumentTooLargeException) {
                        skipped += "${path.fileName}: ${e.message}"
                    } catch (e: DocumentParseException) {
                        skipped += "${path.fileName}: ${e.message}"
                    } catch (e: EmptyDocumentException) {
                        skipped += "${path.fileName}: ${e.message}"
                    }
                }
        }

        return SaveDocsResult(
            indexed = indexed,
            skipped = skipped,
            storedDocuments = store.countDocuments(userId),
            storedChunks = store.countChunks(userId),
            storedVectors = store.countVectors(userId),
            embeddingModel = embeddings.modelName,
            vectorSearchBackend = store.vectorSearchBackend,
        )
    }
}
