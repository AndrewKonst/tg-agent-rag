package rag

import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.sql.SQLException
import kotlin.math.min

private val ragStoreLogger = KotlinLogging.logger {}

class SqliteRagStore(
    databasePath: Path = Path.of(DEFAULT_DB_PATH),
    sqliteVecExtensionPath: String? = System.getenv("SQLITE_VEC_EXTENSION_PATH")?.takeIf { it.isNotBlank() },
    private val embeddingDimension: Int = HashingEmbeddingService.DEFAULT_DIMENSION,
    private val embeddingModel: String = HashingEmbeddingService.MODEL_NAME,
) : RagStore, AutoCloseable {

    private val connection: Connection
    private val sqliteVecEnabled: Boolean

    override val vectorSearchBackend: String
        get() = if (sqliteVecEnabled) "sqlite-vec" else "sqlite-blob-fallback"

    init {
        databasePath.toAbsolutePath().parent?.let { Files.createDirectories(it) }
        connection = DriverManager.getConnection(
            "jdbc:sqlite:${databasePath.toAbsolutePath()}?enable_load_extension=true",
        )
        sqliteVecEnabled = loadSqliteVec(sqliteVecExtensionPath)
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA journal_mode=WAL")
            statement.execute("PRAGMA synchronous=NORMAL")
            statement.execute("PRAGMA foreign_keys=ON")
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL,
                    filename   TEXT    NOT NULL,
                    file_type  TEXT    NOT NULL,
                    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, filename)
                )
                """.trimIndent(),
            )
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text        TEXT    NOT NULL,
                    UNIQUE(document_id, chunk_index)
                )
                """.trimIndent(),
            )
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    chunk_id   INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    dimension  INTEGER NOT NULL,
                    embedding  BLOB    NOT NULL
                )
                """.trimIndent(),
            )
            if (sqliteVecEnabled) {
                statement.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS rag_vectors USING vec0(
                        user_id INTEGER PARTITION KEY,
                        embedding FLOAT[$embeddingDimension]
                    )
                    """.trimIndent(),
                )
            }
            statement.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)")
            statement.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_meta (
                    id        INTEGER PRIMARY KEY CHECK (id = 1),
                    model     TEXT    NOT NULL,
                    dimension INTEGER NOT NULL
                )
                """.trimIndent(),
            )
        }
        resetIndexIfEmbeddingChanged()
        ragStoreLogger.info {
            "RAG store: sqlite at ${databasePath.toAbsolutePath()}, vectorSearch=$vectorSearchBackend"
        }
    }

    override fun saveIndexedDocuments(userId: Long, documents: List<DocumentChunkingResult>) {
        if (documents.isEmpty()) return

        connection.autoCommit = false
        try {
            documents.forEach { result ->
                require(result.chunks.size == result.embeddings.size) {
                    "Document ${result.document.filename} has ${result.chunks.size} chunks " +
                        "but ${result.embeddings.size} embeddings."
                }
                deleteDocumentInternal(userId, result.document.filename)
                val documentId = insertDocument(userId, result.document)
                insertChunksAndVectors(documentId, result.chunks, result.embeddings)
            }
            connection.commit()
        } catch (e: Throwable) {
            connection.rollback()
            throw e
        } finally {
            connection.autoCommit = true
        }
    }

    override fun countDocuments(userId: Long): Int =
        count("SELECT COUNT(*) FROM documents WHERE user_id = ?", userId)

    override fun listDocuments(userId: Long): List<RagDocument> {
        val documents = mutableListOf<RagDocument>()
        connection.prepareStatement(
            """
            SELECT d.filename, d.file_type, d.created_at, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.user_id = ?
            GROUP BY d.id
            ORDER BY d.created_at DESC, d.filename
            """.trimIndent(),
        ).use { statement ->
            statement.setLong(1, userId)
            statement.executeQuery().use { rows ->
                while (rows.next()) {
                    documents += RagDocument(
                        filename = rows.getString("filename"),
                        fileType = rows.getString("file_type"),
                        chunkCount = rows.getInt("chunk_count"),
                        createdAt = rows.getString("created_at"),
                    )
                }
            }
        }
        return documents
    }

    override fun deleteDocument(userId: Long, filename: String): Boolean {
        connection.autoCommit = false
        return try {
            val deleted = deleteDocumentInternal(userId, filename)
            connection.commit()
            deleted
        } catch (e: Throwable) {
            connection.rollback()
            throw e
        } finally {
            connection.autoCommit = true
        }
    }

    override fun search(userId: Long, queryEmbedding: FloatArray, topK: Int): List<RagSearchResult> {
        require(topK > 0) { "topK must be positive" }
        if (sqliteVecEnabled) return searchWithSqliteVec(userId, queryEmbedding, topK)
        return searchWithBlobFallback(userId, queryEmbedding, topK)
    }

    private fun searchWithSqliteVec(userId: Long, queryEmbedding: FloatArray, topK: Int): List<RagSearchResult> {
        val results = mutableListOf<RagSearchResult>()

        connection.prepareStatement(
            """
            SELECT d.filename, c.chunk_index, c.text, rv.distance
            FROM rag_vectors rv
            JOIN chunks c ON c.id = rv.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE rv.embedding MATCH ?
              AND rv.k = ?
              AND rv.user_id = ?
            ORDER BY rv.distance
            """.trimIndent(),
        ).use { statement ->
            statement.setBytes(1, queryEmbedding.toBlob())
            statement.setInt(2, topK)
            statement.setLong(3, userId)
            statement.executeQuery().use { rows ->
                while (rows.next()) {
                    results += RagSearchResult(
                        filename = rows.getString("filename"),
                        chunkIndex = rows.getInt("chunk_index"),
                        text = rows.getString("text"),
                        // sqlite-vec returns a distance where smaller is better.
                        score = -rows.getDouble("distance"),
                    )
                }
            }
        }

        return results
    }

    private fun searchWithBlobFallback(userId: Long, queryEmbedding: FloatArray, topK: Int): List<RagSearchResult> {
        val results = mutableListOf<RagSearchResult>()

        connection.prepareStatement(
            """
            SELECT d.filename, c.chunk_index, c.text, v.embedding, v.dimension
            FROM vectors v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE d.user_id = ?
            """.trimIndent(),
        ).use { statement ->
            statement.setLong(1, userId)
            statement.executeQuery().use { rows ->
                while (rows.next()) {
                    val embedding = rows.getBytes("embedding").toFloatArray(rows.getInt("dimension"))
                    results += RagSearchResult(
                        filename = rows.getString("filename"),
                        chunkIndex = rows.getInt("chunk_index"),
                        text = rows.getString("text"),
                        score = dot(queryEmbedding, embedding),
                    )
                }
            }
        }

        return results
            .sortedByDescending { it.score }
            .take(topK)
    }

    override fun countChunks(userId: Long): Int =
        count(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.user_id = ?
            """.trimIndent(),
            userId,
        )

    override fun countVectors(userId: Long): Int =
        count(
            """
            SELECT COUNT(*)
            FROM vectors v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE d.user_id = ?
            """.trimIndent(),
            userId,
        )

    override fun close() {
        connection.close()
    }

    /**
     * Drops everything indexed with a different embedding, then records the current one.
     *
     * Vectors from two different models share no geometry: mixing them makes every
     * search silently wrong, and a changed dimension would make an insert fail
     * outright. Both are worse than asking for the documents again, so the index is
     * rebuilt from scratch — loudly — whenever the embedding changes.
     */
    private fun resetIndexIfEmbeddingChanged() {
        val previous = connection.createStatement().use { statement ->
            statement.executeQuery("SELECT model, dimension FROM embedding_meta WHERE id = 1").use { rows ->
                if (rows.next()) rows.getString("model") to rows.getInt("dimension") else null
            }
        }

        if (previous != null && previous.first == embeddingModel && previous.second == embeddingDimension) return

        val indexedDocuments = connection.createStatement().use { statement ->
            statement.executeQuery("SELECT COUNT(*) FROM documents").use { rows ->
                if (rows.next()) rows.getInt(1) else 0
            }
        }
        if (indexedDocuments == 0) {
            // Nothing to invalidate: an empty index, or a database this version has not seen.
            writeEmbeddingMeta()
            return
        }

        ragStoreLogger.warn {
            val from = previous?.let { "${it.first}/${it.second}d" } ?: "an unrecorded embedding"
            "Embedding changed from $from to $embeddingModel/${embeddingDimension}d; " +
                "clearing $indexedDocuments indexed document(s). They must be uploaded again."
        }
        connection.createStatement().use { statement ->
            statement.execute("DELETE FROM documents")
            if (sqliteVecEnabled) {
                statement.execute("DROP TABLE IF EXISTS rag_vectors")
                statement.execute(
                    """
                    CREATE VIRTUAL TABLE rag_vectors USING vec0(
                        user_id INTEGER PARTITION KEY,
                        embedding FLOAT[$embeddingDimension]
                    )
                    """.trimIndent(),
                )
            }
        }
        writeEmbeddingMeta()
    }

    private fun writeEmbeddingMeta() {
        connection.prepareStatement(
            "INSERT INTO embedding_meta (id, model, dimension) VALUES (1, ?, ?) " +
                "ON CONFLICT(id) DO UPDATE SET model = excluded.model, dimension = excluded.dimension",
        ).use { statement ->
            statement.setString(1, embeddingModel)
            statement.setInt(2, embeddingDimension)
            statement.executeUpdate()
        }
    }

    private fun deleteDocumentInternal(userId: Long, filename: String): Boolean {
        val chunkIds = chunkIdsForDocument(userId, filename)
        if (sqliteVecEnabled && chunkIds.isNotEmpty()) {
            connection.prepareStatement("DELETE FROM rag_vectors WHERE rowid = ?").use { statement ->
                chunkIds.forEach { chunkId ->
                    statement.setLong(1, chunkId)
                    statement.addBatch()
                }
                statement.executeBatch()
            }
        }
        connection.prepareStatement("DELETE FROM documents WHERE user_id = ? AND filename = ?").use { statement ->
            statement.setLong(1, userId)
            statement.setString(2, filename)
            return statement.executeUpdate() > 0
        }
    }

    private fun insertDocument(userId: Long, document: RawDocument): Long {
        connection.prepareStatement(
            "INSERT INTO documents (user_id, filename, file_type) VALUES (?, ?, ?)",
            java.sql.Statement.RETURN_GENERATED_KEYS,
        ).use { statement ->
            statement.setLong(1, userId)
            statement.setString(2, document.filename)
            statement.setString(3, document.fileType)
            statement.executeUpdate()
            statement.generatedKeys.use { keys ->
                check(keys.next()) { "SQLite did not return a generated document id." }
                return keys.getLong(1)
            }
        }
    }

    private fun insertChunksAndVectors(
        documentId: Long,
        chunks: List<TextChunk>,
        embeddings: List<FloatArray>,
    ) {
        connection.prepareStatement(
            "INSERT INTO chunks (document_id, chunk_index, text) VALUES (?, ?, ?)",
            java.sql.Statement.RETURN_GENERATED_KEYS,
        ).use { chunkStatement ->
            connection.prepareStatement(
                "INSERT INTO vectors (chunk_id, dimension, embedding) VALUES (?, ?, ?)",
            ).use { vectorStatement ->
                val vecStatement = if (sqliteVecEnabled) {
                    connection.prepareStatement(
                        "INSERT INTO rag_vectors(rowid, user_id, embedding) VALUES (?, ?, ?)",
                    )
                } else {
                    null
                }
                try {
                    val userId = userIdForDocument(documentId)
                    chunks.zip(embeddings).forEach { (chunk, embedding) ->
                        require(embedding.size == embeddingDimension) {
                            "Embedding dimension ${embedding.size} does not match sqlite-vec table dimension $embeddingDimension."
                        }
                        chunkStatement.setLong(1, documentId)
                        chunkStatement.setInt(2, chunk.chunkIndex)
                        chunkStatement.setString(3, chunk.text)
                        chunkStatement.executeUpdate()

                        val chunkId = chunkStatement.generatedKeys.use { keys ->
                            check(keys.next()) { "SQLite did not return a generated chunk id." }
                            keys.getLong(1)
                        }

                        val blob = embedding.toBlob()
                        vectorStatement.setLong(1, chunkId)
                        vectorStatement.setInt(2, embedding.size)
                        vectorStatement.setBytes(3, blob)
                        vectorStatement.addBatch()

                        vecStatement?.setLong(1, chunkId)
                        vecStatement?.setLong(2, userId)
                        vecStatement?.setBytes(3, blob)
                        vecStatement?.addBatch()
                    }
                    vectorStatement.executeBatch()
                    vecStatement?.executeBatch()
                } finally {
                    vecStatement?.close()
                }
            }
        }
    }

    private fun loadSqliteVec(extensionPath: String?): Boolean {
        if (extensionPath == null) {
            ragStoreLogger.warn {
                "SQLITE_VEC_EXTENSION_PATH is not set; using slower SQLite BLOB fallback for vector search."
            }
            return false
        }

        return try {
            connection.prepareStatement("SELECT load_extension(?)").use { statement ->
                statement.setString(1, extensionPath)
                statement.execute()
            }
            connection.createStatement().use { statement ->
                statement.executeQuery("SELECT vec_version()").use { rows ->
                    if (rows.next()) {
                        ragStoreLogger.info { "Loaded sqlite-vec ${rows.getString(1)} from $extensionPath" }
                    }
                }
            }
            true
        } catch (e: SQLException) {
            ragStoreLogger.error(e) {
                "Failed to load sqlite-vec from $extensionPath; using SQLite BLOB fallback."
            }
            false
        }
    }

    private fun chunkIdsForDocument(userId: Long, filename: String): List<Long> {
        val ids = mutableListOf<Long>()
        connection.prepareStatement(
            """
            SELECT c.id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.user_id = ? AND d.filename = ?
            """.trimIndent(),
        ).use { statement ->
            statement.setLong(1, userId)
            statement.setString(2, filename)
            statement.executeQuery().use { rows ->
                while (rows.next()) ids += rows.getLong(1)
            }
        }
        return ids
    }

    private fun userIdForDocument(documentId: Long): Long =
        connection.prepareStatement("SELECT user_id FROM documents WHERE id = ?").use { statement ->
            statement.setLong(1, documentId)
            statement.executeQuery().use { rows ->
                check(rows.next()) { "Document $documentId does not exist." }
                rows.getLong(1)
            }
        }

    private fun count(sql: String, userId: Long): Int =
        connection.prepareStatement(sql).use { statement ->
            statement.setLong(1, userId)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getInt(1) else 0 }
        }

    private fun FloatArray.toBlob(): ByteArray {
        val buffer = ByteBuffer.allocate(size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        forEach { buffer.putFloat(it) }
        return buffer.array()
    }

    private fun ByteArray.toFloatArray(dimension: Int): FloatArray {
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
        val count = min(dimension, size / Float.SIZE_BYTES)
        return FloatArray(count) { buffer.getFloat() }
    }

    private fun dot(left: FloatArray, right: FloatArray): Double {
        val count = min(left.size, right.size)
        var score = 0.0
        for (index in 0 until count) {
            score += left[index] * right[index]
        }
        return score
    }

    companion object {
        const val DEFAULT_DB_PATH = "data/rag.db"
    }
}
