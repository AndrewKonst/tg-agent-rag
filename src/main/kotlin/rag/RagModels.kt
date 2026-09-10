package rag

import java.nio.file.Path

data class RawDocument(
    val path: Path,
    val filename: String,
    val fileType: String,
    val text: String,
)

data class TextChunk(
    val documentName: String,
    val chunkIndex: Int,
    val text: String,
)

data class DocumentChunkingResult(
    val document: RawDocument,
    val chunks: List<TextChunk>,
    val embeddings: List<FloatArray> = emptyList(),
)

data class SaveDocsResult(
    val indexed: List<DocumentChunkingResult>,
    val skipped: List<String>,
    val storedDocuments: Int = 0,
    val storedChunks: Int = 0,
    val storedVectors: Int = 0,
    val embeddingModel: String = "unknown",
    val vectorSearchBackend: String = "unknown",
) {
    val totalChunks: Int get() = indexed.sumOf { it.chunks.size }
    val totalEmbeddings: Int get() = indexed.sumOf { it.embeddings.size }
}

data class IndexDocumentResult(
    val document: RawDocument,
    val chunks: List<TextChunk>,
    val embeddings: List<FloatArray>,
    val embeddingModel: String,
    val vectorSearchBackend: String,
)

data class RagSearchResult(
    val filename: String,
    val chunkIndex: Int,
    val text: String,
    val score: Double,
)

data class RagDocument(
    val filename: String,
    val fileType: String,
    val chunkCount: Int,
    val createdAt: String,
)
