package rag

interface RagStore {
    val vectorSearchBackend: String

    fun saveIndexedDocuments(userId: Long, documents: List<DocumentChunkingResult>)
    fun search(userId: Long, queryEmbedding: FloatArray, topK: Int): List<RagSearchResult>
    fun listDocuments(userId: Long): List<RagDocument>
    fun deleteDocument(userId: Long, filename: String): Boolean
    fun countDocuments(userId: Long): Int
    fun countChunks(userId: Long): Int
    fun countVectors(userId: Long): Int
}
