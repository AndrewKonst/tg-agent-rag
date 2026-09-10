package rag

class DocumentSearchService(
    private val store: RagStore = SqliteRagStore(),
    private val embeddings: EmbeddingService = HashingEmbeddingService(),
    private val topK: Int = DEFAULT_TOP_K,
) {

    fun searchDocuments(userId: Long, query: String): List<RagSearchResult> {
        val queryEmbedding = embeddings.embed(listOf(query)).single()
        return store.search(userId, queryEmbedding, topK)
    }

    fun listDocuments(userId: Long): List<RagDocument> =
        store.listDocuments(userId)

    fun deleteDocument(userId: Long, filename: String): Boolean =
        store.deleteDocument(userId, filename)

    companion object {
        const val DEFAULT_TOP_K = 5
    }
}
