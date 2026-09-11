package rag

class TextChunker(
    private val chunkSize: Int = DEFAULT_CHUNK_SIZE,
    private val overlap: Int = DEFAULT_OVERLAP,
) {

    init {
        require(chunkSize > 0) { "chunkSize must be positive" }
        require(overlap >= 0) { "overlap must not be negative" }
        require(overlap < chunkSize) { "overlap must be smaller than chunkSize" }
    }

    fun chunk(document: RawDocument): List<TextChunk> {
        val normalized = document.text.replace(Regex("\\s+"), " ").trim()
        if (normalized.isBlank()) return emptyList()

        val chunks = mutableListOf<TextChunk>()
        var start = 0
        while (start < normalized.length) {
            val end = findChunkEnd(normalized, start)
            val text = normalized.substring(start, end).trim()
            if (text.isNotBlank()) {
                chunks += TextChunk(
                    documentName = document.filename,
                    chunkIndex = chunks.size,
                    text = text,
                )
            }
            if (end == normalized.length) break
            start = (end - overlap).coerceAtLeast(start + 1)
        }
        return chunks
    }

    private fun findChunkEnd(text: String, start: Int): Int {
        val hardEnd = (start + chunkSize).coerceAtMost(text.length)
        if (hardEnd == text.length) return hardEnd

        val window = text.substring(start, hardEnd)
        val breakAt = window.lastIndexOf(". ").takeIf { it > chunkSize / 2 }
            ?: window.lastIndexOf("? ").takeIf { it > chunkSize / 2 }
            ?: window.lastIndexOf("! ").takeIf { it > chunkSize / 2 }
            ?: window.lastIndexOf(' ').takeIf { it > chunkSize / 2 }
            ?: window.length

        return start + breakAt + 1
    }

    companion object {
        const val DEFAULT_CHUNK_SIZE = 1_000
        const val DEFAULT_OVERLAP = 150
    }
}
