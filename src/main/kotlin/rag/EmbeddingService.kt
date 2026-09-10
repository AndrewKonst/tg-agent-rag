package rag

import kotlin.math.sqrt

interface EmbeddingService {
    val modelName: String
    val dimension: Int

    fun embed(texts: List<String>): List<FloatArray>
}

/**
 * Lightweight deterministic embeddings, used when no embedding server is available.
 *
 * These are lexical, not semantic: a paraphrase that shares no words with a chunk
 * will not find it. [OllamaEmbeddingService] is the default for that reason. This
 * one keeps the pipeline — and the tests — working with no server, no download and
 * no network.
 */
class HashingEmbeddingService(
    override val dimension: Int = DEFAULT_DIMENSION,
) : EmbeddingService {

    override val modelName: String = MODEL_NAME

    init {
        require(dimension > 0) { "dimension must be positive" }
    }

    override fun embed(texts: List<String>): List<FloatArray> =
        texts.map { embedOne(it) }

    private fun embedOne(text: String): FloatArray {
        val vector = FloatArray(dimension)
        TOKEN.findAll(text.lowercase()).forEach { match ->
            val token = match.value
            val index = (token.hashCode() and Int.MAX_VALUE) % dimension
            vector[index] += 1.0f
        }
        normalize(vector)
        return vector
    }

    private fun normalize(vector: FloatArray) {
        val norm = sqrt(vector.sumOf { (it * it).toDouble() }).toFloat()
        if (norm == 0.0f) return
        for (index in vector.indices) {
            vector[index] = vector[index] / norm
        }
    }

    companion object {
        const val MODEL_NAME = "local-hashing-embedding"
        const val DEFAULT_DIMENSION = 384
        private val TOKEN = Regex("[\\p{L}\\p{N}][\\p{L}\\p{N}_-]*")
    }
}
