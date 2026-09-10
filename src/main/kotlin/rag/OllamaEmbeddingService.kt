package rag

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import kotlin.math.sqrt

/** Thrown when the embedding model cannot turn text into vectors. */
class EmbeddingException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

private val embeddingLogger = KotlinLogging.logger {}

/**
 * Real semantic embeddings from a local Ollama server.
 *
 * The default model is `all-minilm`, which is `all-MiniLM-L6-v2` — the sentence
 * transformer the homework suggests, at 384 dimensions. Vectors come back
 * normalized, so a dot product is a cosine similarity and sqlite-vec's L2 distance
 * ranks chunks in the same order.
 *
 * Requests are batched because one HTTP round trip per chunk dominates indexing time
 * for a document of any size.
 */
class OllamaEmbeddingService(
    baseUrl: String = DEFAULT_BASE_URL,
    override val modelName: String = DEFAULT_MODEL,
    override val dimension: Int = DEFAULT_DIMENSION,
    private val requestTimeout: Duration = Duration.ofSeconds(DEFAULT_TIMEOUT_SECONDS),
    private val batchSize: Int = DEFAULT_BATCH_SIZE,
) : EmbeddingService {

    private val embedUrl = URI.create("${baseUrl.trimEnd('/')}/api/embed")
    private val tagsUrl = URI.create("${baseUrl.trimEnd('/')}/api/tags")

    private val client: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(CONNECT_TIMEOUT_SECONDS))
        .build()

    init {
        require(dimension > 0) { "dimension must be positive" }
        require(batchSize > 0) { "batchSize must be positive" }
    }

    override fun embed(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) return emptyList()

        return texts.chunked(batchSize).flatMap { batch ->
            embedBatch(batch)
        }
    }

    /**
     * Reports whether this Ollama has the model, without embedding anything.
     *
     * Used at startup so an unreachable server or an unpulled model is a clear log
     * line and a documented fallback, instead of a failure on the first upload.
     */
    fun isAvailable(): Boolean = try {
        val response = client.send(
            HttpRequest.newBuilder(tagsUrl).timeout(Duration.ofSeconds(PROBE_TIMEOUT_SECONDS)).GET().build(),
            HttpResponse.BodyHandlers.ofString(),
        )
        response.statusCode() == 200 && installedModels(response.body()).any { it.matchesModel(modelName) }
    } catch (e: Exception) {
        embeddingLogger.debug(e) { "Ollama embedding probe failed for $tagsUrl" }
        false
    }

    private fun embedBatch(batch: List<String>): List<FloatArray> {
        val payload = json.encodeToString(EmbedRequest(model = modelName, input = batch))

        val response = try {
            client.send(
                HttpRequest.newBuilder(embedUrl)
                    .timeout(requestTimeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload))
                    .build(),
                HttpResponse.BodyHandlers.ofString(),
            )
        } catch (e: Exception) {
            throw EmbeddingException(
                "Embedding model '$modelName' is unreachable at $embedUrl. Is Ollama running?",
                e,
            )
        }

        if (response.statusCode() != 200) {
            throw EmbeddingException(
                "Embedding model '$modelName' returned HTTP ${response.statusCode()}: " +
                    response.body().take(ERROR_BODY_LIMIT),
            )
        }

        val vectors = try {
            json.decodeFromString<EmbedResponse>(response.body()).embeddings
        } catch (e: Exception) {
            throw EmbeddingException("Could not parse the reply from embedding model '$modelName'.", e)
        }

        if (vectors.size != batch.size) {
            throw EmbeddingException(
                "Embedding model '$modelName' returned ${vectors.size} vectors for ${batch.size} texts.",
            )
        }

        return vectors.map { it.toNormalizedVector() }
    }

    private fun List<Float>.toNormalizedVector(): FloatArray {
        if (size != dimension) {
            throw EmbeddingException(
                "Embedding model '$modelName' returned $size dimensions, but this store expects $dimension. " +
                    "Set EMBEDDING_DIMENSION to $size, or pick a different EMBEDDING_MODEL.",
            )
        }
        val vector = toFloatArray()
        val norm = sqrt(vector.sumOf { (it * it).toDouble() }).toFloat()
        if (norm > 0.0f) {
            for (index in vector.indices) vector[index] = vector[index] / norm
        }
        return vector
    }

    private fun installedModels(body: String): List<String> =
        Regex("\"name\"\\s*:\\s*\"([^\"]+)\"").findAll(body).map { it.groupValues[1] }.toList()

    /** `all-minilm` in the config should match the `all-minilm:latest` Ollama reports. */
    private fun String.matchesModel(wanted: String): Boolean =
        this == wanted || substringBefore(':') == wanted.substringBefore(':')

    @Serializable
    private data class EmbedRequest(
        val model: String,
        val input: List<String>,
    )

    @Serializable
    private data class EmbedResponse(
        @SerialName("embeddings") val embeddings: List<List<Float>> = emptyList(),
    )

    companion object {
        const val DEFAULT_BASE_URL = "http://localhost:11434"

        /** `all-MiniLM-L6-v2`, as Ollama names it. */
        const val DEFAULT_MODEL = "all-minilm"
        const val DEFAULT_DIMENSION = 384

        private const val DEFAULT_BATCH_SIZE = 32
        private const val DEFAULT_TIMEOUT_SECONDS = 120L
        private const val CONNECT_TIMEOUT_SECONDS = 5L
        private const val PROBE_TIMEOUT_SECONDS = 3L
        private const val ERROR_BODY_LIMIT = 300

        private val json = Json { ignoreUnknownKeys = true }
    }
}
