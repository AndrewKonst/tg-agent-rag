package benchmark

import agent.AgentFactory
import config.AppConfig
import config.EmbeddingProvider
import conversation.InMemoryConversationStore
import io.github.oshai.kotlinlogging.KotlinLogging
import observability.ObservabilityStore
import observability.TokenMeter
import observability.TokenPrices
import observability.TokenReport
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.HashingEmbeddingService
import rag.OllamaEmbeddingService
import rag.SqliteRagStore
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

/**
 * Runs [BenchmarkTasks] against the real agent and records what each task cost.
 *
 * Everything about a run is real except Telegram: the same factory, the same loop,
 * the same tools, the same local model. Only the transport is missing, because a
 * benchmark that measures a mock measures the mock.
 *
 * Each task gets a fresh chat id and a fresh history, so one task's leftovers cannot
 * inflate — or rescue — the next one's context.
 */
class BenchmarkRunner(
    private val config: AppConfig,
    private val observability: ObservabilityStore,
    private val prices: TokenPrices,
    private val ragDatabase: Path = Files.createTempFile("benchmark-rag", ".db"),
) {

    suspend fun run(variant: String, tasks: List<BenchmarkTask> = BenchmarkTasks.all): BenchmarkResult {
        val embeddings = if (config.embeddingProvider == EmbeddingProvider.OLLAMA) {
            OllamaEmbeddingService(
                baseUrl = config.embeddingBaseUrl,
                modelName = config.embeddingModel,
                dimension = config.embeddingDimension,
            ).takeIf { it.isAvailable() } ?: HashingEmbeddingService()
        } else {
            HashingEmbeddingService()
        }

        return SqliteRagStore(
            databasePath = ragDatabase,
            sqliteVecExtensionPath = config.sqliteVecExtensionPath,
            embeddingDimension = embeddings.dimension,
            embeddingModel = embeddings.modelName,
        ).use { ragStore ->
            val search = DocumentSearchService(store = ragStore, embeddings = embeddings)
            val meter = TokenMeter(observability, prices)
            val outcomes = mutableListOf<TaskOutcome>()

            tasks.forEachIndexed { index, task ->
                // A chat per task: the documents belong to whoever uploaded them, so each
                // benchmark chat has to be given its own copy of the corpus.
                val chatId = CHAT_ID_BASE + index
                DocumentIndexingService(embeddings = embeddings, store = ragStore)
                    .indexTestDocuments(chatId)

                val agent = AgentFactory.create(
                    config = config,
                    store = InMemoryConversationStore(),
                    sandbox = null,
                    documentSearchService = search,
                    meter = meter,
                )

                task.setUpTurns.forEach { turn ->
                    runCatching { agent.ask(chatId, turn) }
                        .onFailure { logger.warn(it) { "Set-up turn failed for ${task.id}" } }
                }

                val outcome = runCatching { agent.ask(chatId, task.prompt) }
                val answer = outcome.getOrElse { error ->
                    logger.error(error) { "Task ${task.id} failed" }
                    ""
                }

                // The run the meter just stored for this chat is this task's run.
                val record = observability.runs(limit = 200).firstOrNull { it.chatId == chatId }
                val toolsCalled = record
                    ?.let { observability.toolCalls(it.runId).map { call -> call.toolName }.toSet() }
                    .orEmpty()
                val succeeded = task.isSatisfiedBy(answer, toolsCalled)

                if (record != null) {
                    observability.markSucceeded(record.runId, succeeded)
                    observability.labelVariant(record.runId, variant)
                }

                logger.info {
                    "[$variant] ${task.id}: ${if (succeeded) "ok" else "FAILED"}" +
                        (record?.let { ", ${it.totalTokens} tokens, ${it.turns} turns" } ?: "")
                }
                outcomes += TaskOutcome(task, answer, succeeded, toolsCalled)
            }

            BenchmarkResult(variant, outcomes)
        }
    }

    private companion object {
        /** Well clear of any real Telegram chat id, so traces never mix. */
        const val CHAT_ID_BASE = 900_000L
    }
}

data class TaskOutcome(
    val task: BenchmarkTask,
    val answer: String,
    val succeeded: Boolean,
    val toolsCalled: Set<String>,
)

data class BenchmarkResult(val variant: String, val outcomes: List<TaskOutcome>) {
    val successRate: Double
        get() = if (outcomes.isEmpty()) 0.0 else outcomes.count { it.succeeded }.toDouble() / outcomes.size

    /** Task-by-task, so a failure can be read rather than inferred from a percentage. */
    fun summary(): String = buildString {
        appendLine("Benchmark: $variant")
        appendLine("─".repeat(56))
        outcomes.forEach { outcome ->
            appendLine(
                "%-26s %-7s %s".format(
                    outcome.task.id,
                    if (outcome.succeeded) "ok" else "FAILED",
                    outcome.toolsCalled.joinToString(),
                ),
            )
        }
        appendLine()
        append("Success rate: ${(successRate * 100).toInt()}% (${outcomes.count { it.succeeded }}/${outcomes.size})")
    }
}

/**
 * Runs the benchmark and prints the dashboard.
 *
 * ```
 * ./gradlew benchmark --args="baseline"
 * ./gradlew benchmark --args="optimized"
 * ./gradlew benchmark --args="report"        # compares the two
 * ```
 */
object BenchmarkMain {

    @JvmStatic
    fun main(args: Array<String>) {
        val variant = args.firstOrNull() ?: "baseline"
        val config = AppConfig.load()
        val prices = TokenPrices(referenceModel = config.costReferenceModel)

        ObservabilityStore(Path.of(config.observabilityDbPath)).use { store ->
            if (variant == "report") {
                printReport(store, prices)
                return
            }

            val result = kotlinx.coroutines.runBlocking {
                BenchmarkRunner(config, store, prices).run(variant)
            }

            println()
            println(result.summary())
            println()
            println(TokenReport.dashboard(store.runs(variant), store.toolCalls(), prices, title = "VARIANT: $variant"))
        }
    }

    private fun printReport(store: ObservabilityStore, prices: TokenPrices) {
        val baseline = store.runs("baseline")
        val optimized = store.runs("optimized")

        println(TokenReport.dashboard(baseline, store.toolCalls(), prices, title = "BASELINE"))
        println()
        if (optimized.isEmpty()) {
            println("No optimized runs recorded yet — run: ./gradlew benchmark --args=\"optimized\"")
            return
        }
        println(TokenReport.comparison(baseline, optimized))
    }
}
