package benchmark

import agent.AgentFactory
import config.AppConfig
import config.EmbeddingProvider
import conversation.InMemoryConversationStore
import io.github.oshai.kotlinlogging.KotlinLogging
import observability.ObservabilityStore
import observability.RunRecord
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
import kotlin.io.path.writeText

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
            var failures = 0

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
                if (outcome.isFailure) failures++

                // The run the meter just stored for this chat is this task's run.
                val record = observability.runs(limit = 200).firstOrNull { it.chatId == chatId }
                val toolsCalled = record
                    ?.let { observability.toolCalls(it.runId).map { call -> call.toolName }.toSet() }
                    .orEmpty()
                val succeeded = task.isSatisfiedBy(answer, toolsCalled)

                if (record != null) {
                    observability.annotate(record.runId, task.id, variant, succeeded)
                }

                logger.info {
                    "[$variant] ${task.id}: ${if (succeeded) "ok" else "FAILED"}" +
                        (record?.let { ", ${it.totalTokens} tokens, ${it.turns} turns" } ?: "")
                }
                outcomes += TaskOutcome(task, answer, succeeded, toolsCalled)
            }

            // Every task erroring is not a 0% score, it is a broken setup — usually the
            // model being unreachable. Saying so beats writing an arm of zeroes that a
            // before/after comparison would then treat as a measurement.
            if (failures == tasks.size && tasks.isNotEmpty()) {
                throw BenchmarkSetupException(
                    "All ${tasks.size} tasks failed to reach the model. Check that the LLM in " +
                        "LLM_PROVIDER/LLM_MODEL is running, then re-run this arm.",
                )
            }

            BenchmarkResult(variant, outcomes)
        }
    }

    private companion object {
        /** Well clear of any real Telegram chat id, so traces never mix. */
        const val CHAT_ID_BASE = 900_000L
    }
}

/** Raised when the benchmark could not run at all, as opposed to running badly. */
class BenchmarkSetupException(message: String) : RuntimeException(message)

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
 * ./gradlew benchmark --args="export"        # writes reports/ for the repository
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
            if (variant == "export") {
                export(store, prices)
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

    /**
     * Writes the measurements out as files the repository can keep.
     *
     * The trace database lives under `data/`, which is git-ignored and is wiped by a
     * re-run — so a "before" that exists only there is a "before" that will be gone
     * exactly when the "after" needs it. The rendered dashboard makes the result
     * readable in a review; the CSV makes it checkable.
     */
    private fun export(store: ObservabilityStore, prices: TokenPrices) {
        val directory = Path.of(REPORTS_DIR)
        Files.createDirectories(directory)

        listOf("baseline", "optimized").forEach { variant ->
            val runs = store.runs(variant)
            if (runs.isEmpty()) return@forEach

            val runIds = runs.map { it.runId }.toSet()
            val tools = store.toolCalls().filter { it.runId in runIds }

            directory.resolve("$variant-dashboard.txt").writeText(
                TokenReport.dashboard(runs, tools, prices, title = variant.uppercase()),
            )
            directory.resolve("$variant-runs.csv").writeText(runsCsv(runs))
            directory.resolve("$variant-calls.csv").writeText(callsCsv(store, runs))
            println("Wrote ${runs.size} $variant run(s) to $REPORTS_DIR/")
        }

        val baseline = store.runs("baseline")
        val optimized = store.runs("optimized")
        if (baseline.isNotEmpty() && optimized.isNotEmpty()) {
            directory.resolve("comparison.txt").writeText(TokenReport.comparison(baseline, optimized))
            println("Wrote the before/after comparison to $REPORTS_DIR/comparison.txt")
        }
    }

    private fun runsCsv(runs: List<RunRecord>): String = buildString {
        appendLine(
            "task_id,turns,tool_calls,input_tokens,output_tokens,reasoning_tokens," +
                "reused_input_tokens,cost_usd,duration_ms,stop_reason,succeeded," +
                "ctx_system,ctx_task,ctx_history,ctx_tools",
        )
        runs.sortedBy { it.startedAtMillis }.forEach { run ->
            appendLine(
                listOf(
                    run.taskId ?: run.runId,
                    run.turns,
                    run.toolCalls,
                    run.inputTokens,
                    run.outputTokens,
                    run.reasoningTokens,
                    run.reusedInputTokens,
                    "%.6f".format(run.estimatedCostUsd),
                    run.durationMillis,
                    run.stopReason,
                    run.succeeded?.toString() ?: "",
                    run.context.systemPrompt,
                    run.context.userTask,
                    run.context.conversationHistory,
                    run.context.toolOutputs,
                ).joinToString(","),
            )
        }
    }

    private fun callsCsv(store: ObservabilityStore, runs: List<RunRecord>): String = buildString {
        appendLine("task_id,turn,kind,name,input_tokens,output_tokens,reasoning_tokens,reused_input_tokens,duration_ms")
        runs.sortedBy { it.startedAtMillis }.forEach { run ->
            val task = run.taskId ?: run.runId
            store.llmCalls(run.runId).forEach { call ->
                appendLine(
                    "$task,${call.turn},llm,${call.model},${call.inputTokens},${call.outputTokens}," +
                        "${call.reasoningTokens},${call.reusedInputTokens},${call.latencyMillis}",
                )
            }
            store.toolCalls(run.runId).forEach { call ->
                appendLine(
                    "$task,${call.turn},tool,${call.toolName},,${call.outputTokens},,,${call.durationMillis}",
                )
            }
        }
    }

    private const val REPORTS_DIR = "reports"

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
