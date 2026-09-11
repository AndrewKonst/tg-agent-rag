package observability

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.message.Message
import harness.AgentTool
import harness.Llm
import harness.visibleText
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import java.util.UUID
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

private val logger = KotlinLogging.logger {}

/**
 * The run a measurement belongs to, carried in the coroutine context.
 *
 * The alternative was to thread a recorder through [harness.AgentLoop] and every
 * tool signature. The loop already defines what a turn is and nothing else needs to
 * know that measurement is happening, so the run identity travels with the coroutine
 * instead: [MeteredLlm] and [MeteredTool] pick it up where they run.
 *
 * One run is one sequential chain of calls — the loop awaits each step — so the
 * mutable state here is never touched concurrently.
 */
class RunScope(
    val runId: String,
    val agentId: String,
    val taskId: String?,
    val chatId: Long,
    val model: String,
) : AbstractCoroutineContextElement(RunScope) {

    companion object Key : CoroutineContext.Key<RunScope>

    val llmCalls = mutableListOf<LlmCallRecord>()
    val toolCalls = mutableListOf<ToolCallRecord>()

    /** The turn currently in flight: 1 during the first LLM call and its tools. */
    var turn: Int = 0
        private set

    /** The previous call's prompt, for measuring how much of this one repeats it. */
    var previousPrompt: List<Message> = emptyList()

    fun nextTurn(): Int = ++turn
}

/** The value a measured run produced, and the record of what it cost. */
data class MeteredRun<T>(val value: T, val record: RunRecord)

/**
 * Opens and closes measured runs, and persists them.
 *
 * A run that throws is still recorded: a failed expensive run is exactly the kind
 * this audit is looking for.
 */
class TokenMeter(
    private val store: ObservabilityStore?,
    private val prices: TokenPrices = TokenPrices(),
    private val clock: () -> Long = System::currentTimeMillis,
) {

    /**
     * Runs [block] as one measured agent run.
     *
     * [stopReasonOf] lets a caller that knows how the run ended — the loop reports a
     * [harness.StopReason] — record that instead of a bare "it returned".
     */
    suspend fun <T> measure(
        agentId: String,
        chatId: Long,
        model: String,
        taskId: String? = null,
        stopReasonOf: (T) -> String = { "COMPLETED" },
        block: suspend () -> T,
    ): MeteredRun<T> {
        val scope = RunScope(UUID.randomUUID().toString(), agentId, taskId, chatId, model)
        val startedAt = clock()

        val value = try {
            withContext(scope) { block() }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            persist(scope, startedAt, stopReason = "ERROR")
            throw e
        }

        return MeteredRun(value, persist(scope, startedAt, stopReasonOf(value)))
    }

    private fun persist(scope: RunScope, startedAt: Long, stopReason: String): RunRecord {
        val record = RunRecord(
            runId = scope.runId,
            agentId = scope.agentId,
            taskId = scope.taskId,
            chatId = scope.chatId,
            model = scope.model,
            startedAtMillis = startedAt,
            durationMillis = clock() - startedAt,
            turns = scope.llmCalls.size,
            toolCalls = scope.toolCalls.size,
            inputTokens = scope.llmCalls.sumOf { it.inputTokens },
            outputTokens = scope.llmCalls.sumOf { it.outputTokens },
            reasoningTokens = scope.llmCalls.sumOf { it.reasoningTokens },
            cachedTokens = scope.llmCalls.sumOf { it.cachedTokens },
            reusedInputTokens = scope.llmCalls.sumOf { it.reusedInputTokens },
            estimatedCostUsd = scope.llmCalls.sumOf { it.estimatedCostUsd },
            stopReason = stopReason,
            succeeded = null,
            context = scope.llmCalls.fold(ContextBreakdown()) { total, call -> total + call.context },
        )

        try {
            store?.save(record, scope.llmCalls, scope.toolCalls)
        } catch (e: Throwable) {
            // Losing a measurement must never cost the user their answer.
            logger.error(e) { "Failed to store the trace of run ${scope.runId}" }
        }
        return record
    }

    /** Wraps [llm] so every call to the model is measured. */
    fun meter(llm: Llm): Llm = MeteredLlm(llm, prices, clock)

    /** Wraps [tool] so every call to it is measured. */
    fun meter(tool: AgentTool): AgentTool = MeteredTool(tool, clock)
}

/**
 * Middleware around one call to the model.
 *
 * Records what the provider reports, and computes the two things it does not: the
 * reasoning tokens we paid for and threw away, and the share of the prompt that
 * repeats the previous turn verbatim.
 */
private class MeteredLlm(
    private val delegate: Llm,
    private val prices: TokenPrices,
    private val clock: () -> Long,
) : Llm {

    override suspend fun complete(
        messages: List<Message>,
        tools: List<ToolDescriptor>,
    ): Message.Assistant {
        val scope = currentCoroutineContext()[RunScope]
        val startedAt = clock()

        val reply = delegate.complete(messages, tools)

        val latency = clock() - startedAt
        if (scope == null) return reply

        val meta = reply.metaInfo
        val reported = meta.inputTokensCount != null || meta.outputTokensCount != null
        val inputTokens = meta.inputTokensCount
            ?: PromptAudit.breakdown(messages, measuredInputTokens = null).total
        val outputTokens = meta.outputTokensCount
            ?: TokenEstimator.countTokens(reply.textContent())

        val record = LlmCallRecord(
            runId = scope.runId,
            turn = scope.nextTurn(),
            timestampMillis = startedAt,
            model = meta.modelId ?: scope.model,
            inputTokens = inputTokens,
            outputTokens = outputTokens,
            // Reasoning is part of the output, so it cannot exceed it — our estimate can,
            // by a few percent, since the output count is the provider's and this one is
            // ours. Capping keeps the report from claiming more thinking than the model
            // produced; the estimate is only ever used to say how much of the output was
            // thrown away, never as a total.
            reasoningTokens = PromptAudit.reasoningTokens(reply).coerceAtMost(outputTokens),
            // No provider on this project reports a cache hit; see TokenRecords.
            cachedTokens = 0,
            reusedInputTokens = PromptAudit.reusedPrefixTokens(scope.previousPrompt, messages)
                .coerceAtMost(inputTokens),
            latencyMillis = latency,
            estimatedCostUsd = prices.cost(scope.model, inputTokens, outputTokens),
            context = PromptAudit.breakdown(messages, inputTokens),
            tokensReportedByProvider = reported,
        )

        scope.llmCalls += record
        // A snapshot, not the caller's list: the loop appends to the same instance it
        // passes in, so keeping the reference would make turn N+1 look identical to
        // turn N and report the whole prompt as repeated.
        scope.previousPrompt = messages.toList()
        logger.debug {
            "run=${scope.runId} turn=${record.turn}: in=${record.inputTokens} " +
                "(reused ${record.reusedInputTokens}) out=${record.outputTokens} " +
                "reasoning=${record.reasoningTokens} ${latency}ms"
        }
        return reply
    }
}

/** Middleware around one tool call. */
private class MeteredTool(
    private val delegate: AgentTool,
    private val clock: () -> Long,
) : AgentTool {

    override val descriptor: ToolDescriptor get() = delegate.descriptor

    override suspend fun execute(args: JsonObject): String {
        val scope = currentCoroutineContext()[RunScope]
        val startedAt = clock()

        val output = try {
            delegate.execute(args)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            scope?.record(startedAt, args, output = e.message.orEmpty(), failed = true)
            throw e
        }

        scope?.record(startedAt, args, output, failed = false)
        return output
    }

    private fun RunScope.record(startedAt: Long, args: JsonObject, output: String, failed: Boolean) {
        val serialisedArgs = args.toString()
        toolCalls += ToolCallRecord(
            runId = runId,
            // The tool runs inside the turn whose LLM call asked for it.
            turn = turn,
            timestampMillis = startedAt,
            toolName = descriptor.name,
            inputSize = serialisedArgs.length,
            outputSize = output.length,
            outputTokens = TokenEstimator.countTokens(output),
            durationMillis = clock() - startedAt,
            failed = failed,
        )
    }
}
