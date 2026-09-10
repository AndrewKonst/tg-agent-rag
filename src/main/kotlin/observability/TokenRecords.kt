package observability

/**
 * What one call to the model cost.
 *
 * [inputTokens] and [outputTokens] come from the provider when it reports them.
 * The two fields the provider never reports are computed here instead, and are
 * named for what they actually are:
 *
 * - [reasoningTokens] — the `<think>` block the model emitted and we stripped before
 *   showing or storing the answer. Paid for, then thrown away.
 * - [reusedInputTokens] — how much of this call's prompt was a byte-identical prefix
 *   of the previous call's prompt in the same run. This is the part a provider cache
 *   would have served, and the part we are paying for twice without one.
 */
data class LlmCallRecord(
    val runId: String,
    val turn: Int,
    val timestampMillis: Long,
    val model: String,
    val inputTokens: Int,
    val outputTokens: Int,
    val reasoningTokens: Int,
    val cachedTokens: Int,
    val reusedInputTokens: Int,
    val latencyMillis: Long,
    val estimatedCostUsd: Double,
    /** Estimated split of [inputTokens] across what the prompt is made of. */
    val context: ContextBreakdown,
    /** False when the numbers are our own estimate because the provider reported none. */
    val tokensReportedByProvider: Boolean,
) {
    val totalTokens: Int get() = inputTokens + outputTokens

    /** Input tokens the model had not seen before on this run. */
    val newInputTokens: Int get() = (inputTokens - reusedInputTokens).coerceAtLeast(0)
}

/**
 * Where a prompt's input tokens went.
 *
 * The total is measured — it is the provider's `input_tokens` — while the split
 * between the parts is estimated per message and scaled to that total. Reports say
 * so rather than presenting the split as exact.
 */
data class ContextBreakdown(
    val systemPrompt: Int = 0,
    val userTask: Int = 0,
    val conversationHistory: Int = 0,
    val toolOutputs: Int = 0,
) {
    val total: Int get() = systemPrompt + userTask + conversationHistory + toolOutputs

    operator fun plus(other: ContextBreakdown) = ContextBreakdown(
        systemPrompt = systemPrompt + other.systemPrompt,
        userTask = userTask + other.userTask,
        conversationHistory = conversationHistory + other.conversationHistory,
        toolOutputs = toolOutputs + other.toolOutputs,
    )
}

/** What one tool call cost. Sizes are characters; tokens are estimated from the text. */
data class ToolCallRecord(
    val runId: String,
    val turn: Int,
    val timestampMillis: Long,
    val toolName: String,
    val inputSize: Int,
    val outputSize: Int,
    val outputTokens: Int,
    val durationMillis: Long,
    val failed: Boolean,
)

/**
 * One agent run, from the user's message to the answer.
 *
 * [taskId] is the benchmark task when a run comes from the harness, and null for a
 * real Telegram request. [succeeded] is only known for benchmark runs, where there is
 * a predicate to check; a real request records null rather than a guess.
 */
data class RunRecord(
    val runId: String,
    val agentId: String,
    val taskId: String?,
    val chatId: Long,
    val model: String,
    val startedAtMillis: Long,
    val durationMillis: Long,
    val turns: Int,
    val toolCalls: Int,
    val inputTokens: Int,
    val outputTokens: Int,
    val reasoningTokens: Int,
    val cachedTokens: Int,
    val reusedInputTokens: Int,
    val estimatedCostUsd: Double,
    val stopReason: String,
    val succeeded: Boolean?,
    val context: ContextBreakdown,
) {
    val totalTokens: Int get() = inputTokens + outputTokens
    val newInputTokens: Int get() = (inputTokens - reusedInputTokens).coerceAtLeast(0)
}
