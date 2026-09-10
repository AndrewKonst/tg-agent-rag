package observability

import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import harness.visibleText

/**
 * Reads a prompt and works out what it is made of.
 *
 * Two things the provider will not tell us are computed here.
 *
 * **The split.** Providers report one number for input tokens. To say *why* a prompt
 * is large, we estimate each message separately with [TokenEstimator] and scale the
 * parts so they add up to the number the provider reported. The total is therefore
 * exact and the proportions are approximate — which is how reports present it.
 *
 * **The reuse.** A prompt on turn 5 is mostly the prompt from turn 4 with more
 * appended. That shared prefix is paid for again on every turn, and is what a
 * provider cache would have served. Comparing prompts message by message from the
 * start gives the size of that prefix exactly, in messages, and approximately in
 * tokens.
 */
object PromptAudit {

    /**
     * Classifies [messages] and returns an estimated breakdown, scaled to
     * [measuredInputTokens] when that is known.
     *
     * The last user message that a person actually typed is the task; earlier ones are
     * history. A user message carrying tool results is not a person talking — it is the
     * loop feeding results back — so it counts as tool output wherever it sits.
     */
    fun breakdown(messages: List<Message>, measuredInputTokens: Int?): ContextBreakdown {
        val lastTypedUserIndex = messages.indexOfLast { it.isTypedByUser() }

        var breakdown = ContextBreakdown()
        messages.forEachIndexed { index, message ->
            val tokens = TokenEstimator.countTokens(message.auditText())
            breakdown = breakdown + when {
                message is Message.System -> ContextBreakdown(systemPrompt = tokens)
                message.carriesToolResults() -> ContextBreakdown(toolOutputs = tokens)
                index == lastTypedUserIndex -> ContextBreakdown(userTask = tokens)
                else -> ContextBreakdown(conversationHistory = tokens)
            }
        }

        return if (measuredInputTokens == null) breakdown else breakdown.scaledTo(measuredInputTokens)
    }

    /**
     * Estimated tokens of the prefix [current] shares with [previous].
     *
     * Only a prefix counts: providers cache from the start of the prompt, and a
     * message that matches after a mismatch would not have been served from cache.
     */
    fun reusedPrefixTokens(previous: List<Message>, current: List<Message>): Int {
        var tokens = 0
        for ((index, message) in current.withIndex()) {
            val earlier = previous.getOrNull(index) ?: break
            if (!earlier.sameContentAs(message)) break
            tokens += TokenEstimator.countTokens(message.auditText())
        }
        return tokens
    }

    /**
     * Thinking the model was paid for and the user never sees.
     *
     * It arrives two ways and both count: as a [MessagePart.Reasoning] part, which is
     * how a provider that separates reasoning reports it, and inline in the text as a
     * `<think>` block, which is what local qwen3 does and what [visibleText] strips.
     */
    fun reasoningTokens(reply: Message.Assistant): Int {
        val separateParts = reply.parts
            .filterIsInstance<MessagePart.Reasoning>()
            .sumOf { TokenEstimator.countTokens(it.content.joinToString("\n")) }
        val strippedFromText = TokenEstimator.countTokens(reply.textContent()) -
            TokenEstimator.countTokens(reply.visibleText())
        return separateParts + strippedFromText.coerceAtLeast(0)
    }

    private fun ContextBreakdown.scaledTo(target: Int): ContextBreakdown {
        val estimated = total
        if (estimated <= 0 || target <= 0) return ContextBreakdown()

        val factor = target.toDouble() / estimated
        val scaled = intArrayOf(
            (systemPrompt * factor).toInt(),
            (userTask * factor).toInt(),
            (conversationHistory * factor).toInt(),
            (toolOutputs * factor).toInt(),
        )

        // The parts must sum to the measured total, and the remainder goes to the
        // largest of them — never to a fixed one, which would put stray tokens into a
        // part that is genuinely empty (history on the first turn, say).
        val remainder = target - scaled.sum()
        if (remainder != 0) {
            val largest = scaled.indices.maxBy { scaled[it] }
            scaled[largest] = (scaled[largest] + remainder).coerceAtLeast(0)
        }

        return ContextBreakdown(
            systemPrompt = scaled[0],
            userTask = scaled[1],
            conversationHistory = scaled[2],
            toolOutputs = scaled[3],
        )
    }

    /** Every part of a message that reaches the provider, tool calls included. */
    private fun Message.auditText(): String = parts.joinToString("\n") { part ->
        when (part) {
            is MessagePart.Text -> part.text
            is MessagePart.Reasoning -> part.content.joinToString("\n")
            is MessagePart.Tool.Call -> "${part.tool}(${part.args})"
            is MessagePart.Tool.Result -> part.output
            else -> ""
        }
    }

    private fun Message.carriesToolResults(): Boolean =
        parts.any { it is MessagePart.Tool.Result }

    private fun Message.isTypedByUser(): Boolean =
        this is Message.User && !carriesToolResults()

    private fun Message.sameContentAs(other: Message): Boolean =
        this::class == other::class && auditText() == other.auditText()
}

/**
 * Estimates tokens from text, for the two jobs where nobody reports them: the split
 * of a prompt into its parts, and the size of a tool's output.
 *
 * Deliberately simple — words plus a share of the punctuation, which lands within a
 * few percent of real tokenizers on prose and errs high on code. It is never used
 * where a provider number exists, and never used for a total that is reported as
 * measured.
 */
object TokenEstimator {

    fun countTokens(text: String): Int {
        if (text.isBlank()) return 0
        val words = WORD.findAll(text).count()
        val symbols = SYMBOL.findAll(text).count()
        // ~1.3 tokens per word covers sub-word splits; symbols are usually a token each.
        return (words * WORD_FACTOR + symbols).toInt().coerceAtLeast(1)
    }

    private const val WORD_FACTOR = 1.3
    private val WORD = Regex("[\\p{L}\\p{N}]+")
    private val SYMBOL = Regex("[^\\p{L}\\p{N}\\s]")
}
