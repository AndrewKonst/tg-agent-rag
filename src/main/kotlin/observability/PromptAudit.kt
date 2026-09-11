package observability

import ai.koog.agents.core.tools.ToolDescriptor
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
    fun breakdown(
        messages: List<Message>,
        tools: List<ToolDescriptor> = emptyList(),
        measuredInputTokens: Int? = null,
    ): ContextBreakdown {
        val lastTypedUserIndex = messages.indexOfLast { it.isTypedByUser() }

        var breakdown = ContextBreakdown(toolSchemas = schemaTokens(tools))
        messages.forEachIndexed { index, message ->
            val tokens = TokenEstimator.countTokens(message.auditText())
            breakdown = breakdown + when {
                message is Message.System -> ContextBreakdown(systemPrompt = tokens)
                message.carriesToolResults() -> ContextBreakdown(toolOutputs = tokens)
                index == lastTypedUserIndex -> ContextBreakdown(userTask = tokens)
                else -> ContextBreakdown(conversationHistory = tokens)
            }
        }

        if (measuredInputTokens == null) return breakdown
        return breakdown.reconciledWith(measuredInputTokens)
    }

    /**
     * Estimates what the tool definitions cost.
     *
     * Providers serialise a tool into JSON — name, description, a schema per
     * parameter — and every call carries all of them. Reconstructing that shape is
     * closer than counting the description alone, and whatever the estimate misses
     * lands in [ContextBreakdown.overhead] rather than in a category it did not come
     * from.
     */
    private fun schemaTokens(tools: List<ToolDescriptor>): Int {
        if (tools.isEmpty()) return 0
        val json = tools.joinToString(",") { tool ->
            val parameters = (tool.requiredParameters + tool.optionalParameters).joinToString(",") { parameter ->
                """"${parameter.name}":{"type":"${parameter.type}","description":"${parameter.description}"}"""
            }
            """{"type":"function","function":{"name":"${tool.name}","description":"${tool.description}",""" +
                """"parameters":{"type":"object","properties":{$parameters},""" +
                """"required":[${tool.requiredParameters.joinToString(",") { "\"${it.name}\"" }}]}}}"""
        }
        return TokenEstimator.countTokens(json)
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

    /**
     * Makes the parts add up to what the provider charged for.
     *
     * Usually the estimate falls short — the chat template's own tokens are real and
     * invisible to us — and the shortfall becomes [ContextBreakdown.overhead]. Naming
     * it is the point: the earlier version spread it across the categories in
     * proportion, which quadrupled the apparent size of the system prompt.
     *
     * When the estimate overshoots instead, there is nothing to attribute it to, so
     * the parts are scaled down to fit.
     */
    private fun ContextBreakdown.reconciledWith(measured: Int): ContextBreakdown {
        if (measured <= 0) return ContextBreakdown()
        val estimated = total
        if (estimated <= 0) return ContextBreakdown(overhead = measured)

        if (estimated <= measured) return copy(overhead = measured - estimated)

        val factor = measured.toDouble() / estimated
        val scaled = intArrayOf(
            (systemPrompt * factor).toInt(),
            (userTask * factor).toInt(),
            (conversationHistory * factor).toInt(),
            (toolOutputs * factor).toInt(),
            (toolSchemas * factor).toInt(),
        )
        val remainder = measured - scaled.sum()
        if (remainder != 0) {
            val largest = scaled.indices.maxBy { scaled[it] }
            scaled[largest] = (scaled[largest] + remainder).coerceAtLeast(0)
        }

        return ContextBreakdown(
            systemPrompt = scaled[0],
            userTask = scaled[1],
            conversationHistory = scaled[2],
            toolOutputs = scaled[3],
            toolSchemas = scaled[4],
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
