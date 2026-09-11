package observability

import kotlin.math.roundToInt

/**
 * Renders what the store holds as plain text.
 *
 * Plain text because both readers are text: a terminal, and a Telegram message. No
 * chart is worth a dependency the bot would then have to carry.
 */
object TokenReport {

    private const val WIDTH = 40

    /** The assignment's targets: at least 30% fewer tokens, at most 2pp worse success. */
    private const val TOKEN_TARGET_PERCENT = 30.0
    private const val SUCCESS_TOLERANCE_PP = 2.0

    /** Slack for double arithmetic, not for the target itself. */
    private const val TOLERANCE = 1e-6

    /** Smallest gap between a label and its value, so they never touch. */
    private const val MIN_GAP = 2

    /** The whole picture: totals, averages, and where the tokens went. */
    fun dashboard(
        runs: List<RunRecord>,
        toolCalls: List<ToolCallRecord>,
        prices: TokenPrices = TokenPrices(),
        title: String = "AI AGENT",
    ): String = buildString {
        appendLine(title)
        appendLine("─".repeat(WIDTH))
        appendLine()

        if (runs.isEmpty()) {
            append("No runs recorded yet.")
            return@buildString
        }

        val input = runs.sumOf { it.inputTokens.toLong() }
        val output = runs.sumOf { it.outputTokens.toLong() }
        val reused = runs.sumOf { it.reusedInputTokens.toLong() }
        val reasoning = runs.sumOf { it.reasoningTokens.toLong() }

        appendLine(row("Tasks completed", runs.size.toString()))
        appendLine()
        appendLine("Total tokens")
        appendLine(row("  Input", compact(input), indent = 2))
        appendLine(row("  Output", compact(output), indent = 2))
        appendLine(row("  Reused input", compact(reused), indent = 2))
        appendLine(row("  Reasoning (discarded)", compact(reasoning), indent = 2))
        appendLine()
        appendLine(row("Estimated cost", money(runs.sumOf { it.estimatedCostUsd })))
        appendLine(row("  billed as", prices.referenceModel, indent = 2))
        appendLine()
        appendLine("Average task")
        appendLine(row("  Tokens", compact((input + output) / runs.size), indent = 2))
        appendLine(row("  Turns", "%.1f".format(runs.map { it.turns }.average()), indent = 2))
        appendLine(row("  Tool calls", "%.1f".format(runs.map { it.toolCalls }.average()), indent = 2))
        appendLine(row("  Latency", "%.1fs".format(runs.map { it.durationMillis }.average() / 1000), indent = 2))
        appendLine()
        appendLine(row("Repeated input share", percent(reused, input)))
        successRate(runs)?.let { appendLine(row("Success rate", percent(it))) }

        val byTool = toolCalls.groupBy { it.toolName }
        if (byTool.isNotEmpty()) {
            val totalToolTokens = toolCalls.sumOf { it.outputTokens.toLong() }.coerceAtLeast(1)
            appendLine()
            appendLine("Most expensive tools:")
            byTool.entries
                .sortedByDescending { entry -> entry.value.sumOf { it.outputTokens.toLong() } }
                .forEach { (name, calls) ->
                    val tokens = calls.sumOf { it.outputTokens.toLong() }
                    appendLine(
                        row(
                            "  $name",
                            "${percent(tokens, totalToolTokens)}  (${compact(tokens)}, ${calls.size} calls)",
                            indent = 2,
                        ),
                    )
                }
        }

        appendLine()
        appendLine("Context growth:")
        append(contextBars(runs))
    }

    /** Turn by turn for one run — where in a single task the tokens went. */
    fun timeline(
        run: RunRecord,
        llmCalls: List<LlmCallRecord>,
        toolCalls: List<ToolCallRecord>,
    ): String = buildString {
        appendLine("Task ${run.taskId ?: run.runId.take(8)}")
        appendLine("─".repeat(WIDTH))
        appendLine()

        val toolsByTurn = toolCalls.groupBy { it.turn }
        llmCalls.sortedBy { it.turn }.forEach { call ->
            appendLine(
                "Turn %-3d LLM   %8s tokens in, %s out%s".format(
                    call.turn,
                    compact(call.inputTokens.toLong()),
                    compact(call.outputTokens.toLong()),
                    if (call.reusedInputTokens > 0) {
                        "  (${percent(call.reusedInputTokens.toLong(), call.inputTokens.toLong())} repeated)"
                    } else {
                        ""
                    },
                ),
            )
            toolsByTurn[call.turn].orEmpty().forEach { tool ->
                // Never truncated: the tool's name is the whole point of the line.
                appendLine(
                    "         %-18s %6s tokens out, %dms%s".format(
                        tool.toolName,
                        compact(tool.outputTokens.toLong()),
                        tool.durationMillis,
                        if (tool.failed) "  (failed)" else "",
                    ),
                )
            }
        }

        appendLine()
        appendLine(row("Total", "${compact(run.totalTokens.toLong())} tokens"))
        appendLine(row("  new input", compact(run.newInputTokens.toLong()), indent = 2))
        appendLine(row("  repeated input", compact(run.reusedInputTokens.toLong()), indent = 2))
        appendLine(row("Cost", money(run.estimatedCostUsd)))
        appendLine(row("Stopped because", run.stopReason))

        val worst = llmCalls.maxByOrNull { it.inputTokens }
        if (worst != null) {
            appendLine()
            appendLine("Most expensive turn: ${worst.turn} → ${compact(worst.inputTokens.toLong())} input tokens")
        }
    }

    /**
     * The before/after table the assignment asks for.
     *
     * Deltas are what the reader is looking for, so they are computed here rather
     * than left as two columns to subtract by eye.
     */
    fun comparison(baseline: List<RunRecord>, optimized: List<RunRecord>): String = buildString {
        appendLine("BEFORE / AFTER")
        appendLine("─".repeat(56))
        appendLine("%-24s %10s %10s %9s".format("", "baseline", "optimized", "change"))
        appendLine()

        fun line(label: String, of: (List<RunRecord>) -> Double, format: (Double) -> String) {
            val before = of(baseline)
            val after = of(optimized)
            val change = if (before == 0.0) 0.0 else (after - before) / before * 100
            appendLine(
                "%-24s %10s %10s %8.1f%%".format(label, format(before), format(after), change),
            )
        }

        val tokens = { runs: List<RunRecord> -> runs.averageOrZero { it.totalTokens.toDouble() } }
        line("Tokens per task", tokens) { compact(it.roundToInt().toLong()) }
        line("  input", { runs -> runs.averageOrZero { it.inputTokens.toDouble() } }) { compact(it.roundToInt().toLong()) }
        line("  output", { runs -> runs.averageOrZero { it.outputTokens.toDouble() } }) { compact(it.roundToInt().toLong()) }
        line("  repeated input", { runs -> runs.averageOrZero { it.reusedInputTokens.toDouble() } }) { compact(it.roundToInt().toLong()) }
        line("Cost per task, USD", { runs -> runs.averageOrZero { it.estimatedCostUsd } }) { "%.4f".format(it) }
        line("Turns per task", { runs -> runs.averageOrZero { it.turns.toDouble() } }) { "%.1f".format(it) }
        line("Tool calls per task", { runs -> runs.averageOrZero { it.toolCalls.toDouble() } }) { "%.1f".format(it) }
        line("Latency per task, s", { runs -> runs.averageOrZero { it.durationMillis / 1000.0 } }) { "%.1f".format(it) }

        val beforeSuccess = successRate(baseline)
        val afterSuccess = successRate(optimized)
        if (beforeSuccess != null && afterSuccess != null) {
            appendLine()
            appendLine(
                "%-24s %10s %10s %8.1f pp".format(
                    "Success rate",
                    percent(beforeSuccess),
                    percent(afterSuccess),
                    (afterSuccess - beforeSuccess) * 100,
                ),
            )
        }

        appendLine()
        val tokenChange = tokens(baseline).let { before ->
            if (before == 0.0) 0.0 else (tokens(optimized) - before) / before * 100
        }
        val successChange = if (beforeSuccess != null && afterSuccess != null) {
            (afterSuccess - beforeSuccess) * 100
        } else {
            null
        }
        appendLine("Target: tokens −30% or better, success rate no worse than −2pp")
        append(
            when {
                // The thresholds are inclusive — exactly -30% passes, exactly -2pp passes —
                // and a rate computed in doubles lands a hair either side of a round number.
                tokenChange > -TOKEN_TARGET_PERCENT + TOLERANCE ->
                    "NOT MET: tokens changed %.1f%%".format(tokenChange)
                successChange != null && successChange < -SUCCESS_TOLERANCE_PP - TOLERANCE ->
                    "NOT MET: success rate fell %.1f pp".format(successChange)
                else -> "MET: tokens %.1f%%%s".format(
                    tokenChange,
                    successChange?.let { ", success rate %+.1f pp".format(it) } ?: "",
                )
            },
        )
    }

    private fun contextBars(runs: List<RunRecord>): String {
        val total = runs.fold(ContextBreakdown()) { sum, run -> sum + run.context }
        val parts = listOf(
            "Tool outputs" to total.toolOutputs,
            "Tool schemas" to total.toolSchemas,
            "Conversation history" to total.conversationHistory,
            "System prompt" to total.systemPrompt,
            "User task" to total.userTask,
            "Chat template etc." to total.overhead,
        ).sortedByDescending { it.second }

        val sum = total.total.coerceAtLeast(1)
        return buildString {
            parts.forEach { (label, tokens) ->
                val share = tokens.toDouble() / sum
                val bar = "█".repeat((share * 20).roundToInt())
                appendLine("  %-22s %-20s %s".format(label, bar, percent(tokens.toLong(), sum.toLong())))
            }
            append(
                "  (the total is measured; the parts are estimated, and what could not be\n" +
                    "   attributed is its own line rather than spread across the others)",
            )
        }
    }

    private fun successRate(runs: List<RunRecord>): Double? {
        val judged = runs.mapNotNull { it.succeeded }
        return if (judged.isEmpty()) null else judged.count { it }.toDouble() / judged.size
    }

    private fun List<RunRecord>.averageOrZero(of: (RunRecord) -> Double): Double =
        if (isEmpty()) 0.0 else sumOf(of) / size

    /**
     * A label on the left, a value on the right.
     *
     * The padding is never allowed to vanish: a long label next to a long value would
     * otherwise run the two together into `search_documents100%`.
     */
    private fun row(label: String, value: String, indent: Int = 0): String {
        val padded = (WIDTH - value.length - indent).coerceAtLeast(label.length + MIN_GAP)
        return label.padEnd(padded).plus(value)
    }

    /**
     * Money, at a precision that shows what was actually spent.
     *
     * Two decimals is right for a bill and useless for one local run, which lands
     * around half a cent — and a report that says `$0.00` after measuring 20k tokens
     * reads as broken.
     */
    private fun money(amount: Double): String = when {
        amount == 0.0 -> "$0"
        amount < 0.01 -> "$" + "%.4f".format(amount)
        else -> "$" + "%.2f".format(amount)
    }

    private fun percent(part: Long, whole: Long): String =
        if (whole <= 0) "0%" else "${(part.toDouble() / whole * 100).roundToInt()}%"

    private fun percent(share: Double): String = "${(share * 100).roundToInt()}%"

    /** 4_200_000 -> "4.2M", 39_400 -> "39.4k". */
    private fun compact(value: Long): String = when {
        value >= 1_000_000 -> "%.1fM".format(value / 1_000_000.0)
        value >= 1_000 -> "%.1fk".format(value / 1_000.0)
        else -> value.toString()
    }
}
