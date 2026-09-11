import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.utils.time.KoogClock
import harness.AgentLoop
import harness.AgentTool
import harness.Llm
import harness.ToolBox
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import observability.ContextBreakdown
import observability.ObservabilityStore
import observability.PromptAudit
import observability.RunRecord
import observability.TokenEstimator
import observability.TokenMeter
import observability.TokenPrices
import observability.TokenReport
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Covers the measurement layer: what it reads from the provider, what it works out
 * for itself, and that the numbers survive a round trip through SQLite.
 *
 * The point of these is that an audit nobody can trust is worse than no audit — a
 * wrong token count sends the optimisation work at the wrong target.
 */
class ObservabilityTest {

    private val clock = KoogClock.System

    private fun system(text: String) = Message.System(text, RequestMetaInfo.create(clock))
    private fun user(text: String) = Message.User(text, RequestMetaInfo.create(clock))
    private fun assistant(text: String) = Message.Assistant(text, ResponseMetaInfo.Empty)
    private fun toolResult(tool: String, output: String) = Message.User(
        listOf(MessagePart.Tool.Result(id = "1", tool = tool, output = output)),
        RequestMetaInfo.create(clock),
    )

    // --- what the prompt is made of ---

    @Test
    fun `a prompt is split into what it is made of`() {
        val messages = listOf(
            system("You are a helpful assistant with a fairly long system prompt."),
            user("An older question from earlier in this conversation."),
            assistant("An older answer that is also part of the history now."),
            toolResult("search_documents", "A very long chunk of retrieved document text. ".repeat(20)),
            user("The question being asked right now."),
        )

        val breakdown = PromptAudit.breakdown(messages)

        assertTrue(breakdown.systemPrompt > 0, "system prompt should be counted")
        assertTrue(breakdown.userTask > 0, "the last typed message is the task")
        assertTrue(breakdown.conversationHistory > 0, "earlier turns are history")
        assertTrue(
            breakdown.toolOutputs > breakdown.systemPrompt,
            "the long tool output should dominate: $breakdown",
        )
    }

    @Test
    fun `the split always adds up to the measured total`() {
        val messages = listOf(
            system("System."),
            user("Earlier."),
            toolResult("exec", "output"),
            user("Now."),
        )

        val breakdown = PromptAudit.breakdown(messages, emptyList(), measuredInputTokens = 1_000)

        // The provider's number is the truth; the parts are reconciled to fit it exactly.
        assertEquals(1_000, breakdown.total)
    }

    @Test
    fun `tool schemas are their own line, not spread across the messages`() {
        val messages = listOf(system("A short system prompt."), user("A short question."))
        val tools = listOf(
            tools.DateTimeTool().descriptor,
            tools.SearchDocumentsTool(1, rag.DocumentSearchService()).descriptor,
        )

        val withoutTools = PromptAudit.breakdown(messages)
        val withTools = PromptAudit.breakdown(messages, tools)

        assertTrue(withTools.toolSchemas > 0, "two tool definitions are not free")
        // The schemas must not inflate the system prompt: that is the bug this fixes.
        assertEquals(withoutTools.systemPrompt, withTools.systemPrompt)
        assertTrue(
            withTools.toolSchemas > withTools.systemPrompt,
            "measured on this project's own prompt, schemas cost 246 tokens against the " +
                "system prompt's 73: ${withTools.toolSchemas} vs ${withTools.systemPrompt}",
        )
    }

    @Test
    fun `what cannot be attributed is named, not spread over the parts`() {
        val messages = listOf(system("A short system prompt."), user("A short question."))

        // The provider charged far more than the messages account for — the chat
        // template's own tokens. The difference belongs to nobody in particular.
        val breakdown = PromptAudit.breakdown(messages, emptyList(), measuredInputTokens = 1_000)

        val attributed = PromptAudit.breakdown(messages)
        assertEquals(attributed.systemPrompt, breakdown.systemPrompt, "the system prompt did not grow")
        assertEquals(1_000 - attributed.total, breakdown.overhead)
        assertEquals(1_000, breakdown.total)
    }

    @Test
    fun `an overshooting estimate is scaled down instead of inventing overhead`() {
        val messages = listOf(system("A system prompt long enough to overshoot a tiny measured total."))

        val breakdown = PromptAudit.breakdown(messages, emptyList(), measuredInputTokens = 5)

        assertEquals(5, breakdown.total)
        assertEquals(0, breakdown.overhead, "there is nothing left over to attribute")
    }

    @Test
    fun `a part that is genuinely empty stays empty after scaling`() {
        // First turn: a system prompt and a question, no history and no tool output.
        val messages = listOf(system("A system prompt."), user("The first question."))

        val breakdown = PromptAudit.breakdown(messages, emptyList(), measuredInputTokens = 343)

        assertEquals(0, breakdown.conversationHistory, "there is no history yet: $breakdown")
        assertEquals(0, breakdown.toolOutputs)
        assertEquals(343, breakdown.total)
    }

    @Test
    fun `tool results count as tool output wherever they sit`() {
        val messages = listOf(
            system("System."),
            toolResult("search_documents", "chunk text"),
            user("The current question."),
        )

        val breakdown = PromptAudit.breakdown(messages)

        assertTrue(breakdown.toolOutputs > 0, "a user message carrying results is not a person talking")
        assertEquals(0, breakdown.conversationHistory)
    }

    // --- what is paid for twice ---

    @Test
    fun `the repeated prefix of a prompt is measured`() {
        val shared = listOf(system("A system prompt of some length."), user("The question."))
        val turnOne = shared
        val turnTwo = shared + assistant("Calling a tool.") + toolResult("exec", "result")

        val reused = PromptAudit.reusedPrefixTokens(turnOne, turnTwo)
        val whole = PromptAudit.breakdown(turnTwo).total

        assertTrue(reused > 0, "turn two repeats turn one's prompt")
        assertTrue(reused < whole, "but not the appended part: reused=$reused whole=$whole")
        assertEquals(PromptAudit.breakdown(turnOne).total, reused)
    }

    @Test
    fun `only a prefix counts, because that is all a cache would serve`() {
        val previous = listOf(system("System."), user("First question."))
        // Same length, different second message: nothing after the mismatch is reusable.
        val current = listOf(system("System."), user("A different question entirely."), assistant("x"))

        val reused = PromptAudit.reusedPrefixTokens(previous, current)

        assertEquals(PromptAudit.breakdown(listOf(system("System."))).total, reused)
    }

    @Test
    fun `discarded reasoning is counted as paid for, inline or separate`() {
        val inline = Message.Assistant(
            "<think>Let me work through this at some length before answering.</think>The answer is 25.",
            ResponseMetaInfo.Empty,
        )
        val separate = Message.Assistant(
            listOf(
                MessagePart.Reasoning(listOf("Working through the same thing as a separate part.")),
                MessagePart.Text("The answer is 25."),
            ),
            ResponseMetaInfo.Empty,
        )

        assertTrue(PromptAudit.reasoningTokens(inline) > 0, "a <think> block was billed and dropped")
        assertTrue(PromptAudit.reasoningTokens(separate) > 0, "so was a reasoning part")
        assertEquals(
            0,
            PromptAudit.reasoningTokens(Message.Assistant("Just the answer.", ResponseMetaInfo.Empty)),
        )
    }

    @Test
    fun `an estimate is never zero for real text and grows with it`() {
        val short = TokenEstimator.countTokens("one short sentence")
        val long = TokenEstimator.countTokens("one short sentence ".repeat(50))

        assertTrue(short > 0)
        assertTrue(long > short * 20, "estimates should scale with length")
        assertEquals(0, TokenEstimator.countTokens("   "))
    }

    // --- money ---

    @Test
    fun `a known model is priced at its own rates`() {
        val prices = TokenPrices()

        // 1M input + 1M output of gpt-4o-mini at $0.15 + $0.60.
        val cost = prices.cost("gpt-4o-mini", inputTokens = 1_000_000, outputTokens = 1_000_000)

        assertEquals(0.75, cost, 1e-9)
        assertTrue(prices.isPriced("gpt-4o-mini"))
    }

    @Test
    fun `a local model is billed at the reference model instead of zero`() {
        val prices = TokenPrices()

        val cost = prices.cost("qwen3:14b", inputTokens = 1_000_000, outputTokens = 0)

        assertTrue(!prices.isPriced("qwen3:14b"), "a local model has no price of its own")
        assertEquals(0.15, cost, 1e-9)
        assertEquals("gpt-4o-mini", prices.referenceModel)
    }

    @Test
    fun `cached input is billed at the cached rate`() {
        val prices = TokenPrices()

        val withoutCache = prices.cost("gpt-4o-mini", 1_000_000, 0)
        val withCache = prices.cost("gpt-4o-mini", 1_000_000, 0, cachedTokens = 1_000_000)

        assertTrue(withCache < withoutCache, "a cache hit has to be cheaper")
        assertEquals(0.075, withCache, 1e-9)
    }

    // --- the middleware, end to end ---

    @Test
    fun `a measured run records every turn, every tool call, and what repeated`() = runTest {
        ObservabilityStore(Files.createTempFile("obs", ".db")).use { store ->
            val meter = TokenMeter(store)
            val tool = EchoTool()
            val tools = ToolBox(listOf(meter.meter(tool)))
            val loop = AgentLoop(llm = meter.meter(ScriptedLlm()), maxSteps = 4)

            val measured = meter.measure(
                agentId = "test-agent",
                chatId = 7,
                model = "qwen3:14b",
                taskId = "task-1",
            ) {
                loop.run(listOf(system("A system prompt."), user("Do the thing.")), tools)
            }

            val record = measured.record
            assertEquals(2, record.turns, "two LLM calls: one asking for the tool, one answering")
            assertEquals(1, record.toolCalls)
            assertEquals(1, tool.invocations, "the wrapper must still run the real tool")
            assertEquals("task-1", record.taskId)
            assertTrue(record.inputTokens > 0)
            assertTrue(record.outputTokens > 0)
            // The loop appends to the very list it hands the model, so a measurement that
            // kept the reference instead of a snapshot would call the whole prompt
            // repeated — and point the optimisation work at nothing.
            val calls = store.llmCalls(record.runId)
            assertEquals(0, calls[0].reusedInputTokens, "nothing precedes the first turn")
            assertTrue(
                calls[1].reusedInputTokens in 1 until calls[1].inputTokens,
                "turn two repeats turn one and adds to it: ${calls[1].reusedInputTokens} " +
                    "of ${calls[1].inputTokens}",
            )
            assertTrue(
                calls[1].reusedInputTokens <= calls[0].inputTokens,
                "the repeat cannot exceed the prompt it repeats: ${calls[1].reusedInputTokens} " +
                    "vs ${calls[0].inputTokens}",
            )
            assertTrue(record.estimatedCostUsd > 0, "a local model is still priced for the report")
            assertTrue(record.reasoningTokens > 0, "the scripted model emits a think block")

            // And it is all readable back out of the database.
            val stored = store.runs().single()
            assertEquals(record.runId, stored.runId)
            assertEquals(record.inputTokens, stored.inputTokens)
            assertEquals(2, store.llmCalls(record.runId).size)
            assertEquals(listOf("echo"), store.toolCalls(record.runId).map { it.toolName })
            assertEquals(record.context.total, stored.context.total)
        }
    }

    @Test
    fun `measurement does not change what the agent returns`() = runTest {
        val plain = AgentLoop(llm = ScriptedLlm(), maxSteps = 4)
            .run(listOf(system("A system prompt."), user("Do the thing.")), ToolBox(listOf(EchoTool())))

        ObservabilityStore(Files.createTempFile("obs-transparent", ".db")).use { store ->
            val meter = TokenMeter(store)
            val metered = AgentLoop(llm = meter.meter(ScriptedLlm()), maxSteps = 4).run(
                listOf(system("A system prompt."), user("Do the thing.")),
                ToolBox(listOf(meter.meter(EchoTool()))),
            )

            assertEquals(plain.answer, metered.answer)
            assertEquals(plain.steps, metered.steps)
            assertEquals(plain.stopReason, metered.stopReason)
        }
    }

    @Test
    fun `reasoning is never reported as larger than the output it is part of`() = runTest {
        ObservabilityStore(Files.createTempFile("obs-reasoning-cap", ".db")).use { store ->
            val meter = TokenMeter(store)

            // A model that thinks at length and reports a small output count: our estimate
            // of the thinking would otherwise exceed the provider's measured total.
            meter.measure(agentId = "test", chatId = 1, model = "qwen3:14b") {
                AgentLoop(llm = meter.meter(LongThinkingLlm()), maxSteps = 2).run(
                    listOf(system("A system prompt."), user("Think hard.")),
                    ToolBox(emptyList()),
                )
            }

            val call = store.llmCalls(store.runs().single().runId).single()
            assertTrue(
                call.reasoningTokens <= call.outputTokens,
                "reasoning ${call.reasoningTokens} cannot exceed output ${call.outputTokens}",
            )
        }
    }

    @Test
    fun `a benchmark run can be tied back to its task and arm`() = runTest {
        ObservabilityStore(Files.createTempFile("obs-annotate", ".db")).use { store ->
            val meter = TokenMeter(store)
            val measured = meter.measure(agentId = "bench", chatId = 900_000, model = "qwen3:14b") {
                AgentLoop(llm = meter.meter(ScriptedLlm()), maxSteps = 4).run(
                    listOf(system("A system prompt."), user("Do the thing.")),
                    ToolBox(listOf(meter.meter(EchoTool()))),
                )
            }

            store.annotate(measured.record.runId, taskId = "apple-one-trial", variant = "baseline", succeeded = true)

            val stored = store.runs("baseline").single()
            // Without the task id, a before/after can only be read in aggregate.
            assertEquals("apple-one-trial", stored.taskId)
            assertEquals(true, stored.succeeded)
            assertTrue(store.runs("optimized").isEmpty(), "the other arm stays empty")
        }
    }

    @Test
    fun `a failing run is still recorded`() = runTest {
        ObservabilityStore(Files.createTempFile("obs-failure", ".db")).use { store ->
            val meter = TokenMeter(store)

            val failure = runCatching {
                meter.measure(agentId = "test-agent", chatId = 1, model = "qwen3:14b") {
                    error("the run blew up")
                }
            }

            assertTrue(failure.isFailure, "the error must reach the caller")
            assertEquals("ERROR", store.runs().single().stopReason)
        }
    }

    @Test
    fun `a run outside any measured scope is left alone`() = runTest {
        ObservabilityStore(Files.createTempFile("obs-unscoped", ".db")).use { store ->
            val meter = TokenMeter(store)

            // No measure { } around it: the wrappers find no run and record nothing.
            AgentLoop(llm = meter.meter(ScriptedLlm()), maxSteps = 4).run(
                listOf(system("A system prompt."), user("Do the thing.")),
                ToolBox(listOf(meter.meter(EchoTool()))),
            )

            assertTrue(store.runs().isEmpty())
        }
    }

    // --- the reports ---

    @Test
    fun `the dashboard reports totals, tools and where context goes`() {
        val runs = listOf(
            run(id = "a", input = 8_000, output = 1_000, reused = 5_000, tools = 2),
            run(id = "b", input = 12_000, output = 2_000, reused = 9_000, tools = 3),
        )
        val toolCalls = listOf(
            toolCall("a", "search_documents", tokens = 3_000),
            toolCall("b", "exec", tokens = 1_000),
        )

        val dashboard = TokenReport.dashboard(runs, toolCalls)

        assertContains(dashboard, "Tasks completed")
        assertContains(dashboard, "20.0k")
        assertContains(dashboard, "search_documents")
        assertContains(dashboard, "75%")
        assertContains(dashboard, "Conversation history")
        assertContains(dashboard, "billed as")
        // A long label next to a long value must not run the two together.
        assertTrue(
            dashboard.lines().none { it.contains(Regex("[a-z_]{6,}\\d+%")) },
            "a label and its value collided:\n$dashboard",
        )
    }

    @Test
    fun `a cost of half a cent is shown, not rounded away to zero`() {
        val runs = listOf(run(id = "a", input = 16_852, output = 3_379, reused = 1_013, tools = 9))

        val dashboard = TokenReport.dashboard(runs, emptyList())

        // Two decimals would print $0.00 after measuring twenty thousand tokens.
        assertTrue(
            dashboard.contains("$0.0"),
            "expected a visible cost, got:\n$dashboard",
        )
        assertTrue(!dashboard.contains("$0.00\n"), "a measured run is not free")
    }

    @Test
    fun `the timeline shows each turn and names the most expensive one`() {
        ObservabilityStore(Files.createTempFile("obs-timeline", ".db")).use { store ->
            val record = run(id = "t", input = 20_000, output = 2_000, reused = 12_000, tools = 1)
            store.save(
                record,
                llmCalls = listOf(
                    observability.LlmCallRecord(
                        runId = "t", turn = 1, timestampMillis = 0, model = "qwen3:14b",
                        inputTokens = 6_000, outputTokens = 1_000, reasoningTokens = 200,
                        cachedTokens = 0, reusedInputTokens = 0, latencyMillis = 900,
                        estimatedCostUsd = 0.001, context = ContextBreakdown(100, 50, 0, 0),
                        tokensReportedByProvider = true,
                    ),
                    observability.LlmCallRecord(
                        runId = "t", turn = 2, timestampMillis = 1, model = "qwen3:14b",
                        inputTokens = 14_000, outputTokens = 1_000, reasoningTokens = 300,
                        cachedTokens = 0, reusedInputTokens = 12_000, latencyMillis = 1_200,
                        estimatedCostUsd = 0.002, context = ContextBreakdown(100, 50, 400, 900),
                        tokensReportedByProvider = true,
                    ),
                ),
                toolCalls = listOf(toolCall("t", "search_documents", tokens = 3_000)),
            )

            val stored = store.runs().single()
            val timeline = TokenReport.timeline(stored, store.llmCalls("t"), store.toolCalls("t"))

            assertContains(timeline, "Turn 1")
            assertContains(timeline, "Turn 2")
            assertContains(timeline, "search_documents")
            assertContains(timeline, "Most expensive turn: 2")
            assertContains(timeline, "repeated input")
        }
    }

    @Test
    fun `the comparison judges the target rather than leaving it to the reader`() {
        val baseline = List(50) { run(id = "b$it", input = 10_000, output = 1_000, reused = 7_000, tools = 2, succeeded = true) }
        // Half the tokens, and exactly one task worse — 98% against 100%, which is the
        // -2pp the assignment allows and not a point more.
        val good = List(50) { run(id = "g$it", input = 5_000, output = 900, reused = 1_000, tools = 2, succeeded = it != 0) }
        val weak = List(50) { run(id = "w$it", input = 9_500, output = 1_000, reused = 6_000, tools = 2, succeeded = true) }

        val met = TokenReport.comparison(baseline, good)
        val notMet = TokenReport.comparison(baseline, weak)

        assertContains(met, "MET: tokens")
        assertContains(notMet, "NOT MET")
    }

    @Test
    fun `a success rate that falls too far fails the target even when tokens drop`() {
        val baseline = List(10) { run(id = "b$it", input = 10_000, output = 1_000, reused = 7_000, tools = 2, succeeded = true) }
        // Half the tokens, but a fifth of the tasks now fail.
        val cheap = List(10) { run(id = "c$it", input = 4_000, output = 500, reused = 500, tools = 1, succeeded = it >= 2) }

        val report = TokenReport.comparison(baseline, cheap)

        assertContains(report, "NOT MET: success rate fell")
    }

    private fun run(
        id: String,
        input: Int,
        output: Int,
        reused: Int,
        tools: Int,
        succeeded: Boolean? = null,
    ) = RunRecord(
        runId = id,
        agentId = "test-agent",
        taskId = "task-$id",
        chatId = 1,
        model = "qwen3:14b",
        startedAtMillis = 0,
        durationMillis = 2_000,
        turns = 2,
        toolCalls = tools,
        inputTokens = input,
        outputTokens = output,
        reasoningTokens = 100,
        cachedTokens = 0,
        reusedInputTokens = reused,
        estimatedCostUsd = input * 0.15 / 1_000_000 + output * 0.60 / 1_000_000,
        stopReason = "COMPLETED",
        succeeded = succeeded,
        context = ContextBreakdown(
            systemPrompt = 200,
            userTask = 100,
            conversationHistory = input / 2,
            toolOutputs = input / 3,
        ),
    )

    private fun toolCall(runId: String, name: String, tokens: Int) = observability.ToolCallRecord(
        runId = runId,
        turn = 1,
        timestampMillis = 0,
        toolName = name,
        inputSize = 40,
        outputSize = tokens * 4,
        outputTokens = tokens,
        durationMillis = 120,
        failed = false,
    )

    /** Thinks at length, and reports a modest output count, as a real provider would. */
    private class LongThinkingLlm : Llm {
        override suspend fun complete(
            messages: List<Message>,
            tools: List<ToolDescriptor>,
        ): Message.Assistant = Message.Assistant(
            "<think>" + "Working through this carefully. ".repeat(60) + "</think>Yes.",
            ResponseMetaInfo.create(KoogClock.System, inputTokensCount = 100, outputTokensCount = 30),
        )
    }

    /** Asks for a tool once, then answers. Reports token counts like a real provider. */
    private class ScriptedLlm : Llm {
        override suspend fun complete(
            messages: List<Message>,
            tools: List<ToolDescriptor>,
        ): Message.Assistant {
            val meta = ResponseMetaInfo.create(
                KoogClock.System,
                inputTokensCount = messages.sumOf { it.textContent().length } / 4 + 10,
                outputTokensCount = 40,
            )
            val hasResults = messages.any { message ->
                message.parts.any { it is MessagePart.Tool.Result }
            }
            return if (hasResults) {
                Message.Assistant("<think>brief</think>Done: the tool said what it said.", meta)
            } else {
                Message.Assistant(
                    listOf(MessagePart.Tool.Call("call-1", "echo", """{"text":"hello"}""")),
                    meta,
                )
            }
        }
    }

    private class EchoTool : AgentTool {
        var invocations = 0
            private set

        override val descriptor = ToolDescriptor(
            name = "echo",
            description = "Echoes the given text back.",
            requiredParameters = listOf(
                ToolParameterDescriptor("text", "Text to echo.", ToolParameterType.String),
            ),
            optionalParameters = emptyList(),
        )

        override suspend fun execute(args: JsonObject): String {
            invocations++
            return "echo: " + (args["text"] as? JsonPrimitive)?.content.orEmpty()
        }
    }
}
