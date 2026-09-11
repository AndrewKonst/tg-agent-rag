import agent.AgentProfile
import agent.HarnessAgentService
import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.ResponseMetaInfo
import conversation.HistoryWindow
import conversation.InMemoryConversationStore
import harness.AgentLoop
import harness.Llm
import harness.ToolBox
import kotlinx.coroutines.test.runTest
import rag.DocumentIndexingService
import rag.DocumentSearchService
import rag.SqliteRagStore
import telegram.MessageHandler
import tools.SearchDocumentsTool
import java.nio.file.Files
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * The whole pipeline in one test, with nothing skipped and nothing to install:
 *
 *     document -> index -> question -> retrieval -> LLM -> reply with a source
 *
 * The model is scripted and Telegram is absent — `MessageHandler.answerOrExplain` is
 * the layer directly beneath it — so what is actually verified is the wiring in
 * between: that the agent's tool reaches this user's chunks, that the retrieved source
 * reaches the model, and that the reply the user would receive carries it.
 *
 * [OllamaIntegrationTest] covers the part a script cannot: that a real model decides
 * to call the tool by itself.
 */
class RagEndToEndTest {

    @Test
    fun `a question about an uploaded document is answered from it, with the source`() = runTest {
        val chatId = 9_001L
        val documents = Files.createTempDirectory("rag-e2e-docs")
        documents.resolve("vacation_policy.md").writeText(
            """
            # Vacation policy

            Employees receive 25 paid vacation days per year. Unused days may be carried
            over into the first quarter of the following year.
            """.trimIndent(),
        )

        SqliteRagStore(Files.createTempFile("rag-e2e", ".db")).use { store ->
            DocumentIndexingService(documentsDir = documents, store = store)
                .indexTestDocuments(userId = chatId)

            val search = DocumentSearchService(store = store, topK = 3)
            val model = ScriptedLlm()
            val handler = MessageHandler(
                agentService = HarnessAgentService(
                    loop = AgentLoop(llm = model, maxSteps = 4),
                    store = InMemoryConversationStore(),
                    window = HistoryWindow(12_000),
                    profileFor = { chat ->
                        AgentProfile(
                            systemPrompt = "Use search_documents for document questions.",
                            tools = ToolBox(listOf(SearchDocumentsTool(chat, search))),
                        )
                    },
                ),
                timeout = 10.seconds,
            )

            val reply = handler.answerOrExplain("How many vacation days do employees get?", chatId)

            // The model asked for retrieval, and retrieval answered with this user's chunk.
            assertEquals(listOf("search_documents"), model.calledTools)
            assertContains(model.toolOutputs.single(), "25 paid vacation days")
            assertContains(model.toolOutputs.single(), "Source: vacation_policy.md")

            // And the reply a Telegram user would see carries the answer and its source.
            assertContains(reply, "25")
            assertContains(reply, "vacation_policy.md")
        }
    }

    @Test
    fun `the same question from another chat retrieves nothing`() = runTest {
        val chatId = 9_002L
        val otherChat = 9_003L
        val documents = Files.createTempDirectory("rag-e2e-isolation")
        documents.resolve("secret.txt").writeText("The launch date is 12 March 2027.")

        SqliteRagStore(Files.createTempFile("rag-e2e-isolation", ".db")).use { store ->
            DocumentIndexingService(documentsDir = documents, store = store)
                .indexTestDocuments(userId = chatId)

            val search = DocumentSearchService(store = store, topK = 3)
            val model = ScriptedLlm()
            val handler = MessageHandler(
                agentService = HarnessAgentService(
                    loop = AgentLoop(llm = model, maxSteps = 4),
                    store = InMemoryConversationStore(),
                    window = HistoryWindow(12_000),
                    profileFor = { chat ->
                        AgentProfile(
                            systemPrompt = "Use search_documents for document questions.",
                            tools = ToolBox(listOf(SearchDocumentsTool(chat, search))),
                        )
                    },
                ),
                timeout = 10.seconds,
            )

            val reply = handler.answerOrExplain("What is the launch date?", otherChat)

            assertTrue(
                model.toolOutputs.single().contains("No relevant chunks found"),
                "another chat must not see the document; got: ${model.toolOutputs.single()}",
            )
            assertTrue("12 March 2027" !in reply, "the other chat's secret leaked into: $reply")
        }
    }

    /**
     * A model that always retrieves once, then answers from what came back.
     *
     * It quotes the tool output into its answer so the test can tell whether the
     * retrieved chunk and its source actually reached the model — which is the part of
     * the pipeline a script can verify and a real model cannot verify deterministically.
     */
    private class ScriptedLlm : Llm {

        val calledTools = mutableListOf<String>()
        val toolOutputs = mutableListOf<String>()

        override suspend fun complete(
            messages: List<Message>,
            tools: List<ToolDescriptor>,
        ): Message.Assistant {
            val results = messages
                .flatMap { it.parts }
                .filterIsInstance<MessagePart.Tool.Result>()

            if (results.isEmpty()) {
                val query = messages.last().parts
                    .filterIsInstance<MessagePart.Text>()
                    .joinToString(" ") { it.text }
                calledTools += "search_documents"
                return Message.Assistant(
                    listOf(MessagePart.Tool.Call("call-1", "search_documents", """{"query":"$query"}""")),
                    ResponseMetaInfo.Empty,
                )
            }

            toolOutputs += results.joinToString("\n") { it.output }
            return Message.Assistant(toolOutputs.last(), ResponseMetaInfo.Empty)
        }
    }
}
