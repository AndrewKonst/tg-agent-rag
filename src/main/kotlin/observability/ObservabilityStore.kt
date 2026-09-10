package observability

import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager

private val storeLogger = KotlinLogging.logger {}

/**
 * Where traces are kept: one row per run, one per LLM call, one per tool call.
 *
 * A separate database from the conversations and the RAG index on purpose — an audit
 * that competes for the same file as the bot's working data is an audit that changes
 * what it measures, and one that can be deleted wholesale when it has served its
 * purpose.
 */
class ObservabilityStore(
    databasePath: Path = Path.of(DEFAULT_DB_PATH),
) : AutoCloseable {

    private val connection: Connection

    init {
        databasePath.toAbsolutePath().parent?.let { Files.createDirectories(it) }
        connection = DriverManager.getConnection("jdbc:sqlite:${databasePath.toAbsolutePath()}")
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA journal_mode=WAL")
            statement.execute("PRAGMA synchronous=NORMAL")
            statement.execute("PRAGMA foreign_keys=ON")
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id          TEXT    PRIMARY KEY,
                    agent_id        TEXT    NOT NULL,
                    task_id         TEXT,
                    chat_id         INTEGER NOT NULL,
                    model           TEXT    NOT NULL,
                    started_at      INTEGER NOT NULL,
                    duration_ms     INTEGER NOT NULL,
                    turns           INTEGER NOT NULL,
                    tool_calls      INTEGER NOT NULL,
                    input_tokens    INTEGER NOT NULL,
                    output_tokens   INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    cached_tokens   INTEGER NOT NULL,
                    reused_tokens   INTEGER NOT NULL,
                    cost_usd        REAL    NOT NULL,
                    stop_reason     TEXT    NOT NULL,
                    succeeded       INTEGER,
                    ctx_system      INTEGER NOT NULL,
                    ctx_task        INTEGER NOT NULL,
                    ctx_history     INTEGER NOT NULL,
                    ctx_tools       INTEGER NOT NULL,
                    variant         TEXT
                )
                """.trimIndent(),
            )
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    turn            INTEGER NOT NULL,
                    timestamp       INTEGER NOT NULL,
                    model           TEXT    NOT NULL,
                    input_tokens    INTEGER NOT NULL,
                    output_tokens   INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    cached_tokens   INTEGER NOT NULL,
                    reused_tokens   INTEGER NOT NULL,
                    latency_ms      INTEGER NOT NULL,
                    cost_usd        REAL    NOT NULL,
                    ctx_system      INTEGER NOT NULL,
                    ctx_task        INTEGER NOT NULL,
                    ctx_history     INTEGER NOT NULL,
                    ctx_tools       INTEGER NOT NULL,
                    provider_tokens INTEGER NOT NULL
                )
                """.trimIndent(),
            )
            statement.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    turn          INTEGER NOT NULL,
                    timestamp     INTEGER NOT NULL,
                    tool_name     TEXT    NOT NULL,
                    input_size    INTEGER NOT NULL,
                    output_size   INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    duration_ms   INTEGER NOT NULL,
                    failed        INTEGER NOT NULL
                )
                """.trimIndent(),
            )
            statement.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id)")
            statement.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id)")
            statement.execute("CREATE INDEX IF NOT EXISTS idx_runs_variant ON runs(variant)")
        }
        storeLogger.info { "Observability store: sqlite at ${databasePath.toAbsolutePath()}" }
    }

    /**
     * Stores one run with its calls, in a single transaction.
     *
     * [variant] labels a benchmark arm — `baseline` or `optimized` — so a before/after
     * comparison is a query rather than a pair of database files.
     */
    fun save(
        run: RunRecord,
        llmCalls: List<LlmCallRecord>,
        toolCalls: List<ToolCallRecord>,
        variant: String? = null,
    ) {
        connection.autoCommit = false
        try {
            insertRun(run, variant)
            insertLlmCalls(llmCalls)
            insertToolCalls(toolCalls)
            connection.commit()
        } catch (e: Throwable) {
            connection.rollback()
            throw e
        } finally {
            connection.autoCommit = true
        }
    }

    /** Records whether a benchmark run produced an acceptable answer. */
    fun markSucceeded(runId: String, succeeded: Boolean) {
        connection.prepareStatement("UPDATE runs SET succeeded = ? WHERE run_id = ?").use { statement ->
            statement.setInt(1, if (succeeded) 1 else 0)
            statement.setString(2, runId)
            statement.executeUpdate()
        }
    }

    fun labelVariant(runId: String, variant: String) {
        connection.prepareStatement("UPDATE runs SET variant = ? WHERE run_id = ?").use { statement ->
            statement.setString(1, variant)
            statement.setString(2, runId)
            statement.executeUpdate()
        }
    }

    /** Every run, newest first. [variant] narrows it to one benchmark arm. */
    fun runs(variant: String? = null, limit: Int = 1_000): List<RunRecord> {
        val sql = buildString {
            append("SELECT * FROM runs ")
            if (variant != null) append("WHERE variant = ? ")
            append("ORDER BY started_at DESC LIMIT ?")
        }
        return connection.prepareStatement(sql).use { statement ->
            var index = 1
            if (variant != null) statement.setString(index++, variant)
            statement.setInt(index, limit)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) add(rows.toRunRecord())
                }
            }
        }
    }

    fun llmCalls(runId: String): List<LlmCallRecord> =
        connection.prepareStatement("SELECT * FROM llm_calls WHERE run_id = ? ORDER BY turn").use { statement ->
            statement.setString(1, runId)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) {
                        add(
                            LlmCallRecord(
                                runId = rows.getString("run_id"),
                                turn = rows.getInt("turn"),
                                timestampMillis = rows.getLong("timestamp"),
                                model = rows.getString("model"),
                                inputTokens = rows.getInt("input_tokens"),
                                outputTokens = rows.getInt("output_tokens"),
                                reasoningTokens = rows.getInt("reasoning_tokens"),
                                cachedTokens = rows.getInt("cached_tokens"),
                                reusedInputTokens = rows.getInt("reused_tokens"),
                                latencyMillis = rows.getLong("latency_ms"),
                                estimatedCostUsd = rows.getDouble("cost_usd"),
                                context = ContextBreakdown(
                                    systemPrompt = rows.getInt("ctx_system"),
                                    userTask = rows.getInt("ctx_task"),
                                    conversationHistory = rows.getInt("ctx_history"),
                                    toolOutputs = rows.getInt("ctx_tools"),
                                ),
                                tokensReportedByProvider = rows.getInt("provider_tokens") == 1,
                            ),
                        )
                    }
                }
            }
        }

    fun toolCalls(runId: String? = null): List<ToolCallRecord> {
        val sql = if (runId == null) {
            "SELECT * FROM tool_calls ORDER BY timestamp"
        } else {
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY turn, timestamp"
        }
        return connection.prepareStatement(sql).use { statement ->
            if (runId != null) statement.setString(1, runId)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) {
                        add(
                            ToolCallRecord(
                                runId = rows.getString("run_id"),
                                turn = rows.getInt("turn"),
                                timestampMillis = rows.getLong("timestamp"),
                                toolName = rows.getString("tool_name"),
                                inputSize = rows.getInt("input_size"),
                                outputSize = rows.getInt("output_size"),
                                outputTokens = rows.getInt("output_tokens"),
                                durationMillis = rows.getLong("duration_ms"),
                                failed = rows.getInt("failed") == 1,
                            ),
                        )
                    }
                }
            }
        }
    }

    override fun close() {
        connection.close()
    }

    private fun insertRun(run: RunRecord, variant: String?) {
        connection.prepareStatement(
            """
            INSERT OR REPLACE INTO runs (
                run_id, agent_id, task_id, chat_id, model, started_at, duration_ms, turns,
                tool_calls, input_tokens, output_tokens, reasoning_tokens, cached_tokens,
                reused_tokens, cost_usd, stop_reason, succeeded,
                ctx_system, ctx_task, ctx_history, ctx_tools, variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.trimIndent(),
        ).use { statement ->
            statement.setString(1, run.runId)
            statement.setString(2, run.agentId)
            statement.setString(3, run.taskId)
            statement.setLong(4, run.chatId)
            statement.setString(5, run.model)
            statement.setLong(6, run.startedAtMillis)
            statement.setLong(7, run.durationMillis)
            statement.setInt(8, run.turns)
            statement.setInt(9, run.toolCalls)
            statement.setInt(10, run.inputTokens)
            statement.setInt(11, run.outputTokens)
            statement.setInt(12, run.reasoningTokens)
            statement.setInt(13, run.cachedTokens)
            statement.setInt(14, run.reusedInputTokens)
            statement.setDouble(15, run.estimatedCostUsd)
            statement.setString(16, run.stopReason)
            run.succeeded?.let { statement.setInt(17, if (it) 1 else 0) } ?: statement.setNull(17, java.sql.Types.INTEGER)
            statement.setInt(18, run.context.systemPrompt)
            statement.setInt(19, run.context.userTask)
            statement.setInt(20, run.context.conversationHistory)
            statement.setInt(21, run.context.toolOutputs)
            statement.setString(22, variant)
            statement.executeUpdate()
        }
    }

    private fun insertLlmCalls(calls: List<LlmCallRecord>) {
        if (calls.isEmpty()) return
        connection.prepareStatement(
            """
            INSERT INTO llm_calls (
                run_id, turn, timestamp, model, input_tokens, output_tokens, reasoning_tokens,
                cached_tokens, reused_tokens, latency_ms, cost_usd,
                ctx_system, ctx_task, ctx_history, ctx_tools, provider_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.trimIndent(),
        ).use { statement ->
            calls.forEach { call ->
                statement.setString(1, call.runId)
                statement.setInt(2, call.turn)
                statement.setLong(3, call.timestampMillis)
                statement.setString(4, call.model)
                statement.setInt(5, call.inputTokens)
                statement.setInt(6, call.outputTokens)
                statement.setInt(7, call.reasoningTokens)
                statement.setInt(8, call.cachedTokens)
                statement.setInt(9, call.reusedInputTokens)
                statement.setLong(10, call.latencyMillis)
                statement.setDouble(11, call.estimatedCostUsd)
                statement.setInt(12, call.context.systemPrompt)
                statement.setInt(13, call.context.userTask)
                statement.setInt(14, call.context.conversationHistory)
                statement.setInt(15, call.context.toolOutputs)
                statement.setInt(16, if (call.tokensReportedByProvider) 1 else 0)
                statement.addBatch()
            }
            statement.executeBatch()
        }
    }

    private fun insertToolCalls(calls: List<ToolCallRecord>) {
        if (calls.isEmpty()) return
        connection.prepareStatement(
            """
            INSERT INTO tool_calls (
                run_id, turn, timestamp, tool_name, input_size, output_size,
                output_tokens, duration_ms, failed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.trimIndent(),
        ).use { statement ->
            calls.forEach { call ->
                statement.setString(1, call.runId)
                statement.setInt(2, call.turn)
                statement.setLong(3, call.timestampMillis)
                statement.setString(4, call.toolName)
                statement.setInt(5, call.inputSize)
                statement.setInt(6, call.outputSize)
                statement.setInt(7, call.outputTokens)
                statement.setLong(8, call.durationMillis)
                statement.setInt(9, if (call.failed) 1 else 0)
                statement.addBatch()
            }
            statement.executeBatch()
        }
    }

    private fun java.sql.ResultSet.toRunRecord(): RunRecord {
        // wasNull() reports on the column read last, so it has to be consulted here
        // and not after the twenty other reads below.
        val succeededValue = getInt("succeeded")
        val succeededWasNull = wasNull()
        return RunRecord(
            runId = getString("run_id"),
            agentId = getString("agent_id"),
            taskId = getString("task_id"),
            chatId = getLong("chat_id"),
            model = getString("model"),
            startedAtMillis = getLong("started_at"),
            durationMillis = getLong("duration_ms"),
            turns = getInt("turns"),
            toolCalls = getInt("tool_calls"),
            inputTokens = getInt("input_tokens"),
            outputTokens = getInt("output_tokens"),
            reasoningTokens = getInt("reasoning_tokens"),
            cachedTokens = getInt("cached_tokens"),
            reusedInputTokens = getInt("reused_tokens"),
            estimatedCostUsd = getDouble("cost_usd"),
            stopReason = getString("stop_reason"),
            succeeded = if (succeededWasNull) null else succeededValue == 1,
            context = ContextBreakdown(
                systemPrompt = getInt("ctx_system"),
                userTask = getInt("ctx_task"),
                conversationHistory = getInt("ctx_history"),
                toolOutputs = getInt("ctx_tools"),
            ),
        )
    }

    companion object {
        const val DEFAULT_DB_PATH = "data/observability.db"
    }
}
