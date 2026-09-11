import benchmark.BenchmarkSetupException
import benchmark.BenchmarkTask
import benchmark.BenchmarkTasks
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The benchmark decides what "the agent still works" means, so its own rules are
 * worth testing: a wrong verdict here would either hide a regression or invent one.
 */
class BenchmarkRunnerTest {

    private val task = BenchmarkTask(
        id = "apple-one-trial",
        prompt = "What do buyers of a new iPhone get?",
        expectAnyOf = listOf("Apple One", "three-month"),
        expectToolCall = "search_documents",
    )

    @Test
    fun `an answer counts only when the tool it needed was actually called`() {
        assertTrue(task.isSatisfiedBy("They get a three-month Apple One trial.", setOf("search_documents")))
        // The right answer from the model's own memory is not retrieval working.
        assertFalse(task.isSatisfiedBy("They get a three-month Apple One trial.", emptySet()))
    }

    @Test
    fun `any of the accepted phrasings counts`() {
        assertTrue(task.isSatisfiedBy("Buyers receive Apple One free for a while.", setOf("search_documents")))
        assertFalse(task.isSatisfiedBy("I am not sure what they get.", setOf("search_documents")))
    }

    @Test
    fun `the suite covers absent answers and a follow-up that needs history`() {
        val ids = BenchmarkTasks.all.map { it.id }

        assertTrue(ids.size >= 10, "the suite should be big enough to average over: ${ids.size}")
        assertTrue(
            BenchmarkTasks.all.any { it.setUpTurns.isNotEmpty() },
            "trimming context is easy if nothing in the suite depends on it",
        )
        assertTrue(
            ids.count { it.startsWith("absent-") } >= 2,
            "an agent that invents an answer has to be able to fail the suite",
        )
    }

    @Test
    fun `a setup failure is an exception, not a zero percent score`() {
        val error = BenchmarkSetupException(
            "All 10 tasks failed to reach the model. Check that the LLM in LLM_PROVIDER/LLM_MODEL is running.",
        )

        // An arm of zeroes would otherwise read as a real measurement in the comparison.
        assertContains(error.message!!, "failed to reach the model")
    }
}
