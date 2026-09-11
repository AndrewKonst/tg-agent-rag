package benchmark

/**
 * One task the agent is asked to do, and how to tell whether it did.
 *
 * The point of a predicate rather than an eyeball is that "the tokens went down"
 * only means something next to "and the answers stayed right". A judgement made by
 * hand after seeing the token numbers is not a judgement.
 *
 * [expectAnyOf] is a list because a model phrases the same correct answer several
 * ways; matching any one of them is what "correct" means here.
 */
data class BenchmarkTask(
    val id: String,
    val prompt: String,
    /** Phrasings that count as the right answer. Empty means nothing is required. */
    val expectAnyOf: List<String> = emptyList(),
    /** A tool the run must have called, when the task is about reaching for it. */
    val expectToolCall: String? = null,
    /**
     * A claim the answer must not make.
     *
     * For a question the documents cannot answer, listing the phrasings of a refusal
     * does not work: "not found", "does not mention" and "None of the chunks mention"
     * are the same answer, and a list of them measures the model's vocabulary rather
     * than its honesty. What the task actually forbids is inventing a figure, so that
     * is what is checked.
     */
    val mustNotClaim: Regex? = null,
    /**
     * How many tool calls the task needs at the least.
     *
     * A task that cannot be done in one lookup is the only thing that catches an
     * optimisation which quietly takes the tools away after the first one.
     */
    val minToolCalls: Int = 0,
    /** Sent before [prompt] in the same chat, to build up history the task depends on. */
    val setUpTurns: List<String> = emptyList(),
) {
    fun isSatisfiedBy(answer: String, toolsCalled: Set<String>, toolCallCount: Int = toolsCalled.size): Boolean {
        val answered = expectAnyOf.isEmpty() || expectAnyOf.any { answer.contains(it, ignoreCase = true) }
        val inventedNothing = mustNotClaim?.containsMatchIn(answer) != true
        val calledWhatItHadTo = expectToolCall == null || expectToolCall in toolsCalled
        val lookedEnough = toolCallCount >= minToolCalls
        return answered && inventedNothing && calledWhatItHadTo && lookedEnough
    }
}

/**
 * The benchmark suite.
 *
 * Every task is answerable from the four documents in `data/test-documents`, which
 * the runner indexes first — except the ones that deliberately are not, because an
 * agent that invents an answer has failed the task even though it produced text.
 *
 * The mix is on purpose: single-hop retrieval, retrieval that needs two documents,
 * a tool that is not retrieval, a follow-up that only makes sense with history, and
 * two questions whose answers are absent. Optimising context is easy if nothing in
 * the suite depends on context.
 */
object BenchmarkTasks {

    val all: List<BenchmarkTask> = listOf(
        BenchmarkTask(
            id = "apple-one-trial",
            prompt = "What do buyers of a new iPhone get, according to my documents?",
            expectAnyOf = listOf("Apple One", "three-month", "three month"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "foldable-name",
            prompt = "What is the foldable iPhone called in my documents?",
            expectAnyOf = listOf("iPhone Duo", "Duo"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "siri-languages",
            prompt = "Which language does the new Siri launch in first, and how many follow?",
            expectAnyOf = listOf("English"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "airpods-anc",
            prompt = "What do my documents say about noise cancellation on the new AirPods?",
            expectAnyOf = listOf("Noise Cancellation", "ANC", "noise"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "macos-release",
            prompt = "What do my documents say about the next macOS release?",
            expectAnyOf = listOf("macOS"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "which-file-apple-one",
            prompt = "Which of my files mentions the Apple One trial? Name the file.",
            expectAnyOf = listOf("9to5mac-services-and-macos.pdf", "services-and-macos"),
            expectToolCall = "search_documents",
        ),
        BenchmarkTask(
            id = "follow-up-trial-length",
            prompt = "And how long does that trial last?",
            expectAnyOf = listOf("three", "3 month", "three-month"),
            // The question is meaningless on its own: it can only be answered by a run
            // that still has the previous exchange. This is the task that catches
            // context trimming taken too far.
            setUpTurns = listOf("What do new iPhone buyers get, according to my documents?"),
        ),
        // A question in two parts, where the second is about whatever the first one
        // found. It was written to force two rounds of lookups and does not: one
        // retrieval returns both facts, and the agent answers in a single round.
        //
        // The requirement is therefore on the answer, not on the mechanism. Demanding
        // two calls would fail an agent for being more efficient than expected, which
        // is the opposite of what this suite is for — and it did exactly that on the
        // baseline before the requirement was dropped.
        BenchmarkTask(
            id = "two-hop-same-file",
            prompt = "Which of my files mentions the Apple One trial? Then tell me what that " +
                "same file says about macOS.",
            expectAnyOf = listOf("macOS"),
            expectToolCall = "search_documents",
            minToolCalls = 1,
        ),
        BenchmarkTask(
            id = "current-year",
            prompt = "What year is it right now? Use your tools.",
            expectAnyOf = listOf("2026"),
            expectToolCall = "current_datetime",
        ),
        // The documents are Apple news; neither of these questions has an answer in
        // them. Both must still search before declining — an agent that refuses
        // without looking is not being careful, it is being useless — and neither may
        // produce a figure, because any figure would be invented.
        BenchmarkTask(
            id = "absent-vacation-policy",
            prompt = "According to my documents, how many vacation days do employees get?",
            expectToolCall = "search_documents",
            mustNotClaim = Regex("""\b\d+([.,]\d+)?\s*(paid\s+)?(vacation|holiday|annual)?\s*days?\b""", RegexOption.IGNORE_CASE),
        ),
        BenchmarkTask(
            id = "absent-salary",
            prompt = "What does my documentation say the CEO's salary is?",
            expectToolCall = "search_documents",
            mustNotClaim = Regex("""[$€£]\s?\d|\b\d[\d,.]*\s*(k\b|thousand|million|usd|dollars|euros)""", RegexOption.IGNORE_CASE),
        ),
    )
}
