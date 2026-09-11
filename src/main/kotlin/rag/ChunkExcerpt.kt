package rag

/**
 * Cuts a retrieved chunk down to the part that answers the question.
 *
 * A chunk is sized for retrieval — a thousand characters, so its embedding describes
 * a whole section — but the model rarely needs all of it. The audit found chunk text
 * to be 64% of everything the agent sends, so this is where the tokens are.
 *
 * The window is chosen by where the question's own words appear, and it is widened
 * to sentence boundaries so the model never receives half a sentence. When nothing
 * matches — the question paraphrases the chunk rather than quoting it, which is the
 * normal case for semantic search — the opening of the chunk is returned, because a
 * chunk that was retrieved at all is about the right subject.
 */
object ChunkExcerpt {

    /**
     * Returns at most [maxChars] of [text], centred on [query]'s words.
     *
     * Returns the whole text when it already fits: trimming something short buys no
     * tokens and can only lose meaning.
     */
    fun excerpt(text: String, query: String, maxChars: Int = DEFAULT_MAX_CHARS): String {
        require(maxChars > 0) { "maxChars must be positive" }
        if (text.length <= maxChars) return text

        val sentences = splitIntoSentences(text)
        if (sentences.isEmpty()) return text.take(maxChars).trimEnd()

        val terms = queryTerms(query)
        val best = sentences.indices.maxBy { index -> sentences[index].score(terms) }

        return window(sentences, around = best, maxChars = maxChars)
    }

    /**
     * Grows a window outward from the best sentence until the budget is spent.
     *
     * Outward rather than forward: the sentence before a match often carries what the
     * match is about ("Buyers of these models…" / "…get three months free").
     */
    private fun window(sentences: List<String>, around: Int, maxChars: Int): String {
        var first = around
        var last = around
        var length = sentences[around].length

        while (true) {
            val previous = (first - 1).takeIf { it >= 0 }?.let { sentences[it].length + 1 }
            val next = (last + 1).takeIf { it < sentences.size }?.let { sentences[it].length + 1 }

            val takePrevious = previous != null && length + previous <= maxChars
            val takeNext = next != null && length + next <= maxChars

            when {
                // Prefer the sentence before, for the context it usually carries.
                takePrevious -> {
                    first--
                    length += previous
                }
                takeNext -> {
                    last++
                    length += next
                }
                else -> break
            }
        }

        val body = sentences.subList(first, last + 1).joinToString(" ").trim()
        return buildString {
            if (first > 0) append(ELLIPSIS)
            append(body)
            if (last < sentences.size - 1) append(ELLIPSIS)
        }
    }

    private fun String.score(terms: Set<String>): Int {
        if (terms.isEmpty()) return 0
        val words = WORD.findAll(lowercase()).map { it.value }.toSet()
        return terms.count { it in words }
    }

    /**
     * The words of the question worth matching on.
     *
     * Short words are dropped: "the", "of" and "is" appear in every sentence and would
     * make every sentence look equally relevant.
     */
    private fun queryTerms(query: String): Set<String> =
        WORD.findAll(query.lowercase())
            .map { it.value }
            .filter { it.length >= MIN_TERM_LENGTH }
            .toSet()

    private fun splitIntoSentences(text: String): List<String> =
        SENTENCE.findAll(text).map { it.value.trim() }.filter { it.isNotEmpty() }.toList()

    private const val DEFAULT_MAX_CHARS = 400
    private const val MIN_TERM_LENGTH = 4
    private const val ELLIPSIS = "…"

    private val WORD = Regex("[\\p{L}\\p{N}]+")

    /** A sentence, or whatever is left at the end of the text. */
    private val SENTENCE = Regex("[^.!?]+[.!?]+|[^.!?]+$")
}
