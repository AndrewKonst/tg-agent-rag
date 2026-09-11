package observability

/**
 * What a thousand tokens costs, per model.
 *
 * This bot's own runs go to a local Ollama, where the marginal cost of a token is
 * zero — and a cost report that only ever says `$0.00` measures nothing. So a model
 * with no price of its own is billed at [referenceModel]'s rates, and every report
 * says which model the money is counted in. The token counts are real either way;
 * the dollars are what those tokens would cost on a hosted provider.
 */
class TokenPrices(
    private val rates: Map<String, ModelRate> = DEFAULT_RATES,
    /** Rates used for a model that has none — a local one, or one newer than this table. */
    val referenceModel: String = DEFAULT_REFERENCE_MODEL,
) {

    /** Dollars per million tokens, as providers quote them. */
    data class ModelRate(
        val inputPerMillion: Double,
        val outputPerMillion: Double,
        /** What a cached input token costs; hosted providers discount it heavily. */
        val cachedInputPerMillion: Double,
    )

    private val reference: ModelRate =
        rates[referenceModel] ?: error("Reference model '$referenceModel' has no rate.")

    /** True when [model] is billed at its own rates rather than the reference model's. */
    fun isPriced(model: String): Boolean = rateFor(model) != null

    fun cost(model: String, inputTokens: Int, outputTokens: Int, cachedTokens: Int = 0): Double {
        val rate = rateFor(model) ?: reference
        val billedInput = (inputTokens - cachedTokens).coerceAtLeast(0)
        return billedInput * rate.inputPerMillion / MILLION +
            cachedTokens * rate.cachedInputPerMillion / MILLION +
            outputTokens * rate.outputPerMillion / MILLION
    }

    private fun rateFor(model: String): ModelRate? =
        rates[model] ?: rates.entries.firstOrNull { model.startsWith(it.key) }?.value

    companion object {
        private const val MILLION = 1_000_000.0

        /**
         * Cheap and widely used, so the numbers stay recognisable — and low enough that
         * the report never overstates what a local run would have cost.
         */
        const val DEFAULT_REFERENCE_MODEL = "gpt-4o-mini"

        /** List prices as of early 2026, in USD per million tokens. */
        val DEFAULT_RATES: Map<String, ModelRate> = mapOf(
            "gpt-4o-mini" to ModelRate(0.15, 0.60, 0.075),
            "gpt-4o" to ModelRate(2.50, 10.00, 1.25),
            "claude-haiku-4-5" to ModelRate(1.00, 5.00, 0.10),
            "claude-sonnet-4-5" to ModelRate(3.00, 15.00, 0.30),
        )
    }
}
