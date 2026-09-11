import telegram.safeUploadFilename
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A filename arrives from Telegram, so it is untrusted input that ends up in a path
 * and in every reply that cites the document.
 */
class UploadFilenameTest {

    @Test
    fun `keeps the name the user recognises`() {
        assertEquals("AI Agent Token Audit.md", safeUploadFilename("AI Agent Token Audit.md", "id"))
        assertEquals("Отчёт за квартал.pdf", safeUploadFilename("Отчёт за квартал.pdf", "id"))
    }

    @Test
    fun `strips path separators so the file cannot escape its directory`() {
        val name = safeUploadFilename("../../etc/passwd", "id")

        assertTrue('/' !in name, "got: $name")
        assertTrue(!name.startsWith("."), "got: $name")
    }

    @Test
    fun `a hidden file stays visible`() {
        assertEquals("env", safeUploadFilename(".env", "id"))
    }

    @Test
    fun `a blank name falls back to the telegram id`() {
        assertEquals("document-abc123", safeUploadFilename("   ", "abc123"))
    }

    @Test
    fun `a very long name is truncated`() {
        assertEquals(120, safeUploadFilename("x".repeat(500), "id").length)
    }
}
