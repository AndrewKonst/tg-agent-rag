package rag

import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.poi.xwpf.usermodel.XWPFDocument
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.inputStream

class UnsupportedDocumentTypeException(message: String) : RuntimeException(message)
class EmptyDocumentException(message: String) : RuntimeException(message)
class DocumentTooLargeException(message: String) : RuntimeException(message)

/** Thrown when a file has a supported extension but its contents cannot be read. */
class DocumentParseException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * Turns a file on disk into plain text.
 *
 * Every failure mode the homework asks about is a distinct exception, so the
 * Telegram layer can tell the user which one happened: an unsupported extension, a
 * file too large to index, a document that parsed but held no text, and a supported
 * format whose bytes are damaged.
 */
class DocumentTextExtractor(
    private val maxBytes: Long = DEFAULT_MAX_BYTES,
) {

    fun extract(path: Path): RawDocument {
        val filename = path.fileName.toString()

        val size = Files.size(path)
        if (size > maxBytes) {
            throw DocumentTooLargeException(
                "File '$filename' is ${size / 1024 / 1024} MB, over the ${maxBytes / 1024 / 1024} MB limit.",
            )
        }
        val fileType = filename.substringAfterLast('.', missingDelimiterValue = "")
            .lowercase()
            .takeIf { it.isNotBlank() }
            ?: throw UnsupportedDocumentTypeException("File '$filename' has no extension.")

        val text = try {
            when (fileType) {
                "txt", "md" -> Files.readString(path)
                "pdf" -> extractPdf(path)
                "docx" -> extractDocx(path)
                else -> throw UnsupportedDocumentTypeException(
                    "File '$filename' has unsupported type '$fileType'.",
                )
            }
        } catch (e: UnsupportedDocumentTypeException) {
            throw e
        } catch (e: Exception) {
            // A truncated PDF, a .docx that is really a .zip of something else, text in
            // an encoding that is not UTF-8 — all arrive here as library-specific noise.
            throw DocumentParseException("File '$filename' could not be read as $fileType.", e)
        }.trim()

        if (text.isBlank()) throw EmptyDocumentException("File '$filename' is empty.")

        return RawDocument(
            path = path,
            filename = filename,
            fileType = fileType,
            text = text,
        )
    }

    private fun extractPdf(path: Path): String =
        Loader.loadPDF(path.toFile()).use { document ->
            PDFTextStripper().getText(document)
        }

    private fun extractDocx(path: Path): String =
        path.inputStream().use { input ->
            XWPFDocument(input).use { document ->
                document.paragraphs.joinToString("\n") { it.text }
            }
        }

    companion object {
        /** 20 MB, mirroring the default of `RAG_MAX_DOCUMENT_BYTES`. */
        const val DEFAULT_MAX_BYTES = 20L * 1024 * 1024
    }
}
