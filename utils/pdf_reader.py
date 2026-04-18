import pdfplumber
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    if not text.strip():
        raise ValueError("No extractable text found. The PDF may be scanned/image-based.")

    # Truncate to ~12000 words to stay within token limits
    words = text.split()
    if len(words) > 12000:
        text = " ".join(words[:12000]) + "\n\n[Document truncated for processing...]"

    return text
