"""OCR support for scanned PDFs.

Heuristic: if pdfplumber extracts < 50 characters per page on average,
the PDF is likely scanned and needs OCR.  Requires:
  - PyMuPDF  (pip install PyMuPDF)
  - pytesseract + Tesseract-OCR system package
  - Pillow
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE = 50   # below this → likely scanned


def is_scanned_pdf(file_bytes: bytes) -> bool:
    """Return True when the PDF has almost no extractable text (scanned image)."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return False
            total_chars = sum(len(p.extract_text() or "") for p in pdf.pages)
            avg = total_chars / len(pdf.pages)
            return avg < _MIN_CHARS_PER_PAGE
    except Exception:
        return False


def ocr_pdf_pages(file_bytes: bytes, dpi: int = 300) -> list[str]:
    """Render each page to an image and OCR it.  Returns list of text strings."""
    try:
        import fitz           # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        logger.warning("OCR dependencies missing (%s); falling back to empty text.", exc)
        return []

    texts: list[str] = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale factor for target DPI

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img  = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng")
            texts.append(text.strip())
        except Exception as exc:
            logger.warning("OCR failed for page %d: %s", page_num + 1, exc)
            texts.append("")

    doc.close()
    return texts
