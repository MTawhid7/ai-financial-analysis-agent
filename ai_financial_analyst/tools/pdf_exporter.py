"""PDF exporter — converts a Markdown report to a styled PDF via weasyprint.

Import is deferred so the module loads without error even when weasyprint
is not installed or its system dependencies are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimal CSS for a clean financial report PDF
_CSS = """
@page { margin: 2cm; size: A4; }
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
}
h1 { font-size: 18pt; border-bottom: 2px solid #1a1a1a; padding-bottom: 6pt; margin-top: 0; }
h2 { font-size: 13pt; border-bottom: 1px solid #cccccc; padding-bottom: 4pt; margin-top: 20pt; }
h3 { font-size: 11pt; font-weight: bold; margin-top: 14pt; }
table { width: 100%; border-collapse: collapse; margin: 12pt 0; font-size: 10pt; }
th { background: #f0f0f0; border: 1px solid #cccccc; padding: 5pt 8pt; text-align: left; }
td { border: 1px solid #dddddd; padding: 4pt 8pt; }
code { font-family: 'Courier New', monospace; background: #f5f5f5; padding: 2pt 4pt;
       font-size: 9pt; border-radius: 2pt; }
blockquote { margin: 0 0 0 16pt; padding-left: 10pt; border-left: 3px solid #cccccc; color: #555; }
ul, ol { margin: 4pt 0; padding-left: 20pt; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 1px solid #cccccc; margin: 16pt 0; }
em { color: #555; }
.disclaimer { font-size: 9pt; color: #777; border-top: 1px solid #cccccc;
              margin-top: 20pt; padding-top: 10pt; }
"""


def is_available() -> bool:
    """Return True if weasyprint can be imported successfully."""
    try:
        import weasyprint  # noqa: F401
        return True
    except Exception:
        return False


def export_to_pdf(markdown_text: str, output_path: Path) -> Path:
    """Convert markdown_text to a PDF and write to output_path.

    Raises ImportError if weasyprint is not installed.
    Raises RuntimeError wrapping any weasyprint error.
    """
    try:
        from weasyprint import HTML, CSS
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise ImportError(
            "PDF export requires weasyprint and markdown-it-py. "
            "Install with: pip install weasyprint markdown-it-py"
        ) from exc

    md = MarkdownIt()
    body_html = md.render(markdown_text)
    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{body_html}</body></html>"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=full_html).write_pdf(str(output_path), stylesheets=[CSS(string=_CSS)])
        return output_path
    except Exception as exc:
        raise RuntimeError(f"PDF generation failed: {exc}") from exc
