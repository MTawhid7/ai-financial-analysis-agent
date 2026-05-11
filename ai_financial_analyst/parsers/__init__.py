"""AI Financial Analyst — file parsers package.

Each parser returns a fixed-schema dict summary.
No raw user content is forwarded to the primary LLM.
"""
from .csv_parser   import parse_csv
from .pdf_parser   import parse_pdf
from .excel_parser import parse_xlsx
from .word_parser  import parse_docx
from .text_parser  import parse_text
from .json_parser  import parse_json

__all__ = ["parse_csv", "parse_pdf", "parse_xlsx", "parse_docx", "parse_text", "parse_json"]
