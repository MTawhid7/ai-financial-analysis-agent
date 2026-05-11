# Backwards-compatibility shim.
# Parser logic lives in ai_financial_analyst/parsers/
from ai_financial_analyst.parsers import (  # noqa: F401
    parse_csv,
    parse_pdf,
    parse_xlsx,
    parse_docx,
    parse_text,
    parse_json,
)
from ai_financial_analyst.parsers._summarise import hierarchical_summarise  # noqa: F401
