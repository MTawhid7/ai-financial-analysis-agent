"""CSV file parser with formula injection scrubbing."""
from __future__ import annotations

import io
import re
from typing import Any

_FORMULA_RE = re.compile(r"^[=+\-@]")


def parse_csv(file_bytes: bytes, filename: str = "file.csv") -> dict[str, Any]:
    """Parse a CSV file. Returns a fixed-schema summary — no raw cell data to LLM."""
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        return {"error": f"Could not parse CSV: {exc}", "filename": filename}

    injection_count = 0
    for col in df.select_dtypes(include="object").columns:
        def _scrub(val):
            nonlocal injection_count
            s = str(val) if val is not None else ""
            if _FORMULA_RE.match(s):
                injection_count += 1
                return "[REMOVED]"
            return s[:200]
        df[col] = df[col].map(_scrub)

    numeric_stats: dict[str, dict] = {}
    for col in df.select_dtypes(include="number").columns:
        stats = df[col].describe()
        numeric_stats[col] = {k: round(float(v), 4) for k, v in stats.items() if k != "count"}

    return {
        "filename": filename,
        "file_type": "csv",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "numeric_stats": numeric_stats,
        "formula_cells_removed": injection_count,
    }
