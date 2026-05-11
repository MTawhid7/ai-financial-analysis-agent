"""JSON file parser — returns structure summary without raw values."""
from __future__ import annotations

import json as _json
from typing import Any


def parse_json(file_bytes: bytes, filename: str = "file.json") -> dict[str, Any]:
    try:
        data = _json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        return {"error": f"Could not parse JSON: {exc}", "filename": filename}

    def _describe(obj: Any, depth: int = 0) -> dict:
        if depth > 3:
            return {"type": type(obj).__name__}
        if isinstance(obj, dict):
            return {"type": "object", "keys": list(obj.keys())[:20], "key_count": len(obj)}
        if isinstance(obj, list):
            return {"type": "array", "length": len(obj),
                    "item_schema": _describe(obj[0], depth + 1) if obj else {}}
        return {"type": type(obj).__name__, "value_preview": str(obj)[:80]}

    return {
        "filename":       filename,
        "file_type":      "json",
        "schema":         _describe(data),
        "top_level_keys": list(data.keys())[:20] if isinstance(data, dict) else None,
        "array_length":   len(data) if isinstance(data, list) else None,
    }
