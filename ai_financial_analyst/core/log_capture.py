"""In-memory log handler for surfacing pipeline logs in the Streamlit UI."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


class ListLogHandler(logging.Handler):
    """Captures log records into an in-memory list during a pipeline run."""

    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self.records: list[dict] = []
        self.setFormatter(logging.Formatter("%(name)s — %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append({
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
            "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
        })


@contextmanager
def capture_logs(logger_name: str = "ai_financial_analyst") -> Iterator[ListLogHandler]:
    """Context manager that captures logs from the given logger into a list.

    Usage::

        with capture_logs() as handler:
            run_pipeline(...)
        for record in handler.records:
            print(record["message"])
    """
    handler = ListLogHandler()
    root_logger = logging.getLogger(logger_name)
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)
