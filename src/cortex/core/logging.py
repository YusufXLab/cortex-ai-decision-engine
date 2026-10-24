"""Structured logging system with correlation ID tracking."""

import contextvars
import logging
import sys
from typing import Any
from pythonjsonlogger import jsonlogger

# Context variable for request correlation ID
correlation_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Injects the current request correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_ctx.get() or "system"
        return True


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configures root logger with formatting and filters."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Clear any existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level.upper())
    handler.addFilter(CorrelationIdFilter())

    if json_format:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(correlation_id)s] | %(name)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence overly verbose external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance bound to a module name."""
    return logging.getLogger(name)
