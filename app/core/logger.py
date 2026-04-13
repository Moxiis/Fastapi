import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import settings

# Ensure logs directory exists
LOG_DIR: Path = settings.log_dir
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


def _make_handlers(level_int: int):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler = RotatingFileHandler(
        str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level_int)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level_int)

    return [file_handler, console_handler]


def configure_logging(level: Optional[str] = None):
    """Configure root logging for the app.

    - Writes rotating logs to `logs/app.log` and prints to console.
    - Respects `LOG_LEVEL` environment variable if `level` is not provided.
    Returns the root logger.
    """
    if level is None:
        level_str = settings.log_level.upper()
    else:
        level_str = level.upper()
    level_int = getattr(logging, level_str, logging.INFO)

    root = logging.getLogger()
    # remove existing handlers to avoid duplicate logs during reloads
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level_int)
    handlers = _make_handlers(level_int)
    for h in handlers:
        root.addHandler(h)

    # Tidy up some noisy library loggers
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(level_int)
    logging.getLogger("uvicorn.error").setLevel(level_int)
    logging.getLogger("uvicorn.access").setLevel(level_int)

    return root


def get_logger(name: Optional[str] = None):
    """Return a configured logger instance for `name` (module name recommended)."""
    return logging.getLogger(name) if name else logging.getLogger()
