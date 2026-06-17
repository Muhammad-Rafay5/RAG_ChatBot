# =============================================================================
# utils.py — Shared Helper Utilities
# =============================================================================
# Small reusable functions used across multiple modules.
# Centralizing them here avoids code duplication and keeps other files clean.
# =============================================================================

import logging
import sys
import time
import functools
from pathlib import Path
from typing import List


# =============================================================================
# LOGGING SETUP
# =============================================================================
# A consistent logger so every module prints in the same format.
# Usage in any file:
#   from src.utils import get_logger
#   logger = get_logger(__name__)
#   logger.info("Something happened")

def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a logger with a standardized format.

    Args:
        name: Typically pass __name__ from the calling module so log lines
              show which file they came from (e.g. 'src.embed_index').

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Only add handlers once — prevents duplicate log lines if the function
    # is called multiple times for the same logger name.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Stream handler → prints to the terminal
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Format: timestamp  [LEVEL]  module_name: message
        fmt = logging.Formatter(
            fmt="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


# =============================================================================
# TIMING DECORATOR
# =============================================================================
# Wrap any function with @timer to automatically log how long it takes.
# Useful for profiling slow steps like embedding or indexing.
#
# Usage:
#   @timer
#   def my_slow_function(): ...

def timer(func):
    """Decorator that logs the execution time of the wrapped function."""
    @functools.wraps(func)          # preserves the original function's name/docstring
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱  {func.__name__}() finished in {elapsed:.2f}s")
        return result
    return wrapper


# =============================================================================
# FILE HELPERS
# =============================================================================

def collect_source_files(data_dir: Path, extensions: List[str] = None) -> List[Path]:
    """
    Recursively collect all files under data_dir matching the given extensions.

    Args:
        data_dir:   Root directory to search (e.g. Path("data/")).
        extensions: List of file extensions to include, e.g. [".txt", ".pdf"].
                    If None, collects every file.

    Returns:
        Sorted list of Path objects for matching files.

    Example:
        files = collect_source_files(Path("data"), [".txt", ".pdf"])
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Create it and place your source files inside."
        )

    extensions = [ext.lower() for ext in (extensions or [])]

    files = [
        path for path in data_dir.rglob("*")
        if path.is_file() and (not extensions or path.suffix.lower() in extensions)
    ]

    if not files:
        raise ValueError(
            f"No matching files found in {data_dir}.\n"
            f"Supported extensions: {extensions or 'any'}"
        )

    return sorted(files)


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and any missing parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path (convenient for chaining).

    Example:
        db_path = ensure_dir(Path("vector_store"))
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# TEXT HELPERS
# =============================================================================

def truncate(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """
    Shorten a string for display purposes (e.g. logging, UI previews).

    Args:
        text:      The string to truncate.
        max_chars: Maximum number of characters to keep.
        suffix:    Appended when text is truncated (default: ellipsis).

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def clean_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive whitespace characters into a single space
    and strip leading/trailing whitespace.

    Args:
        text: Raw string.

    Returns:
        Normalized string.
    """
    import re
    return re.sub(r"\s+", " ", text).strip()
