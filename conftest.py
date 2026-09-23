"""Pytest portability shim.

The repository's text artifacts are UTF-8.  On Windows, pathlib.Path.read_text()
without an explicit encoding follows the active ANSI code page (often cp1252),
which can fail on UTF-8 Markdown.  Keep historical frozen test files byte-for-byte
unchanged and make their implicit text reads deterministic during pytest.
"""
from __future__ import annotations

from pathlib import Path

_ORIGINAL_READ_TEXT = Path.read_text


def _read_text_utf8_default(self: Path, encoding=None, errors=None, newline=None):
    if encoding is None:
        encoding = "utf-8"
    return _ORIGINAL_READ_TEXT(self, encoding=encoding, errors=errors, newline=newline)


Path.read_text = _read_text_utf8_default
