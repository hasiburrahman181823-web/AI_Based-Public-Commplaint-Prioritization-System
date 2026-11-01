"""Shared text cleaning and conservative, extensible Banglish normalization.

Only runs of three or more identical Latin letters are collapsed. Ordinary
double letters (for example ``school``) are kept; dictionary entries handle
known double-letter spelling variants such as ``biddut``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


DEFAULT_VARIANTS_PATH = Path(__file__).with_name("banglish_variants.json")
_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_REPEATED_LETTERS_RE = re.compile(r"([a-z])\1{2,}")
_LATIN_TOKEN_RE = re.compile(r"\b[a-z]+\b")
_WORD_RE = re.compile(r"[a-z]+\Z")


def clean_text(text: str) -> str:
    """NFKC-normalize text, translate Bangla digits, remove URLs and spaces.

    Latin text is lowercased; Bengali letters and punctuation are preserved.
    A non-string raises ``TypeError`` instead of converting missing values to
    literal strings such as ``nan``. The preparation pipeline validates missing
    input values before calling this function.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, received {type(text).__name__}")
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.translate(_BANGLA_DIGITS).lower()
    cleaned = _URL_RE.sub(" ", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _collapse_repeats(text: str) -> str:
    return _REPEATED_LETTERS_RE.sub(r"\1", text)


def load_variants(variants_path: str | Path | None = None) -> dict[str, str]:
    """Load an alias-to-canonical JSON object and resolve spelling chains.

    ``None`` uses the adjacent ``banglish_variants.json`` file. An explicit path
    replaces that dictionary. Keys and values must each be one lowercase ASCII
    word. Canonical chains are resolved so normalization remains idempotent;
    cycles and aliases that conflict after repeated-letter collapse are errors.
    Parsed dictionaries are cached by file modification time and size so ordinary
    edits take effect without restarting and processing many rows stays fast.
    A fresh copy is returned to protect the cache from caller modifications.
    """
    path = Path(variants_path) if variants_path is not None else DEFAULT_VARIANTS_PATH
    try:
        path = path.resolve()
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"Cannot load Banglish variants from {path}: {exc}") from exc
    return dict(_load_variants_cached(path, stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=16)
def _load_variants_cached(path: Path, mtime_ns: int, size: int) -> dict[str, str]:
    # mtime_ns and size are intentional cache-key arguments.
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load Banglish variants from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Banglish variants in {path} must be a JSON object of alias: canonical pairs")

    aliases: dict[str, str] = {}
    for alias, canonical in raw.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            raise ValueError(f"Banglish variants in {path} must map strings to strings")
        if not _WORD_RE.fullmatch(alias) or not _WORD_RE.fullmatch(canonical):
            raise ValueError(
                f"Invalid Banglish variant {alias!r}: {canonical!r} in {path}; "
                "use single lowercase Latin words (a-z)"
            )
        key, value = _collapse_repeats(alias), _collapse_repeats(canonical)
        if key in aliases and aliases[key] != value:
            raise ValueError(f"Conflicting Banglish aliases after letter collapse for {key!r} in {path}")
        aliases[key] = value

    resolved: dict[str, str] = {}
    for alias in aliases:
        current = alias
        seen: set[str] = set()
        while current in aliases and aliases[current] != current:
            if current in seen:
                raise ValueError(f"Cyclic Banglish spelling variants involving {current!r} in {path}")
            seen.add(current)
            current = aliases[current]
        resolved[alias] = current
    return resolved


def normalize_banglish(text: str, variants_path: str | Path | None = None) -> str:
    """Clean text, shorten expressive Latin letter runs, and map spellings.

    Examples: ``paniiii`` becomes ``pani``; ``PANY NEI biddut`` becomes
    ``pani nai bidyut``. Token replacement preserves surrounding punctuation and
    does not transliterate Bengali. Pass a JSON path to use a custom dictionary.
    """
    variants = load_variants(variants_path)
    cleaned = _collapse_repeats(clean_text(text))
    return _LATIN_TOKEN_RE.sub(lambda match: variants.get(match.group(), match.group()), cleaned)
