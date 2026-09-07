"""Text cleaning utilities for ingestion.

Removes boilerplate, normalises unicode, collapses redundant whitespace,
and strips content that would skew embeddings (e.g. page footers, repeated
headers).
"""

from __future__ import annotations

import re
import unicodedata

# Characters that are guaranteed noise in any enterprise document.
_DISCARD_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # C0 control chars except \n \r \t
)

# Lines that are pure pagination artefacts.
_PAGE_BREAK_LINES_RE = re.compile(
    r"^(?:\d+\s*$|---+$|^\*{3,}$|^\_{3,}$)", re.MULTILINE
)

# Repeated document titles that appear on every page (header boilerplate).
# Matches common patterns like:
#   "AskMyDocs | Some Page"
#   "Some Doc | Some Page"
#   "Company Confidential"
#   "Confidential | Foo"
_BOILERPLATE_HEADER_RE = re.compile(
    r"""^(
        AskMyDocs\s*[\||\:].*              # "AskMyDocs | ..." or "AskMyDocs: ..."
      | .+\s*[\||\:]\s*(?:Page|Guide|Chapter|Section)\b.*   # "<anything> | Page/Guide/..."
      | Company\s+Confidential\b.*        # "Company Confidential ..."
      | Confidential\s*[\||\:].*          # "Confidential | ..."
    )$""",
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)


def clean_text(raw: str) -> str:
    """Apply all cleaning steps to raw document text.

    Parameters
    ----------
    raw:
        The raw text as extracted by a parser.

    Returns
    -------
    str
        Cleaned text, safe to chunk and embed.
    """
    text = _normalize_unicode(raw)
    text = _strip_control_chars(text)
    text = _remove_page_breaks(text)
    text = _collapse_whitespace(text)
    return text


def _normalize_unicode(text: str) -> str:
    """Replace non-ASCII punctuation with ASCII equivalents."""
    # NB: this is deliberately conservative — we don't strip accented letters.
    replacements = {
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
        "\xa0": " ",     # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Normalise to NFKC for consistent encoding.
    return unicodedata.normalize("NFKC", text)


def _strip_control_chars(text: str) -> str:
    return _DISCARD_RE.sub("", text)


def _remove_page_breaks(text: str) -> str:
    text = _PAGE_BREAK_LINES_RE.sub("", text)
    text = _BOILERPLATE_HEADER_RE.sub("", text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and newlines, trim edges."""
    text = re.sub(r"[ \t]+", " ", text)          # internal spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # too many newlines
    return text.strip()
