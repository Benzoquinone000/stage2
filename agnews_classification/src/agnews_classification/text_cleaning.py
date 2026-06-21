"""Text cleaning helpers shared by AG News classification and MLM data prep."""

from __future__ import annotations

import html
import re


_MISSING_NUMERIC_ENTITY_RE = re.compile(r"(?<!&)#([0-9]+);")
_MISSING_HEX_ENTITY_RE = re.compile(r"(?<!&)#x([0-9a-fA-F]+);")
_MISSING_NAMED_ENTITY_RE = re.compile(r"(?<!&)\b(quot|apos|amp|lt|gt|nbsp);")
_ANCHOR_TAG_RE = re.compile(r"</?\s*a\b[^>]*>", re.IGNORECASE)
_MALFORMED_ANCHOR_RE = re.compile(
    r'\(?\s*a\s+href\s*=\s*(?:"[^"]*"|\S+)?\s*>\s*([^)]+)\)?',
    re.IGNORECASE,
)
_KNOWN_HTML_TAG_RE = re.compile(
    r"</?\s*(?:b|i|em|strong|cite|font|br|p|span|div|ul|ol|li)\b[^>]*>",
    re.IGNORECASE,
)
_ANGLE_CONTENT_RE = re.compile(r"<\s*([^<>]{1,80})\s*>")
_LEFTOVER_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S*|\bwww\.[A-Za-z0-9]\S*|\bwww\.(?=\s|$|[),.;:!?])", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _restore_missing_entities(text: str) -> str:
    text = _MISSING_NUMERIC_ENTITY_RE.sub(r"&#\1;", text)
    text = _MISSING_HEX_ENTITY_RE.sub(r"&#x\1;", text)
    return _MISSING_NAMED_ENTITY_RE.sub(r"&\1;", text)


def _unescape_repeatedly(text: str, rounds: int = 2) -> str:
    for _ in range(rounds):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def _replace_angle_content(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not inner:
        return " "
    if re.fullmatch(r"[A-Z0-9][A-Z0-9._/-]{0,30}", inner):
        return f" {inner} "
    return " "


def clean_text(*parts: str) -> str:
    """Normalize common AG News markup, escapes, and whitespace artifacts."""

    text = " ".join(str(part).strip() for part in parts if part and str(part).strip())
    text = _restore_missing_entities(text)
    text = _unescape_repeatedly(text)

    text = re.sub(r"\bAT\s*&\s*T;", "AT&T", text)
    text = re.sub(r"\bAT\s*&\s*T\b", "AT&T", text)
    text = text.replace('\\"', '"').replace("\\'", "'").replace("\\$", "$")
    text = re.sub(r"\\[nrt]+", " ", text)
    text = text.replace("\\", " ")

    text = _ANCHOR_TAG_RE.sub(" ", text)
    text = _MALFORMED_ANCHOR_RE.sub(r" \1 ", text)
    text = _KNOWN_HTML_TAG_RE.sub(" ", text)
    text = _ANGLE_CONTENT_RE.sub(_replace_angle_content, text)
    text = _LEFTOVER_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)

    text = text.replace("\u00a0", " ")
    text = re.sub(r"(\w)\s+'\s*s\b", r"\1's", text)
    text = re.sub(r"(\w)\s+'\s+", r"\1' ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\$\s+([0-9])", r"$\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    return clean_text(text).casefold()
