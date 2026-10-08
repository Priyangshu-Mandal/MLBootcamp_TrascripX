"""Split long transcripts into chunks at paragraph / sentence boundaries, with no overlap
(so refined text can be joined back without duplication)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    sep: str  # separator that followed this chunk in the original ("\n\n", " " or "")


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Last resort for a single 'sentence' longer than max_chars: split on words."""
    parts, cur = [], ""
    for word in sentence.split():
        while len(word) > max_chars:  # absurdly long token
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(word[:max_chars])
            word = word[max_chars:]
        if cur and len(cur) + 1 + len(word) > max_chars:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return parts


def _units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Break text into (unit_text, separator_after) pieces, each <= max_chars."""
    units: list[tuple[str, str]] = []
    for para in (p.strip() for p in _PARA_SPLIT.split(text)):
        if not para:
            continue
        para = re.sub(r"[ \t]*\n[ \t]*", " ", para)  # single newlines inside a paragraph
        if len(para) <= max_chars:
            units.append((para, "\n\n"))
            continue
        pieces: list[str] = []
        for sent in _SENT_SPLIT.split(para):
            pieces.extend([sent] if len(sent) <= max_chars else _hard_split(sent, max_chars))
        units.extend((pc, " ") for pc in pieces)
        units[-1] = (units[-1][0], "\n\n")
    return units


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    """Pack paragraphs/sentences into chunks of at most `max_chars` characters."""
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    chunks: list[Chunk] = []
    cur = ""
    cur_sep = ""
    for unit, sep in _units(text, max_chars):
        if cur and len(cur) + len(cur_sep) + len(unit) > max_chars:
            chunks.append(Chunk(cur, cur_sep))
            cur, cur_sep = unit, sep
        elif cur:
            cur = f"{cur}{cur_sep}{unit}"
            cur_sep = sep
        else:
            cur, cur_sep = unit, sep
    if cur:
        chunks.append(Chunk(cur, ""))
    elif not chunks:
        return []
    return chunks


def join_chunks(parts: list[str], chunks: list[Chunk]) -> str:
    """Reassemble processed chunk texts using the original separators."""
    if len(parts) != len(chunks):
        raise ValueError("parts and chunks must have the same length")
    out = []
    for part, ch in zip(parts, chunks):
        out.append(part.strip())
        out.append(ch.sep)
    return "".join(out).strip()
