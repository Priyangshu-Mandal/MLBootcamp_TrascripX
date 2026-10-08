"""Word-level highlighting of what refinement changed (optional UI feature)."""
from __future__ import annotations

import html
import re
from difflib import SequenceMatcher

_TOK = re.compile(r"\s+|\S+")


def highlight_changes(raw: str, refined: str) -> str:
    """HTML of `refined` with changed words wrapped in <mark>. All text is HTML-escaped."""
    a, b = _TOK.findall(raw), _TOK.findall(refined)
    out: list[str] = []
    for op, _i1, _i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        chunk = "".join(b[j1:j2])
        if op == "equal":
            out.append(html.escape(chunk))
        elif op in ("replace", "insert") and chunk.strip():
            out.append(f"<mark>{html.escape(chunk.strip())}</mark>"
                       + html.escape(chunk[len(chunk.rstrip()):]))
        else:
            out.append(html.escape(chunk))
    # "$" would start LaTeX math in Streamlit markdown; use the HTML entity.
    return "".join(out).replace("$", "&#36;").replace("\n", "<br>")


def count_changes(raw: str, refined: str) -> int:
    a, b = raw.split(), refined.split()
    return sum(1 for op, *_ in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal")
