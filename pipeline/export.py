"""Exports. Markdown and JSON are both generated from the same validated MeetingRecord, so
they always contain exactly the same decisions and tasks. JSON always carries evidence;
Markdown omits it unless asked."""
from __future__ import annotations

import json
from typing import Any

from .schemas import MeetingRecord

NOT_VERIFIED = "evidence not verified"


def to_json(record: MeetingRecord, metadata: dict[str, Any] | None = None) -> str:
    """Machine-readable record (always includes evidence and evidence_verified)."""
    payload: dict[str, Any] = {"metadata": metadata or {}, **record.model_dump()}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _quote_lines(quotes: list[str], indent: str = "") -> list[str]:
    return [f'{indent}> "{q.strip()}"' for q in quotes]


def to_markdown(record: MeetingRecord, metadata: dict[str, Any] | None = None,
                include_evidence: bool = False) -> str:
    """Human-readable record. Evidence is excluded by default."""
    md = metadata or {}
    out: list[str] = ["# Meeting Record", ""]
    if md.get("source_file"):
        out.append(f"*Source recording:* `{md['source_file']}`  ")
    if md.get("generated_at"):
        out.append(f"*Generated:* {md['generated_at']}")
    if md.get("models"):
        out.append("*Models:* " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in md["models"].items()))
    out += ["", "## Summary", "", record.summary or "_No summary._", "", "## Minutes", ""]
    if record.minutes:
        for sec in record.minutes:
            out.append(f"### {sec.title}")
            out += [f"- {p}" for p in sec.points] or ["- _(no points)_"]
            out.append("")
    else:
        out += ["_No minutes._", ""]

    out += ["## Key Decisions", ""]
    if record.key_decisions:
        for i, d in enumerate(record.key_decisions, 1):
            line = f"{i}. **{d.decision.strip()}**"
            if d.context:
                line += f" — {d.context.strip()}"
            out.append(line)
            if include_evidence:
                if not d.evidence_verified:
                    out.append(f"   - ⚠ {NOT_VERIFIED}")
                out += _quote_lines(d.evidence, "   ")
    else:
        out.append("_No decisions were reached._")
    out.append("")

    out += ["## Action Items", ""]
    if record.action_items:
        has_condition = any(a.condition for a in record.action_items)
        header = "| # | Task | Owner | Deadline | Status |" + (" Condition / note |" if has_condition else "")
        sep = "|---|------|-------|----------|--------|" + ("------------------|" if has_condition else "")
        out += [header, sep]
        for i, a in enumerate(record.action_items, 1):
            row = f"| {i} | {_cell(a.task)} | {_cell(a.owner)} | {_cell(a.deadline)} | {_cell(a.status)} |"
            out.append(row + (f" {_cell(a.condition)} |" if has_condition else ""))
        if include_evidence:
            out += ["", "### Evidence for action items", ""]
            for i, a in enumerate(record.action_items, 1):
                out.append(f"**Task {i}: {a.task.strip()}**" + ("" if a.evidence_verified else f" (⚠ {NOT_VERIFIED})"))
                out += _quote_lines(a.evidence)
                out.append("")
    else:
        out.append("_No action items were assigned._")
    out.append("")
    return "\n".join(out).rstrip() + "\n"
