"""Data models. `Llm*` models are what Gemini is asked to return (all fields required, no
defaults, flat types). `MeetingRecord` is the validated object every export is built from."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

UNSPECIFIED = "unspecified"

_EMPTY_MARKERS = {
    "", "n/a", "na", "n.a.", "none", "null", "nil", "tbd", "tba", "tbc", "unknown",
    "unassigned", "not specified", "not stated", "not mentioned", "unspecified", "unclear",
    "to be determined", "to be decided", "no owner", "no deadline", "no one", "nobody",
    "-", "--", "—", "?", "??",
}


def normalize_unspecified(value: Any) -> str:
    """Map empty/null/'N/A'/'TBD'-style values to exactly "unspecified"; strip others."""
    if value is None:
        return UNSPECIFIED
    text = str(value).strip().strip(".").strip()
    return UNSPECIFIED if text.lower() in _EMPTY_MARKERS else str(value).strip()


# ---------------------------------------------------------------- final record
class MinutesSection(BaseModel):
    title: str
    points: list[str] = Field(default_factory=list)


class KeyDecision(BaseModel):
    decision: str
    context: str | None = None
    evidence: list[str] = Field(default_factory=list)
    evidence_verified: bool = False

    @field_validator("context")
    @classmethod
    def _blank_context(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


ActionStatus = Literal["confirmed", "conditional", "unassigned"]


class ActionItem(BaseModel):
    task: str
    owner: str = UNSPECIFIED
    deadline: str = UNSPECIFIED
    status: ActionStatus = "confirmed"      # confirmed | conditional | unassigned
    condition: str | None = None            # e.g. "only if the load test passes"; deadline corrections
    evidence: list[str] = Field(default_factory=list)
    evidence_verified: bool = False

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> str:
        return normalize_unspecified(v)

    @field_validator("condition")
    @classmethod
    def _blank_condition(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


class MeetingRecord(BaseModel):
    summary: str
    minutes: list[MinutesSection] = Field(default_factory=list)
    key_decisions: list[KeyDecision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


# ---------------------------------------------------------------- LLM-facing schema (Stage 3)
# Mirrors the JSON in prompts/document_system.txt exactly: all fields required, flat types.
class ScanItem(BaseModel):
    quote: str
    kind: Literal["task", "decision", "rejected", "proposal", "info"]


class LlmMinute(BaseModel):
    topic: str
    points: list[str]


class LlmDecision(BaseModel):
    decision: str
    evidence: str


class LlmActionItem(BaseModel):
    task: str
    owner: str
    deadline: str
    status: ActionStatus
    condition: str
    evidence: str


class LlmRecord(BaseModel):
    commitment_scan: list[ScanItem]      # first on purpose: forces the model to find everything
    summary: str
    minutes: list[LlmMinute]
    decisions: list[LlmDecision]
    action_items: list[LlmActionItem]


def llm_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $ref inlined and title/default keywords removed, to stay
    inside the JSON-schema subset Gemini accepts. Property names are never dropped."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        if in_properties:  # keys are property names here, keep them all
            return {k: walk(v) for k, v in node.items()}
        return {k: walk(v, in_properties=(k == "properties"))
                for k, v in node.items() if k not in ("title", "default")}

    return walk(schema)
