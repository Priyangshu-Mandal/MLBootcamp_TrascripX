"""Typed data models used by the meeting pipeline."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionItem:
    task: str
    owner: str
    deadline: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Documentation:
    minutes: str
    decisions: list[str]
    action_items: list[ActionItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "minutes": self.minutes,
            "decisions": self.decisions,
            "action_items": [item.to_dict() for item in self.action_items],
        }


@dataclass(frozen=True)
class MeetingResult:
    raw_transcript: str
    refined_transcript: str
    documentation: Documentation

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_transcript": self.raw_transcript,
            "refined_transcript": self.refined_transcript,
            **self.documentation.to_dict(),
        }
