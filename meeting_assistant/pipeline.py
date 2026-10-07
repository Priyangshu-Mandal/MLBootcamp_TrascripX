"""The strict three-stage meeting processing pipeline."""

from __future__ import annotations

from typing import Protocol

from .models import Documentation, MeetingResult


class MeetingProvider(Protocol):
    def transcribe(self, filename: str, audio_bytes: bytes) -> str: ...

    def refine(self, raw_transcript: str) -> str: ...

    def document(self, refined_transcript: str) -> Documentation: ...


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


class InputError(ValueError):
    """Raised when an uploaded recording cannot be processed."""


class MeetingAssistant:
    """Orchestrates ingestion, refinement, and documentation in order."""

    def __init__(self, provider: MeetingProvider) -> None:
        self.provider = provider

    def process(self, filename: str, audio_bytes: bytes) -> MeetingResult:
        self._validate_audio(filename, audio_bytes)
        raw = self.provider.transcribe(filename, audio_bytes)
        if not raw.strip():
            raise InputError("The recording produced no speech transcript.")
        refined = self.provider.refine(raw)
        if not refined.strip():
            raise RuntimeError("The refinement stage returned an empty transcript.")
        documentation = self.provider.document(refined)
        return MeetingResult(raw, refined, documentation)

    @staticmethod
    def _validate_audio(filename: str, audio_bytes: bytes) -> None:
        if not filename or "." not in filename:
            raise InputError("Unsupported file: please upload an audio recording.")
        extension = "." + filename.rsplit(".", 1)[1].lower()
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise InputError(
                f"Unsupported file type '{extension}'. Supported types: "
                f"{', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}."
            )
        if not audio_bytes:
            raise InputError("The uploaded recording is empty.")
