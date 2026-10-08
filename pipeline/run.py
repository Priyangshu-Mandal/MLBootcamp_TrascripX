"""Orchestrator: runs the three stages in order and reports progress to the UI/CLI.

Plain function calls, no framework. Each stage is also usable on its own (stt.py, refine.py,
document.py). Failures raise `MeetingAssistantError` subclasses whose `.stage` names the stage.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Config, load_config
from .document import DocumentResult, generate_record
from .errors import MeetingAssistantError, MissingAPIKeyError, MissingGeminiKeyError, UnexpectedError
from .refine import RefinementResult, refine_transcript
from .stt import TranscriptResult, transcribe_audio

log = logging.getLogger(__name__)

STAGES = ("Transcribing", "Refining", "Generating record")
# event callback: (stage, status, detail) where status is "start" | "progress" | "done" | "error"
EventCallback = Callable[[str, str, Any], None]


@dataclass
class PipelineResult:
    source_name: str
    transcript: TranscriptResult
    refinement: RefinementResult
    document: DocumentResult
    timings: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""

    @property
    def raw_text(self) -> str:
        return self.transcript.text

    @property
    def refined_text(self) -> str:
        return self.refinement.text

    @property
    def record(self):
        return self.document.record

    @property
    def warnings(self) -> list[str]:
        return [*self.transcript.warnings, *self.refinement.warnings, *self.document.warnings]

    def metadata(self) -> dict[str, Any]:
        t = self.transcript
        return {
            "source_file": self.source_name,
            "generated_at": self.generated_at,
            "audio_duration_seconds": round(t.duration, 1),
            "models": {
                "speech_to_text": f"faster-whisper {t.model_name} ({t.device}, {t.compute_type})",
                "refinement": f"{self.refinement.model} (Groq)",
                "documentation": f"{self.document.model} (Gemini)",
            },
            "corrections": {"applied": len(self.refinement.applied), "rejected": len(self.refinement.rejected)},
            "warnings": self.warnings,
        }


def preflight(cfg: Config) -> None:
    """Fail fast (before the slow transcription) if the LLM stages cannot possibly run."""
    if not cfg.groq_api_key:
        raise MissingAPIKeyError("GROQ_API_KEY is not set.",
                                 "Add it to your .env file (free key: https://console.groq.com/keys).")
    if not cfg.gemini_api_key:
        raise MissingGeminiKeyError("GEMINI_API_KEY is not set.",
                                    "Add it to your .env file (free key: https://aistudio.google.com/apikey).")


def run_pipeline(path: str | Path, original_name: str | None = None, config: Config | None = None,
                 on_event: EventCallback | None = None, *, check_keys: bool = True) -> PipelineResult:
    """Run STT -> refinement -> documentation on one recording."""
    cfg = config or load_config()
    name = original_name or Path(path).name
    emit = on_event or (lambda *_: None)
    timings: dict[str, float] = {}
    current = STAGES[0]

    def stage(index: int, fn: Callable[[], Any], progress_name: str):
        nonlocal current
        current = STAGES[index]
        emit(current, "start", None)
        t0 = time.time()
        try:
            out = fn()
        except MeetingAssistantError as exc:
            emit(current, "error", exc)
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected failure in stage %s", current)
            err = UnexpectedError(f"unexpected {type(exc).__name__}: {exc}")
            err.stage = current
            emit(current, "error", err)
            raise err from exc
        timings[progress_name] = round(time.time() - t0, 1)
        emit(current, "done", out)
        return out

    try:
        if check_keys:
            preflight(cfg)
    except MeetingAssistantError as exc:
        emit("Setup", "error", exc)
        raise

    progress = lambda s: (lambda frac: emit(s, "progress", frac))  # noqa: E731
    transcript = stage(0, lambda: transcribe_audio(path, name, cfg, progress(STAGES[0])), "transcribing")
    refinement = stage(1, lambda: refine_transcript(transcript.text, cfg, progress=progress(STAGES[1])),
                       "refining")
    document = stage(2, lambda: generate_record(refinement.text, cfg, progress=progress(STAGES[2])),
                     "generating_record")
    return PipelineResult(source_name=name, transcript=transcript, refinement=refinement,
                          document=document, timings=timings,
                          generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
