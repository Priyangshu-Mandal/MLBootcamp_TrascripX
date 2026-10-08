"""Custom exceptions. Each carries the pipeline stage so the UI can name what failed."""
from __future__ import annotations


class MeetingAssistantError(Exception):
    """Base class. `message` is user-facing; `hint` suggests a fix."""

    stage = "Pipeline"

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        text = f"{self.stage} failed: {self.message}"
        return f"{text} {self.hint}" if self.hint else text


class TranscribingError(MeetingAssistantError):
    stage = "Transcribing"


class UnsupportedFileTypeError(TranscribingError):
    """File extension is not an accepted audio/video container."""


class EmptyFileError(TranscribingError):
    """Uploaded file has zero bytes."""


class AudioDecodeError(TranscribingError):
    """ffmpeg could not decode the file (corrupt, truncated, or not really audio)."""


class FFmpegNotFoundError(TranscribingError):
    """ffmpeg binary is not installed or not on PATH."""


class NoSpeechError(TranscribingError):
    """Audio decoded fine but contains no detectable speech."""


class ModelLoadError(TranscribingError):
    """Whisper model could not be loaded (download failure, corrupt cache, out of memory)."""


class GPUError(TranscribingError):
    """CUDA problem that could not be recovered by falling back to CPU."""


class TranscriptionError(TranscribingError):
    """Unexpected failure while running the speech-to-text model."""


class RefiningError(MeetingAssistantError):
    stage = "Refining"


class MissingAPIKeyError(RefiningError):
    """GROQ_API_KEY is not set."""


class InvalidAPIKeyError(RefiningError):
    """Groq rejected the API key (401/403)."""


class RateLimitedError(RefiningError):
    """Groq rate limit still hit after all retries (or the daily quota is exhausted)."""


class NetworkError(RefiningError):
    """Could not reach the Groq API (connection, timeout, or 5xx after all retries)."""


class MissingDependencyError(RefiningError):
    """The groq package is not installed."""


class RefinementError(RefiningError):
    """Any other refinement failure (bad request, unknown model, empty transcript)."""


class DocumentingError(MeetingAssistantError):
    stage = "Generating record"


class MissingGeminiKeyError(DocumentingError):
    """GEMINI_API_KEY is not set."""


class InvalidGeminiKeyError(DocumentingError):
    """Gemini rejected the API key or the request is not permitted."""


class GeminiRateLimitedError(DocumentingError):
    """Gemini rate limit/quota still hit after all retries."""


class GeminiNetworkError(DocumentingError):
    """Could not reach the Gemini API (connection, timeout, 5xx after all retries)."""


class MissingGeminiDependencyError(DocumentingError):
    """The google-genai package is not installed."""


class ExtractionError(DocumentingError):
    """Gemini output could not be turned into a valid meeting record."""


class UnexpectedError(MeetingAssistantError):
    """Any unforeseen failure; `stage` is set on the instance by the orchestrator."""
