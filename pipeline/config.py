"""Central configuration. Non-secret settings have defaults and can be overridden via
environment variables or `.env`; API keys come only from the environment (`.env`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time; requirements.txt installs it.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

ALLOWED_EXTENSIONS = (
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".mp4", ".webm", ".aac",
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Stage 1: speech-to-text (faster-whisper, local)
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"            # auto | cuda | cpu
    whisper_compute_type: str = "float16"   # used on CUDA
    whisper_cpu_model: str = "small.en"     # fallback when no usable GPU
    whisper_cpu_compute_type: str = "int8"
    whisper_language: str = "en"
    whisper_beam_size: int = 5
    vad_min_silence_ms: int = 500
    paragraph_gap_s: float = 2.0            # pause that starts a new paragraph in the raw text
    ffmpeg_timeout_s: int = 600
    max_upload_mb: int = 500
    min_audio_seconds: float = 0.5
    allowed_extensions: tuple[str, ...] = ALLOWED_EXTENSIONS

    # Stage 2: refinement (Groq).
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_models: tuple[str, ...] = ()   # used only if the main model is unavailable
    refine_temperature: float = 0.0
    refine_chunk_chars: int = 12000          # transcripts longer than this are corrected in several parts
    refine_max_retries: int = 5
    refine_backoff_base_s: float = 2.0
    refine_backoff_cap_s: float = 60.0

    # Stage 3: documentation (Gemini). Verify the current model ID in Gemini docs.
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_fallback_models: tuple[str, ...] = ("gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.8-flash")
    document_max_retries: int = 5
    document_timeout_s: float = 120.0
    document_backoff_base_s: float = 3.0
    document_backoff_cap_s: float = 60.0
    document_temperature: float = 0.1
    document_max_chars: int = 400000         # whole transcript goes to Gemini in one call (about 5 hours of speech)
    evidence_fuzzy_threshold: float = 0.92

    # Secrets: read from the environment only; never logged.
    groq_api_key: str = field(default="", repr=False)
    gemini_api_key: str = field(default="", repr=False)


def load_config() -> Config:
    """Build a Config from defaults + environment."""
    d = Config()
    return Config(
        whisper_model=_env("WHISPER_MODEL", d.whisper_model),
        whisper_device=_env("WHISPER_DEVICE", d.whisper_device).lower(),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
        whisper_cpu_model=_env("WHISPER_CPU_MODEL", d.whisper_cpu_model),
        whisper_cpu_compute_type=_env("WHISPER_CPU_COMPUTE_TYPE", d.whisper_cpu_compute_type),
        whisper_language=_env("WHISPER_LANGUAGE", d.whisper_language),
        whisper_beam_size=_env_int("WHISPER_BEAM_SIZE", d.whisper_beam_size),
        vad_min_silence_ms=_env_int("VAD_MIN_SILENCE_MS", d.vad_min_silence_ms),
        paragraph_gap_s=_env_float("PARAGRAPH_GAP_S", d.paragraph_gap_s),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", d.max_upload_mb),
        groq_model=_env("GROQ_MODEL", d.groq_model),
        groq_fallback_models=tuple(m.strip() for m in _env("GROQ_FALLBACK_MODELS", "").split(",") if m.strip()),
        refine_temperature=_env_float("REFINE_TEMPERATURE", d.refine_temperature),
        refine_chunk_chars=_env_int("REFINE_CHUNK_CHARS", d.refine_chunk_chars),
        refine_max_retries=_env_int("REFINE_MAX_RETRIES", d.refine_max_retries),
        gemini_model=_env("GEMINI_MODEL", d.gemini_model),
        gemini_fallback_models=tuple(m.strip() for m in _env(
            "GEMINI_FALLBACK_MODELS", ",".join(d.gemini_fallback_models)).split(",") if m.strip()),
        document_max_retries=_env_int("DOCUMENT_MAX_RETRIES", d.document_max_retries),
        document_timeout_s=_env_float("DOCUMENT_TIMEOUT_S", d.document_timeout_s),
        document_temperature=_env_float("DOCUMENT_TEMPERATURE", d.document_temperature),
        document_max_chars=_env_int("DOCUMENT_MAX_CHARS", d.document_max_chars),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
    )
