"""Stage 1: ingestion and speech-to-text with local faster-whisper.

Flow: validate upload -> ffmpeg to 16 kHz mono WAV -> faster-whisper (VAD on) ->
TranscriptResult. GPU (large-v3-turbo, float16) is tried first when available; any CUDA
problem falls back to CPU (small.en, int8) with a warning. The model is released afterwards.
"""
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config, load_config
from .errors import (
    AudioDecodeError, EmptyFileError, FFmpegNotFoundError, GPUError, ModelLoadError,
    NoSpeechError, TranscriptionError, UnsupportedFileTypeError,
)

log = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"
_CUDA_MARKERS = ("cuda", "cublas", "cudnn", "cudart", "nvrtc", "out of memory", "gpu")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str                      # raw transcript, paragraphs separated by blank lines
    segments: list[Segment]        # timestamps kept internally
    language: str
    duration: float
    model_name: str
    device: str
    compute_type: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- validation
def validate_audio_file(path: str | Path, original_name: str | None = None,
                        config: Config | None = None) -> None:
    """Raise a clear error if the file type is unsupported or the file is empty/missing."""
    cfg = config or load_config()
    p = Path(path)
    name = original_name or p.name
    ext = Path(name).suffix.lower()
    if ext not in cfg.allowed_extensions:
        raise UnsupportedFileTypeError(
            f"'{name}' is not a supported file type ({ext or 'no extension'}).",
            f"Upload one of: {', '.join(cfg.allowed_extensions)}.")
    if not p.is_file():
        raise AudioDecodeError(f"'{name}' could not be found or read.")
    size = p.stat().st_size
    if size == 0:
        raise EmptyFileError(f"'{name}' is empty (0 bytes).", "Upload a file that contains audio.")
    if size > cfg.max_upload_mb * 1024 * 1024:
        raise UnsupportedFileTypeError(
            f"'{name}' is larger than the {cfg.max_upload_mb} MB limit.",
            "Trim or compress the recording and try again.")


# ---------------------------------------------------------------- ffmpeg
def convert_to_wav(src: str | Path, dst: str | Path, config: Config | None = None) -> float:
    """Convert any ffmpeg-readable file to 16 kHz mono WAV. Returns duration in seconds."""
    cfg = config or load_config()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegNotFoundError(
            "ffmpeg is not installed or not on PATH.",
            "Install it (Windows: `winget install Gyan.FFmpeg`; Linux: `sudo apt install ffmpeg`).")
    cmd = [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.ffmpeg_timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out while converting the audio.") from exc
    except OSError as exc:
        raise FFmpegNotFoundError(f"ffmpeg could not be started: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-300:]
        raise AudioDecodeError(
            "the audio is unreadable or corrupt (ffmpeg could not decode it).",
            f"Details: {tail}" if tail else None)
    try:
        with wave.open(str(dst), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, FileNotFoundError) as exc:
        raise AudioDecodeError("the audio contains no decodable audio stream.") from exc
    if duration < cfg.min_audio_seconds:
        raise NoSpeechError(f"the audio is too short ({duration:.2f}s) to contain speech.")
    return duration


# ---------------------------------------------------------------- device / model
def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # missing package, driver problem, etc.
        return False


def register_nvidia_dll_dirs() -> list[str]:
    """Windows: make the cuBLAS/cuDNN DLLs from the `nvidia-*-cu12` pip wheels findable.

    CTranslate2 loads these libraries lazily, at the first GPU inference, via the DLL search
    path. Does nothing off Windows or when those wheels are not installed.
    """
    if not _IS_WINDOWS:
        return []
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    added: list[str] = []
    for base in spec.submodule_search_locations:
        for bin_dir in sorted(Path(base).glob("*/bin")):
            d = str(bin_dir)
            if d in os.environ.get("PATH", "").split(os.pathsep):
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            added.append(d)
    if added:
        log.info("Added NVIDIA DLL directories: %s", added)
    return added


def _looks_like_cuda_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _CUDA_MARKERS)


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ModelLoadError(
            "faster-whisper is not installed.", "Run `pip install -r requirements.txt`.") from exc
    return WhisperModel


def _plans(cfg: Config) -> list[tuple[str, str, str]]:
    """Ordered (device, model, compute_type) attempts."""
    gpu = (cfg.whisper_device, cfg.whisper_model, cfg.whisper_compute_type)
    cpu = ("cpu", cfg.whisper_cpu_model, cfg.whisper_cpu_compute_type)
    if cfg.whisper_device == "cpu":
        return [cpu]
    if cfg.whisper_device == "cuda":
        return [("cuda", cfg.whisper_model, cfg.whisper_compute_type), cpu]
    return ([("cuda", cfg.whisper_model, cfg.whisper_compute_type), cpu]
            if cuda_available() else [cpu])


def _release(model) -> None:
    """Drop the model and free GPU memory."""
    del model
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _group_paragraphs(segments: list[Segment], gap: float) -> str:
    paras: list[list[str]] = []
    prev_end: float | None = None
    for seg in segments:
        if prev_end is None or seg.start - prev_end >= gap:
            paras.append([])
        paras[-1].append(seg.text)
        prev_end = seg.end
    return "\n\n".join(" ".join(p) for p in paras)


def _load_wav_array(wav: Path):
    """Read our own 16 kHz mono PCM16 WAV into a float32 array.

    Passing an array to faster-whisper means it never has to decode the file itself, which
    avoids version problems in its PyAV (`av`) dependency.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ModelLoadError("numpy is not installed.", "Run `pip install -r requirements.txt`.") from exc
    try:
        with wave.open(str(wav), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise AudioDecodeError("the converted audio has an unexpected format.")
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDecodeError("the converted audio could not be read.") from exc
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _run_plan(audio, duration: float, plan: tuple[str, str, str], cfg: Config,
              progress: Callable[[float], None] | None) -> tuple[list[Segment], str]:
    device, model_name, compute = plan
    whisper_cls = _import_whisper_model()
    log.info("Loading faster-whisper %s on %s (%s)", model_name, device, compute)
    try:
        model = whisper_cls(model_name, device=device, compute_type=compute)
    except Exception as exc:
        if _looks_like_cuda_error(exc):
            raise GPUError(f"CUDA problem while loading {model_name}: {exc}") from exc
        raise ModelLoadError(
            f"could not load Whisper model '{model_name}': {exc}",
            "The first run downloads the model; check your internet connection and disk space.") from exc
    try:
        seg_iter, info = model.transcribe(
            audio, language=cfg.whisper_language or None, beam_size=cfg.whisper_beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms,
                            "min_speech_duration_ms": 250})
        segments: list[Segment] = []
        for s in seg_iter:  # lazy generator: decoding errors surface here
            text = (s.text or "").strip()
            if text:
                segments.append(Segment(float(s.start), float(s.end), text))
            if progress and duration > 0:
                progress(min(float(s.end) / duration, 1.0))
        return segments, getattr(info, "language", cfg.whisper_language)
    finally:
        _release(model)


def transcribe_wav(wav: str | Path, duration: float, config: Config | None = None,
                   progress: Callable[[float], None] | None = None) -> TranscriptResult:
    """Transcribe an already-converted WAV, falling back from GPU to CPU on CUDA trouble."""
    cfg = config or load_config()
    register_nvidia_dll_dirs()
    audio = _load_wav_array(Path(wav))
    plans = _plans(cfg)
    warnings: list[str] = []
    for i, plan in enumerate(plans):
        is_last = i == len(plans) - 1
        try:
            segments, language = _run_plan(audio, duration, plan, cfg, progress)
        except (GPUError, ModelLoadError) as exc:
            if not is_last and isinstance(exc, GPUError):
                warnings.append(f"GPU unavailable ({exc.message}); falling back to CPU "
                                f"with {plans[i + 1][1]} (slower, less accurate).")
                log.warning(warnings[-1])
                continue
            raise
        except Exception as exc:
            if _looks_like_cuda_error(exc):
                if not is_last:
                    warnings.append(f"CUDA error during transcription ({exc}); retrying on CPU "
                                    f"with {plans[i + 1][1]} (slower, less accurate).")
                    log.warning(warnings[-1])
                    continue
                raise GPUError(f"CUDA error during transcription: {exc}",
                               "Set WHISPER_DEVICE=cpu or update your NVIDIA driver/cuDNN.") from exc
            raise TranscriptionError(f"speech-to-text failed: {exc}") from exc
        if not segments:
            raise NoSpeechError("no speech was detected in the recording.",
                                "Check that the file has audible English speech.")
        device, model_name, compute = plan
        text = _group_paragraphs(segments, cfg.paragraph_gap_s)
        return TranscriptResult(text=text, segments=segments, language=language,
                                duration=duration, model_name=model_name, device=device,
                                compute_type=compute, warnings=warnings)
    raise TranscriptionError("no transcription plan could be run.")  # pragma: no cover


def transcribe_audio(path: str | Path, original_name: str | None = None,
                     config: Config | None = None,
                     progress: Callable[[float], None] | None = None) -> TranscriptResult:
    """Stage 1 entry point: validate, convert, transcribe. Raises TranscribingError subclasses."""
    cfg = config or load_config()
    validate_audio_file(path, original_name, cfg)
    with tempfile.TemporaryDirectory(prefix="meeting_stt_") as tmp:
        wav = Path(tmp) / "audio.wav"
        duration = convert_to_wav(path, wav, cfg)
        return transcribe_wav(wav, duration, cfg, progress)


if __name__ == "__main__":  # quick manual check: python -m pipeline.stt recording.mp3
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pipeline.stt <audio file>")
    try:
        res = transcribe_audio(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        sys.exit(getattr(e, "user_message", str(e)))
    print(f"[{res.model_name} on {res.device}/{res.compute_type}, {res.duration:.1f}s]")
    for w in res.warnings:
        print("WARNING:", w)
    print(res.text)
