"""Checks the Stage 1 setup: faster-whisper import, ffmpeg, GPU model load (downloads the
model on first run, ~1.5 GB), then CPU fallback model. Run: python scripts/check_stt_setup.py"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    print("ffmpeg:", shutil.which("ffmpeg") or "NOT FOUND (winget install Gyan.FFmpeg, then reopen the terminal)")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is NOT installed. Run: pip install -r requirements.txt")
        return 1
    try:
        import ctranslate2
        print("CUDA devices seen by CTranslate2:", ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001
        print("Could not query CUDA:", exc)
    for device, model, compute in (("cuda", cfg.whisper_model, cfg.whisper_compute_type),
                                   ("cpu", cfg.whisper_cpu_model, cfg.whisper_cpu_compute_type)):
        print(f"\nLoading {model} on {device} ({compute}); the first run downloads the model...")
        t = time.time()
        try:
            WhisperModel(model, device=device, compute_type=compute)
            print(f"OK: {model} on {device} loaded in {time.time() - t:.1f}s")
            if device == "cuda":
                print("GPU setup is working. CPU fallback not needed.")
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {model} on {device}: {exc}")
            if device == "cuda":
                print("-> CUDA 12 cuBLAS / cuDNN 9 is probably missing. The app will use the CPU fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
