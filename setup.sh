#!/usr/bin/env bash
# Linux / macOS setup: virtual environment, dependencies, ffmpeg check, .env, tests, model check.
# Run from the project folder:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n== %s ==\n' "$1"; }

step "1/5 Python virtual environment"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "ERROR: $PY not found. Install Python 3.10-3.12."; exit 1; }
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

step "2/5 ffmpeg"
if command -v ffmpeg >/dev/null; then echo "ffmpeg found."; else
  echo "WARNING: ffmpeg not found. Install it (Ubuntu/Debian: sudo apt install ffmpeg; macOS: brew install ffmpeg)."
fi

step "3/5 .env"
if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - add GROQ_API_KEY and GEMINI_API_KEY."; else echo ".env already exists."; fi

step "4/5 GPU (optional)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv
  echo "If the GPU model fails to load, install the CUDA 12 runtime libraries:"
  echo "  pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"
else
  echo "No NVIDIA driver detected: transcription will use the CPU fallback (small.en)."
fi

step "5/5 Tests, then model check"
python -m unittest discover -s tests || echo "WARNING: some unit tests failed (see above)."
python scripts/check_stt_setup.py || true

printf '\nDone. Activate the environment with:  source .venv/bin/activate\n'
printf 'Start the app with:                    streamlit run app.py\n'
