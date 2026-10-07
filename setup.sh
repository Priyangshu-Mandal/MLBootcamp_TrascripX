#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Setup complete. Install ffmpeg, export QWEN_API_KEY, then run: streamlit run app.py"
