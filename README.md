# Inter IIT Meeting Assistant

This project implements the design in `ml_ps_design.md` as a Streamlit app with
three explicit stages:

1. Validate the uploaded recording and transcribe it locally with `openai-whisper`.
2. Refine technical terms without summarizing or changing intended meaning.
3. Extract minutes, explicitly agreed decisions, and explicitly assigned action
   items using separate Qwen language-model calls.

Missing owners and deadlines are represented exactly as `"unspecified"`. Model
output is validated before it reaches the UI or downloads.

## Run

```bash
./setup.sh
source .venv/bin/activate
export QWEN_API_KEY="your-qwen-api-key"
streamlit run app.py
```

The transcription stage is local and does not use an OpenAI API key. The
refinement and documentation stages use Qwen through its OpenAI-compatible API.
The default endpoint is DashScope; use `QWEN_BASE_URL` for another compatible
endpoint.

Optional model settings:

```bash
export WHISPER_MODEL="small"
export WHISPER_LANGUAGE="en"
export WHISPER_INITIAL_PROMPT="Cline Hackathon, Kubernetes, TensorFlow, PyTorch"
export QWEN_MODEL="qwen-plus"
export QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

Install `ffmpeg` separately and ensure it is on `PATH`; Whisper requires it to
decode uploaded audio. On Ubuntu: `sudo apt-get install ffmpeg`.

For better recognition of names and project terms, use `small` or `medium`
instead of `base` and put known vocabulary in `WHISPER_INITIAL_PROMPT`. Larger
models are slower and use more VRAM. The prompt helps Whisper recognize terms
but cannot guarantee a correction when the audio is unclear.

The JSON and readable-text downloads are generated from the same validated
`MeetingResult`, so decisions and action items cannot diverge between formats.

## Test

```bash
pytest -q
```
