"""Check that the Stage 3 model set in .env really works for this app.

Run (with the project's venv Python):
    python scripts/check_model.py                 # tests GEMINI_MODEL from .env
    python scripts/check_model.py gemini-3.5-flash   # tests a model ID given on the command line

It sends a short made-up meeting through the app's own Stage 3 code (same prompt, same JSON-schema
mode, same parsing), with fallbacks switched OFF so only this one model is tested.
"""
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import document  # noqa: E402
from pipeline.config import load_config  # noqa: E402

SAMPLE = (
    "Okay, let's start with the release plan. I suggest we move the launch to the twelfth. "
    "Sounds good, let's do that.\n\n"
    "Maya, can you update the pricing page? Yes, I will do it by Thursday. "
    "Somebody should also write the release notes. Should we add a dark mode? "
    "Maybe later, let's see after the launch."
)


def check(model: str, client=None) -> bool:
    cfg = dataclasses.replace(load_config(), gemini_model=model, gemini_fallback_models=(),
                              document_max_retries=1)
    if client is None and not cfg.gemini_api_key:
        print("FAILED: GEMINI_API_KEY is not set in .env")
        return False
    print(f"Testing Stage 3 with model: {model}")
    t = time.time()
    try:
        res = document.generate_record(SAMPLE, cfg, client=client)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED after {time.time() - t:.1f}s: "
              f"{getattr(exc, 'user_message', None) or f'{type(exc).__name__}: {exc}'}")
        return False
    rec = res.record
    print(f"\nOK in {time.time() - t:.1f}s. The model answered with a valid record:")
    print(f"  decisions   : {len(rec.key_decisions)}")
    for d in rec.key_decisions:
        print(f"      - {d.decision}")
    print(f"  action items: {len(rec.action_items)}")
    for a in rec.action_items:
        print(f"      - {a.task}  [owner: {a.owner}, deadline: {a.deadline}, status: {a.status}]")
    for w in res.warnings:
        print("  note:", w)
    print("\nExpected for this sample: 1 decision (move launch to the twelfth), 2 action items "
          "(Maya: pricing page by Thursday; release notes: unassigned), and dark mode NOT a decision.")
    print(f"\nRESULT: you can use it. Put this in .env:  GEMINI_MODEL={model}")
    return True


if __name__ == "__main__":
    chosen = sys.argv[1] if len(sys.argv) > 1 else load_config().gemini_model
    sys.exit(0 if check(chosen) else 1)
