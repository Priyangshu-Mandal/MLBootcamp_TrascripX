"""List the Groq models your API key can actually use.  Run: python scripts/list_groq_models.py"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import load_config  # noqa: E402


def try_model(key: str, model: str) -> tuple[bool, str]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                       "max_tokens": 16}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "meeting-assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            json.load(resp)
        return True, "works"
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read().decode(errors="replace"))["error"]["message"]
        except Exception:  # noqa: BLE001
            msg = ""
        return False, f"HTTP {exc.code}: {msg[:200]}"
    except OSError as exc:
        return False, f"could not reach Groq: {exc}"


def main() -> int:
    cfg = load_config()
    if not cfg.groq_api_key:
        print("GROQ_API_KEY is not set in .env")
        return 1
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {cfg.groq_api_key}", "User-Agent": "meeting-assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"Groq returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}")
        return 1
    except OSError as exc:
        print(f"Could not reach Groq: {exc}")
        return 1
    ids = sorted(m["id"] for m in data.get("data", []))
    print(f"Models available to your key ({len(ids)}):")
    for i in ids:
        print("  ", i, "  <-- configured GROQ_MODEL" if i == cfg.groq_model else "")
    if cfg.groq_model not in ids:
        print(f"\nYour configured GROQ_MODEL '{cfg.groq_model}' is NOT in this list.")
        print("Pick a chat model from the list above and set GROQ_MODEL=<id> in your .env file.")
    print(f"\nTesting the configured model '{cfg.groq_model}' with a real request...")
    ok, detail = try_model(cfg.groq_api_key, cfg.groq_model)
    print("  RESULT: works." if ok else f"  RESULT: NOT usable on your key ({detail}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
