"""Test Gemini directly and print the exact result.  Run: python scripts/check_gemini.py"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import load_config  # noqa: E402


def redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def list_models(key: str) -> bool:
    print("1) Reaching the Gemini API and listing models for your key...")
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key, "User-Agent": "meeting-assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"   FAILED: HTTP {exc.code}: {redact(exc.read().decode(errors='replace')[:400], key)}")
        return False
    except OSError as exc:
        print(f"   FAILED to connect: {redact(str(exc), key)}")
        print("   -> network, firewall, proxy or VPN problem reaching generativelanguage.googleapis.com")
        return False
    names = sorted(m["name"].removeprefix("models/") for m in data.get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", []))
    print(f"   OK. {len(names)} models support generateContent. Flash models:")
    for n in names:
        if "flash" in n:
            print("     ", n)
    return True


def rest_call(key: str, model: str, timeout: int = 60) -> bool:
    import time
    print(f"\n2) Plain HTTP call to '{model}' ({timeout}s timeout)...")
    body = json.dumps({"contents": [{"parts": [{"text": "Reply with the single word: OK"}]}]}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, method="POST",
        headers={"x-goog-api-key": key, "Content-Type": "application/json", "User-Agent": "meeting-assistant/1.0"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.load(resp)
        text = out["candidates"][0]["content"]["parts"][0].get("text", "")
        print(f"   OK in {time.time() - t:.1f}s, model replied: {text.strip()[:60]}")
        return True
    except urllib.error.HTTPError as exc:
        body = redact(exc.read().decode(errors='replace'), key)
        try:
            body = json.loads(body)["error"]["message"]
        except Exception:  # noqa: BLE001
            body = body[:200]
        print(f"   FAILED: HTTP {exc.code}: {body}")
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED after {time.time() - t:.1f}s: {type(exc).__name__}: {redact(str(exc), key)[:300]}")
    return False


def sdk_call(key: str, model: str) -> None:
    import time
    print(f"\n3) Same call through the google-genai SDK (60s timeout)...")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("   google-genai is not installed. Run: pip install -r requirements.txt")
        return
    t = time.time()
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
        resp = client.models.generate_content(model=model, contents="Reply with the single word: OK")
        print(f"   OK in {time.time() - t:.1f}s, model replied:", (resp.text or "").strip()[:60])
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED after {time.time() - t:.1f}s: {type(exc).__name__} (code {getattr(exc, 'code', None)}): "
              f"{redact(str(exc), key)[:600]}")


def main() -> int:
    cfg = load_config()
    if not cfg.gemini_api_key:
        print("GEMINI_API_KEY is not set in .env")
        return 1
    if not list_models(cfg.gemini_api_key):
        return 0
    candidates = [cfg.gemini_model, *cfg.gemini_fallback_models, "gemini-flash-latest"]
    working = []
    for m in dict.fromkeys(candidates):
        if rest_call(cfg.gemini_api_key, m, timeout=30):
            working.append(m)
    print("\n=== RESULT ===")
    if working:
        print("Working models:", ", ".join(working))
        print(f"Put this in .env:  GEMINI_MODEL={working[0]}")
    else:
        print("No model answered. Try again in a few minutes (Google-side overload) or create a key in a new AI Studio project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
