"""Stage 2: domain-aware transcript refinement with openai/gpt-oss-120b on Groq.

The model does NOT rewrite the transcript. For each section of the raw transcript it returns a JSON list
of proposed corrections (prompts/refine_system.txt), and Python applies them:

  * vet_corrections (here): drops malformed corrections, no-ops, anything that changes a negation,
    hedge or commitment word, and corrections whose "support" quote is not in the transcript.
  * apply_corrections (here): each correction must quote its exact context in the transcript, and a
    day / number / date may only change to a value mentioned elsewhere in the transcript.

Because only the listed snippets are replaced, nothing else in the transcript can change. Rejected
corrections are kept in the result and shown in the interface. 429s and transient errors are retried
with exponential backoff.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .chunking import Chunk, chunk_text, join_chunks
from .config import Config, load_config
from .errors import (
    InvalidAPIKeyError, MissingAPIKeyError, MissingDependencyError, NetworkError,
    RateLimitedError, RefinementError,
)
from .evidence import tokenize

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "refine_system.txt"

_WORD_RE = re.compile(r"[a-z']+")
_NEGATIONS = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "cannot",
              "without"}
_HEDGES = {"might", "maybe", "perhaps", "probably", "possibly", "likely", "unlikely",
           "could", "guess", "suppose", "think", "should", "only"}
_COMMITS = {"will", "shall", "gonna", "wanna"}


class ModelState:
    """Tracks the Groq model in use; moves to a configured fallback if one is unavailable."""

    def __init__(self, cfg: Config) -> None:
        self.models = [cfg.groq_model, *[m for m in cfg.groq_fallback_models if m != cfg.groq_model]]
        self.index = 0
        self.notes: list[str] = []

    @property
    def model(self) -> str:
        return self.models[self.index]

    def advance(self) -> bool:
        if self.index + 1 >= len(self.models):
            return False
        self.notes.append(f"Groq model '{self.model}' was not available for this API key; "
                          f"used '{self.models[self.index + 1]}' instead.")
        log.warning(self.notes[-1])
        self.index += 1
        return True


@dataclass
class RefinementResult:
    text: str
    model: str
    chunks_total: int
    fallback_chunks: list[int] = field(default_factory=list)  # 1-based; raw text kept (unusable model reply)
    warnings: list[str] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)     # corrections that were applied
    rejected: list[dict] = field(default_factory=list)    # corrections that were refused, each with "why"


# ------------------------------------------------------------------ correction vetting
def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _is_negation(w: str) -> bool:
    return w in _NEGATIONS or w.endswith("n't")


def _is_commit(w: str) -> bool:
    return w in _COMMITS or w.endswith("'ll")


def _modal_profile(text: str) -> tuple[Counter, Counter, Counter]:
    toks = _tokens(text)
    return (Counter(w for w in toks if _is_negation(w)),
            Counter(w for w in toks if w in _HEDGES),
            Counter(w for w in toks if _is_commit(w)))


def vet_corrections(corrections: list, text: str) -> tuple[list[dict], list[dict]]:
    """Split proposed corrections into (kept, rejected). Rejected ones carry a "why"."""
    kept: list[dict] = []
    rejected: list[dict] = []
    norm_text = " ".join(tokenize(text))

    def reject(c: Any, why: str) -> None:
        base = c if isinstance(c, dict) else {"context": str(c)[:80]}
        rejected.append({**base, "why": why})

    for c in corrections:
        if not isinstance(c, dict) or not all(isinstance(c.get(k), str) for k in ("context", "original", "replacement")):
            reject(c, "malformed correction")
            continue
        original, replacement = c["original"], c["replacement"]
        if not original.strip() or original.strip() == replacement.strip():
            reject(c, "empty or no-op correction")
        elif _modal_profile(original) != _modal_profile(replacement):
            reject(c, "would change a negation, hedge or commitment word")
        else:
            support = str(c.get("support") or "context").strip()
            if support.lower() not in ("context", "'context'") and " ".join(tokenize(support)) not in norm_text:
                reject(c, "support quote not found in the transcript")
            else:
                kept.append(c)
    return kept, rejected


_PROTECTED_RE = re.compile(
    r"\d|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|"
    r"june|july|august|september|october|november|december|zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|first|second|third|fourth|fifth|tomorrow|today|yesterday)\b", re.I)


def _is_protected(text: str) -> bool:
    """True if the text holds a digit, a day, a month or a number word: values that must not be guessed."""
    return bool(_PROTECTED_RE.search(text))


def apply_corrections(text: str, corrections: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Apply vetted corrections to `text`, each only at its quoted context.

    Returns (new_text, applied, rejected); rejected ones carry a "why". A correction is refused when its
    context is not found verbatim, when its original words are not inside that context, or when it
    changes a protected value (digit, day, month, number word) to something not corroborated by another
    place in the transcript. Everything not quoted stays byte-for-byte identical.
    """
    applied: list[dict] = []
    rejected: list[dict] = []
    for c in corrections:
        context, original, replacement = c["context"], c["original"], c["replacement"]
        pos = text.find(context) if context.strip() else -1
        if pos < 0:
            rejected.append({**c, "why": "context not found verbatim in the transcript"})
            continue
        if original not in context:
            rejected.append({**c, "why": "the original words are not inside the quoted context"})
            continue
        if _is_protected(original) or _is_protected(replacement):
            elsewhere = text[:pos] + " " + text[pos + len(context):]
            if not re.search(r"(?<!\w)" + re.escape(replacement.strip()) + r"(?!\w)", elsewhere, re.I):
                rejected.append({**c, "why": "protected value (day/number/date) not corroborated elsewhere"})
                continue
        text = text[:pos] + context.replace(original, replacement, 1) + text[pos + len(context):]
        applied.append(c)
    return text, applied, rejected


def parse_corrections(reply: str) -> list:
    """Extract the {"corrections": [...]} object from a model reply (tolerates code fences and prose)."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(reply[start:end + 1])
    corrections = data.get("corrections") if isinstance(data, dict) else None
    if not isinstance(corrections, list):
        raise ValueError('the JSON object has no "corrections" list')
    return corrections


# ------------------------------------------------------------------ API plumbing
def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RefinementError(f"refinement prompt file not found: {path}") from exc


def _make_client(cfg: Config):
    if not cfg.groq_api_key:
        raise MissingAPIKeyError("GROQ_API_KEY is not set.",
                                 "Add it to your .env file (free key: https://console.groq.com/keys).")
    try:
        from groq import Groq
    except ImportError as exc:
        raise MissingDependencyError("the 'groq' package is not installed.",
                                     "Run `pip install -r requirements.txt`.") from exc
    return Groq(api_key=cfg.groq_api_key, max_retries=0, timeout=90.0)  # we do our own retries


def _status(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers else None
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _complete(client: Any, cfg: Config, messages: list[dict], sleep: Callable[[float], None],
              state: ModelState):
    """chat.completions.create with exponential backoff for 429 and transient errors."""
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=state.model, messages=messages, temperature=cfg.refine_temperature)
        except Exception as exc:  # noqa: BLE001 - classified below
            status = _status(exc)
            name = type(exc).__name__
            text = str(exc).lower()
            if status in (401, 403):
                raise InvalidAPIKeyError("Groq rejected the API key.",
                                         "Check GROQ_API_KEY in your .env file.") from exc
            if status == 429 and ("per day" in text or "tokens per day" in text):
                raise RateLimitedError("the Groq daily quota is exhausted.",
                                       "Wait for the quota to reset or use another key.") from exc
            is_rate = status == 429
            is_transient = (status is not None and (status >= 500 or status == 408)) or \
                name in ("APIConnectionError", "APITimeoutError") or \
                isinstance(exc, (ConnectionError, TimeoutError))
            if not (is_rate or is_transient):
                if status == 404 or "model_not_found" in text:
                    if state.advance():
                        continue
                    raise RefinementError(
                        f"Groq cannot find or give this API key access to the model '{state.model}'.",
                        "Run `python scripts/list_groq_models.py` to see which models your key can use, "
                        "then set GROQ_MODEL in your .env file.") from exc
                if status in (400, 422):
                    raise RefinementError(
                        f"Groq rejected the request ({status}): {exc}",
                        f"Check GROQ_MODEL ('{state.model}') is a valid, available model ID.") from exc
                raise RefinementError(f"unexpected error from Groq: {name}: {exc}") from exc
            if attempt >= cfg.refine_max_retries:
                if is_rate:
                    raise RateLimitedError("Groq rate limit still hit after retries.",
                                           "Wait a minute and try again, or use a shorter recording.") from exc
                raise NetworkError("could not reach the Groq API.",
                                   "Check your internet connection and try again.") from exc
            delay = _retry_after(exc)
            if delay is None:
                delay = min(cfg.refine_backoff_base_s * 2 ** attempt, cfg.refine_backoff_cap_s)
            delay = min(delay, 120.0) + random.uniform(0, 0.25)
            attempt += 1
            log.warning("Groq %s (attempt %d/%d); retrying in %.1fs",
                        "rate limit" if is_rate else "transient error", attempt,
                        cfg.refine_max_retries, delay)
            sleep(delay)


def refine_chunk(client: Any, cfg: Config, system_prompt: str, raw_chunk: str,
                 sleep: Callable[[float], None] = time.sleep, state: ModelState | None = None) -> list | None:
    """Ask the model for corrections to one chunk. Returns the proposed list, or None if the model
    twice failed to return usable JSON."""
    state = state or ModelState(cfg)
    user = f"<transcript>\n{raw_chunk}\n</transcript>"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    for attempt in range(2):
        resp = _complete(client, cfg, messages, sleep, state)
        try:
            reply = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise RefinementError("Groq returned an unexpected response format.") from exc
        try:
            return parse_corrections(reply)
        except ValueError as exc:  # includes json.JSONDecodeError
            log.warning("Refinement reply was not usable JSON (%s)%s", exc, "; retrying" if attempt == 0 else "")
            messages = messages + [
                {"role": "assistant", "content": reply[:4000]},
                {"role": "user", "content": 'Your reply was not valid JSON in the required schema. Reply with ONLY '
                                            'the JSON object {"corrections": [...]} and nothing else.'}]
    return None


def refine_transcript(raw_text: str, config: Config | None = None, *, client: Any = None,
                      progress: Callable[[float], None] | None = None,
                      sleep: Callable[[float], None] = time.sleep) -> RefinementResult:
    """Stage 2 entry point: raw transcript -> refined transcript (raw text with vetted corrections applied)."""
    cfg = config or load_config()
    if not raw_text or not raw_text.strip():
        raise RefinementError("the raw transcript is empty, so there is nothing to refine.")
    system_prompt = load_system_prompt()
    if client is None:
        client = _make_client(cfg)
    chunks: list[Chunk] = chunk_text(raw_text, cfg.refine_chunk_chars)
    state = ModelState(cfg)
    outputs: list[str] = []
    fallback: list[int] = []
    warnings: list[str] = []
    applied_all: list[dict] = []
    rejected_all: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        proposed = refine_chunk(client, cfg, system_prompt, chunk.text, sleep, state)
        if proposed is None:
            msg = (f"Section {i} of {len(chunks)} was kept as the raw transcript because the model did not "
                   "return usable corrections.")
            log.warning(msg)
            warnings.append(msg)
            fallback.append(i)
            outputs.append(chunk.text)
        else:
            kept, vetoed = vet_corrections(proposed, chunk.text)
            refined, applied, rejected = apply_corrections(chunk.text, kept)
            outputs.append(refined)
            applied_all += applied
            rejected_all += vetoed + rejected
        if progress:
            progress(i / len(chunks))
    return RefinementResult(text=join_chunks(outputs, chunks), model=state.model,
                            chunks_total=len(chunks), fallback_chunks=fallback,
                            warnings=state.notes + warnings, applied=applied_all, rejected=rejected_all)


if __name__ == "__main__":  # python -m pipeline.refine raw_transcript.txt
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pipeline.refine <raw transcript .txt>")
    try:
        res = refine_transcript(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        sys.exit(getattr(e, "user_message", str(e)))
    print(f"[{res.model}; {res.chunks_total} section(s); {len(res.applied)} correction(s) applied, "
          f"{len(res.rejected)} rejected]")
    for c in res.applied:
        print(f"  applied : {c['original']!r} -> {c['replacement']!r}  ({c.get('reason', '')})")
    for c in res.rejected:
        print(f"  rejected: {c.get('original', '')!r} -> {c.get('replacement', '')!r}  [{c['why']}]")
    for w in res.warnings:
        print("WARNING:", w)
    print(res.text)
