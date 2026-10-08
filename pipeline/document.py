"""Stage 3: meeting minutes, decisions and action items with Gemini Flash.

Input is the *refined* transcript only, sent whole in one call (so recap statements at the end of a
meeting can be matched with earlier discussion). The model first fills a `commitment_scan`, then the
record (prompts/document_system.txt). Its output is then checked in code:

  * if the model's own scan found more tasks than it listed as action items, the call is repeated
    once with feedback and the better result is kept.
  * evidence.verify_record: every quote is verified against the transcript and flagged if not found;
    items with no evidence at all are dropped.
The result is a validated `MeetingRecord`; Markdown and JSON are both built from it.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .config import Config, load_config
from .errors import (
    DocumentingError, ExtractionError, GeminiNetworkError, GeminiRateLimitedError,
    InvalidGeminiKeyError, MissingGeminiDependencyError, MissingGeminiKeyError,
)
from .evidence import TranscriptIndex, clean_quotes, drop_unsupported_items, verify_record
from .schemas import (
    UNSPECIFIED, ActionItem, KeyDecision, LlmRecord, MeetingRecord, MinutesSection, llm_json_schema,
)

log = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CONNECTION_ERRORS = {"ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException",
                      "RemoteProtocolError", "ReadError", "WriteError", "PoolTimeout"}


@dataclass
class DocumentResult:
    record: MeetingRecord
    model: str
    warnings: list[str] = field(default_factory=list)
    dropped_items: list[str] = field(default_factory=list)
    downgraded_fields: list[str] = field(default_factory=list)
    scan_tasks: int = 0                                      # tasks the model's own scan found

    @property
    def unverified_items(self) -> int:
        return sum(not d.evidence_verified for d in self.record.key_decisions) + \
               sum(not a.evidence_verified for a in self.record.action_items)


# ------------------------------------------------------------------ Gemini plumbing
def load_prompt(name: str) -> str:
    try:
        return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DocumentingError(f"prompt file not found: {PROMPT_DIR / name}") from exc


def _make_client(cfg: Config):
    if not cfg.gemini_api_key:
        raise MissingGeminiKeyError("GEMINI_API_KEY is not set.",
                                    "Add it to your .env file (free key: https://aistudio.google.com/apikey).")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise MissingGeminiDependencyError("the 'google-genai' package is not installed.",
                                           "Run `pip install -r requirements.txt`.") from exc
    return genai.Client(api_key=cfg.gemini_api_key,
                        http_options=types.HttpOptions(timeout=int(cfg.document_timeout_s * 1000)))


def _build_config(system: str, schema: dict, temperature: float) -> Any:
    params = {"system_instruction": system, "temperature": temperature,
              "response_mime_type": "application/json", "response_json_schema": schema}
    try:
        from google.genai import types
        return types.GenerateContentConfig(**params)
    except ImportError:
        return params


def _code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_delay(exc: BaseException) -> float | None:
    m = re.search(r"retry(?:\s+in|Delay)\D{0,6}([\d.]+)\s*s", str(exc), re.I)
    return float(m.group(1)) if m else None


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


class GeminiCaller:
    """Calls Gemini with backoff for 429/5xx/network errors and model-ID fallback."""

    def __init__(self, client: Any, cfg: Config, sleep: Callable[[float], None] = time.sleep) -> None:
        self.client, self.cfg, self.sleep = client, cfg, sleep
        self.models = [cfg.gemini_model, *[m for m in cfg.gemini_fallback_models if m != cfg.gemini_model]]
        self.model = self.models[0]
        self.notes: list[str] = []

    def call(self, system: str, contents: str, schema: dict) -> str:
        attempt = 0
        while True:
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents,
                    config=_build_config(system, schema, self.cfg.document_temperature))
                text = getattr(resp, "text", None)
                if not text or not text.strip():
                    raise ExtractionError("Gemini returned an empty or blocked response.",
                                          "Try again; if it keeps happening the content may be filtered.")
                return text
            except DocumentingError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                code, msg, name = _code(exc), str(exc), type(exc).__name__
                low = msg.lower()
                if code in (401, 403) or (code == 400 and ("api key not valid" in low or "api_key_invalid" in low)):
                    raise InvalidGeminiKeyError("Gemini rejected the API key.",
                                                "Check GEMINI_API_KEY in your .env file.") from exc
                if code == 404 or "is not found" in low or "not supported for generatecontent" in low:
                    nxt = self.models.index(self.model) + 1
                    if nxt < len(self.models):
                        self.notes.append(f"Model '{self.model}' was not available; used '{self.models[nxt]}'.")
                        log.warning(self.notes[-1])
                        self.model = self.models[nxt]
                        continue
                    raise ExtractionError(f"Gemini model '{self.model}' was not found.",
                                          "Set GEMINI_MODEL in .env to a current Flash model ID.") from exc
                is_rate = code == 429
                if is_rate and ("per day" in low or "perday" in low):
                    raise GeminiRateLimitedError("the Gemini daily quota is exhausted.",
                                                 "Wait for the quota to reset or use another key.") from exc
                transient = is_rate or (code is not None and (code >= 500 or code == 408)) or \
                    name in _CONNECTION_ERRORS or isinstance(exc, (ConnectionError, TimeoutError))
                if not transient:
                    raise ExtractionError(f"Gemini request failed ({code or name}): {msg[:300]}") from exc
                if attempt >= self.cfg.document_max_retries:
                    last = _redact(f"{name}{f' {code}' if code else ''}: {msg[:250]}", self.cfg.gemini_api_key)
                    if is_rate:
                        raise GeminiRateLimitedError(f"Gemini rate limit still hit after retries (last error: {last}).",
                                                     "Wait a minute and try again.") from exc
                    raise GeminiNetworkError(f"could not get a response from the Gemini API (last error: {last}).",
                                             "Check your internet connection, or try again in a minute "
                                             "(a 503 means Gemini is overloaded). "
                                             "Run `python scripts/check_gemini.py` for details.") from exc
                nxt = self.models.index(self.model) + 1
                if code is not None and code >= 500 and attempt >= 2 and nxt < len(self.models):
                    self.notes.append(f"Model '{self.model}' was overloaded ({code}); used '{self.models[nxt]}'.")
                    log.warning(self.notes[-1])
                    self.model, attempt = self.models[nxt], 0
                    continue
                delay = _retry_delay(exc)
                if delay is None:
                    delay = min(self.cfg.document_backoff_base_s * 2 ** attempt, self.cfg.document_backoff_cap_s)
                delay = min(delay, 120.0) + random.uniform(0, 0.25)
                attempt += 1
                log.warning("Gemini %s (attempt %d/%d); retrying in %.1fs",
                            "rate limit" if is_rate else "transient error", attempt,
                            self.cfg.document_max_retries, delay)
                self.sleep(delay)


def _parse(model_cls: type[BaseModel], text: str) -> BaseModel:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return model_cls.model_validate_json(text)


def _generate_validated(caller: GeminiCaller, system: str, user: str, model_cls: type[BaseModel]):
    """Generate + validate; one repair retry if the JSON is invalid."""
    schema = llm_json_schema(model_cls)
    text = caller.call(system, user, schema)
    try:
        return _parse(model_cls, text)
    except (ValidationError, ValueError) as first:
        log.warning("Gemini output failed validation; retrying with a repair prompt")
        repair = (f"{user}\n\nYour previous output was invalid:\n{text[:3000]}\n\n"
                  f"Validation error: {str(first)[:600]}\nReturn the corrected JSON object only.")
        text2 = caller.call(system, repair, schema)
        try:
            return _parse(model_cls, text2)
        except (ValidationError, ValueError) as second:
            raise ExtractionError("Gemini did not return a valid meeting record after a repair retry.",
                                  f"Details: {str(second)[:200]}") from second


# ------------------------------------------------------------------ extraction
def _wrap(text: str) -> str:
    return f"<transcript>\n{text}\n</transcript>"


def split_evidence(text: str) -> list[str]:
    """The prompt asks for 'exact quote(s)' as one string; several quotes are separated by ' / ' or '...'."""
    return [p.strip().strip('"“”').strip() for p in re.split(r"\s+/\s+|\.\.\.|…", text or "") if p.strip()]


def _recall_gap(data: dict) -> int:
    n_tasks = sum(1 for s in data.get("commitment_scan", []) if s.get("kind") == "task")
    return max(0, n_tasks - len(data.get("action_items", [])))


@dataclass
class _Attempt:
    data: dict
    needs_retry: bool


def _attempt(caller: GeminiCaller, system: str, text: str, extra: str = "") -> _Attempt:
    data = _generate_validated(caller, system, _wrap(text) + extra, LlmRecord).model_dump()
    return _Attempt(data, _recall_gap(data) > 0)


def _to_record(data: dict, index: TranscriptIndex) -> tuple[MeetingRecord, list[str]]:
    """Model output dict -> validated MeetingRecord with evidence flags."""
    notes: list[str] = []
    items: list[ActionItem] = []
    for a in data["action_items"]:
        owner = a["owner"]
        if a["status"] == "unassigned" and owner != UNSPECIFIED:
            notes.append(f"owner '{owner}' removed from unassigned task '{a['task']}'")
            owner = UNSPECIFIED
        items.append(ActionItem(task=a["task"].strip(), owner=owner, deadline=a["deadline"],
                                status=a["status"], condition=a["condition"],
                                evidence=split_evidence(a["evidence"])))
    record = MeetingRecord(
        summary=data["summary"].strip(),
        minutes=[MinutesSection(title=m["topic"].strip(), points=list(m["points"])) for m in data["minutes"]],
        key_decisions=[KeyDecision(decision=d["decision"].strip(), evidence=split_evidence(d["evidence"]))
                       for d in data["decisions"]],
        action_items=items)
    verify_record(record, index)
    return record, notes


def generate_record(refined_text: str, config: Config | None = None, *, client: Any = None,
                    progress: Callable[[float], None] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> DocumentResult:
    """Stage 3 entry point: refined transcript -> verified MeetingRecord."""
    cfg = config or load_config()
    if not refined_text or not refined_text.strip():
        raise DocumentingError("the refined transcript is empty, so there is nothing to document.")
    if len(refined_text) > cfg.document_max_chars:
        raise ExtractionError(f"the transcript is too long to document in one pass ({len(refined_text):,} characters).",
                              "Raise DOCUMENT_MAX_CHARS in .env, or process a shorter recording.")
    system = load_prompt("document_system.txt")
    caller = GeminiCaller(client if client is not None else _make_client(cfg), cfg, sleep)
    index = TranscriptIndex(refined_text, cfg.evidence_fuzzy_threshold)

    best = _attempt(caller, system, refined_text)
    if progress:
        progress(0.6)
    if best.needs_retry:
        gap = _recall_gap(best.data)
        feedback = (f"\n\nIMPORTANT: your commitment_scan lists {gap} more task(s) than your action_items. "
                    "Every scan item of kind \"task\" must become exactly one action item. "
                    "Return the complete JSON again, with no task missing.")
        try:
            second = _attempt(caller, system, refined_text, feedback)
            if _recall_gap(second.data) < _recall_gap(best.data):
                best = second
        except DocumentingError as exc:
            log.warning("Recall retry failed (%s); keeping the first extraction", exc.message)

    record, notes = _to_record(best.data, index)
    dropped = drop_unsupported_items(record)
    result = DocumentResult(record=record, model=caller.model, warnings=list(caller.notes),
                            dropped_items=dropped, downgraded_fields=notes,
                            scan_tasks=sum(1 for s in best.data["commitment_scan"] if s["kind"] == "task"))
    if dropped:
        result.warnings.append(f"Dropped {len(dropped)} item(s) with no supporting evidence: "
                               + "; ".join(d[:80] for d in dropped[:5]))
    if result.unverified_items:
        result.warnings.append(f"{result.unverified_items} item(s) have evidence quotes that could not be "
                               "verified against the transcript and are marked 'evidence not verified'.")
    if result.downgraded_fields:
        result.warnings.append("Adjusted to match the rules for unassigned tasks: "
                               + "; ".join(result.downgraded_fields[:5]))
    gap = _recall_gap(best.data)
    if gap:
        result.warnings.append(f"The model's own scan found {gap} more task(s) than it listed as action items; "
                               "some tasks may be missing.")
    if progress:
        progress(1.0)
    return result


if __name__ == "__main__":  # python -m pipeline.document refined_transcript.txt
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pipeline.document <refined transcript .txt>")
    try:
        res = generate_record(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        sys.exit(getattr(e, "user_message", str(e)))
    print(f"[{res.model}]")
    for w in res.warnings:
        print("WARNING:", w)
    print(res.record.model_dump_json(indent=2))
