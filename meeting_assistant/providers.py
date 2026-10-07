"""Local Whisper transcription and Qwen language-model providers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any

from .models import ActionItem, Documentation
from .prompts import (
    DOCUMENTATION_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    documentation_user_prompt,
    refinement_user_prompt,
)


class ProviderError(RuntimeError):
    """Raised when an external AI provider cannot complete a stage."""


class LocalWhisperQwenProvider:
    """Runs Whisper locally and sends only text to a Qwen-compatible API."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        whisper_model: Any | None = None,
        language_model: str | None = None,
    ) -> None:
        api_key = os.getenv("QWEN_API_KEY")
        if client is None and not api_key:
            raise ProviderError(
                "QWEN_API_KEY is not configured. Add it to the environment and retry."
            )
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "The OpenAI-compatible client is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "QWEN_BASE_URL",
                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                ),
            )
        self.client = client
        self.whisper_model = whisper_model
        self.language_model = language_model or os.getenv(
            "QWEN_MODEL", "qwen-plus"
        )

    def transcribe(self, filename: str, audio_bytes: bytes) -> str:
        temporary_path: str | None = None
        try:
            model = self._get_whisper_model()
            suffix = os.path.splitext(filename)[1].lower() or ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio_bytes)
                temporary_path = audio_file.name
            transcribe_options: dict[str, Any] = {
                "language": os.getenv("WHISPER_LANGUAGE", "en"),
                "fp16": self._whisper_uses_cuda(),
            }
            initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()
            if initial_prompt:
                transcribe_options["initial_prompt"] = initial_prompt
            result = model.transcribe(temporary_path, **transcribe_options)
        except Exception as exc:
            raise ProviderError(
                "Local Whisper transcription failed. Ensure Whisper and ffmpeg are installed. "
                f"Details: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        text = result.get("text") if isinstance(result, dict) else result
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Local Whisper returned an empty transcript.")
        return text.strip()

    def refine(self, raw_transcript: str) -> str:
        return self._chat(
            REFINEMENT_SYSTEM_PROMPT,
            refinement_user_prompt(raw_transcript),
        )

    def document(self, refined_transcript: str) -> Documentation:
        text = self._chat(
            DOCUMENTATION_SYSTEM_PROMPT,
            documentation_user_prompt(refined_transcript),
        )
        try:
            payload = json.loads(_strip_code_fences(text))
            return _parse_documentation(payload, refined_transcript)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Documentation model returned invalid JSON: {exc}") from exc

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.language_model,
                temperature=0,
                response_format={"type": "text"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise ProviderError(f"Language-model stage failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Language-model stage returned empty content.")
        return content.strip()

    def _get_whisper_model(self) -> Any:
        if self.whisper_model is not None:
            return self.whisper_model
        try:
            import whisper
        except ImportError as exc:
            raise ProviderError(
                "openai-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self.whisper_model = whisper.load_model(
            os.getenv("WHISPER_MODEL", "base")
        )
        return self.whisper_model

    def _whisper_uses_cuda(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())


# Kept as a migration alias for callers that imported the original provider.
OpenAIProvider = LocalWhisperQwenProvider


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_documentation(payload: Any, transcript: str = "") -> Documentation:
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    minutes = payload.get("minutes")
    decisions = payload.get("decisions")
    action_items = payload.get("action_items")
    if not isinstance(minutes, str) or not isinstance(decisions, list) or not isinstance(action_items, list):
        raise ValueError("minutes, decisions, and action_items have invalid types")
    if not all(isinstance(decision, str) for decision in decisions):
        raise ValueError("decisions must contain only strings")
    parsed_items: list[ActionItem] = []
    for item in action_items:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) for key in ("task", "owner", "deadline")
        ):
            raise ValueError("each action item needs task, owner, and deadline strings")
        task = item["task"]
        owner = item["owner"]
        deadline = item["deadline"]
        if owner == "unspecified":
            owner = _explicit_owner_for_task(task, transcript)
        if deadline == "unspecified":
            deadline = _explicit_deadline_for_task(task, transcript)
        parsed_items.append(
            ActionItem(
                task=task,
                owner=owner,
                deadline=deadline,
            )
        )
    return Documentation(
        minutes=minutes,
        decisions=decisions,
        action_items=parsed_items,
    )


_DEADLINE_PATTERN = re.compile(
    r"\b(?:by|on|before|within|next|this|tomorrow|today)\b"
    r"[^,.;!?]*(?:day|week|month|year|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|morning|evening|finalizing|finalising)?",
    re.IGNORECASE,
)


def _explicit_deadline_for_task(task: str, transcript: str) -> str:
    """Copy a deadline from the task's sentence without inventing one."""
    task_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", task)
        if len(word) > 2
    }
    if not task_words:
        return "unspecified"
    for sentence in re.split(r"(?<=[.!?])\s+", transcript):
        sentence_words = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
        if len(task_words & sentence_words) < min(2, len(task_words)):
            continue
        match = _DEADLINE_PATTERN.search(sentence)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip(" ,;:.")
    return "unspecified"


def _explicit_owner_for_task(task: str, transcript: str) -> str:
    """Copy a directly stated owner, including explicit module assignments."""
    task_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", task)
        if len(word) > 2
    }
    if not task_words:
        return "unspecified"

    direct_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
        r"(?:will|is responsible for|owns|must|should)\s+"
        r"([^.;!?]+)",
    )
    for match in direct_pattern.finditer(transcript):
        candidate_task = set(re.findall(r"[A-Za-z0-9]+", match.group(2).lower()))
        if len(task_words & candidate_task) >= min(2, len(task_words)):
            return match.group(1)

    module_match = re.search(r"\bmodule\s+([a-z0-9]+)", task, re.IGNORECASE)
    if module_match:
        module_number = module_match.group(1).lower()
        assignment_pattern = re.compile(
            r"\b([A-Z][a-z]+)\s+"
            r"(?:is\s+working\s+on|works\s+on|working\s+on|on)\s+"
            r"([^,.;!?]+)",
            re.IGNORECASE,
        )
        for match in assignment_pattern.finditer(transcript):
            assignment = match.group(2).lower()
            if (
                re.search(rf"\bmodule\s+{re.escape(module_number)}\b", assignment)
                or (
                    re.search(r"\bmodules?\s+", assignment)
                    and module_number in assignment
                )
            ):
                return match.group(1)
    return "unspecified"
