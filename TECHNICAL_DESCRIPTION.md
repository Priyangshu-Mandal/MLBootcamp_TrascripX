# Technical description

## Models and roles

| Stage | Model | Where it runs | Role |
|-------|-------|---------------|------|
| 1 | **faster-whisper `large-v3-turbo`** (CTranslate2). CPU fallback: `small.en`, int8 | Locally; CUDA float16 when an NVIDIA GPU is usable | Speech-to-text on the uploaded recording |
| 2 | **`openai/gpt-oss-120b`** | Groq API | Language model for **transcript refinement**: proposes corrections for recognition errors in technical terms, acronyms, product names and misspelt names, using meeting context |
| 3 | **Gemini Flash** (default `gemini-3.1-flash-lite`; fallbacks `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-3.8-flash`) | Gemini API | Separate language model for **meeting documentation**: summary, minutes, key decisions, action items |

The two language models are different models from different providers, called by different modules
(`pipeline/refine.py`, `pipeline/document.py`) with different prompts (`prompts/`). Orchestration is plain
Python function calls in `pipeline/run.py`.

## How outputs move between stages

1. **Upload -> audio.** The file type, size and content are validated, then ffmpeg converts it to 16 kHz mono
   WAV. Empty, unsupported, undecodable or silent files stop here with a specific message.
2. **Stage 1 -> raw transcript.** faster-whisper (with voice-activity filtering) returns timed segments.
   They are joined into the raw transcript, with paragraph breaks at long pauses. The raw text is kept
   unchanged for display and download.
3. **Raw transcript -> Stage 2.** The raw text is split into sections (12,000 characters by default; most
   meetings are one section) and each section is sent to the refinement model at temperature 0. The model
   does **not** return a transcript: it returns a JSON list of corrections, each with an exact context
   snippet, the wrong words, the replacement, a reason and a supporting quote (`prompts/refine_system.txt`).
   Python then
   vets and applies the list: corrections that change a negation, hedge or commitment word, or whose
   supporting quote is not in the transcript, are refused; `apply_corrections` (in `refine.py`) refuses any
   correction whose context is not found verbatim, and allows a day, number or date to change only to a
   value that appears elsewhere in the transcript. Only the quoted occurrences are replaced, so the rest of
   the transcript is unchanged by construction. Applied and refused corrections are both kept and shown.
   If the model twice fails to return usable JSON for a section, that section stays as raw text and a
   warning is shown.
4. **Refined transcript -> Stage 3.** Only the refined transcript, whole, is given to the Gemini model with a
   JSON schema (validated by Pydantic; one repair retry on invalid JSON). The model first fills a
   `commitment_scan` listing every promise, request, decision, rejection and proposal, then the summary,
   minutes, decisions and action items (`prompts/document_system.txt`). Each action item has a status
   (`confirmed`, `conditional` or `unassigned`), an optional condition, and `owner`/`deadline` that are
   `unspecified` unless the transcript states them.
5. **Verification.** If the model's own scan found more tasks than it listed, the call is repeated once
   with feedback (the better of the two results is kept; a remaining gap is shown as a warning). Code
   then verifies every evidence quote against the transcript (exact match, or a
   close fuzzy match only when numbers, negations and hedges are identical): unverifiable quotes are flagged
   "evidence not verified", items with no quote at all are dropped, and an `unassigned` task never keeps an
   owner.
6. **Record -> outputs.** One validated `MeetingRecord` object is rendered to Markdown and to JSON, so both
   contain the same decisions and tasks. The interface shows raw and refined transcripts side by side, the
   list of corrections, and the record, and offers raw transcript, refined transcript, record (.md) and
   record (.json) downloads.

## Reliability

* Rate limits (429), server errors (5xx) and network errors are retried with exponential backoff, honouring
  `retry-after`. A persistently overloaded or missing Gemini model falls back to the next configured model;
  the switch is reported as a warning. Calls have timeouts.
* A GPU/CUDA failure at load time or during transcription falls back to CPU automatically.
* Every failure is raised as a typed error that names the stage; the interface shows the message, and keeps
  already completed results (for example the raw transcript) visible and downloadable.
* API keys come only from `.env`/environment and are never logged, displayed or exported.

## Prompts

`prompts/refine_system.txt` (what may and may not be corrected, the recap rule) and
`prompts/document_system.txt` (the scan-first procedure, what counts as a decision or task, owner and deadline
rules, the `unspecified` rule). Transcript text is passed inside `<transcript>` tags.
