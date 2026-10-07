# Inter IIT Meeting Assistant - Dev Doc

we need to build an end-to-end AI meeting assistant. The vibe is a clean Streamlit web app that processes audio through a strict three-stage pipeline. 

## The Stack
* **Frontend:** Streamlit. Keep it simple and interactive.
* **Audio Processing (Stage 1):** OpenAI Whisper (either local `openai-whisper` or API) for the speech-to-text.
* **LLM Engine (Stages 2 & 3):** We need to run two distinct language model calls. Wire this up to use Qwen (via an API or local quantized pipeline if it fits) or standard OpenAI endpoints. I want to lean on a clarification-aware prompting approach to tightly control the outputs.

## Pipeline Architecture
It has to be one seamless run from upload to final output, but logically split into these exact steps:

### 1. Ingestion & STT
* Let the user upload a meeting recording.
* **Crucial:** Add robust error handling right here. If the file is unsupported, empty, or unreadable, throw a clear Streamlit error message and halt.
* Run it through the STT model to get the **Raw Transcript**.

### 2. Domain-Aware Refinement
* Pass the raw transcript to the first LLM prompt.
* **Prompt Goal:** Act as a smart corrector. Fix likely transcription errors in technical terms, acronyms, and domain-specific language. 
* **Strict Constraint:** It must preserve the speaker's exact intended meaning, including names, numbers, negations, and commitments. No summarizing here, just output the **Refined Transcript**.

### 3. Documentation & Extraction
* Feed the *Refined Transcript* to a separate LLM prompt.
* **Prompt Goal:** Generate a structured object containing:
  * `minutes`: A concise summary of the discussion.
  * `decisions`: A list of agreed-upon decisions (empty list if none were reached).
  * `action_items`: A list of actionable tasks.
* **Anti-Hallucination Rule:** This is critical. For action items, only include an owner or deadline if the transcript explicitly states one. If unstated, output exactly `"unspecified"` rather than guessing. Do not present a proposal as an agreed decision or an unstated assignment as a confirmed task.

## UI & Deliverables
* Use `st.columns` to show the Raw Transcript and Refined Transcript side-by-side so users can compare them.
* Render the minutes, decisions, and tasks cleanly below that.
* Add `st.download_button`s for the final outputs. We need to export a human-readable format and a machine-readable structured format (like JSON). Both formats must convey the exact same decisions and tasks.
* Add a quick `setup.sh` bash script and a `requirements.txt` so the pipeline can be run on a standard Linux environment.