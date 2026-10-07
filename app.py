"""Streamlit entry point for the Inter IIT Meeting Assistant."""

from __future__ import annotations

import json

import streamlit as st

from meeting_assistant.pipeline import InputError, MeetingAssistant
from meeting_assistant.providers import LocalWhisperQwenProvider, ProviderError


def _human_readable(result) -> str:
    lines = [
        "MEETING MINUTES",
        "",
        result.documentation.minutes,
        "",
        "DECISIONS",
        *([f"- {item}" for item in result.documentation.decisions] or ["- None"]),
        "",
        "ACTION ITEMS",
    ]
    if result.documentation.action_items:
        lines.extend(
            f"- {item.task} | Owner: {item.owner} | Deadline: {item.deadline}"
            for item in result.documentation.action_items
        )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


st.set_page_config(page_title="Inter IIT Meeting Assistant", layout="wide")
st.title("Inter IIT Meeting Assistant")
st.caption("Upload a recording to transcribe, refine, and extract faithful meeting documentation.")

uploaded_file = st.file_uploader(
    "Meeting recording",
    type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
)

if st.button("Process meeting", type="primary", disabled=uploaded_file is None):
    try:
        provider = LocalWhisperQwenProvider()
        result = MeetingAssistant(provider).process(
            uploaded_file.name, uploaded_file.getvalue()
        )
    except InputError as exc:
        st.error(str(exc))
    except ProviderError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Meeting processing failed: {exc}")
    else:
        st.success("Meeting processed successfully.")
        raw_column, refined_column = st.columns(2)
        with raw_column:
            st.subheader("Raw Transcript")
            st.text_area("Raw transcript", result.raw_transcript, height=400, label_visibility="collapsed")
        with refined_column:
            st.subheader("Refined Transcript")
            st.text_area("Refined transcript", result.refined_transcript, height=400, label_visibility="collapsed")

        st.subheader("Minutes")
        st.write(result.documentation.minutes)
        st.subheader("Decisions")
        for decision in result.documentation.decisions:
            st.markdown(f"- {decision}")
        if not result.documentation.decisions:
            st.write("None")
        st.subheader("Action Items")
        if result.documentation.action_items:
            st.table(
                [
                    {
                        "Task": item.task,
                        "Owner": item.owner,
                        "Deadline": item.deadline,
                    }
                    for item in result.documentation.action_items
                ]
            )
        else:
            st.write("None")

        json_output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        st.download_button("Download readable minutes", _human_readable(result), "meeting_minutes.txt", "text/plain")
        st.download_button("Download structured JSON", json_output, "meeting_minutes.json", "application/json")
