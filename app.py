"""Streamlit UI for the Inter IIT meeting assistant.

Run:  streamlit run app.py
Upload a meeting recording -> Transcribing -> Refining -> Generating record -> view + download.
Results live in st.session_state, so download clicks never re-run the pipeline.
"""
from __future__ import annotations

import html
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.config import load_config
from pipeline.diffview import count_changes, highlight_changes
from pipeline.errors import MeetingAssistantError
from pipeline.export import to_json, to_markdown
from pipeline.run import STAGES, PipelineResult, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")
cfg = load_config()


# ------------------------------------------------------------------ sidebar
def sidebar() -> tuple[bool, bool]:
    result: PipelineResult | None = st.session_state.get("result")
    with st.sidebar:
        st.header("Pipeline")
        if result:
            t = result.transcript
            stt_line = f"faster-whisper **{t.model_name}** ({t.device}, {t.compute_type})"
            ref_line = f"**{result.refinement.model}** via Groq"
            doc_line = f"**{result.document.model}** via Gemini"
        else:
            stt_line = f"faster-whisper **{cfg.whisper_model}** (CPU fallback: {cfg.whisper_cpu_model})"
            ref_line = f"**{cfg.groq_model}** via Groq"
            doc_line = f"**{cfg.gemini_model}** via Gemini"
        st.markdown(f"1. **Transcribing** — {stt_line}")
        st.markdown(f"2. **Refining** — {ref_line}")
        st.markdown(f"3. **Generating record** — {doc_line}")
        st.divider()
        st.subheader("API keys")
        st.write(("✅" if cfg.groq_api_key else "❌ missing —") + " GROQ_API_KEY")
        st.write(("✅" if cfg.gemini_api_key else "❌ missing —") + " GEMINI_API_KEY")
        st.caption("Keys are read from the .env file and are never displayed.")
        st.divider()
        st.subheader("Display and export")
        show_evidence = st.toggle("Show evidence", value=False,
                                  help="Expand every “Why this?” panel at once.")
        md_evidence = st.checkbox("Include evidence in Markdown download", value=False)
    return show_evidence, md_evidence


# ------------------------------------------------------------------ running the pipeline
def _stage_line(stage: str, state: str, detail: str = "") -> str:
    icon = {"wait": "⚪", "run": "⏳", "done": "✅", "error": "❌"}[state]
    return f"{icon} **{stage}**" + (f" — {detail}" if detail else "")


def _done_detail(stage: str, out) -> str:
    if stage == STAGES[0]:
        return f"{out.duration:.0f}s of audio, {out.model_name} on {out.device}"
    if stage == STAGES[1]:
        kept = f", {len(out.fallback_chunks)} kept raw" if out.fallback_chunks else ""
        return f"{len(out.applied)} correction(s) applied, {len(out.rejected)} rejected by the safety checks{kept}"
    return f"{len(out.record.key_decisions)} decision(s), {len(out.record.action_items)} action item(s)"


def process_upload(uploaded) -> None:
    """Run the pipeline on an uploaded file, showing three labelled stages."""
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    st.session_state.pop("partial", None)
    partial: dict = {}
    suffix = Path(uploaded.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="meeting_upload_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(uploaded.getvalue())
        with st.status("Processing recording…", expanded=True) as status:
            slots = {s: st.empty() for s in STAGES}
            for s in STAGES:
                slots[s].markdown(_stage_line(s, "wait"))
            bar = st.progress(0.0)

            def on_event(stage: str, kind: str, detail) -> None:
                if stage not in slots:  # setup/preflight failure
                    return
                if kind == "start":
                    status.update(label=f"{stage}…")
                    slots[stage].markdown(_stage_line(stage, "run"))
                    bar.progress(0.0)
                elif kind == "progress":
                    bar.progress(float(min(max(detail, 0.0), 1.0)))
                elif kind == "done":
                    partial[stage] = detail
                    slots[stage].markdown(_stage_line(stage, "done", _done_detail(stage, detail)))
                    bar.progress(1.0)
                elif kind == "error":
                    slots[stage].markdown(_stage_line(stage, "error", "failed"))

            try:
                result = run_pipeline(tmp_path, uploaded.name, cfg, on_event)
            except MeetingAssistantError as exc:
                status.update(label=f"{exc.stage} failed", state="error", expanded=True)
                st.session_state["error"] = exc.user_message
                st.session_state["partial"] = partial
                return
            status.update(label="Done", state="complete", expanded=False)
            st.session_state["result"] = result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def render_partial(partial: dict) -> None:
    """Show what finished before a later stage failed, so nothing completed is lost."""
    transcript = partial.get(STAGES[0])
    if transcript is None:
        return
    st.subheader("Completed before the failure")
    for warning in transcript.warnings:
        st.warning(md(warning))
    st.markdown(f"**Raw transcript** — {transcript.model_name} on {transcript.device} "
                f"({transcript.duration:.0f}s of audio)")
    st.text_area("Raw transcript (partial result)", transcript.text, height=300, disabled=True,
                 label_visibility="collapsed")
    st.download_button("Raw transcript (.txt)", transcript.text, file_name="raw_transcript.txt",
                       mime="text/plain", key="dl_partial_raw")
    refinement = partial.get(STAGES[1])
    if refinement is not None:
        st.markdown(f"**Refined transcript** — {refinement.model}")
        st.text_area("Refined transcript (partial result)", refinement.text, height=300, disabled=True,
                     label_visibility="collapsed")
        st.download_button("Refined transcript (.txt)", refinement.text,
                           file_name="refined_transcript.txt", mime="text/plain", key="dl_partial_refined")


# ------------------------------------------------------------------ rendering results
STYLE = """
<style>
.block-container {padding-top: 2.2rem;}
.chip {display:inline-block; padding:2px 11px; margin:0 6px 4px 0; border-radius:999px; font-size:0.95rem;
       border:1px solid rgba(128,128,128,.35); background:rgba(128,128,128,.12); white-space:nowrap;}
.chip-green {color:#2ea043; background:rgba(46,160,67,.14); border-color:rgba(46,160,67,.45);}
.chip-amber {color:#d29922; background:rgba(210,153,34,.14); border-color:rgba(210,153,34,.45);}
.chip-red   {color:#f85149; background:rgba(248,81,73,.12); border-color:rgba(248,81,73,.45);}
.chip-muted {opacity:.75;}
.step-num {font-size:1.6rem; font-weight:700; opacity:.55;}
/* scale everything up: Streamlit sizes are rem-based, so this enlarges all text */
html {font-size: 112.5%;}
/* section tabs: large labels, roomy hit area */
[data-baseweb="tab-list"] {gap: 0.75rem;}
button[data-baseweb="tab"] {padding: 0.9rem 1.3rem !important; height: auto !important;}
button[data-baseweb="tab"] *, [data-baseweb="tab-list"] button p
    {font-size: 1.35rem !important; font-weight: 600 !important; line-height: 1.3;}
[data-baseweb="tab-highlight"] {height: 3px !important;}
/* body text, captions, expanders, buttons */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {font-size: 1.05rem; line-height: 1.55;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {font-size: 1rem !important;}
[data-testid="stExpander"] summary p {font-size: 1.05rem !important; font-weight: 600;}
.stButton button, .stDownloadButton button {font-size: 1.05rem !important;}
h3 {font-size: 1.5rem !important;}
/* key numbers row: equal-height cards, centred, nothing truncated */
[data-testid="stMetric"] {
    background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.28);
    border-radius: 12px; padding: 1rem 1rem; min-height: 7rem;
    display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
}
[data-testid="stMetric"] > div {width: 100%; justify-content: center;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p
    {justify-content: center; font-size: 0.95rem !important; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div
    {font-size: 2rem !important; font-weight: 700; justify-content: center;
     white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.2;}

/* visual refresh */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(99, 102, 241, .14), transparent 28rem),
        radial-gradient(circle at 100% 18%, rgba(14, 165, 233, .10), transparent 26rem);
}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1440px; padding: 3.5rem 3rem 5rem;}
h1 {font-size: clamp(2.2rem, 4vw, 3.5rem) !important; letter-spacing: -0.045em; margin-bottom: .35rem !important;}
h2 {letter-spacing: -0.025em;}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(148, 163, 184, .23) !important;
    border-radius: 20px !important;
    background: rgba(255, 255, 255, .035);
    box-shadow: 0 14px 40px rgba(15, 23, 42, .08);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(99, 102, 241, .13), rgba(14, 165, 233, .07));
    border: 1px solid rgba(129, 140, 248, .28);
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, .08);
    transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: 0 16px 34px rgba(15, 23, 42, .14);}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p
    {font-size: .82rem !important; letter-spacing: .09em;}
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed rgba(99, 102, 241, .55);
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(99, 102, 241, .10), rgba(14, 165, 233, .05));
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {border-color: #818cf8; background: rgba(99, 102, 241, .16);}
.stButton button, .stDownloadButton button {
    border-radius: 11px !important;
    font-weight: 650 !important;
    transition: transform .18s ease, box-shadow .18s ease;
}
.stButton button:hover, .stDownloadButton button:hover
    {transform: translateY(-2px); box-shadow: 0 8px 20px rgba(99, 102, 241, .20);}
[data-testid="stProgressBar"] > div > div {background: linear-gradient(90deg, #6366f1, #06b6d4);}
[data-testid="stExpander"] {border: 1px solid rgba(148, 163, 184, .22) !important; border-radius: 14px !important;}
[data-baseweb="tab-list"] {gap: .35rem; border-bottom: 1px solid rgba(148, 163, 184, .2); padding-bottom: .3rem;}
button[data-baseweb="tab"] {padding: .8rem 1rem !important; border-radius: 10px 10px 0 0;}
button[data-baseweb="tab"] *, [data-baseweb="tab-list"] button p
    {font-size: 1.04rem !important; font-weight: 650 !important;}
button[data-baseweb="tab"]:hover {background: rgba(99, 102, 241, .10);}
[data-baseweb="tab-highlight"] {background: linear-gradient(90deg, #6366f1, #06b6d4) !important;}
.chip {
    display: inline-flex; align-items: center; padding: 4px 12px; margin: 0 6px 6px 0;
    border-radius: 999px; font-weight: 600; border-color: rgba(148, 163, 184, .3);
}
.chip-green {color: #22c55e; background: rgba(34, 197, 94, .13); border-color: rgba(34, 197, 94, .4);}
.chip-amber {color: #f59e0b; background: rgba(245, 158, 11, .13); border-color: rgba(245, 158, 11, .4);}
.chip-red {color: #f87171; background: rgba(248, 113, 113, .13); border-color: rgba(248, 113, 113, .4);}
[data-testid="stDataFrame"] {border: 1px solid rgba(148, 163, 184, .22); border-radius: 12px; overflow: hidden;}
[data-testid="stTextArea"] textarea {border-radius: 12px !important; line-height: 1.6 !important;}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, .18);
    background: linear-gradient(180deg, rgba(99, 102, 241, .08), rgba(14, 165, 233, .025) 55%, transparent);
}
@media (max-width: 900px) {
    .block-container {padding: 2.5rem 1.2rem 4rem;}
    h1 {font-size: 2.35rem !important;}
    [data-baseweb="tab-list"] {overflow-x: auto; flex-wrap: nowrap;}
    button[data-baseweb="tab"] {white-space: nowrap; padding: .7rem .8rem !important;}
}
</style>
"""

STATUS_CHIP = {"confirmed": ("✅ confirmed", "chip-green"),
               "conditional": ("⏳ conditional", "chip-amber"),
               "unassigned": ("❔ unassigned", "chip-red")}


def md(text: str) -> str:
    """Escape '$' so Streamlit does not render text between two dollar signs as LaTeX."""
    return text.replace("$", "\\$")


def chip(text: str, css: str = "") -> str:
    """One small rounded label (HTML). The text is HTML-escaped; '$' cannot start LaTeX."""
    return f'<span class="chip {css}">{html.escape(text).replace("$", "&#36;")}</span>'


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def evidence_panel(item, show_evidence: bool) -> None:
    """Hidden-by-default 'Why this?' panel with the supporting quotes."""
    if not item.evidence_verified:
        st.warning("⚠ Evidence not verified: at least one quote could not be found in the "
                   "refined transcript, or no quote was given.")
    with st.expander("Why this?", expanded=show_evidence):
        if not item.evidence:
            st.write("_No supporting quotes._")
        for quote in item.evidence:
            st.markdown("> " + md(" ".join(quote.split())))


def _table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df


def render_overview(result: PipelineResult) -> None:
    """Key numbers and run notices, above the tabs."""
    t, record = result.transcript, result.record
    cols = st.columns(5)
    cols[0].metric("Audio length", fmt_duration(t.duration))
    cols[1].metric("Speech model", t.model_name, help=f"Ran on {t.device} ({t.compute_type})")
    cols[2].metric("Corrections applied", len(result.refinement.applied))
    cols[3].metric("Key decisions", len(record.key_decisions))
    cols[4].metric("Action items", len(record.action_items))
    notices = result.warnings
    if notices:
        with st.expander(f"⚠ {len(notices)} notice(s) about this run", expanded=False):
            for notice in notices:
                st.warning(md(notice))


def render_transcripts(result: PipelineResult) -> None:
    t, ref = result.transcript, result.refinement
    highlight = st.checkbox("Highlight what refinement changed", value=False)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Raw transcript** · speech-to-text (`{t.model_name}` on `{t.device}`)")
        st.text_area("Raw transcript", result.raw_text, height=420, disabled=True,
                     label_visibility="collapsed")
    with right:
        n = count_changes(result.raw_text, result.refined_text)
        st.markdown(f"**Refined transcript** · corrected by `{ref.model}` ({n} change(s))")
        if highlight:
            with st.container(height=420, border=True):
                st.markdown(highlight_changes(result.raw_text, result.refined_text), unsafe_allow_html=True)
        else:
            st.text_area("Refined transcript", result.refined_text, height=420, disabled=True,
                         label_visibility="collapsed")
    with st.expander(f"Corrections made by refinement ({len(ref.applied)} applied, {len(ref.rejected)} rejected)"):
        if ref.applied:
            st.dataframe(_table([{"Original": c["original"], "Replacement": c["replacement"],
                                  "Reason": c.get("reason", "")} for c in ref.applied]))
        else:
            st.write("_No corrections were applied._")
        if ref.rejected:
            st.caption("Proposed by the model but refused by the safety checks (the transcript was not changed):")
            st.dataframe(_table([{"Original": c.get("original", ""), "Replacement": c.get("replacement", ""),
                                  "Why refused": c.get("why", "")} for c in ref.rejected]))
    with st.expander("Raw transcript with timestamps"):
        st.text("\n".join(f"[{int(s.start) // 60:02d}:{int(s.start) % 60:02d}] {s.text}" for s in t.segments))


def render_summary(record) -> None:
    with st.container(border=True):
        st.markdown(md(record.summary) if record.summary else "_No summary._")


def render_minutes(record) -> None:
    if not record.minutes:
        st.info("No minutes were produced.")
    for section in record.minutes:
        with st.container(border=True):
            st.markdown(f"**{md(section.title)}**")
            st.markdown("\n".join(f"- {md(p)}" for p in section.points) or "_No points._")


def render_decisions(record, show_evidence: bool) -> None:
    if not record.key_decisions:
        st.info("No decisions were reached in this meeting.")
    for i, d in enumerate(record.key_decisions, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {md(d.decision)}**")
            if d.context:
                st.caption(md(d.context))
            evidence_panel(d, show_evidence)


def render_actions(record, show_evidence: bool) -> None:
    if not record.action_items:
        st.info("No action items were assigned in this meeting.")
        return
    st.caption("“unspecified” means the recording did not state an owner or deadline. Status: "
               "confirmed = someone accepted or volunteered; conditional = depends on a condition; "
               "unassigned = needed, but nobody took it.")
    for i, a in enumerate(record.action_items, start=1):
        label, css = STATUS_CHIP.get(a.status, (a.status, ""))
        with st.container(border=True):
            st.markdown(f"**{i}. {md(a.task)}**")
            st.markdown(chip(label, css)
                        + chip(f"👤 {a.owner}", "chip-muted" if a.owner == "unspecified" else "")
                        + chip(f"📅 {a.deadline}", "chip-muted" if a.deadline == "unspecified" else ""),
                        unsafe_allow_html=True)
            if a.condition:
                st.caption("Condition / note: " + md(a.condition))
            evidence_panel(a, show_evidence)
    with st.expander("Table view"):
        st.dataframe(_table([{
            "Task": a.task, "Owner": a.owner, "Deadline": a.deadline, "Status": a.status,
            "Condition / note": a.condition or "",
            "Evidence": "verified" if a.evidence_verified else "NOT VERIFIED",
        } for a in record.action_items]))


def render_downloads(result: PipelineResult, md_evidence: bool) -> None:
    base = Path(result.source_name).stem or "meeting"
    meta, record = result.metadata(), result.record
    st.caption("Everything below was generated from this recording. The Markdown and JSON files come from the "
               "same validated record, so they list identical decisions and tasks.")
    row1 = st.columns(2)
    row1[0].markdown("**Raw transcript** — speech-to-text output, before refinement")
    row1[0].download_button("Raw transcript (.txt)", result.raw_text, file_name=f"{base}_raw_transcript.txt",
                            mime="text/plain", key="dl_raw")
    row1[1].markdown("**Refined transcript** — after domain-term correction")
    row1[1].download_button("Refined transcript (.txt)", result.refined_text,
                            file_name=f"{base}_refined_transcript.txt", mime="text/plain", key="dl_refined")
    row2 = st.columns(2)
    row2[0].markdown("**Meeting record** — human-readable (Markdown)")
    row2[0].download_button("Meeting record (.md)", to_markdown(record, meta, include_evidence=md_evidence),
                            file_name=f"{base}_meeting_record.md", mime="text/markdown", key="dl_md")
    row2[1].markdown("**Meeting record** — machine-readable (JSON, always includes evidence)")
    row2[1].download_button("Meeting record (.json)", to_json(record, meta),
                            file_name=f"{base}_meeting_record.json", mime="application/json", key="dl_json")
    st.caption("Tip: the sidebar option “Include evidence in Markdown download” adds the supporting quotes to the .md file.")


def render_results(result: PipelineResult, show_evidence: bool, md_evidence: bool) -> None:
    record = result.record
    render_overview(result)
    tabs = st.tabs(["📄 Transcripts", "📝 Summary", "🗂️ Minutes",
                    f"✅ Key decisions ({len(record.key_decisions)})",
                    f"📌 Action items ({len(record.action_items)})", "⬇️ Downloads"])
    with tabs[0]:
        render_transcripts(result)
    with tabs[1]:
        render_summary(record)
    with tabs[2]:
        render_minutes(record)
    with tabs[3]:
        render_decisions(record, show_evidence)
    with tabs[4]:
        render_actions(record, show_evidence)
    with tabs[5]:
        render_downloads(result, md_evidence)


def render_landing() -> None:
    """Shown before the first run: what will happen and with which models."""
    steps = [("1", "Transcribe", f"faster-whisper **{cfg.whisper_model}** turns the speech into a raw transcript, "
                                 "locally on your GPU (CPU fallback)."),
             ("2", "Refine", f"**{cfg.groq_model}** on Groq proposes corrections for technical terms and names; "
                             "safety checks apply only the verifiable ones."),
             ("3", "Document", f"**{cfg.gemini_model}** on Gemini writes the summary, minutes, decisions and "
                               "action items, with supporting quotes.")]
    for col, (num, title, text) in zip(st.columns(3), steps):
        with col:
            with st.container(border=True):
                st.markdown(f'<span class="step-num">{num}</span>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(text)


# ------------------------------------------------------------------ page
def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("📝 AI Meeting Assistant")
    st.caption("Upload a meeting recording to get a transcript, a domain-corrected transcript, "
               "minutes, key decisions and action items.")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Meeting recording", help="Supported: " + ", ".join(cfg.allowed_extensions))
        if st.button("Process recording", type="primary"):
            if uploaded is None:
                st.session_state["error"] = "Please upload a meeting recording first."
            else:
                process_upload(uploaded)

    show_evidence, md_evidence = sidebar()  # after processing so it shows the models actually used
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
        render_partial(st.session_state.get("partial") or {})
    result = st.session_state.get("result")
    if result is not None:
        render_results(result, show_evidence, md_evidence)
    elif not st.session_state.get("error"):
        render_landing()


main()
