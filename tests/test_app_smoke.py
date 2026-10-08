"""Smoke test of app.py against a *stubbed* streamlit module.

This catches logic errors in app.py (wrong names, wrong data flow, download contents, error
handling). It does NOT prove the real Streamlit API calls are valid; run `streamlit run app.py`
to confirm that.
"""
import json
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas  # noqa: F401  (import once: numpy cannot be re-imported after sys.modules is restored)

from pipeline.errors import UnsupportedFileTypeError
from tests.helpers import make_result

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def make_stub(uploaded=None, button=False, toggles=None):
    st = mock.MagicMock(name="streamlit")
    st.session_state = SessionState()
    st.col = mock.MagicMock(name="column")   # every column is this one shared mock
    st.columns.side_effect = lambda spec, **kw: [st.col] * (spec if isinstance(spec, int) else len(spec))
    st.tab = mock.MagicMock(name="tab")      # every tab is this one shared mock (usable in `with`)
    st.tabs.side_effect = lambda labels, **kw: [st.tab] * len(labels)
    st.file_uploader.return_value = uploaded
    st.button.return_value = button
    toggles = toggles or {}
    st.toggle.side_effect = lambda label, **kw: toggles.get(label, False)
    st.checkbox.side_effect = lambda label, **kw: toggles.get(label, False)
    return st


def run_app(st):
    with mock.patch.dict(sys.modules, {"streamlit": st}):
        runpy.run_path(APP, run_name="__main__")


def fake_upload(name="meeting.mp3", data=b"abc"):
    return SimpleNamespace(name=name, getvalue=lambda: data, size=len(data))


def frame_with(st, column):
    return next(c.args[0] for c in st.dataframe.call_args_list if column in c.args[0].columns)


def collect_downloads(st):
    return {c.kwargs["key"]: c for m in (st, st.col) for c in m.download_button.call_args_list}


class AppSmokeTests(unittest.TestCase):
    def test_first_load_renders_without_error(self):
        st = make_stub()
        run_app(st)
        st.title.assert_called()
        st.error.assert_not_called()
        st.set_page_config.assert_called_once()

    def test_button_without_file_shows_clear_error(self):
        st = make_stub(button=True)
        run_app(st)
        st.error.assert_called_once()
        self.assertIn("upload a meeting recording", st.error.call_args.args[0])

    def test_successful_run_stores_result_and_offers_four_downloads(self):
        result = make_result()

        def fake_run(path, name, cfg, on_event):
            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).suffix, ".mp3")          # original extension preserved
            for stage, kind, detail in (("Transcribing", "start", None), ("Transcribing", "progress", 0.5),
                                        ("Transcribing", "done", result.transcript),
                                        ("Refining", "start", None), ("Refining", "done", result.refinement),
                                        ("Generating record", "start", None),
                                        ("Generating record", "done", result.document)):
                on_event(stage, kind, detail)
            self.fake_path = path
            return result

        st = make_stub(uploaded=fake_upload(), button=True)
        with mock.patch("pipeline.run.run_pipeline", fake_run):
            run_app(st)
        self.assertIs(st.session_state["result"], result)
        self.assertFalse(Path(self.fake_path).exists())          # temp upload cleaned up
        st.error.assert_not_called()
        dl = collect_downloads(st)
        self.assertEqual(set(dl), {"dl_raw", "dl_refined", "dl_md", "dl_json"})
        self.assertEqual(dl["dl_raw"].args[1], result.raw_text)
        self.assertEqual(dl["dl_refined"].args[1], result.refined_text)
        self.assertEqual(dl["dl_raw"].kwargs["file_name"], "standup_raw_transcript.txt")
        js = json.loads(dl["dl_json"].args[1])
        self.assertIn("evidence", js["key_decisions"][0])
        self.assertNotIn("Evidence for action items", dl["dl_md"].args[1])   # excluded by default
        # warnings from every stage are shown
        shown = [c.args[0] for c in st.warning.call_args_list]
        self.assertTrue(any("could not be verified" in w for w in shown))

    def test_markdown_evidence_checkbox_is_honoured(self):
        st = make_stub(toggles={"Include evidence in Markdown download": True})
        st.session_state["result"] = make_result()
        run_app(st)
        md = collect_downloads(st)["dl_md"].args[1]
        self.assertIn("Evidence for action items", md)

    def test_rerun_with_existing_result_does_not_rerun_pipeline(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        boom = mock.Mock(side_effect=AssertionError("pipeline must not re-run"))
        with mock.patch("pipeline.run.run_pipeline", boom):
            run_app(st)
        boom.assert_not_called()
        self.assertEqual(len(collect_downloads(st)), 4)

    def test_show_evidence_toggle_expands_every_panel(self):
        st = make_stub(toggles={"Show evidence": True})
        st.session_state["result"] = make_result()
        run_app(st)
        flags = [c.kwargs.get("expanded") for c in st.expander.call_args_list if c.args and c.args[0] == "Why this?"]
        self.assertEqual(len(flags), 3)           # 1 decision + 2 tasks
        self.assertTrue(all(flags))

    def test_evidence_hidden_by_default_and_unverified_warned(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        flags = [c.kwargs.get("expanded") for c in st.expander.call_args_list if c.args and c.args[0] == "Why this?"]
        self.assertFalse(any(flags))
        self.assertTrue(any("Evidence not verified" in c.args[0] for c in st.warning.call_args_list))

    def test_unspecified_visible_in_action_table(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        df = frame_with(st, "Task")
        self.assertIn("unspecified", df["Owner"].tolist())
        self.assertIn("unspecified", df["Deadline"].tolist())
        self.assertEqual(list(df.index), [1, 2])

    def test_action_table_shows_status_and_condition(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        df = frame_with(st, "Task")
        self.assertEqual(df["Status"].tolist(), ["confirmed", "conditional"])
        self.assertIn("only if the load test passes", df["Condition / note"].tolist())

    def test_refinement_corrections_and_refusals_are_listed(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        frames = [c.args[0] for c in st.dataframe.call_args_list]
        applied = next(f for f in frames if "Replacement" in f.columns and "Reason" in f.columns)
        refused = next(f for f in frames if "Why refused" in f.columns)
        self.assertEqual(applied["Replacement"].tolist(), ["Kubernetes"])
        self.assertIn("protected value", refused["Why refused"].tolist()[0])
        labels = [c.args[0] for c in st.expander.call_args_list]
        self.assertTrue(any("1 applied, 1 rejected" in label for label in labels))

    def test_results_are_split_into_tabs_with_counts(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        labels = st.tabs.call_args.args[0]
        self.assertEqual(len(labels), 6)
        for word in ("Transcripts", "Summary", "Minutes", "Key decisions (1)", "Action items (2)", "Downloads"):
            self.assertTrue(any(word in label for label in labels), word)

    def test_overview_shows_run_numbers(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        metrics = {c.args[0]: c.args[1] for c in st.col.metric.call_args_list}
        self.assertEqual(metrics["Audio length"], "1m 05s")
        self.assertEqual(metrics["Speech model"], "large-v3-turbo")
        self.assertEqual((metrics["Key decisions"], metrics["Action items"], metrics["Corrections applied"]), (1, 2, 1))

    def test_warnings_are_grouped_in_one_notice_expander(self):
        st = make_stub()
        st.session_state["result"] = make_result()
        run_app(st)
        labels = [c.args[0] for c in st.expander.call_args_list]
        self.assertTrue(any("notice(s) about this run" in label for label in labels))

    def test_status_chips_escape_html_and_dollar_signs(self):
        import importlib.util
        self.assertIn("&lt;b&gt;", self._chip("<b>"))
        self.assertNotIn("$", self._chip("$4500 cap"))

    @staticmethod
    def _chip(text):
        st = make_stub()
        with mock.patch.dict(sys.modules, {"streamlit": st}):
            ns = runpy.run_path(APP, run_name="app_under_test")
        return ns["chip"](text)

    def test_landing_page_before_first_run(self):
        st = make_stub()
        run_app(st)
        st.tabs.assert_not_called()
        captions = " ".join(c.args[0] for c in st.caption.call_args_list)
        for needle in ("faster-whisper", "openai/gpt-oss-120b", "gemini"):
            self.assertIn(needle, captions)

    def test_pipeline_error_is_shown_and_halts(self):
        err = UnsupportedFileTypeError("'notes.txt' is not a supported file type.")

        def failing(path, name, cfg, on_event):
            on_event("Transcribing", "start", None)
            on_event("Transcribing", "error", err)
            raise err

        st = make_stub(uploaded=fake_upload("notes.txt"), button=True)
        st.session_state["result"] = make_result()           # stale result must be cleared
        with mock.patch("pipeline.run.run_pipeline", failing):
            run_app(st)
        self.assertNotIn("result", st.session_state)
        st.error.assert_called_once_with(err.user_message)
        self.assertIn("Transcribing failed", err.user_message)
        self.assertEqual(len(collect_downloads(st)), 0)

    def test_completed_stages_are_kept_when_a_later_stage_fails(self):
        from pipeline.errors import RateLimitedError
        result = make_result()
        result.transcript.warnings.append("GPU unavailable (cublas missing); falling back to CPU")
        err = RateLimitedError("Groq rate limit still hit after retries.")

        def failing(path, name, cfg, on_event):
            on_event("Transcribing", "done", result.transcript)
            on_event("Refining", "error", err)
            raise err

        st = make_stub(uploaded=fake_upload(), button=True)
        with mock.patch("pipeline.run.run_pipeline", failing):
            run_app(st)
        st.error.assert_called_once_with(err.user_message)
        shown = [c.args[0] for c in st.warning.call_args_list]
        self.assertTrue(any("falling back to CPU" in w for w in shown))   # explains why CPU was used
        texts = [c.args[1] for c in st.text_area.call_args_list]
        self.assertIn(result.raw_text, texts)
        keys = [c.kwargs.get("key") for c in st.download_button.call_args_list]
        self.assertIn("dl_partial_raw", keys)

    def test_empty_results_show_info_not_crash(self):
        from pipeline.schemas import MeetingRecord
        st = make_stub()
        st.session_state["result"] = make_result(MeetingRecord(summary="Nothing decided."))
        run_app(st)
        infos = [c.args[0] for c in st.info.call_args_list]
        self.assertTrue(any("No decisions" in i for i in infos))
        self.assertTrue(any("No action items" in i for i in infos))
        for call in st.dataframe.call_args_list:              # only the corrections tables may appear
            self.assertNotIn("Task", list(call.args[0].columns))


if __name__ == "__main__":
    unittest.main()
