import unittest
from unittest import mock

from pipeline import run
from pipeline.config import Config
from pipeline.errors import MeetingAssistantError, MissingAPIKeyError, MissingGeminiKeyError, NoSpeechError
from tests.helpers import make_result

CFG = Config(groq_api_key="g", gemini_api_key="m")


def patched(stt=None, refine=None, doc=None):
    r = make_result()
    return mock.patch.multiple(
        "pipeline.run",
        transcribe_audio=stt or mock.Mock(return_value=r.transcript),
        refine_transcript=refine or mock.Mock(return_value=r.refinement),
        generate_record=doc or mock.Mock(return_value=r.document))


class RunPipelineTests(unittest.TestCase):
    def test_stages_run_in_order_and_feed_each_other(self):
        events = []
        r = make_result()
        stt, ref, doc = (mock.Mock(return_value=r.transcript), mock.Mock(return_value=r.refinement),
                         mock.Mock(return_value=r.document))
        with patched(stt, ref, doc):
            out = run.run_pipeline("a.mp3", "meeting.mp3", CFG, lambda *e: events.append(e[:2]))
        self.assertEqual([e for e in events if e[1] in ("start", "done")],
                         [("Transcribing", "start"), ("Transcribing", "done"), ("Refining", "start"),
                          ("Refining", "done"), ("Generating record", "start"), ("Generating record", "done")])
        self.assertEqual(ref.call_args.args[0], r.transcript.text)       # stage 2 gets the RAW text
        self.assertEqual(doc.call_args.args[0], r.refinement.text)       # stage 3 gets the REFINED text
        self.assertEqual(out.source_name, "meeting.mp3")
        self.assertEqual(set(out.timings), {"transcribing", "refining", "generating_record"})
        self.assertTrue(out.generated_at)

    def test_missing_keys_fail_before_any_stage(self):
        stt = mock.Mock()
        with patched(stt):
            with self.assertRaises(MissingAPIKeyError):
                run.run_pipeline("a.mp3", config=Config(groq_api_key="", gemini_api_key="m"))
            with self.assertRaises(MissingGeminiKeyError):
                run.run_pipeline("a.mp3", config=Config(groq_api_key="g", gemini_api_key=""))
        stt.assert_not_called()

    def test_stage_failure_halts_and_reports_stage(self):
        events, refine = [], mock.Mock()
        with patched(stt=mock.Mock(side_effect=NoSpeechError("no speech"))):
            with mock.patch("pipeline.run.refine_transcript", refine):
                with self.assertRaises(NoSpeechError):
                    run.run_pipeline("a.mp3", config=CFG, on_event=lambda *e: events.append(e))
        refine.assert_not_called()
        self.assertEqual(events[-1][:2], ("Transcribing", "error"))

    def test_later_stage_failure_reports_that_stage(self):
        events = []
        err = MeetingAssistantError("boom")
        with patched(refine=mock.Mock(side_effect=err)):
            with self.assertRaises(MeetingAssistantError):
                run.run_pipeline("a.mp3", config=CFG, on_event=lambda *e: events.append(e))
        self.assertEqual(events[-1][:2], ("Refining", "error"))

    def test_unexpected_exception_is_wrapped_with_stage(self):
        with patched(doc=mock.Mock(side_effect=KeyError("x"))):
            with self.assertRaises(MeetingAssistantError) as cm:
                run.run_pipeline("a.mp3", config=CFG)
        self.assertEqual(cm.exception.stage, "Generating record")
        self.assertIn("Generating record failed", cm.exception.user_message)

    def test_progress_events_forwarded(self):
        events = []
        r = make_result()

        def fake_stt(path, name, cfg, progress):
            progress(0.5)
            return r.transcript

        with patched(stt=fake_stt):
            run.run_pipeline("a.mp3", config=CFG, on_event=lambda *e: events.append(e))
        self.assertIn(("Transcribing", "progress", 0.5), events)

    def test_warnings_aggregated(self):
        res = make_result()
        res.transcript.warnings.append("gpu fallback")
        self.assertIn("gpu fallback", res.warnings)
        self.assertEqual(res.warnings[-1], res.document.warnings[-1])


if __name__ == "__main__":
    unittest.main()
