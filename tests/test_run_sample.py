import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.errors import NoSpeechError
from tests.helpers import make_result

_spec = importlib.util.spec_from_file_location(
    "run_sample", Path(__file__).resolve().parent.parent / "scripts" / "run_sample.py")
run_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_sample)


class RunSampleTests(unittest.TestCase):
    def test_saves_the_four_outputs_from_one_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "standup.mp3"
            audio.write_bytes(b"x")
            out = Path(tmp) / "out"
            code = run_sample.main([str(audio), "--out", str(out)], runner=lambda *a: make_result())
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["meeting_record.json", "meeting_record.md", "raw_transcript.txt",
                              "refined_transcript.txt"])
            self.assertIn("cube are net ease", (out / "raw_transcript.txt").read_text(encoding="utf-8"))
            self.assertIn("Kubernetes", (out / "refined_transcript.txt").read_text(encoding="utf-8"))
            data = json.loads((out / "meeting_record.json").read_text(encoding="utf-8"))
            self.assertIn("key_decisions", data["record"] if "record" in data else data)

    def test_pipeline_error_gives_clear_message_and_nonzero_exit(self):
        def boom(*a):
            raise NoSpeechError("no speech detected.")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stderr"):
                self.assertEqual(run_sample.main([str(Path(tmp) / "a.mp3"), "--out", tmp], runner=boom), 1)


if __name__ == "__main__":
    unittest.main()
