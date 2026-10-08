"""Stage 1 tests. faster-whisper is faked (no model download, no network); ffmpeg is real."""
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipeline import stt
from pipeline.config import Config, load_config
from pipeline.errors import (
    AudioDecodeError, EmptyFileError, FFmpegNotFoundError, GPUError, ModelLoadError,
    NoSpeechError, TranscriptionError, UnsupportedFileTypeError,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def seg(start, end, text, no_speech_prob=None, avg_logprob=None):
    return SimpleNamespace(start=start, end=end, text=text,
                           no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


def make_fake_whisper(segments=(), load_error=None, run_error=None, log=None):
    class FakeModel:
        def __init__(self, name, device, compute_type):
            if log is not None:
                log.append((name, device, compute_type))
            if load_error and device == "cuda":
                raise load_error

        def transcribe(self, audio, **kw):
            FakeModel.kwargs = kw
            FakeModel.audio = audio
            if run_error and FakeModel.device_of_last_load() == "cuda":
                raise run_error
            return iter(segments), SimpleNamespace(language="en")

        @staticmethod
        def device_of_last_load():
            return log[-1][1] if log else "cpu"
    return FakeModel


def make_silent_wav(path, seconds=1, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * seconds * channels)


def make_tone(path, seconds=2):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-ar", "44100", "-ac", "2",
                    str(path)], check=True)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_unsupported_extension(self):
        f = self.dir / "notes.txt"
        f.write_text("hello")
        with self.assertRaises(UnsupportedFileTypeError):
            stt.validate_audio_file(f)

    def test_no_extension_unsupported(self):
        f = self.dir / "recording"
        f.write_bytes(b"abc")
        with self.assertRaises(UnsupportedFileTypeError):
            stt.validate_audio_file(f)

    def test_empty_file(self):
        f = self.dir / "a.wav"
        f.write_bytes(b"")
        with self.assertRaises(EmptyFileError):
            stt.validate_audio_file(f)

    def test_original_name_used_for_extension(self):
        f = self.dir / "upload_tmp"
        f.write_bytes(b"abc")
        stt.validate_audio_file(f, "meeting.MP3")  # accepted, case-insensitive
        with self.assertRaises(UnsupportedFileTypeError):
            stt.validate_audio_file(f, "meeting.pdf")

    def test_too_large(self):
        f = self.dir / "a.wav"
        f.write_bytes(b"x" * 2048)
        cfg = Config(max_upload_mb=0)
        with self.assertRaises(UnsupportedFileTypeError):
            stt.validate_audio_file(f, config=cfg)

    def test_user_message_names_stage(self):
        try:
            stt.validate_audio_file(self.dir / "x.txt")
        except UnsupportedFileTypeError as e:
            self.assertIn("Transcribing failed", e.user_message)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class FFmpegTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_converts_to_16k_mono(self):
        src = self.dir / "tone.mp3"
        make_tone(src)
        dst = self.dir / "out.wav"
        dur = stt.convert_to_wav(src, dst)
        with wave.open(str(dst)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
        self.assertAlmostEqual(dur, 2.0, delta=0.2)

    def test_corrupt_audio(self):
        src = self.dir / "bad.wav"
        src.write_bytes(os.urandom(4096))
        with self.assertRaises(AudioDecodeError):
            stt.convert_to_wav(src, self.dir / "o.wav")

    def test_truncated_header_only(self):
        src = self.dir / "trunc.mp3"
        src.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")
        with self.assertRaises((AudioDecodeError, NoSpeechError)):
            stt.convert_to_wav(src, self.dir / "o.wav")

    def test_ffmpeg_missing(self):
        with mock.patch("pipeline.stt.shutil.which", return_value=None):
            with self.assertRaises(FFmpegNotFoundError):
                stt.convert_to_wav(self.dir / "a.wav", self.dir / "o.wav")

    def test_too_short_audio(self):
        src = self.dir / "short.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=0.1", str(src)], check=True)
        with self.assertRaises(NoSpeechError):
            stt.convert_to_wav(src, self.dir / "o.wav")


class TranscribeTests(unittest.TestCase):
    CPU = Config(whisper_device="cpu")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wav = Path(self.tmp.name) / "audio.wav"
        make_silent_wav(self.wav)

    def run_wav(self, fake, cfg, duration=10.0, progress=None):
        with mock.patch("pipeline.stt._import_whisper_model", return_value=fake):
            return stt.transcribe_wav(self.wav, duration, cfg, progress)

    def test_model_receives_float32_array_not_a_path(self):
        import numpy as np
        fake = make_fake_whisper([seg(0, 1, "Hi.")], log=[])
        self.run_wav(fake, self.CPU)
        self.assertIsInstance(fake.audio, np.ndarray)
        self.assertEqual(fake.audio.dtype, np.float32)
        self.assertEqual(len(fake.audio), 16000)

    def test_unexpected_wav_format_rejected(self):
        make_silent_wav(self.wav, channels=2)
        with self.assertRaises(AudioDecodeError):
            self.run_wav(make_fake_whisper([seg(0, 1, "x")], log=[]), self.CPU)

    def test_cpu_transcription_and_paragraphs(self):
        log = []
        fake = make_fake_whisper([seg(0, 2, " Hello team. "), seg(2.2, 4, "Let's start."),
                                  seg(9, 11, "Next topic.")], log=log)
        res = self.run_wav(fake, self.CPU)
        self.assertEqual(res.text, "Hello team. Let's start.\n\nNext topic.")
        self.assertEqual(log, [("small.en", "cpu", "int8")])
        self.assertEqual(fake.kwargs["vad_filter"], True)
        self.assertEqual(fake.kwargs["language"], "en")
        self.assertEqual(len(res.segments), 3)
        self.assertEqual(res.warnings, [])

    def test_no_speech_raises(self):
        with self.assertRaises(NoSpeechError):
            self.run_wav(make_fake_whisper([]), self.CPU)

    def test_blank_segments_are_no_speech(self):
        with self.assertRaises(NoSpeechError):
            self.run_wav(make_fake_whisper([seg(0, 1, "   ")]), self.CPU)

    def test_cuda_load_failure_falls_back_to_cpu(self):
        log = []
        fake = make_fake_whisper([seg(0, 1, "Hi.")], log=log,
                                 load_error=RuntimeError("Library libcublas.so.12 is not found"))
        res = self.run_wav(fake, Config(whisper_device="cuda"))
        self.assertEqual(log, [("large-v3-turbo", "cuda", "float16"), ("small.en", "cpu", "int8")])
        self.assertEqual(res.device, "cpu")
        self.assertEqual(res.model_name, "small.en")
        self.assertTrue(res.warnings and "falling back to CPU" in res.warnings[0])

    def test_cuda_runtime_failure_retries_on_cpu(self):
        log = []
        fake = make_fake_whisper([seg(0, 1, "Hi.")], log=log,
                                 run_error=RuntimeError("CUDA failed with error out of memory"))
        res = self.run_wav(fake, Config(whisper_device="cuda"))
        self.assertEqual(res.device, "cpu")
        self.assertIn("retrying on CPU", res.warnings[0])

    def test_cuda_error_on_cpu_only_config_is_gpu_error(self):
        fake = make_fake_whisper([], run_error=RuntimeError("cuda boom"), log=[])
        # force run_error to trigger even on cpu plan
        fake.device_of_last_load = staticmethod(lambda: "cuda")
        with self.assertRaises(GPUError):
            self.run_wav(fake, self.CPU)

    def test_non_cuda_error_becomes_transcription_error(self):
        fake = make_fake_whisper([], run_error=ValueError("bad tensor"), log=[("m", "cuda", "x")])
        with self.assertRaises(TranscriptionError):
            self.run_wav(fake, Config(whisper_device="cuda"))

    def test_model_download_failure_is_model_load_error(self):
        class Boom:
            def __init__(self, *a, **k):
                raise OSError("Connection error while downloading")
        with self.assertRaises(ModelLoadError):
            self.run_wav(Boom, self.CPU)

    def test_missing_package(self):
        with mock.patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(ModelLoadError):
                stt.transcribe_wav(self.wav, 5.0, self.CPU)

    def test_model_released_after_transcription(self):
        fake = make_fake_whisper([seg(0, 1, "Hi.")], log=[])
        with mock.patch("pipeline.stt._release") as rel:
            self.run_wav(fake, self.CPU)
        rel.assert_called_once()

    def test_model_released_even_on_failure(self):
        fake = make_fake_whisper([], run_error=ValueError("x"), log=[("m", "cuda", "x")])
        with mock.patch("pipeline.stt._release") as rel:
            with self.assertRaises(TranscriptionError):
                self.run_wav(fake, Config(whisper_device="cuda"))
        rel.assert_called_once()

    def test_progress_reported(self):
        seen = []
        fake = make_fake_whisper([seg(0, 5, "a."), seg(5, 10, "b.")], log=[])
        self.run_wav(fake, self.CPU, duration=10.0, progress=seen.append)
        self.assertEqual(seen, [0.5, 1.0])

    def test_auto_without_cuda_uses_cpu_plan(self):
        with mock.patch("pipeline.stt.cuda_available", return_value=False):
            self.assertEqual(stt._plans(Config(whisper_device="auto")),
                             [("cpu", "small.en", "int8")])
        with mock.patch("pipeline.stt.cuda_available", return_value=True):
            self.assertEqual(stt._plans(Config(whisper_device="auto"))[0],
                             ("cuda", "large-v3-turbo", "float16"))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class EndToEndStage1Tests(unittest.TestCase):
    def test_audio_file_to_transcript_with_fake_model(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "meeting.m4a"
            make_tone(src, 3)
            fake = make_fake_whisper([seg(0, 2, "We will use gRPC.")], log=[])
            with mock.patch("pipeline.stt._import_whisper_model", return_value=fake):
                res = stt.transcribe_audio(src, "meeting.m4a", Config(whisper_device="cpu"))
            self.assertEqual(res.text, "We will use gRPC.")
            self.assertAlmostEqual(res.duration, 3.0, delta=0.3)

    def test_corrupt_upload_halts_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "x.mp3"
            src.write_bytes(os.urandom(2000))
            loader = mock.Mock()
            with mock.patch("pipeline.stt._import_whisper_model", loader):
                with self.assertRaises(AudioDecodeError):
                    stt.transcribe_audio(src, config=Config(whisper_device="cpu"))
            loader.assert_not_called()


class NvidiaDllTests(unittest.TestCase):
    def test_adds_bin_dirs_on_windows_only(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cublas" / "bin").mkdir(parents=True)
            (Path(d) / "cudnn" / "bin").mkdir(parents=True)
            (Path(d) / "cudnn" / "include").mkdir(parents=True)
            spec = SimpleNamespace(submodule_search_locations=[d])
            with mock.patch.dict(os.environ, {"PATH": "C:\\base"}):
                with mock.patch("pipeline.stt._IS_WINDOWS", True), \
                        mock.patch("pipeline.stt.importlib.util.find_spec", return_value=spec):
                    added = stt.register_nvidia_dll_dirs()
                    again = stt.register_nvidia_dll_dirs()   # idempotent
                    path = os.environ["PATH"]
            self.assertEqual(len(added), 2)
            self.assertEqual(again, [])
            self.assertIn(str(Path(d) / "cublas" / "bin"), path)
            self.assertNotIn("include", path)
            with mock.patch("pipeline.stt._IS_WINDOWS", False):
                self.assertEqual(stt.register_nvidia_dll_dirs(), [])

    def test_no_nvidia_package_is_fine(self):
        with mock.patch("pipeline.stt._IS_WINDOWS", True), \
                mock.patch("pipeline.stt.importlib.util.find_spec", return_value=None):
            self.assertEqual(stt.register_nvidia_dll_dirs(), [])


class ConfigTests(unittest.TestCase):
    def test_env_overrides_and_keys(self):
        env = {"WHISPER_DEVICE": "CPU", "GROQ_API_KEY": " k1 ", "WHISPER_BEAM_SIZE": "abc"}
        with mock.patch.dict(os.environ, env):
            cfg = load_config()
        self.assertEqual(cfg.whisper_device, "cpu")
        self.assertEqual(cfg.groq_api_key, "k1")
        self.assertEqual(cfg.whisper_beam_size, 5)  # invalid value -> default

    def test_secrets_not_in_repr(self):
        self.assertNotIn("secret", repr(Config(groq_api_key="secret")))


if __name__ == "__main__":
    unittest.main()
