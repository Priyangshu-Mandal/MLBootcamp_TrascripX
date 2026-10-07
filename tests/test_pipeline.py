import pytest

from meeting_assistant.models import ActionItem, Documentation
from meeting_assistant.pipeline import InputError, MeetingAssistant


class FakeProvider:
    def __init__(self):
        self.calls = []

    def transcribe(self, filename, audio_bytes):
        self.calls.append(("transcribe", filename, audio_bytes))
        return "Raw words"

    def refine(self, raw_transcript):
        self.calls.append(("refine", raw_transcript))
        return "Refined words"

    def document(self, refined_transcript):
        self.calls.append(("document", refined_transcript))
        return Documentation(
            "A concise summary.",
            ["The release was approved."],
            [ActionItem("Publish the release", "Asha", "Friday")],
        )


def test_pipeline_runs_stages_in_order():
    provider = FakeProvider()
    result = MeetingAssistant(provider).process("meeting.wav", b"audio")

    assert result.raw_transcript == "Raw words"
    assert result.refined_transcript == "Refined words"
    assert provider.calls == [
        ("transcribe", "meeting.wav", b"audio"),
        ("refine", "Raw words"),
        ("document", "Refined words"),
    ]


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("meeting.txt", b"audio", "Unsupported file type"),
        ("meeting.wav", b"", "empty"),
        ("", b"audio", "Unsupported file"),
    ],
)
def test_invalid_uploads_halt_before_provider(filename, data, message):
    provider = FakeProvider()

    with pytest.raises(InputError, match=message):
        MeetingAssistant(provider).process(filename, data)
    assert provider.calls == []
