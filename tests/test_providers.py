from types import SimpleNamespace

from meeting_assistant.providers import LocalWhisperQwenProvider


class FakeClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                system = kwargs["messages"][0]["content"]
                content = (
                    "Refined transcript"
                    if "technical transcript editor" in system
                    else '{"minutes":"Summary","decisions":[],"action_items":[]}'
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        completions = Completions()

    chat = Chat()

class FakeWhisper:
    def transcribe(self, audio_path, **options):
        with open(audio_path, "rb") as audio_file:
            assert audio_file.read() == b"audio"
        return {"text": "Transcript"}


class RecordingWhisper:
    def __init__(self):
        self.options = None

    def transcribe(self, audio_path, **options):
        self.options = options
        return {"text": "Transcript"}


def test_local_whisper_qwen_provider_maps_all_stages():
    provider = LocalWhisperQwenProvider(
        client=FakeClient(), whisper_model=FakeWhisper()
    )

    assert provider.transcribe("meeting.wav", b"audio") == "Transcript"
    assert provider.refine("raw") == "Refined transcript"
    assert provider.document("refined").to_dict() == {
        "minutes": "Summary",
        "decisions": [],
        "action_items": [],
    }


def test_transcription_passes_language_and_vocabulary_prompt(monkeypatch):
    model = RecordingWhisper()
    monkeypatch.setenv("WHISPER_LANGUAGE", "en")
    monkeypatch.setenv(
        "WHISPER_INITIAL_PROMPT", "Cline Hackathon, Kubernetes, TensorFlow"
    )
    provider = LocalWhisperQwenProvider(client=FakeClient(), whisper_model=model)

    assert provider.transcribe("meeting.wav", b"audio") == "Transcript"
    assert model.options["language"] == "en"
    assert model.options["initial_prompt"] == (
        "Cline Hackathon, Kubernetes, TensorFlow"
    )


def test_documentation_copies_explicit_deadline_to_matching_action():
    class DocumentationClient(FakeClient):
        class Chat:
            class Completions:
                calls = 0

                @classmethod
                def create(cls, **kwargs):
                    cls.calls += 1
                    content = (
                        "Refined transcript"
                        if cls.calls == 1
                        else '{"minutes":"Summary","decisions":[],'
                        '"action_items":[{"task":"Check loopholes in module one",'
                        '"owner":"unspecified","deadline":"unspecified"}]}'
                    )
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=content)
                            )
                        ]
                    )

            completions = Completions()

        chat = Chat()

    provider = LocalWhisperQwenProvider(
        client=DocumentationClient(), whisper_model=FakeWhisper()
    )
    provider.refine("raw")
    result = provider.document(
        "The team agreed to check loopholes in module one and meet the next day."
    )

    assert result.action_items[0].deadline == "next day"


def test_documentation_copies_explicit_module_owner_to_matching_action():
    class OwnerClient(FakeClient):
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=(
                                        '{"minutes":"Summary","decisions":[],"'
                                        'action_items":[{"task":"Check loopholes in module one",'
                                        '"owner":"unspecified","deadline":"unspecified"}]}'
                                    )
                                )
                            )
                        ]
                    )

            completions = Completions()

        chat = Chat()

    provider = LocalWhisperQwenProvider(
        client=OwnerClient(), whisper_model=FakeWhisper()
    )
    result = provider.document(
        "Mohit is working on module four, Prashad on modules one and two."
    )

    assert result.action_items[0].owner == "Prashad"
