"""Shared builders for Stage 4 tests."""
from pipeline.document import DocumentResult
from pipeline.refine import RefinementResult
from pipeline.run import PipelineResult
from pipeline.schemas import ActionItem, KeyDecision, MeetingRecord, MinutesSection
from pipeline.stt import Segment, TranscriptResult

RAW = "We will deploy on cube are net ease next week.\n\nPriya, can you take the migration? Sure, I will finish it by Friday."
REFINED = "We will deploy on Kubernetes next week.\n\nPriya, can you take the migration? Sure, I will finish it by Friday."


def make_record() -> MeetingRecord:
    return MeetingRecord(
        summary="The team planned a Kubernetes deployment and a database migration.",
        minutes=[MinutesSection(title="Deployment", points=["Deploy on Kubernetes next week"])],
        key_decisions=[KeyDecision(decision="Deploy on Kubernetes", context="Agreed in the meeting",
                                   evidence=["We will deploy on Kubernetes next week."],
                                   evidence_verified=True)],
        action_items=[
            ActionItem(task="Finish the migration | phase 1", owner="Priya", deadline="by Friday",
                       evidence=["Priya, can you take the migration? Sure, I will finish it by Friday."],
                       evidence_verified=True),
            ActionItem(task="Send the report", status="conditional", condition="only if the load test passes",
                       evidence=["I'll send the report."],
                       evidence_verified=False),
        ])


def make_result(record: MeetingRecord | None = None, name: str = "standup.mp3") -> PipelineResult:
    seg = Segment(0.0, 4.0, "We will deploy on cube are net ease next week.")
    transcript = TranscriptResult(text=RAW, segments=[seg], language="en", duration=65.0,
                                  model_name="large-v3-turbo", device="cuda", compute_type="float16")
    refinement = RefinementResult(
        text=REFINED, model="openai/gpt-oss-120b", chunks_total=1,
        applied=[{"context": "deploy on cube are net ease next", "original": "cube are net ease",
                  "replacement": "Kubernetes", "reason": "tool name", "support": "context"}],
        rejected=[{"context": "latency is 40 milliseconds", "original": "40", "replacement": "50",
                   "why": "protected value (day/number/date) not corroborated elsewhere"}])
    document = DocumentResult(record=record or make_record(), model="gemini-3.8-flash",
                              warnings=["1 item(s) have evidence quotes that could not be verified."])
    return PipelineResult(source_name=name, transcript=transcript, refinement=refinement,
                          document=document, timings={"transcribing": 1.0},
                          generated_at="2026-10-07T12:00:00+00:00")
