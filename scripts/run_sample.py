"""Run the full pipeline on one recording from the command line and save every output.

    python scripts/run_sample.py path/to/meeting.mp3
    python scripts/run_sample.py path/to/meeting.mp3 --out samples/my_meeting

Writes raw_transcript.txt, refined_transcript.txt, meeting_record.md and meeting_record.json
(the same files the web app offers for download). Everything is produced by the pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.errors import MeetingAssistantError  # noqa: E402
from pipeline.export import to_json, to_markdown  # noqa: E402
from pipeline.run import run_pipeline  # noqa: E402


def save_outputs(result, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = result.metadata()
    files = {
        "raw_transcript.txt": result.raw_text,
        "refined_transcript.txt": result.refined_text,
        "meeting_record.md": to_markdown(result.record, meta, include_evidence=True),
        "meeting_record.json": to_json(result.record, meta),
    }
    paths = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None, runner=run_pipeline) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="path to the meeting recording")
    parser.add_argument("--out", help="output folder (default: samples/<recording name>)")
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "samples" / audio.stem

    def on_event(stage: str, status: str, detail) -> None:
        if status == "start":
            print(f"[{stage}] started ...", flush=True)
        elif status == "done":
            print(f"[{stage}] done", flush=True)

    try:
        result = runner(audio, audio.name, load_config(), on_event)
    except MeetingAssistantError as exc:
        print(f"\nFAILED - {exc.user_message}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}")
    paths = save_outputs(result, out_dir)
    print(f"\nSaved {len(paths)} files to {out_dir}:")
    for path in paths:
        print("  ", path.name)
    print("Models:", "; ".join(f"{k}: {v}" for k, v in result.metadata()["models"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
