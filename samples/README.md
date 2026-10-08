# Sample recording and generated outputs

This folder holds a recording that may be shared and the outputs the application produced for it.
Nothing here is written by hand: every output file comes from `scripts/run_sample.py`, which runs
the same pipeline as the web app (transcription, refinement, documentation).

Regenerate the outputs for any recording:

```
python scripts/run_sample.py samples/<recording file>
```

| File | Content |
|------|---------|
| `raw_transcript.txt` | Speech-to-text output before refinement |
| `refined_transcript.txt` | Transcript after domain-term correction |
| `meeting_record.md` | Summary, minutes, key decisions, action items (with supporting quotes) |
| `meeting_record.json` | The same record as structured data, plus model and run metadata |
