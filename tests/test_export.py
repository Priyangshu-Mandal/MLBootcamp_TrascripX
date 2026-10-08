import json
import re
import unittest

from pipeline.diffview import count_changes, highlight_changes
from pipeline.export import NOT_VERIFIED, to_json, to_markdown
from pipeline.schemas import MeetingRecord
from tests.helpers import make_record, make_result


def md_action_rows(md: str) -> list[tuple[str, str, str]]:
    """Parse (task, owner, deadline) from the Markdown action-items table."""
    section = md.split("## Action Items", 1)[1]
    rows = []
    for line in section.splitlines():
        if re.match(r"^\| \d+ \|", line):
            cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            rows.append((cells[1], cells[2], cells[3]))
    return rows


def md_decisions(md: str) -> list[str]:
    section = md.split("## Key Decisions", 1)[1].split("## Action Items", 1)[0]
    return [re.match(r"^\d+\. \*\*(.*?)\*\*", ln).group(1) for ln in section.splitlines()
            if re.match(r"^\d+\. \*\*", ln)]


class ExportConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.res = make_result()
        self.rec = self.res.record
        self.md = to_markdown(self.rec, self.res.metadata())
        self.js = json.loads(to_json(self.rec, self.res.metadata()))

    def test_same_decisions_in_both_formats(self):
        self.assertEqual(md_decisions(self.md), [d["decision"] for d in self.js["key_decisions"]])

    def test_same_tasks_owners_deadlines_in_both_formats(self):
        json_rows = [(a["task"], a["owner"], a["deadline"]) for a in self.js["action_items"]]
        self.assertEqual(md_action_rows(self.md), json_rows)
        self.assertEqual(json_rows[1][1:], ("unspecified", "unspecified"))

    def test_highlight_escapes_dollar_signs_so_markdown_cannot_render_math(self):
        out = highlight_changes("budget $4500 left, cap at $4500 total", "budget $4500 left, cap at $4500 total now")
        self.assertNotIn("$", out)
        self.assertIn("&#36;4500", out)

    def test_pipe_in_task_is_escaped_not_lost(self):
        self.assertIn("Finish the migration | phase 1", [r[0] for r in md_action_rows(self.md)])

    def test_json_always_has_evidence_and_flag(self):
        for item in (*self.js["key_decisions"], *self.js["action_items"]):
            self.assertIn("evidence", item)
            self.assertIn("evidence_verified", item)
        self.assertEqual(self.js["action_items"][0]["evidence"][0][:5], "Priya")

    def test_markdown_excludes_evidence_by_default(self):
        self.assertNotIn("We will deploy on Kubernetes next week.\"", self.md)
        self.assertNotIn("Why", self.md)
        self.assertNotIn(NOT_VERIFIED, self.md)

    def test_markdown_optionally_includes_evidence_without_changing_items(self):
        with_ev = to_markdown(self.rec, self.res.metadata(), include_evidence=True)
        self.assertIn('> "We will deploy on Kubernetes next week."', with_ev)
        self.assertIn(NOT_VERIFIED, with_ev)
        self.assertEqual(md_decisions(with_ev), md_decisions(self.md))
        self.assertEqual(md_action_rows(with_ev), md_action_rows(self.md))

    def test_json_roundtrips_to_same_record(self):
        again = MeetingRecord(**{k: v for k, v in self.js.items() if k != "metadata"})
        self.assertEqual(again, self.rec)

    def test_metadata_in_json(self):
        md = self.js["metadata"]
        self.assertEqual(md["source_file"], "standup.mp3")
        self.assertIn("faster-whisper large-v3-turbo", md["models"]["speech_to_text"])
        self.assertIn("Groq", md["models"]["refinement"])
        self.assertIn("Gemini", md["models"]["documentation"])

    def test_empty_record(self):
        rec = MeetingRecord(summary="Nothing decided.")
        md = to_markdown(rec)
        self.assertIn("No decisions were reached", md)
        self.assertIn("No action items", md)
        js = json.loads(to_json(rec))
        self.assertEqual((js["key_decisions"], js["action_items"]), ([], []))

    def test_status_always_shown_and_condition_column_only_when_present(self):
        self.assertIn("| Status |", self.md)
        self.assertIn("conditional | only if the load test passes", self.md)
        rec = MeetingRecord(summary="s", action_items=[make_record().action_items[0]])
        md = to_markdown(rec)
        self.assertIn("| confirmed |", md)
        self.assertNotIn("Condition", md)

    def test_json_has_status_and_condition(self):
        js = json.loads(to_json(make_record()))
        self.assertEqual([a["status"] for a in js["action_items"]], ["confirmed", "conditional"])
        self.assertEqual(js["action_items"][1]["condition"], "only if the load test passes")
        self.assertNotIn("notes", js["action_items"][0])

    def test_unicode_preserved_in_json(self):
        rec = MeetingRecord(summary="Café résumé — 日本")
        self.assertIn("Café résumé — 日本", to_json(rec))


class DiffViewTests(unittest.TestCase):
    def test_highlights_only_changed_words_and_escapes_html(self):
        out = highlight_changes("use cube are net ease <script>x</script>", "use Kubernetes <script>x</script>")
        self.assertIn("<mark>Kubernetes</mark>", out)
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_identical_text_has_no_marks(self):
        self.assertNotIn("<mark>", highlight_changes("same text", "same text"))
        self.assertEqual(count_changes("same text", "same text"), 0)

    def test_count_changes(self):
        self.assertEqual(count_changes("a b c d", "a X c Y"), 2)


if __name__ == "__main__":
    unittest.main()
