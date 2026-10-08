"""Stage 3 flow tests with a scripted fake Gemini client (no network, no google-genai needed)."""
import json
import re
import unittest
from types import SimpleNamespace

from pipeline import document
from pipeline.config import Config
from pipeline.errors import (
    DocumentingError, ExtractionError, GeminiNetworkError, GeminiRateLimitedError,
    InvalidGeminiKeyError, MissingGeminiKeyError,
)
from pipeline.schemas import UNSPECIFIED

CFG = Config(gemini_api_key="gem-test-key", document_max_retries=3, gemini_model="gemini-3.8-flash",
             gemini_fallback_models=("gemini-3.5-flash", "gemini-3.1-flash-lite"))

TRANSCRIPT = (
    "Okay, let's start with the deployment plan. I think we should move the API to Kubernetes. "
    "Sounds good, let's do that.\n\n"
    "Priya, can you take the database migration? Sure, I will finish it by Friday.\n\n"
    "We also talked about adding a caching layer with Redis. Maybe next quarter, nobody has "
    "agreed on that yet. Somebody should update the documentation. I'll send the report."
)
Q_DEC = "I think we should move the API to Kubernetes. Sounds good, let's do that."
Q_MIG = "Priya, can you take the database migration? Sure, I will finish it by Friday."
Q_REPORT = "I'll send the report."
Q_DOC = "Somebody should update the documentation."


def scan(*kinds_quotes):
    return [{"quote": q, "kind": k} for k, q in kinds_quotes]


def record(decisions=(), actions=(), scan_items=None, summary="Deployment and migration were discussed."):
    return {"commitment_scan": scan_items if scan_items is not None else [],
            "summary": summary,
            "minutes": [{"topic": "Deployment", "points": ["Move API to Kubernetes"]}],
            "decisions": list(decisions), "action_items": list(actions)}


def dec(text, evidence):
    return {"decision": text, "evidence": evidence}


def act(task, evidence, owner="unspecified", deadline="unspecified", status="confirmed", condition=""):
    return {"task": task, "owner": owner, "deadline": deadline, "status": status,
            "condition": condition, "evidence": evidence}


SCAN2 = scan(("decision", Q_DEC), ("task", Q_MIG), ("task", Q_REPORT))
GOOD = record([dec("Move the API to Kubernetes", Q_DEC)],
              [act("Finish the database migration", Q_MIG, "Priya", "by Friday"),
               act("Send the report", Q_REPORT)], SCAN2)


class FakeGeminiError(Exception):
    def __init__(self, code, msg="err"):
        super().__init__(msg)
        self.code = code


class ConnectError(Exception):
    pass


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(contents)
        if isinstance(item, dict):
            item = json.dumps(item)
        return SimpleNamespace(text=item)


def run(script, text=TRANSCRIPT, cfg=CFG, sleeps=None):
    sleeps = [] if sleeps is None else sleeps
    client = FakeClient(script)
    res = document.generate_record(text, cfg, client=client, sleep=sleeps.append)
    return res, client


class ExtractionTests(unittest.TestCase):
    def test_happy_path(self):
        res, client = run([GOOD])
        rec = res.record
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(rec.key_decisions), 1)
        self.assertTrue(rec.key_decisions[0].evidence_verified)
        migration, report = rec.action_items
        self.assertEqual((migration.owner, migration.deadline, migration.status), ("Priya", "by Friday", "confirmed"))
        self.assertEqual((report.owner, report.deadline), (UNSPECIFIED, UNSPECIFIED))
        self.assertEqual(rec.minutes[0].title, "Deployment")
        self.assertEqual(res.unverified_items, 0)
        self.assertEqual(res.scan_tasks, 2)
        self.assertEqual(res.warnings, [])
        self.assertEqual(client.calls[0]["model"], "gemini-3.8-flash")
        self.assertTrue(client.calls[0]["contents"].startswith("<transcript>"))

    def test_no_decisions_or_tasks_gives_empty_lists(self):
        res, _ = run([record()])
        self.assertEqual(res.record.key_decisions, [])
        self.assertEqual(res.record.action_items, [])

    def test_tbd_values_normalized(self):
        res, _ = run([record(actions=[act("Send the report", Q_REPORT, "TBD", "N/A")],
                             scan_items=scan(("task", Q_REPORT)))])
        a = res.record.action_items[0]
        self.assertEqual((a.owner, a.deadline), (UNSPECIFIED, UNSPECIFIED))

    def test_unassigned_task_never_keeps_an_owner(self):
        t = act("Update the documentation", Q_DOC, owner="Priya", status="unassigned")
        res, _ = run([record(actions=[t], scan_items=scan(("task", Q_DOC)))])
        a = res.record.action_items[0]
        self.assertEqual((a.owner, a.status), (UNSPECIFIED, "unassigned"))
        self.assertTrue(any("unassigned task" in d for d in res.downgraded_fields))

    def test_conditional_status_and_condition_kept(self):
        t = act("Finish the database migration", Q_MIG, "Priya", "by Friday", "conditional", "only if tests pass")
        res, _ = run([record(actions=[t], scan_items=scan(("task", Q_MIG)))])
        a = res.record.action_items[0]
        self.assertEqual((a.status, a.condition), ("conditional", "only if tests pass"))

    def test_item_without_evidence_dropped_with_warning(self):
        res, _ = run([record([dec("Adopt Redis next quarter", "")])])
        self.assertEqual(res.record.key_decisions, [])
        self.assertEqual(len(res.dropped_items), 1)
        self.assertTrue(any("Dropped" in w for w in res.warnings))

    def test_fabricated_quote_is_flagged_not_hidden(self):
        fabricated = record([dec("Move to Kubernetes", Q_DEC + " / We all agreed to migrate today, no debate.")])
        res, client = run([fabricated])
        self.assertEqual(len(client.calls), 1)
        d = res.record.key_decisions[0]
        self.assertFalse(d.evidence_verified)
        self.assertEqual(len(d.evidence), 2)     # quotes kept unchanged for the user to inspect
        self.assertEqual(res.unverified_items, 1)
        self.assertTrue(any("not verified" in w for w in res.warnings))

    def test_several_quotes_in_one_string_are_split_and_verified(self):
        d = dec("Migration owner", "Priya, can you take the database migration? / Sure, I will finish it by Friday.")
        res, _ = run([record([d])])
        self.assertEqual(len(res.record.key_decisions[0].evidence), 2)
        self.assertTrue(res.record.key_decisions[0].evidence_verified)

    def test_missing_tasks_trigger_one_retry_that_fixes_it(self):
        incomplete = record(actions=[act("Finish the database migration", Q_MIG, "Priya", "by Friday")],
                            scan_items=SCAN2)
        res, client = run([incomplete, GOOD])
        self.assertEqual(len(client.calls), 2)
        self.assertIn("commitment_scan", client.calls[1]["contents"])
        self.assertEqual(len(res.record.action_items), 2)
        self.assertFalse(any("missing" in w for w in res.warnings))

    def test_missing_tasks_persisting_is_reported(self):
        incomplete = record(actions=[act("Finish the database migration", Q_MIG, "Priya", "by Friday")],
                            scan_items=SCAN2)
        res, client = run([incomplete])
        self.assertEqual(len(client.calls), 2)   # one retry only
        self.assertEqual(len(res.record.action_items), 1)
        self.assertTrue(any("some tasks may be missing" in w for w in res.warnings))

    def test_worse_retry_is_ignored(self):
        first = record(actions=[act("Finish the database migration", Q_MIG)], scan_items=SCAN2)
        worse = record(actions=[], scan_items=SCAN2)
        res, _ = run([first, worse])
        self.assertEqual(len(res.record.action_items), 1)

    def test_failed_retry_keeps_first_result(self):
        first = record(actions=[act("Finish the database migration", Q_MIG)], scan_items=SCAN2)
        res, _ = run([first, "not json", "still not json"])
        self.assertEqual(len(res.record.action_items), 1)

    def test_invalid_json_gets_one_repair_retry(self):
        res, client = run(["not json at all", GOOD])
        self.assertEqual(len(client.calls), 2)
        self.assertIn("invalid", client.calls[1]["contents"])
        self.assertEqual(len(res.record.action_items), 2)

    def test_schema_violation_repaired(self):
        res, _ = run([{"summary": "only a summary"}, GOOD])
        self.assertEqual(len(res.record.key_decisions), 1)

    def test_invalid_twice_raises(self):
        with self.assertRaises(ExtractionError):
            run(["nope"])

    def test_code_fenced_json_accepted(self):
        res, _ = run(["```json\n" + json.dumps(GOOD) + "\n```"])
        self.assertEqual(len(res.record.action_items), 2)

    def test_empty_response_error(self):
        with self.assertRaises(ExtractionError):
            run([""])

    def test_empty_transcript(self):
        with self.assertRaises(DocumentingError):
            run([GOOD], text="  ")

    def test_over_long_transcript_is_a_clear_error(self):
        with self.assertRaises(ExtractionError) as cm:
            run([GOOD], cfg=Config(gemini_api_key="k", document_max_chars=50))
        self.assertIn("DOCUMENT_MAX_CHARS", cm.exception.user_message)

    def test_missing_key(self):
        with self.assertRaises(MissingGeminiKeyError) as cm:
            document.generate_record(TRANSCRIPT, Config(gemini_api_key=""))
        self.assertIn("Generating record failed", cm.exception.user_message)

    def test_api_key_not_sent_in_contents(self):
        _, client = run([GOOD])
        self.assertNotIn("gem-test-key", repr(client.calls))

    def test_request_uses_json_mode_and_schema(self):
        _, client = run([GOOD])
        cfg = client.calls[0]["config"]
        get = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
        self.assertEqual(get("response_mime_type"), "application/json")
        self.assertEqual(get("temperature"), 0.1)
        schema = get("response_json_schema")
        self.assertEqual(next(iter(schema["properties"])), "commitment_scan")   # scan comes first
        self.assertIn("action_items", schema["properties"])
        self.assertIn("commitment_scan", get("system_instruction"))


class ApiErrorTests(unittest.TestCase):
    def test_429_retries_with_backoff_then_succeeds(self):
        sleeps = []
        res, client = run([FakeGeminiError(429, "quota exceeded, retry in 7.5s"), FakeGeminiError(429), GOOD],
                          sleeps=sleeps)
        self.assertEqual(len(res.record.action_items), 2)
        self.assertGreaterEqual(sleeps[0], 7.5)
        self.assertGreaterEqual(sleeps[1], 6.0)  # exponential: base 3 * 2**1

    def test_429_exhausted(self):
        sleeps = []
        with self.assertRaises(GeminiRateLimitedError):
            run([FakeGeminiError(429)], sleeps=sleeps)
        self.assertEqual(len(sleeps), CFG.document_max_retries)

    def test_daily_quota_not_retried(self):
        sleeps = []
        with self.assertRaises(GeminiRateLimitedError):
            run([FakeGeminiError(429, "Quota exceeded for metric ... PerDay")], sleeps=sleeps)
        self.assertEqual(sleeps, [])

    def test_invalid_key_variants(self):
        for err in (FakeGeminiError(400, "API key not valid. Please pass a valid API key."),
                    FakeGeminiError(403, "permission denied"), FakeGeminiError(401)):
            with self.assertRaises(InvalidGeminiKeyError):
                run([err])

    def test_503_then_success(self):
        sleeps = []
        res, _ = run([FakeGeminiError(503, "overloaded"), GOOD], sleeps=sleeps)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(len(res.record.key_decisions), 1)

    def test_connection_error_exhausted_shows_underlying_error_without_key(self):
        with self.assertRaises(GeminiNetworkError) as cm:
            run([ConnectError("down: https://x/?key=gem-test-key")])
        msg = cm.exception.user_message
        self.assertIn("ConnectError", msg)
        self.assertIn("down", msg)
        self.assertNotIn("gem-test-key", msg)
        self.assertIn("check_gemini.py", msg)

    def test_persistent_503_switches_to_fallback_model(self):
        errs = [FakeGeminiError(503, "high demand")] * 3
        res, client = run([*errs, GOOD])
        self.assertEqual(client.calls[0]["model"], "gemini-3.8-flash")
        self.assertEqual(client.calls[-1]["model"], "gemini-3.5-flash")
        self.assertTrue(any("overloaded" in w for w in res.warnings))

    def test_503_exhausted_reports_code(self):
        with self.assertRaises(GeminiNetworkError) as cm:
            run([FakeGeminiError(503, "The model is overloaded")])
        self.assertIn("503", cm.exception.user_message)

    def test_unknown_model_falls_back(self):
        res, client = run([FakeGeminiError(404, "models/gemini-3.8-flash is not found"), GOOD])
        self.assertEqual([c["model"] for c in client.calls], ["gemini-3.8-flash", "gemini-3.5-flash"])
        self.assertEqual(res.model, "gemini-3.5-flash")
        self.assertTrue(any("not available" in w for w in res.warnings))

    def test_all_models_missing(self):
        with self.assertRaises(ExtractionError) as cm:
            run([FakeGeminiError(404, "not found")])
        self.assertIn("GEMINI_MODEL", cm.exception.user_message)

    def test_other_400_is_clear_error(self):
        with self.assertRaises(ExtractionError):
            run([FakeGeminiError(400, "Invalid JSON payload")])


class PromptTests(unittest.TestCase):
    def test_extraction_prompt_has_required_rules(self):
        p = document.load_prompt("document_system.txt")
        for needle in ("commitment_scan", "STEP 1, SCAN", "unspecified", "conditional", "unassigned", "rejected",
                       "Never include a \"proposal\""):
            self.assertIn(needle, p)

    def test_missing_prompt(self):
        with self.assertRaises(DocumentingError):
            document.load_prompt("nope.txt")


if __name__ == "__main__":
    unittest.main()
