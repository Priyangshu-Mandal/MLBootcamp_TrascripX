import json
import unittest

from pipeline.config import Config
from pipeline.evidence import TranscriptIndex, clean_quotes, drop_unsupported_items, verify_record
from pipeline.schemas import (
    UNSPECIFIED, ActionItem, KeyDecision, LlmRecord, MeetingRecord, llm_json_schema,
    normalize_unspecified,
)

TRANSCRIPT = (
    "Okay, let's start with the deployment plan. I think we should move the API to Kubernetes. "
    "Sounds good, let's do that.\n\n"
    "Priya, can you take the database migration? Sure, I will finish it by Friday.\n\n"
    "We also talked about adding a caching layer with Redis. Maybe next quarter, nobody has "
    "agreed on that yet. The latency target is 40 milliseconds and we will not exceed it."
)


class NormalizeTests(unittest.TestCase):
    def test_empty_like_values_become_unspecified(self):
        for v in (None, "", "  ", "N/A", "n/a", "TBD", "tba", "None", "null", "Unknown",
                  "Not specified", "unassigned", "-", "—", "To be determined", "Unspecified", "Nobody."):
            self.assertEqual(normalize_unspecified(v), UNSPECIFIED, v)

    def test_real_values_kept(self):
        self.assertEqual(normalize_unspecified("  Priya "), "Priya")
        self.assertEqual(normalize_unspecified("by Friday"), "by Friday")

    def test_action_item_validator_applies(self):
        a = ActionItem(task="x", owner="TBD", deadline=None)
        self.assertEqual((a.owner, a.deadline), (UNSPECIFIED, UNSPECIFIED))


class SchemaTests(unittest.TestCase):
    def test_llm_schema_is_flat_and_matches_the_prompt(self):
        raw = json.dumps(llm_json_schema(LlmRecord))
        self.assertNotIn("$ref", raw)
        self.assertNotIn("$defs", raw)
        schema = llm_json_schema(LlmRecord)
        self.assertEqual(list(schema["properties"]),
                         ["commitment_scan", "summary", "minutes", "decisions", "action_items"])
        self.assertEqual(schema["properties"]["minutes"]["items"]["required"], ["topic", "points"])
        item = schema["properties"]["action_items"]["items"]
        self.assertEqual(set(item["required"]), {"task", "owner", "deadline", "status", "condition", "evidence"})
        self.assertEqual(set(item["properties"]["status"]["enum"]), {"confirmed", "conditional", "unassigned"})
        scan_kind = schema["properties"]["commitment_scan"]["items"]["properties"]["kind"]
        self.assertEqual(set(scan_kind["enum"]), {"task", "decision", "rejected", "proposal", "info"})

    def test_action_item_defaults_and_condition_cleanup(self):
        a = ActionItem(task="x", condition="   ")
        self.assertEqual((a.status, a.condition), ("confirmed", None))
        with self.assertRaises(Exception):
            ActionItem(task="x", status="maybe")

    def test_record_roundtrip_and_defaults(self):
        rec = MeetingRecord(summary="s", key_decisions=[KeyDecision(decision="d", context="  ")])
        self.assertIsNone(rec.key_decisions[0].context)
        again = MeetingRecord.model_validate_json(rec.model_dump_json())
        self.assertEqual(rec, again)
        self.assertFalse(again.key_decisions[0].evidence_verified)

    def test_invalid_llm_output_rejected(self):
        with self.assertRaises(Exception):
            LlmRecord.model_validate_json('{"summary": "x"}')


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.idx = TranscriptIndex(TRANSCRIPT, 0.92)

    def test_exact_quote(self):
        self.assertTrue(self.idx.verify("I think we should move the API to Kubernetes. Sounds good, let's do that."))

    def test_case_punctuation_whitespace_insensitive(self):
        self.assertTrue(self.idx.verify("  i THINK we should   move the API to Kubernetes  "))
        self.assertTrue(self.idx.verify("“Sounds good, let’s do that.”"))

    def test_fabricated_quote_fails(self):
        self.assertFalse(self.idx.verify("We agreed to move everything to AWS by December."))

    def test_paraphrase_fails(self):
        self.assertFalse(self.idx.verify("Priya agreed to complete the migration on Friday."))

    def test_fuzzy_accepts_tiny_asr_style_difference(self):
        q = "Priya, can you take the database migrations? Sure, I will finish it by Friday."
        self.assertTrue(self.idx.verify(q))

    def test_fuzzy_rejects_changed_number(self):
        q = "The latency target is 50 milliseconds and we will not exceed it."
        self.assertFalse(self.idx.verify(q))

    def test_fuzzy_rejects_dropped_negation(self):
        q = "The latency target is 40 milliseconds and we will exceed it."
        self.assertFalse(self.idx.verify(q))

    def test_fuzzy_rejects_added_hedge(self):
        q = "Sure, I will probably finish it by Friday."
        self.assertFalse(self.idx.verify(q))

    def test_short_quote_needs_exact(self):
        self.assertTrue(self.idx.verify("by Friday"))
        self.assertFalse(self.idx.verify("by Fridays"))

    def test_empty_quote_fails(self):
        self.assertFalse(self.idx.verify(""))
        self.assertFalse(self.idx.verify("... !!"))

    def test_quote_must_match_word_boundaries(self):
        self.assertFalse(self.idx.verify("rnetes"))

    def test_clean_quotes_dedupes_and_caps(self):
        qs = ["a b c", " a  b c ", '"d e f"', "", "g h i", "j k l"]
        self.assertEqual(clean_quotes(qs), ["a b c", "d e f", "g h i"])
        self.assertEqual(len(clean_quotes(qs, limit=None)), 4)


class VerifyRecordTests(unittest.TestCase):
    def setUp(self):
        self.idx = TranscriptIndex(TRANSCRIPT, 0.92)
        self.good_q = "Priya, can you take the database migration? Sure, I will finish it by Friday."

    def rec(self, **kw):
        return MeetingRecord(summary="s", **kw)

    def test_verified_item_keeps_fields_untouched(self):
        r = self.rec(action_items=[ActionItem(task="Migrate DB", owner="Priya", deadline="by Friday",
                                              evidence=[self.good_q])])
        rep = verify_record(r, self.idx)
        a = r.action_items[0]
        self.assertTrue(a.evidence_verified)
        self.assertEqual((a.owner, a.deadline), ("Priya", "by Friday"))
        self.assertEqual(rep.problem_count, 0)

    def test_fabricated_quote_marks_unverified_but_does_not_edit_owner(self):
        # this pass only flags evidence
        r = self.rec(action_items=[ActionItem(task="t", owner="Priya", deadline="by Friday",
                                              evidence=["Priya promised to finish everything by Friday evening."])])
        rep = verify_record(r, self.idx)
        a = r.action_items[0]
        self.assertFalse(a.evidence_verified)
        self.assertEqual((a.owner, a.deadline), ("Priya", "by Friday"))
        self.assertEqual(len(rep.unverified_quotes), 1)

    def test_one_bad_quote_among_good_marks_unverified(self):
        r = self.rec(action_items=[ActionItem(task="t", evidence=[self.good_q, "totally made up sentence here ok"])])
        verify_record(r, self.idx)
        self.assertFalse(r.action_items[0].evidence_verified)

    def test_decision_verification(self):
        r = self.rec(key_decisions=[
            KeyDecision(decision="Move API to Kubernetes", evidence=["Sounds good, let's do that."]),
            KeyDecision(decision="Adopt Redis", evidence=["We agreed to adopt Redis."])])
        verify_record(r, self.idx)
        self.assertTrue(r.key_decisions[0].evidence_verified)
        self.assertFalse(r.key_decisions[1].evidence_verified)

    def test_items_without_evidence_flagged_and_droppable(self):
        r = self.rec(key_decisions=[KeyDecision(decision="x")],
                     action_items=[ActionItem(task="y", evidence=[self.good_q])])
        rep = verify_record(r, self.idx)
        self.assertEqual(rep.items_without_evidence, ["x"])
        dropped = drop_unsupported_items(r)
        self.assertEqual(dropped, ["x"])
        self.assertEqual((len(r.key_decisions), len(r.action_items)), (0, 1))


if __name__ == "__main__":
    unittest.main()
