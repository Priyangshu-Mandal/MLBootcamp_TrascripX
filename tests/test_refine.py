"""Stage 2 tests: Groq is faked (no network, no groq package needed)."""
import json
import re
import unittest
from types import SimpleNamespace

from pipeline import refine
from pipeline.config import Config
from pipeline.errors import (
    InvalidAPIKeyError, MissingAPIKeyError, NetworkError, RateLimitedError, RefinementError,
)

CFG = Config(groq_api_key="test-key", refine_chunk_chars=300, refine_max_retries=3)


class FakeAPIError(Exception):
    def __init__(self, msg="err", status_code=None, headers=None):
        super().__init__(msg)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class APIConnectionError(Exception):  # same class name the groq SDK uses
    pass


def reply(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=finish)])


class FakeClient:
    """Scripted client: each item is a function(user_text)->text, an Exception, or a reply."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            user = kw["messages"][1]["content"]
            inner = user.split("<transcript>\n", 1)[1].rsplit("\n</transcript>", 1)[0]
            return reply(item(inner))
        if isinstance(item, str):
            return reply(item)
        return item


def corr(context, original, replacement, reason="fix recognition error", support="context"):
    return {"context": context, "original": original, "replacement": replacement,
            "reason": reason, "support": support}


def corrections(*cs):
    return json.dumps({"corrections": list(cs)})


def fix_jargon(inner):
    """Fake model: proposes a correction for every jargon error in the text it is given."""
    cs = []
    for bad, good in (("cube are net ease", "Kubernetes"), ("A P I", "API")):
        for m in re.finditer(r"[^.\n]*" + re.escape(bad) + r"[^.\n]*", inner):
            cs.append(corr(m.group(0).strip(), bad, good))
    return corrections(*cs)


class VetCorrectionsTests(unittest.TestCase):
    TEXT = "We will deploy on cube are net ease. Maybe we should not change the A P I. Priya owns the A P I."

    def vet(self, *cs):
        return refine.vet_corrections(list(cs), self.TEXT)

    def test_good_correction_kept(self):
        kept, rejected = self.vet(corr("deploy on cube are net ease.", "cube are net ease", "Kubernetes"))
        self.assertEqual((len(kept), len(rejected)), (1, 0))

    def test_malformed_and_noop_rejected(self):
        kept, rejected = self.vet({"context": "x"}, "not a dict",
                                  corr("the A P I", "A P I", "A P I"), corr("the A P I", " ", "API"))
        self.assertEqual(kept, [])
        self.assertEqual([r["why"] for r in rejected],
                         ["malformed correction", "malformed correction",
                          "empty or no-op correction", "empty or no-op correction"])

    def test_negation_hedge_or_commitment_change_rejected(self):
        for original, replacement in (("should not change", "should change"), ("Maybe we", "We"),
                                      ("will deploy", "deploy"), ("should not", "must not")):
            kept, rejected = self.vet(corr("x", original, replacement))
            self.assertEqual(kept, [], (original, replacement))
            self.assertIn("negation, hedge or commitment", rejected[0]["why"])

    def test_support_quote_must_exist_in_transcript(self):
        ok, _ = self.vet(corr("c", "A P I", "API", support="Priya owns the A P I."))
        bad, rejected = self.vet(corr("c", "A P I", "API", support="the team uses the API daily"))
        self.assertEqual(len(ok), 1)
        self.assertEqual(bad, [])
        self.assertIn("support quote", rejected[0]["why"])


class ParseCorrectionsTests(unittest.TestCase):
    def test_plain_fenced_and_surrounded_json(self):
        body = corrections(corr("a b c", "b", "B"))
        for reply_text in (body, "```json\n" + body + "\n```", "Here you go:\n" + body + "\nDone."):
            self.assertEqual(len(refine.parse_corrections(reply_text)), 1)

    def test_empty_list_is_valid(self):
        self.assertEqual(refine.parse_corrections('{"corrections": []}'), [])

    def test_unusable_replies_raise_valueerror(self):
        for bad in ("", "no json here", "{broken", '{"fixes": []}', '{"corrections": "none"}', "[1, 2]"):
            with self.assertRaises(ValueError):
                refine.parse_corrections(bad)


class RefineFlowTests(unittest.TestCase):
    RAW = ("We will deploy on cube are net ease next week. Maybe the A P I is slow.\n\n"
           "Priya said latency is 40 milliseconds. We should not drop the cache.")

    def run_refine(self, client, raw=None, cfg=CFG, sleeps=None):
        sleeps = [] if sleeps is None else sleeps
        return refine.refine_transcript(raw or self.RAW, cfg, client=client, sleep=sleeps.append)

    def test_corrections_applied_and_everything_else_untouched(self):
        c = FakeClient([fix_jargon])
        res = self.run_refine(c)
        self.assertEqual(res.text, self.RAW.replace("cube are net ease", "Kubernetes").replace("A P I", "API"))
        self.assertEqual(len(res.applied), 2)
        self.assertEqual(res.rejected, [])
        self.assertEqual(res.fallback_chunks, [])
        self.assertEqual(c.calls[0]["temperature"], 0.0)
        self.assertEqual(c.calls[0]["model"], "openai/gpt-oss-120b")
        self.assertIn("corrections", c.calls[0]["messages"][0]["content"])
        self.assertTrue(c.calls[0]["messages"][1]["content"].startswith("<transcript>"))

    def test_empty_correction_list_leaves_transcript_identical(self):
        res = self.run_refine(FakeClient([corrections()]))
        self.assertEqual(res.text, self.RAW)
        self.assertEqual((res.applied, res.rejected), ([], []))

    def test_only_the_quoted_occurrence_is_changed(self):
        raw = "Use the A P I here. Later we mention A P I again in a different sense."
        one = corrections(corr("Use the A P I here", "A P I", "API"))
        res = self.run_refine(FakeClient([one]), raw=raw)
        self.assertEqual(res.text, "Use the API here. Later we mention A P I again in a different sense.")

    def test_context_not_in_transcript_is_rejected(self):
        res = self.run_refine(FakeClient([corrections(corr("this sentence is invented", "invented", "real"))]))
        self.assertEqual(res.text, self.RAW)
        self.assertIn("not found verbatim", res.rejected[0]["why"])

    def test_number_change_without_support_is_refused(self):
        res = self.run_refine(FakeClient([corrections(corr("latency is 40 milliseconds", "40", "50"))]))
        self.assertIn("40 milliseconds", res.text)
        self.assertIn("protected value", res.rejected[0]["why"])

    def test_day_corrected_only_when_recap_mentions_it(self):
        raw = "Run the test by Wensday. Later recap: the test is due Wednesday."
        good = corr("Run the test by Wensday", "Wensday", "Wednesday", support="the test is due Wednesday")
        self.assertIn("by Wednesday", self.run_refine(FakeClient([corrections(good)]), raw=raw).text)
        lone = "Run the test by Wensday. Nothing else."
        res = self.run_refine(FakeClient([corrections(good)]), raw=lone)
        self.assertIn("Wensday", res.text)
        self.assertEqual(len(res.rejected), 1)

    def test_negation_change_is_refused(self):
        res = self.run_refine(FakeClient([corrections(corr("We should not drop the cache", "should not", "should"))]))
        self.assertIn("should not drop", res.text)
        self.assertIn("negation", res.rejected[0]["why"])

    def test_invalid_json_gets_one_retry(self):
        c = FakeClient([reply("sorry, here is the fixed text"), fix_jargon])
        res = self.run_refine(c)
        self.assertEqual(len(c.calls), 2)
        self.assertIn("valid JSON", c.calls[1]["messages"][-1]["content"])
        self.assertIn("Kubernetes", res.text)

    def test_unusable_replies_fall_back_to_raw_with_warning(self):
        res = self.run_refine(FakeClient([reply("no json at all")]))
        self.assertEqual(res.text, self.RAW)
        self.assertEqual(res.fallback_chunks, [1])
        self.assertIn("kept as the raw transcript", res.warnings[0])

    def test_code_fenced_reply_accepted(self):
        res = self.run_refine(FakeClient([lambda t: "```json\n" + fix_jargon(t) + "\n```"]))
        self.assertIn("Kubernetes", res.text)

    def test_multi_chunk_order_and_no_duplication(self):
        raw = "\n\n".join(f"Paragraph {i}: in the review the team said we use cube are net ease here for staging."
                           for i in range(12))
        res = self.run_refine(FakeClient([fix_jargon]), raw=raw)
        self.assertGreater(res.chunks_total, 1)
        for i in range(12):
            self.assertEqual(res.text.count(f"Paragraph {i}:"), 1)
        self.assertEqual(res.text.count("Kubernetes"), 12)
        self.assertEqual(len(res.applied), 12)
        self.assertLess(res.text.index("Paragraph 2:"), res.text.index("Paragraph 9:"))

    def test_retry_on_429_then_success_uses_retry_after(self):
        c = FakeClient([FakeAPIError("rate", 429, {"retry-after": "7"}), FakeAPIError("rate", 429), fix_jargon])
        sleeps = []
        res = self.run_refine(c, sleeps=sleeps)
        self.assertIn("Kubernetes", res.text)
        self.assertEqual(len(sleeps), 2)
        self.assertGreaterEqual(sleeps[0], 7.0)
        self.assertGreaterEqual(sleeps[1], 4.0)  # exponential: base 2 * 2**1

    def test_429_exhausted(self):
        sleeps = []
        with self.assertRaises(RateLimitedError):
            self.run_refine(FakeClient([FakeAPIError("rate", 429)]), sleeps=sleeps)
        self.assertEqual(len(sleeps), CFG.refine_max_retries)

    def test_daily_quota_not_retried(self):
        sleeps = []
        err = FakeAPIError("Rate limit reached for model: tokens per day (TPD)", 429)
        with self.assertRaises(RateLimitedError):
            self.run_refine(FakeClient([err]), sleeps=sleeps)
        self.assertEqual(sleeps, [])

    def test_invalid_key_not_retried(self):
        sleeps = []
        with self.assertRaises(InvalidAPIKeyError):
            self.run_refine(FakeClient([FakeAPIError("bad key", 401)]), sleeps=sleeps)
        self.assertEqual(sleeps, [])

    def test_server_error_retried_then_network_error(self):
        with self.assertRaises(NetworkError):
            self.run_refine(FakeClient([FakeAPIError("boom", 503)]))

    def test_connection_error_retried_then_recovers(self):
        sleeps = []
        c = FakeClient([APIConnectionError("down"), fix_jargon])
        res = self.run_refine(c, sleeps=sleeps)
        self.assertEqual(len(sleeps), 1)
        self.assertIn("Kubernetes", res.text)

    def test_bad_model_gives_clear_error(self):
        with self.assertRaises(RefinementError) as cm:
            self.run_refine(FakeClient([FakeAPIError("model not found", 404)]))
        self.assertIn("GROQ_MODEL", cm.exception.user_message)

    def test_unavailable_model_uses_configured_fallback_and_says_so(self):
        cfg = Config(groq_api_key="k", refine_chunk_chars=300, groq_fallback_models=("openai/gpt-oss-20b",))
        c = FakeClient([FakeAPIError("model_not_found", 404), fix_jargon])
        res = self.run_refine(c, cfg=cfg)
        self.assertEqual([x["model"] for x in c.calls], ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
        self.assertEqual(res.model, "openai/gpt-oss-20b")
        self.assertTrue(any("was not available" in w for w in res.warnings))

    def test_no_fallback_configured_gives_actionable_error(self):
        with self.assertRaises(RefinementError) as cm:
            self.run_refine(FakeClient([FakeAPIError("The model does not exist", 404)]))
        self.assertIn("list_groq_models.py", cm.exception.user_message)
        self.assertIn("GROQ_MODEL", cm.exception.user_message)

    def test_missing_key(self):
        with self.assertRaises(MissingAPIKeyError) as cm:
            refine.refine_transcript(self.RAW, Config(groq_api_key=""))
        self.assertIn("Refining failed", cm.exception.user_message)

    def test_empty_transcript(self):
        with self.assertRaises(RefinementError):
            self.run_refine(FakeClient([fix_jargon]), raw="   ")

    def test_malformed_response(self):
        with self.assertRaises(RefinementError):
            self.run_refine(FakeClient([SimpleNamespace(choices=[])]))

    def test_progress_callback(self):
        raw = "\n\n".join(f"Paragraph {i}: the team said cube are net ease is fine for us." for i in range(12))
        seen = []
        refine.refine_transcript(raw, CFG, client=FakeClient([fix_jargon]), progress=seen.append,
                                 sleep=lambda s: None)
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_prompt_file_has_required_constraints(self):
        p = refine.load_system_prompt()
        for needle in ("conservative ASR-error corrector", "THE RECAP RULE", "NEVER DO", '"corrections"',
                       "An empty list is a valid answer", "Never change negations"):
            self.assertIn(needle, p)

    def test_api_key_never_in_messages(self):
        c = FakeClient([fix_jargon])
        self.run_refine(c)
        self.assertNotIn("test-key", repr(c.calls))


if __name__ == "__main__":
    unittest.main()
