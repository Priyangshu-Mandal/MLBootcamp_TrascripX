"""Code-side verification of evidence quotes (the model is never trusted).

A quote is verified if, after normalizing case, punctuation and whitespace, it appears in the
transcript as an exact word sequence, or (for quotes of 5+ words) as a near-identical window
whose numbers, negations and hedges are exactly the same. Fabricated or altered quotes fail.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .schemas import MeetingRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_NEGATIONS = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "cannot", "without"}
_HEDGES = {"might", "maybe", "perhaps", "probably", "possibly", "likely", "unlikely",
           "could", "guess", "suppose", "think"}
MAX_QUOTES = 3
MIN_FUZZY_WORDS = 5
MAX_ANCHOR_OCCURRENCES = 8
MAX_CANDIDATES = 300


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; quotes, dashes and punctuation are ignored."""
    t = (text or "").lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", " "), ("—", " ")):
        t = t.replace(a, b)
    return _TOKEN_RE.findall(t)


def _invariants(tokens: list[str]) -> tuple[Counter, Counter, Counter]:
    return (Counter(t for t in tokens if any(c.isdigit() for c in t)),
            Counter(t for t in tokens if t in _NEGATIONS or t.endswith("n't")),
            Counter(t for t in tokens if t in _HEDGES))


class TranscriptIndex:
    """Precomputed token index of a transcript for fast quote verification."""

    def __init__(self, text: str, threshold: float = 0.92) -> None:
        self.tokens = tokenize(text)
        self.joined = " " + " ".join(self.tokens) + " "
        self.threshold = threshold
        self.positions: dict[str, list[int]] = defaultdict(list)
        for i, tok in enumerate(self.tokens):
            self.positions[tok].append(i)

    def verify(self, quote: str) -> bool:
        q = tokenize(quote)
        if not q:
            return False
        qjoined = " ".join(q)
        if f" {qjoined} " in self.joined:
            return True
        if len(q) < MIN_FUZZY_WORDS:
            return False
        starts: set[int] = set()
        for j, tok in enumerate(q):
            occ = self.positions.get(tok, [])
            if 0 < len(occ) <= MAX_ANCHOR_OCCURRENCES:
                for p in occ:
                    starts.add(p - j)
            if len(starts) >= MAX_CANDIDATES:
                break
        want = _invariants(q)
        for start in starts:
            for length in (len(q), len(q) - 1, len(q) + 1):
                window = self.tokens[max(start, 0):max(start, 0) + length]
                if not window:
                    continue
                wjoined = " ".join(window)
                sm = SequenceMatcher(None, qjoined, wjoined, autojunk=False)
                if sm.real_quick_ratio() < self.threshold or sm.quick_ratio() < self.threshold:
                    continue
                if sm.ratio() >= self.threshold and _invariants(window) == want:
                    return True
        return False


def clean_quotes(quotes: list[str], limit: int | None = MAX_QUOTES) -> list[str]:
    """Strip, drop empties and duplicates (order preserved); keep at most `limit` if set."""
    seen, out = set(), []
    for q in quotes or []:
        q = (q or "").strip().strip('"“”').strip()
        key = " ".join(tokenize(q))
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out if limit is None else out[:limit]


class EvidenceReport:
    """What the evidence pass found; used to warn the user."""

    def __init__(self) -> None:
        self.unverified_quotes: list[str] = []
        self.items_without_evidence: list[str] = []

    @property
    def problem_count(self) -> int:
        return len(self.unverified_quotes) + len(self.items_without_evidence)


def verify_record(record: MeetingRecord, index: TranscriptIndex) -> EvidenceReport:
    """Verify every item's quotes against `index` and set evidence_verified. Mutates `record`;
    drops nothing."""
    report = EvidenceReport()
    for item, label in ([(d, d.decision) for d in record.key_decisions] +
                        [(a, a.task) for a in record.action_items]):
        item.evidence = clean_quotes(item.evidence, limit=None)
        flags = [index.verify(q) for q in item.evidence]
        item.evidence_verified = bool(flags) and all(flags)
        if not item.evidence:
            report.items_without_evidence.append(label)
        report.unverified_quotes += [q for q, ok in zip(item.evidence, flags) if not ok]
    return report


def drop_unsupported_items(record: MeetingRecord) -> list[str]:
    """Remove decisions/tasks that carry no evidence at all; returns descriptions of drops."""
    dropped = [d.decision for d in record.key_decisions if not d.evidence] + \
              [a.task for a in record.action_items if not a.evidence]
    record.key_decisions = [d for d in record.key_decisions if d.evidence]
    record.action_items = [a for a in record.action_items if a.evidence]
    return dropped
