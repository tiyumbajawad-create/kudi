"""Format detection. Sniffs, in order: an OFX tag probe, then CSV header
fingerprinting -- exact match first, fuzzy match as a fallback with a
confidence score. Below the confidence threshold the pipeline refuses to
guess and reports the headers it saw (design doc §4.2): failing loudly
beats silently mis-parsing money data.

Detection is data-driven (this FORMAT_SPECS registry), so adding format
#6 is one spec here plus one parser module -- no changes to the
detection algorithm itself.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from rapidfuzz import fuzz

CONFIDENCE_THRESHOLD = 0.75
_MAX_HEADER_SCAN_LINES = 15


@dataclass(frozen=True)
class FormatSpec:
    name: str
    header_signature: list[str]


FORMAT_SPECS: list[FormatSpec] = [
    FormatSpec("chase_csv", ["Details", "Posting Date", "Description", "Amount", "Type"]),
    FormatSpec("boa_csv", ["Date", "Description", "Debit", "Credit"]),
    FormatSpec(
        "capital_one_csv",
        ["Transaction Date", "Posted Date", "Description", "Category", "Debit", "Credit"],
    ),
    FormatSpec("generic_credit_csv", ["Date", "Description", "Amount", "Type"]),
]


@dataclass
class DetectionResult:
    format_name: str | None
    confidence: float
    headers_seen: list[str] | None = None


def _looks_like_ofx(text: str) -> bool:
    head = text.lstrip()[:200]
    return "<OFX>" in text or head.startswith("OFXHEADER")


def _header_score(line: list[str], spec: FormatSpec) -> float:
    line_fields = [c.strip() for c in line]
    if line_fields == spec.header_signature:
        return 1.0
    if not line_fields:
        return 0.0

    # per-field best fuzzy match, not a naive comma-joined string compare --
    # joining with commas defeats rapidfuzz's whitespace tokenization and
    # produces false positives between headers of different column counts.
    field_scores = [
        max((fuzz.ratio(spec_field.lower(), lf.lower()) for lf in line_fields), default=0.0)
        for spec_field in spec.header_signature
    ]
    avg_field_score = sum(field_scores) / len(field_scores) / 100.0

    count_penalty = min(len(line_fields), len(spec.header_signature)) / max(
        len(line_fields), len(spec.header_signature)
    )
    return avg_field_score * count_penalty


def detect_format(text: str) -> DetectionResult:
    if _looks_like_ofx(text):
        return DetectionResult(format_name="ofx", confidence=1.0)

    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return DetectionResult(format_name=None, confidence=0.0, headers_seen=None)

    best: tuple[float, str | None, list[str] | None] = (0.0, None, None)

    for line in rows[:_MAX_HEADER_SCAN_LINES]:
        if not line or all(not c.strip() for c in line):
            continue
        for spec in FORMAT_SPECS:
            score = _header_score(line, spec)
            if score > best[0]:
                best = (score, spec.name, line)

    score, name, headers_seen = best
    if name is None or score < CONFIDENCE_THRESHOLD:
        return DetectionResult(format_name=None, confidence=score, headers_seen=headers_seen)
    return DetectionResult(format_name=name, confidence=score)
