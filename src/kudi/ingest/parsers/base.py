"""Shared types every format parser produces. A parser's job stops here --
turning a RawRecord into a canonical `Transaction` (hashing, txn_id,
source_format, ingested_at) is the pipeline's job, not the parser's
(design doc §4.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class RawRecord:
    """A source-faithful record: one bank transaction, minimally parsed."""

    posted_date: date
    amount: Decimal  # signed; negative = outflow
    raw_description: str
    account_id: str | None = None  # only OFX self-identifies its account
    fitid: str | None = None  # stable id from the source file, if present
    category_hint: str | None = None  # bank-supplied category; unverified


@dataclass
class ParseResult:
    """A parser's full output: the good rows, plus an honest account of
    what it skipped (preamble/summary rows -- expected noise) and what it
    flatly failed to parse (row-level, doesn't abort the whole file)."""

    records: list[RawRecord] = field(default_factory=list)
    skipped: int = 0
    failures: list[str] = field(default_factory=list)
