"""Generic credit-card CSV: DD/MM/YYYY (the classic date-ambiguity
trap), thousands separators in amounts, and a trailing summary row that
isn't a transaction at all (design doc §4.1) -- exercises summary-row
skipping."""

from __future__ import annotations

from datagen.events import GroundTruthEvent
from datagen.formats.common import csv_quote, ddmmyyyy, thousands

EXTENSION = "csv"
FORMAT_NAME = "generic_credit_csv"


def render(events: list[GroundTruthEvent]) -> str:
    lines = ["Date,Description,Amount,Type"]
    total = 0.0
    for e in events:
        kind = "DEBIT" if e.amount < 0 else "CREDIT"
        amt = csv_quote(thousands(abs(e.amount)))
        lines.append(f"{ddmmyyyy(e.txn_date)},{csv_quote(e.raw_descriptor)},{amt},{kind}")
        total += e.amount
    total_amt = csv_quote(thousands(abs(total)))
    lines.append(f"TOTAL,,{total_amt},{'DEBIT' if total < 0 else 'CREDIT'}")
    return "\n".join(lines) + "\n"
