"""Chase-style CSV: MM/DD/YYYY, single signed Amount column, quoted
descriptions. No preamble, no summary rows -- the "clean" baseline format
that the others deviate from."""

from __future__ import annotations

from datagen.events import GroundTruthEvent
from datagen.formats.common import csv_quote, mmddyyyy

EXTENSION = "csv"
FORMAT_NAME = "chase_csv"


def render(events: list[GroundTruthEvent]) -> str:
    lines = ["Details,Posting Date,Description,Amount,Type"]
    for e in events:
        detail = "DEBIT" if e.amount < 0 else "CREDIT"
        lines.append(
            f"{detail},{mmddyyyy(e.txn_date)},{csv_quote(e.raw_descriptor)},{e.amount:.2f},{detail}"
        )
    return "\n".join(lines) + "\n"
