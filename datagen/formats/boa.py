"""Bank of America-style CSV: several preamble lines before the real
header, and separate Debit/Credit columns instead of one signed Amount
(design doc §4.1) -- exercises preamble-skipping in the parser."""

from __future__ import annotations

from datagen.events import GroundTruthEvent
from datagen.formats.common import csv_quote, mmddyyyy

EXTENSION = "csv"
FORMAT_NAME = "boa_csv"


def render(events: list[GroundTruthEvent], account_id: str) -> str:
    lines = [
        "Bank of America",
        f"Account,{account_id}",
        "Description,,,",
        "Beginning balance as of statement period,,,",
        ",,,",
        "Date,Description,Debit,Credit",
    ]
    for e in events:
        debit = f"{-e.amount:.2f}" if e.amount < 0 else ""
        credit = f"{e.amount:.2f}" if e.amount > 0 else ""
        lines.append(f"{mmddyyyy(e.txn_date)},{csv_quote(e.raw_descriptor)},{debit},{credit}")
    return "\n".join(lines) + "\n"
