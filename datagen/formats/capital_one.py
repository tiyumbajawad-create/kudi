"""Capital One-style CSV: YYYY-MM-DD dates and a bank-supplied 'Category'
column that's deliberately wrong some of the time (design doc §4.1) --
exercises the ingestion pipeline's job of treating bank-provided
categories as a hint, not ground truth."""

from __future__ import annotations

import random

from datagen.events import GroundTruthEvent
from datagen.formats.common import csv_quote, yyyymmdd_dash

EXTENSION = "csv"
FORMAT_NAME = "capital_one_csv"

_WRONG_CATEGORY_PROB = 0.25
_DECOY_CATEGORIES = [
    "Merchandise",
    "Dining",
    "Travel-Airfare",
    "Groceries",
    "Payment/Credit",
    "Fees",
    "Gas/Automotive",
    "Entertainment",
]


def render(events: list[GroundTruthEvent], rng: random.Random) -> str:
    lines = ["Transaction Date,Posted Date,Description,Category,Debit,Credit"]
    for e in events:
        leaf = e.category.split(">")[-1]
        if rng.random() < _WRONG_CATEGORY_PROB:
            category_hint = rng.choice(_DECOY_CATEGORIES)
        else:
            category_hint = leaf
        debit = f"{-e.amount:.2f}" if e.amount < 0 else ""
        credit = f"{e.amount:.2f}" if e.amount > 0 else ""
        lines.append(
            f"{yyyymmdd_dash(e.txn_date)},{yyyymmdd_dash(e.txn_date)},"
            f"{csv_quote(e.raw_descriptor)},{category_hint},{debit},{credit}"
        )
    return "\n".join(lines) + "\n"
