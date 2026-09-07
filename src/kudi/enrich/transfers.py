"""Transfer detection (design doc §5.4): matched-pair heuristic.
Opposite-sign, equal-magnitude amounts across two different accounts
within a short window, with transfer-ish keywords in the descriptor,
get categorized Transfers and excluded from spend analytics and
anomaly baselines -- otherwise a single credit-card payment would show
up as both a huge "expense" on checking and a huge "income" on the
card, corrupting everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

TRANSFER_CATEGORY = "Transfers>Internal Transfer"
_TRANSFER_KEYWORDS = ("TRANSFER", "ZELLE", "PAYMENT THANK YOU", "PMT TO", "PAYMENT")
_MAX_DAY_GAP = 3


@dataclass
class TransferCandidate:
    txn_id: str
    account_id: str
    posted_date: date
    amount: Decimal
    raw_description: str


def _looks_transfer_ish(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in _TRANSFER_KEYWORDS)


def detect_transfer_pairs(
    candidates: list[TransferCandidate],
) -> list[tuple[str, str]]:
    """Returns (txn_id_a, txn_id_b) pairs identified as the two legs of
    one transfer. Greedy matching: each transaction is used in at most
    one pair."""
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []

    eligible = [c for c in candidates if _looks_transfer_ish(c.raw_description)]

    for i, a in enumerate(eligible):
        if a.txn_id in used:
            continue
        for b in eligible[i + 1 :]:
            if b.txn_id in used:
                continue
            if a.account_id == b.account_id:
                continue
            if a.amount + b.amount != 0:
                continue
            if abs((a.posted_date - b.posted_date).days) > _MAX_DAY_GAP:
                continue
            pairs.append((a.txn_id, b.txn_id))
            used.add(a.txn_id)
            used.add(b.txn_id)
            break

    return pairs
