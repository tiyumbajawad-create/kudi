"""Deterministic transaction IDs so re-ingesting a file -- or an
overlapping export of the same account -- is a no-op (design doc §4.4).

Two legitimately identical transactions on the same day (two identical
coffees) would collide under a pure content hash, so callers must pass an
`occurrence` index computed per-file: the Nth time this exact
(account, date, amount, description) combination has been seen while
processing *this* file, starting at 0.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal


def compute_txn_id(
    account_id: str,
    posted_date: date,
    amount: Decimal,
    normalized_raw_description: str,
    fitid: str | None = None,
    occurrence: int = 0,
) -> str:
    """FITID wins when the source format provides one (OFX); otherwise a
    content hash, salted with the per-file occurrence counter so same-day
    duplicates don't collide."""
    if fitid:
        return hashlib.sha256(f"fitid|{fitid}".encode()).hexdigest()[:16]

    key = (
        f"{account_id}|{posted_date.isoformat()}|{amount}|{normalized_raw_description}|{occurrence}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]
