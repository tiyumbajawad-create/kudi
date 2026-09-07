"""Lightweight non-text features that ride alongside the TF-IDF text
features (design doc §5.2 step 3): amount bucket, sign, day-of-week,
is-weekend. Cheap signals that a merchant string alone doesn't carry --
a $4 charge at 7am on a Tuesday looks a lot like coffee regardless of
what the descriptor says.
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import numpy as np

N_AMOUNT_BUCKETS = 10


def amount_bucket(amount: Decimal | float) -> int:
    """Log-scale bucket of |amount|, clipped to N_AMOUNT_BUCKETS."""
    magnitude = abs(float(amount))
    if magnitude < 1:
        return 0
    bucket = int(math.log2(magnitude)) + 1
    return max(0, min(bucket, N_AMOUNT_BUCKETS - 1))


def numeric_features(amount: Decimal | float, posted_date: date) -> list[float]:
    return [
        float(amount_bucket(amount)),
        1.0 if float(amount) < 0 else 0.0,
        float(posted_date.weekday()),
        1.0 if posted_date.weekday() >= 5 else 0.0,
    ]


NUMERIC_FEATURE_NAMES = ["amount_bucket", "is_outflow", "day_of_week", "is_weekend"]


def numeric_feature_matrix(amounts: list[Decimal | float], dates: list[date]) -> np.ndarray:
    return np.array(
        [numeric_features(a, d) for a, d in zip(amounts, dates, strict=True)],
        dtype=float,
    )
