"""Orchestrates the full two-layer anomaly detector (design doc §6.2):
Layer 1 (interpretable signals) + Layer 2 (Isolation Forest over
engineered features), combined as `max(rule_score, forest_score)`.
Handles cold start (§6.3) and maps raw scores to percentile-of-history
so a 0.98 means "more unusual than 98% of this account's own
transactions" -- a claim a user can actually trust, since it's
calibrated against their own life, not some global population.

The Isolation Forest is fit fresh per account, at analysis time -- this
is *personal* anomaly detection (§6.1): there's no single global model
to ship, because "unusual" is defined relative to one account's own
history.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from kudi.detect.signals import (
    amount_outlier,
    day_of_week_deviation,
    duplicate_charge,
    extreme_absolute_amount,
    geo_mismatch,
    merge_signals,
    new_merchant,
    velocity,
)

COLD_START_MIN_TRANSACTIONS = 30
COLD_START_MIN_DAYS = 30
ISOLATION_FOREST_CONTAMINATION = "auto"
UNUSUAL_COMBINATION_REASON = "unusual combination of factors"


@dataclass
class AnomalyResult:
    scores: pd.Series  # percentile-of-history, indexed like the input df
    reasons: dict[Hashable, list[str]] = field(default_factory=dict)
    cold_start: bool = False


def _is_cold_start(df: pd.DataFrame) -> bool:
    if len(df) < COLD_START_MIN_TRANSACTIONS:
        return True
    span_days = (df["posted_date"].max() - df["posted_date"].min()).days
    return bool(span_days < COLD_START_MIN_DAYS)


def _isolation_forest_features(df: pd.DataFrame) -> np.ndarray:
    log_amount = np.log1p(df["amount"].abs())

    category_freq = df["category"].value_counts(normalize=True)
    category_freq_rank = df["category"].map(category_freq).fillna(0.0)

    first_seen: dict[str, int] = {}
    merchant_novelty = []
    for i, merchant in enumerate(df["merchant_norm"]):
        merchant_novelty.append(1.0 if merchant not in first_seen else 0.0)
        first_seen.setdefault(merchant, i)

    day_of_week = df["posted_date"].apply(lambda d: d.weekday())

    sorted_dates = df["posted_date"].sort_values()
    gap_since_last = sorted_dates.diff().dt.days.reindex(df.index).fillna(0.0)

    return np.column_stack(
        [
            log_amount.to_numpy(),
            category_freq_rank.to_numpy(),
            np.array(merchant_novelty),
            day_of_week.to_numpy(),
            gap_since_last.to_numpy(),
        ]
    )


def _isolation_forest_scores(df: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """Returns scores in [0, 1], higher = more anomalous (sklearn's raw
    convention is the opposite, so we flip and min-max normalize)."""
    X = _isolation_forest_features(df)
    model = IsolationForest(contamination=ISOLATION_FOREST_CONTAMINATION, random_state=random_state)
    model.fit(X)
    raw = -model.score_samples(X)  # higher = more anomalous after negation
    lo, hi = raw.min(), raw.max()
    if hi - lo < 1e-9:
        return np.zeros_like(raw)
    normalized: np.ndarray = (raw - lo) / (hi - lo)
    return normalized


def detect_anomalies(
    df: pd.DataFrame,
    random_state: int = 42,
    known_recurring_merchants: set[str] | None = None,
) -> AnomalyResult:
    """`df` is one account's full transaction history, with columns
    posted_date, amount, category, merchant_norm, raw_description.

    `known_recurring_merchants`, when provided (typically the output of
    `kudi.detect.recurring.infer_recurring_groups` run first), prevents
    a recurring bill's own onboarding month from being flagged as a
    "new merchant" anomaly -- see signals.new_merchant for why that
    matters.
    """
    if df.empty:
        return AnomalyResult(scores=pd.Series(dtype=float), cold_start=False)

    df = df.copy()
    df["posted_date"] = pd.to_datetime(df["posted_date"])

    cold_start = _is_cold_start(df)

    if cold_start:
        merged = merge_signals(
            duplicate_charge(df),
            extreme_absolute_amount(df),
        )
        forest_scores = np.zeros(len(df))
    else:
        merged = merge_signals(
            amount_outlier(df),
            new_merchant(df, known_recurring_merchants),
            velocity(df),
            duplicate_charge(df),
            geo_mismatch(df),
            day_of_week_deviation(df),
        )
        forest_scores = _isolation_forest_scores(df, random_state=random_state)

    rule_scores = np.array([merged.get(idx, (0.0, []))[0] for idx in df.index])
    final_raw = np.maximum(rule_scores, forest_scores)

    reasons: dict[Hashable, list[str]] = {}
    for pos, idx in enumerate(df.index):
        row_reasons = list(merged.get(idx, (0.0, []))[1])
        if forest_scores[pos] > rule_scores[pos] and not row_reasons:
            row_reasons = [UNUSUAL_COMBINATION_REASON]
        if row_reasons:
            reasons[idx] = row_reasons

    # percentile-of-history: rank among this account's own scores
    percentiles = pd.Series(final_raw, index=df.index).rank(pct=True)

    return AnomalyResult(scores=percentiles, reasons=reasons, cold_start=cold_start)
