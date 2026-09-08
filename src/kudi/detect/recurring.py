"""Recurring-charge inference (design doc §7.1): deliberately
statistical rather than ML -- periodicity is a signal-processing
problem, and this method is explainable and testable in a way a
black-box model wouldn't be.

Per (account_id, merchant_norm) group with >= 3 transactions:
  1. inter-arrival gaps
  2. match the gap distribution against period templates (weekly,
     biweekly, monthly, quarterly, annual), scored by the fraction of
     gaps that fall within tolerance
  3. amount consistency via coefficient of variation
  4. combined confidence = period score * amount-consistency score,
     weighted by observation count
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

PERIOD_TEMPLATES: list[tuple[str, int, int]] = [
    ("weekly", 7, 2),
    ("biweekly", 14, 3),
    ("monthly", 30, 3),
    ("quarterly", 91, 7),
    ("annual", 365, 14),
]

MIN_OBSERVATIONS = 3
FIXED_PRICE_CV_THRESHOLD = 0.15
VARIABLE_BILL_CV_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
STRONG_PERIOD_SCORE = 0.85  # can overcome a merely-fair amount fit


@dataclass
class RecurringGroupResult:
    account_id: str
    merchant_norm: str
    period_name: str
    period_days: int
    confidence: float
    n_observations: int
    predicted_next_date: date
    predicted_amount_low: float
    predicted_amount_high: float
    transaction_indices: list[int]


def _best_period_fit(gaps: np.ndarray) -> tuple[str, int, float]:
    best_name, best_period, best_score = "none", 0, 0.0
    for name, period, tolerance in PERIOD_TEMPLATES:
        within = np.abs(gaps - period) <= tolerance
        score = float(within.mean())
        if score > best_score:
            best_name, best_period, best_score = name, period, score
    return best_name, best_period, best_score


def _amount_consistency(amounts: np.ndarray) -> tuple[float, float]:
    """Returns (score, coefficient_of_variation)."""
    mean_abs = np.abs(amounts).mean()
    if mean_abs == 0:
        return 0.0, float("inf")
    cv = np.abs(amounts).std() / mean_abs
    if cv <= FIXED_PRICE_CV_THRESHOLD:
        return 1.0, cv
    if cv <= VARIABLE_BILL_CV_THRESHOLD:
        return 0.6, cv
    return 0.2, cv


def infer_recurring_groups(df: pd.DataFrame) -> list[RecurringGroupResult]:
    """`df` is one account's transaction history with columns
    account_id, posted_date, amount, merchant_norm. Returns one result
    per (account_id, merchant_norm) group that clears the confidence
    threshold."""
    results: list[RecurringGroupResult] = []

    for (account_id, merchant), group in df.groupby(["account_id", "merchant_norm"]):
        if len(group) < MIN_OBSERVATIONS:
            continue

        sorted_group = group.sort_values("posted_date")
        dates = pd.to_datetime(sorted_group["posted_date"]).tolist()
        amounts = sorted_group["amount"].to_numpy()
        gaps = np.array([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))])

        period_name, period_days, period_score = _best_period_fit(gaps)
        amount_score, cv = _amount_consistency(amounts)

        if period_score < STRONG_PERIOD_SCORE and amount_score < 0.6:
            confidence = period_score * amount_score
        else:
            confidence = max(period_score * amount_score, min(period_score, amount_score) * 0.9)

        n = len(sorted_group)
        weight = min(1.0, n / 6.0)  # ramps up to full weight by 6 observations
        confidence *= weight

        if confidence < CONFIDENCE_THRESHOLD or period_name == "none":
            continue

        last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
        recent_amounts = amounts[-3:]
        median_recent = float(np.median(recent_amounts))
        spread = (
            float(np.std(recent_amounts)) if len(recent_amounts) > 1 else abs(median_recent) * cv
        )

        results.append(
            RecurringGroupResult(
                account_id=str(account_id),
                merchant_norm=str(merchant),
                period_name=period_name,
                period_days=period_days,
                confidence=round(min(confidence, 1.0), 3),
                n_observations=n,
                predicted_next_date=last_date + timedelta(days=period_days),
                predicted_amount_low=round(median_recent - spread, 2),
                predicted_amount_high=round(median_recent + spread, 2),
                transaction_indices=list(sorted_group.index),
            )
        )

    return results
