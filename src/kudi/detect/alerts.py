"""Derived alerts from recurring groups (design doc §7.2) -- the
highest-value output of recurring detection, arguably more useful to a
user day-to-day than the raw group detection itself: a price hike they
didn't notice, a bill that silently stopped getting charged (often a
sign of a failed payment method), or being billed twice for one cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from kudi.detect.recurring import RecurringGroupResult

PRICE_HIKE_PCT_THRESHOLD = 0.10
PRICE_HIKE_ABS_THRESHOLD = 2.0
MISSED_CHARGE_GRACE_DAYS = 5
DUPLICATE_BILLING_MAX_GAP_DAYS = 3


@dataclass
class PriceHikeAlert:
    merchant_norm: str
    account_id: str
    old_amount: float
    new_amount: float
    pct_increase: float
    detected_date: date


@dataclass
class MissedChargeAlert:
    merchant_norm: str
    account_id: str
    expected_date: date
    days_overdue: int


@dataclass
class DuplicateBillingAlert:
    merchant_norm: str
    account_id: str
    first_date: date
    second_date: date
    amount: float


def detect_price_hikes(group: RecurringGroupResult, df: pd.DataFrame) -> list[PriceHikeAlert]:
    """New amount exceeds the trailing median by both a percentage and
    an absolute floor -- the absolute floor matters because a 10% hike
    on a $3 charge isn't worth alerting on."""
    alerts = []
    rows = df.loc[group.transaction_indices].sort_values("posted_date")
    amounts = rows["amount"].abs().to_numpy()
    dates = pd.to_datetime(rows["posted_date"]).tolist()

    for i in range(1, len(amounts)):
        trailing_median = float(np.median(amounts[:i]))
        current = amounts[i]
        if trailing_median <= 0:
            continue
        pct_increase = (current - trailing_median) / trailing_median
        abs_increase = current - trailing_median
        if pct_increase > PRICE_HIKE_PCT_THRESHOLD and abs_increase > PRICE_HIKE_ABS_THRESHOLD:
            alerts.append(
                PriceHikeAlert(
                    merchant_norm=group.merchant_norm,
                    account_id=group.account_id,
                    old_amount=round(trailing_median, 2),
                    new_amount=round(current, 2),
                    pct_increase=round(pct_increase, 3),
                    detected_date=dates[i].date(),
                )
            )
    return alerts


def detect_missed_charge(group: RecurringGroupResult, as_of: date) -> MissedChargeAlert | None:
    """Predicted date has passed with grace, and no transaction showed
    up -- a useful early signal for a failed payment method, distinct
    from the user having simply cancelled (which this can't
    distinguish, and says so)."""
    overdue = (as_of - group.predicted_next_date).days
    if overdue > MISSED_CHARGE_GRACE_DAYS:
        return MissedChargeAlert(
            merchant_norm=group.merchant_norm,
            account_id=group.account_id,
            expected_date=group.predicted_next_date,
            days_overdue=overdue,
        )
    return None


def detect_duplicate_billing(
    group: RecurringGroupResult, df: pd.DataFrame
) -> list[DuplicateBillingAlert]:
    """Two charges from the same recurring group closer together than
    its own established cadence should ever produce."""
    alerts = []
    rows = df.loc[group.transaction_indices].sort_values("posted_date")
    dates = pd.to_datetime(rows["posted_date"]).tolist()
    amounts = rows["amount"].tolist()

    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap <= DUPLICATE_BILLING_MAX_GAP_DAYS:
            alerts.append(
                DuplicateBillingAlert(
                    merchant_norm=group.merchant_norm,
                    account_id=group.account_id,
                    first_date=dates[i - 1].date(),
                    second_date=dates[i].date(),
                    amount=round(abs(amounts[i]), 2),
                )
            )
    return alerts


@dataclass
class SubscriptionAuditRow:
    merchant_norm: str
    account_id: str
    period_name: str
    monthly_equivalent: float
    annualized_cost: float
    confidence: float


_PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "monthly": 12, "quarterly": 4, "annual": 1}


def subscription_audit(
    groups: list[RecurringGroupResult], df: pd.DataFrame
) -> list[SubscriptionAuditRow]:
    """All recurring charges, with a monthly-equivalent and annualized
    cost -- the CLI-facing report a user actually reads (§7.2)."""
    rows = []
    for g in groups:
        rows_for_group = df.loc[g.transaction_indices]
        avg_amount = float(rows_for_group["amount"].abs().mean())
        per_year = _PERIODS_PER_YEAR.get(g.period_name, 12)
        annualized = avg_amount * per_year
        rows.append(
            SubscriptionAuditRow(
                merchant_norm=g.merchant_norm,
                account_id=g.account_id,
                period_name=g.period_name,
                monthly_equivalent=round(annualized / 12, 2),
                annualized_cost=round(annualized, 2),
                confidence=g.confidence,
            )
        )
    return sorted(rows, key=lambda r: -r.annualized_cost)
