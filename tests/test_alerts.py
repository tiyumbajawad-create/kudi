"""Tests for derived recurring alerts (§7.2)."""

from datetime import date

import pandas as pd

from kudi.detect.alerts import (
    detect_duplicate_billing,
    detect_missed_charge,
    detect_price_hikes,
    subscription_audit,
)
from kudi.detect.recurring import RecurringGroupResult


def _group(indices: list[int], **overrides: object) -> RecurringGroupResult:
    defaults: dict[str, object] = dict(
        account_id="acct-1",
        merchant_norm="Netflix",
        period_name="monthly",
        period_days=30,
        confidence=0.95,
        n_observations=len(indices),
        predicted_next_date=date(2026, 1, 1),
        predicted_amount_low=14.0,
        predicted_amount_high=16.0,
        transaction_indices=indices,
    )
    defaults.update(overrides)
    return RecurringGroupResult(**defaults)  # type: ignore[arg-type]


def test_detects_price_hike() -> None:
    df = pd.DataFrame(
        {
            "posted_date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)],
            "amount": [-15.49, -15.49, -17.99],
        }
    )
    group = _group(list(df.index))
    alerts = detect_price_hikes(group, df)
    assert len(alerts) == 1
    assert alerts[0].new_amount == 17.99


def test_no_price_hike_below_thresholds() -> None:
    df = pd.DataFrame(
        {
            "posted_date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)],
            "amount": [-15.49, -15.49, -15.99],  # tiny bump, below both thresholds
        }
    )
    group = _group(list(df.index))
    assert detect_price_hikes(group, df) == []


def test_missed_charge_after_grace_period() -> None:
    group = _group([0, 1, 2], predicted_next_date=date(2026, 1, 1))
    alert = detect_missed_charge(group, as_of=date(2026, 1, 10))
    assert alert is not None
    assert alert.days_overdue == 9


def test_no_missed_charge_within_grace_period() -> None:
    group = _group([0, 1, 2], predicted_next_date=date(2026, 1, 1))
    assert detect_missed_charge(group, as_of=date(2026, 1, 3)) is None


def test_detects_duplicate_billing() -> None:
    df = pd.DataFrame(
        {
            "posted_date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 2, 1)],
            "amount": [-15.49, -15.49, -15.49],
        }
    )
    group = _group(list(df.index))
    alerts = detect_duplicate_billing(group, df)
    assert len(alerts) == 1
    assert alerts[0].amount == 15.49


def test_subscription_audit_sorts_by_annualized_cost() -> None:
    df = pd.DataFrame(
        {
            "posted_date": [date(2025, 1, 1), date(2025, 2, 1)] * 2,
            "amount": [-15.49, -15.49, -1450.0, -1450.0],
        }
    )
    cheap = _group([0, 1], merchant_norm="Netflix")
    expensive = _group([2, 3], merchant_norm="Meridian Properties")
    rows = subscription_audit([cheap, expensive], df)
    assert rows[0].merchant_norm == "Meridian Properties"
    assert rows[0].annualized_cost > rows[1].annualized_cost
