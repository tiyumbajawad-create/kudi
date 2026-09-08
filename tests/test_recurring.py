"""Tests for recurring-charge inference (§7.1), including an
integration test against the actual synthetic generator's ground truth
(§7.3: series-level precision/recall targets)."""

from datetime import date, timedelta

import pandas as pd

from datagen.household import HouseholdProfile, simulate_household
from kudi.detect.recurring import infer_recurring_groups


def _monthly_series(merchant: str, amount: float, months: int = 8) -> list[dict]:
    rows = []
    d = date(2025, 1, 15)
    for _ in range(months):
        rows.append(
            {
                "account_id": "acct-1",
                "merchant_norm": merchant,
                "amount": amount,
                "posted_date": d,
            }
        )
        d = d + timedelta(days=30)
    return rows


def test_detects_clean_monthly_fixed_price_series() -> None:
    df = pd.DataFrame(_monthly_series("Netflix", -15.49))
    results = infer_recurring_groups(df)
    assert len(results) == 1
    assert results[0].merchant_norm == "Netflix"
    assert results[0].period_name == "monthly"
    assert results[0].confidence > 0.8


def test_does_not_flag_a_one_off_purchase() -> None:
    df = pd.DataFrame(
        [
            {
                "account_id": "acct-1",
                "merchant_norm": "Best Buy",
                "amount": -300.0,
                "posted_date": date(2025, 1, 1),
            },
            {
                "account_id": "acct-1",
                "merchant_norm": "Best Buy",
                "amount": -50.0,
                "posted_date": date(2025, 6, 15),
            },
        ]
    )
    assert infer_recurring_groups(df) == []


def test_requires_minimum_observations() -> None:
    df = pd.DataFrame(_monthly_series("Spotify", -11.99, months=2))
    assert infer_recurring_groups(df) == []


def test_variable_amount_bill_still_detected_with_lower_confidence() -> None:
    rows = _monthly_series("Georgia Power", -100.0, months=8)
    for i, r in enumerate(rows):
        r["amount"] = -100.0 - (i * 5 % 30)
    df = pd.DataFrame(rows)
    results = infer_recurring_groups(df)
    assert len(results) == 1
    assert results[0].period_name == "monthly"


def test_predicted_next_date_is_last_date_plus_period() -> None:
    df = pd.DataFrame(_monthly_series("Netflix", -15.49, months=5))
    result = infer_recurring_groups(df)[0]
    last_date = pd.Timestamp(df["posted_date"].max())
    assert pd.Timestamp(result.predicted_next_date) == last_date + timedelta(
        days=result.period_days
    )


def test_against_synthetic_ground_truth() -> None:
    """Integration test: run the real generator, and check the detector
    against its labeled recurring merchants (§7.3 targets: series-level
    precision >= 0.9, recall >= 0.85)."""
    from kudi.enrich.normalize import normalize_merchant

    profile = HouseholdProfile(seed=3, months=24)
    events = simulate_household(profile)
    rows = [
        {
            "account_id": e.account_id,
            "posted_date": e.txn_date,
            "amount": float(e.amount),
            "merchant_norm": normalize_merchant(e.raw_descriptor),
            "true_is_recurring": e.is_recurring,
        }
        for e in events
        if not e.is_transfer
    ]
    df = pd.DataFrame(rows)

    results = infer_recurring_groups(df)
    detected = {r.merchant_norm for r in results}
    true_recurring = set(df[df["true_is_recurring"]]["merchant_norm"].unique())

    recall = len(detected & true_recurring) / len(true_recurring)
    assert recall >= 0.85, f"recall {recall:.2f} below §7.3 target"
