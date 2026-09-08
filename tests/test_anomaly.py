"""Tests for the two-layer anomaly detector orchestrator (§6.2, §6.3)."""

from datetime import date, timedelta

import pandas as pd

from kudi.detect.anomaly import COLD_START_MIN_TRANSACTIONS, detect_anomalies


def _make_account_history(n: int, start: date = date(2025, 1, 1)) -> pd.DataFrame:
    rows = [
        {
            "posted_date": start + timedelta(days=i),
            "amount": -20.0,
            "category": "Food & Drink>Groceries",
            "merchant_norm": "Kroger",
            "raw_description": "KROGER #123",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_empty_dataframe_returns_empty_result() -> None:
    result = detect_anomalies(
        pd.DataFrame(
            columns=["posted_date", "amount", "category", "merchant_norm", "raw_description"]
        )
    )
    assert result.scores.empty
    assert not result.cold_start


def test_short_history_is_cold_start() -> None:
    df = _make_account_history(COLD_START_MIN_TRANSACTIONS - 5)
    result = detect_anomalies(df)
    assert result.cold_start


def test_long_history_is_not_cold_start() -> None:
    df = _make_account_history(60)
    result = detect_anomalies(df)
    assert not result.cold_start


def test_scores_are_percentiles_between_zero_and_one() -> None:
    df = _make_account_history(60)
    result = detect_anomalies(df)
    assert (result.scores >= 0).all()
    assert (result.scores <= 1).all()


def test_duplicate_charge_scores_highest_in_cold_start_mode() -> None:
    """Cold start only runs high-precision rules (§6.3) -- a duplicate
    charge should still be caught even with very little history."""
    rows = [
        {
            "posted_date": date(2026, 1, 1),
            "amount": -50.0,
            "category": "Shopping>Electronics",
            "merchant_norm": "Best Buy",
            "raw_description": "BEST BUY #1",
        },
        {
            "posted_date": date(2026, 1, 1),
            "amount": -50.0,
            "category": "Shopping>Electronics",
            "merchant_norm": "Best Buy",
            "raw_description": "BEST BUY #1",
        },
        {
            "posted_date": date(2026, 1, 2),
            "amount": -10.0,
            "category": "Food & Drink>Coffee",
            "merchant_norm": "Starbucks",
            "raw_description": "STARBUCKS #1",
        },
    ]
    df = pd.DataFrame(rows)
    result = detect_anomalies(df)
    assert result.cold_start
    dup_scores = result.scores.loc[[0, 1]]
    other_score = result.scores.loc[2]
    assert dup_scores.min() > other_score


def test_known_recurring_merchants_suppresses_new_merchant_flag() -> None:
    df = _make_account_history(60)
    new_row = pd.DataFrame(
        [
            {
                "posted_date": date(2025, 1, 1),
                "amount": -1450.0,
                "category": "Housing>Rent",
                "merchant_norm": "Meridian Properties",
                "raw_description": "MERIDIAN PROPERTIES",
            }
        ]
    )
    df = pd.concat([df, new_row], ignore_index=True)

    without_suppression = detect_anomalies(df)
    with_suppression = detect_anomalies(df, known_recurring_merchants={"Meridian Properties"})

    rent_idx = df[df["merchant_norm"] == "Meridian Properties"].index[0]
    assert with_suppression.scores[rent_idx] <= without_suppression.scores[rent_idx]


def test_reasons_present_for_scored_anomalies() -> None:
    rows = [
        {
            "posted_date": date(2026, 1, 1),
            "amount": -50.0,
            "category": "Shopping>Electronics",
            "merchant_norm": "Best Buy",
            "raw_description": "BEST BUY #1",
        },
        {
            "posted_date": date(2026, 1, 1),
            "amount": -50.0,
            "category": "Shopping>Electronics",
            "merchant_norm": "Best Buy",
            "raw_description": "BEST BUY #1",
        },
    ]
    df = pd.DataFrame(rows)
    result = detect_anomalies(df)
    assert len(result.reasons) == 2
    assert all("duplicate" in r[0].lower() for r in result.reasons.values())
