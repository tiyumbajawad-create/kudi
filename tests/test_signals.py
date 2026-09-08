"""Unit tests for Layer 1 anomaly signals (§6.2)."""

from datetime import date

import pandas as pd

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


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    return df


def test_amount_outlier_flags_extreme_value_in_category() -> None:
    rows = [
        {"category": "Food & Drink>Groceries", "amount": -50.0, "posted_date": date(2026, 1, i)}
        for i in range(1, 11)
    ]
    rows.append(
        {"category": "Food & Drink>Groceries", "amount": -900.0, "posted_date": date(2026, 1, 11)}
    )
    df = _df(rows)
    result = amount_outlier(df)
    assert df.index[-1] in result
    assert result[df.index[-1]][0] > 0


def test_new_merchant_scores_first_occurrence_weighted_by_amount() -> None:
    df = _df(
        [
            {"merchant_norm": "Kroger", "amount": -600.0, "posted_date": date(2026, 1, 1)},
            {"merchant_norm": "Kroger", "amount": -50.0, "posted_date": date(2026, 1, 8)},
        ]
    )
    result = new_merchant(df)
    assert df.index[0] in result
    assert df.index[1] not in result  # second occurrence, not new anymore


def test_new_merchant_suppressed_for_known_recurring() -> None:
    df = _df(
        [
            {
                "merchant_norm": "Meridian Properties",
                "amount": -1450.0,
                "posted_date": date(2026, 1, 1),
            }
        ]
    )
    result = new_merchant(df, known_recurring_merchants={"Meridian Properties"})
    assert result == {}


def test_new_merchant_ignores_small_amounts() -> None:
    df = _df([{"merchant_norm": "Corner Store", "amount": -3.0, "posted_date": date(2026, 1, 1)}])
    assert new_merchant(df) == {}


def test_velocity_flags_burst_day() -> None:
    rows = [{"posted_date": date(2026, 1, i)} for i in range(1, 30)]  # 1/day baseline
    rows += [{"posted_date": date(2026, 2, 1)}] * 6  # burst day
    df = _df(rows)
    result = velocity(df)
    burst_idx = df[df["posted_date"] == pd.Timestamp("2026-02-01")].index
    assert any(idx in result for idx in burst_idx)


def test_duplicate_charge_flags_both_sides_of_pair() -> None:
    df = _df(
        [
            {"merchant_norm": "Coffee Shop", "amount": -4.75, "posted_date": date(2026, 1, 1)},
            {"merchant_norm": "Coffee Shop", "amount": -4.75, "posted_date": date(2026, 1, 1)},
        ]
    )
    result = duplicate_charge(df)
    assert len(result) == 2
    assert all(score == 1.0 for score, _ in result.values())


def test_duplicate_charge_ignores_wide_gaps() -> None:
    df = _df(
        [
            {"merchant_norm": "Netflix", "amount": -15.49, "posted_date": date(2026, 1, 1)},
            {"merchant_norm": "Netflix", "amount": -15.49, "posted_date": date(2026, 2, 1)},
        ]
    )
    assert duplicate_charge(df) == {}


def test_geo_mismatch_flags_minority_state() -> None:
    rows = [{"raw_description": "SHELL OIL ATLANTA GA"} for _ in range(5)]
    rows.append({"raw_description": "SHELL OIL SEATTLE WA"})
    df = pd.DataFrame(rows)
    result = geo_mismatch(df)
    assert len(result) == 1
    assert "WA" in result[df.index[-1]][1]


def test_geo_mismatch_ignores_descriptors_without_state() -> None:
    df = pd.DataFrame([{"raw_description": "NETFLIX.COM"}, {"raw_description": "SPOTIFY USA"}])
    assert geo_mismatch(df) == {}


def test_geo_mismatch_never_embeds_nan_in_reason() -> None:
    """Regression test: pandas can coerce a Python None into float NaN,
    and `nan is not None` is True -- an `is not None` guard alone lets
    a NaN state slip through and get embedded, literally, in the
    reason string."""
    rows = [{"raw_description": "SHELL OIL ATLANTA GA"}] * 3
    rows += [{"raw_description": "NETFLIX.COM"}] * 5  # no state token at all
    df = pd.DataFrame(rows)
    result = geo_mismatch(df)
    for _score, reason in result.values():
        assert "nan" not in reason.lower()


def test_day_of_week_deviation_needs_minimum_history() -> None:
    df = _df([{"merchant_norm": "Gym", "posted_date": date(2026, 1, 1)}])
    assert day_of_week_deviation(df) == {}


def test_extreme_absolute_amount_flags_large_jump() -> None:
    rows = [{"amount": -20.0, "posted_date": date(2026, 1, i)} for i in range(1, 10)]
    rows.append({"amount": -500.0, "posted_date": date(2026, 1, 10)})
    df = _df(rows)
    result = extreme_absolute_amount(df)
    assert df.index[-1] in result


def test_merge_signals_takes_max_score_and_unions_reasons() -> None:
    a = {1: (0.3, "signal A fired")}
    b = {1: (0.8, "signal B fired"), 2: (0.5, "signal C fired")}
    merged = merge_signals(a, b)
    assert merged[1][0] == 0.8
    assert set(merged[1][1]) == {"signal A fired", "signal B fired"}
    assert merged[2] == (0.5, ["signal C fired"])
