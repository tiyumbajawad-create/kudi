"""Layer 1 of anomaly detection (design doc §6.2): interpretable
per-signal scores, each producing a score in [0, 1] and a human-readable
reason string. These run before the Isolation Forest layer and are what
a user actually sees when an alert fires -- "unusual combination of
factors" from a forest alone is much less trustworthy than "duplicate
charge: same merchant and amount two days apart".

Every function takes one account's full transaction history as a
DataFrame (columns: posted_date, amount, category, merchant_norm,
raw_description) sorted by date, and returns a dict[index, (score,
reason)] for the rows where that signal fired.

Note: the canonical schema stores `posted_date` as a date, not a
datetime, so there is no hour-of-day in this data model -- the design
doc's "timing" signal is implemented here as day-of-week deviation
only, not hour-of-day. That's a real, documented limitation of the
schema, not an oversight.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Hashable

import pandas as pd
from scipy import stats

SignalResult = dict[Hashable, tuple[float, str]]

MAD_FLOOR = 1.0
AMOUNT_Z_THRESHOLD = 3.5
NEW_MERCHANT_AMOUNT_SCALE = 500.0
NEW_MERCHANT_MIN_WEIGHT = 0.1
VELOCITY_MIN_COUNT = 3
VELOCITY_P_THRESHOLD = 0.01
DUPLICATE_MAX_GAP_DAYS = 1
_STATE_RE = re.compile(r"\b(GA|TX|CO|WA|ID|FL|NY|CA)\b")
GEO_MISMATCH_SCORE = 0.15
DAY_OF_WEEK_MIN_HISTORY = 5
DAY_OF_WEEK_SCORE = 0.2


def amount_outlier(df: pd.DataFrame) -> SignalResult:
    """Robust z-score (median/MAD) of |amount| within category, for this
    account. The MAD floor prevents zero-variance blowups on categories
    that are normally a near-constant amount (e.g. rent)."""
    result: SignalResult = {}
    for category, group in df.groupby("category"):
        amounts = group["amount"].abs()
        median = amounts.median()
        mad = (amounts - median).abs().median()
        mad = max(mad, MAD_FLOOR)
        z = 0.6745 * (amounts - median) / mad
        for idx, val in z.items():
            if val > AMOUNT_Z_THRESHOLD:
                score = min(1.0, val / 10.0)
                reason = (
                    f"amount ${amounts.loc[idx]:.2f} is far outside the usual range for "  # type: ignore[call-overload]
                    f"{category} (robust z-score {val:.1f})"
                )
                result[idx] = (score, reason)
    return result


def new_merchant(
    df: pd.DataFrame, known_recurring_merchants: set[str] | None = None
) -> SignalResult:
    """First-ever transaction with this merchant, weighted by amount --
    a new $4 coffee isn't interesting; a new $600 charge is.

    `known_recurring_merchants`, when provided, suppresses this signal
    for a merchant's first occurrence if that merchant turns out to be
    an established recurring bill/subscription. In hindsight (this
    detector scores a full historical ledger, not a live stream), the
    first month's rent or a new streaming subscription isn't the kind
    of "new merchant" this signal exists to catch -- without this, every
    recurring bill's onboarding month floods the top of the ranking and
    buries genuine one-off anomalies under ties.
    """
    result: SignalResult = {}
    known_recurring_merchants = known_recurring_merchants or set()
    seen: set[str] = set()
    for idx, row in df.sort_values("posted_date").iterrows():
        merchant = row["merchant_norm"]
        if merchant not in seen and merchant not in known_recurring_merchants:
            weight = min(1.0, abs(row["amount"]) / NEW_MERCHANT_AMOUNT_SCALE)
            if weight > NEW_MERCHANT_MIN_WEIGHT:
                result[idx] = (
                    weight,
                    f"first-ever transaction with {merchant!r}, amount ${abs(row['amount']):.2f}",
                )
        seen.add(merchant)
    return result


def velocity(df: pd.DataFrame) -> SignalResult:
    """Transactions-per-day vs this account's own baseline rate, scored
    via a Poisson tail probability -- catches card-testing-style bursts
    without needing a fixed 'more than N transactions' threshold."""
    result: SignalResult = {}
    daily_counts = df.groupby("posted_date").size()
    mean_rate = daily_counts.mean()
    if mean_rate <= 0:
        return result

    for idx, row in df.iterrows():
        count_that_day = daily_counts[row["posted_date"]]
        if count_that_day < VELOCITY_MIN_COUNT:
            continue
        p = stats.poisson.sf(count_that_day - 1, mean_rate)
        if p < VELOCITY_P_THRESHOLD:
            result[idx] = (
                min(1.0, 1 - p),
                f"{count_that_day} transactions on {row['posted_date']} vs a "
                f"typical {mean_rate:.1f}/day for this account (p={p:.4f})",
            )
    return result


def duplicate_charge(df: pd.DataFrame, recurring_group_ids: set[str] | None = None) -> SignalResult:
    """Same merchant + amount within a very short window. A recurring
    group's own cadence is always weeks or more, so any same-merchant,
    same-amount pair less than 2 days apart can't be explained by a
    legitimate recurring cycle regardless of whether it happens to
    belong to one.

    Both transactions in a matched pair are flagged, not just the
    later one -- from a user's perspective "you were double-charged"
    implicates both charges, and crucially, when two transactions land
    on the exact same date, pandas' sort is not guaranteed to break the
    tie the same way every time, so flagging only one side risks
    (non-deterministically) flagging the wrong member of the pair.
    """
    result: SignalResult = {}
    grouped = df.sort_values("posted_date").groupby(["merchant_norm", "amount"])
    for (_merchant, _amount), group in grouped:
        if len(group) < 2:
            continue
        rows = list(group.iterrows())
        for i in range(1, len(rows)):
            prev_idx, prev_row = rows[i - 1]
            idx, row = rows[i]
            gap = (row["posted_date"] - prev_row["posted_date"]).days
            if gap <= DUPLICATE_MAX_GAP_DAYS:
                reason = (
                    f"duplicate charge: same merchant and amount "
                    f"(${abs(row['amount']):.2f}) {gap} day(s) apart"
                )
                result[idx] = (1.0, reason)
                result[prev_idx] = (1.0, reason)
    return result


def geo_mismatch(df: pd.DataFrame) -> SignalResult:
    """Soft signal: a transaction's descriptor geo tail doesn't match
    the account's dominant region. Descriptor geo is unreliable (a lot
    of merchants don't include it at all), so this is capped low.

    Uses pd.isna() rather than `is not None`: pandas can silently
    coerce a Python None into a float NaN when building the Series
    (depending on the mix of types encountered), and `float('nan') is
    not None` evaluates True -- an `is not None` guard alone would
    let a NaN "state" slip through and get embedded, literally, in the
    reason string.
    """
    result: SignalResult = {}
    states = df["raw_description"].apply(
        lambda s: m.group(1) if (m := _STATE_RE.search(s)) else None
    )
    known = states.dropna()
    if known.empty:
        return result
    dominant = known.mode().iloc[0]

    for idx, state in states.items():
        if not pd.isna(state) and state != dominant:
            result[idx] = (
                GEO_MISMATCH_SCORE,
                f"descriptor geo {state!r} doesn't match this account's usual {dominant!r}",
            )
    return result


def day_of_week_deviation(df: pd.DataFrame) -> SignalResult:
    """Day-of-week pattern deviation, only for merchants with enough
    history to have an established pattern. (Hour-of-day is not
    available in this schema -- see module docstring.)"""
    result: SignalResult = {}
    for merchant, group in df.groupby("merchant_norm"):
        if len(group) < DAY_OF_WEEK_MIN_HISTORY:
            continue
        dow_counts = group["posted_date"].apply(lambda d: d.weekday()).value_counts()
        common_days = set(dow_counts[dow_counts >= 2].index)
        if not common_days:
            continue
        for idx, row in group.iterrows():
            dow = row["posted_date"].weekday()
            if dow not in common_days:
                result[idx] = (
                    DAY_OF_WEEK_SCORE,
                    f"{merchant} usually seen on different days of the week",
                )
    return result


def extreme_absolute_amount(df: pd.DataFrame, multiple: float = 5.0) -> SignalResult:
    """Cold-start-safe signal (§6.3): flags an amount far beyond
    anything in this account's own history so far, without needing
    per-category baselines that a brand-new account doesn't have yet."""
    result: SignalResult = {}
    sorted_df = df.sort_values("posted_date")
    running_max = 0.0
    for idx, row in sorted_df.iterrows():
        magnitude = abs(row["amount"])
        if running_max > 0 and magnitude > running_max * multiple:
            result[idx] = (
                1.0,
                f"${magnitude:.2f} is more than {multiple:.0f}x anything seen "
                "on this account so far",
            )
        running_max = max(running_max, magnitude)
    return result


def merge_signals(*signal_results: SignalResult) -> dict[Hashable, tuple[float, list[str]]]:
    """Combines multiple signals per row: score is the max across
    signals that fired; reasons carries every signal that fired."""
    merged: dict[Hashable, tuple[float, list[str]]] = defaultdict(lambda: (0.0, []))
    for signals in signal_results:
        for idx, (score, reason) in signals.items():
            current_score, current_reasons = merged[idx]
            merged[idx] = (max(current_score, score), current_reasons + [reason])
    return dict(merged)
