"""Generates recurring-charge series (subscriptions, bills) with the
messiness that makes recurring-charge *inference* (a later milestone) a
real problem: date jitter, occasional skipped cycles, drifting amounts,
and scheduled price hikes. Ground truth for all of it is recorded here so
§7.3's evaluation has something honest to score against.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta

from datagen.events import GroundTruthEvent
from datagen.merchants import Merchant, render_descriptor


def generate_recurring_series(
    merchant: Merchant,
    account_id: str,
    start: date,
    end: date,
    rng: random.Random,
    event_id_fn: Callable[[], str],
) -> list[GroundTruthEvent]:
    """One merchant's full billing history over [start, end)."""
    assert merchant.recurring is not None
    spec = merchant.recurring
    group_id = f"rec-{account_id}-{merchant.name}".replace(" ", "_").lower()

    events: list[GroundTruthEvent] = []
    current = start
    cycle = 0
    amount = spec.base_amount

    while current < end:
        cycle += 1
        jitter = timedelta(days=rng.randint(-spec.jitter_days, spec.jitter_days))
        txn_date = current + jitter

        skip = rng.random() < spec.skip_prob
        if not skip and start <= txn_date < end:
            # amount drift within the coefficient of variation
            if spec.amount_cv > 0:
                drift = rng.gauss(0, spec.amount_cv) * amount
                this_amount = max(round(amount + drift, 2), 1.0)
            else:
                this_amount = amount

            hiked = False
            if spec.hike_after_months and cycle == spec.hike_after_months and spec.hike_pct > 0:
                amount = round(amount * (1 + spec.hike_pct), 2)
                this_amount = amount
                hiked = True

            events.append(
                GroundTruthEvent(
                    event_id=event_id_fn(),
                    account_id=account_id,
                    txn_date=txn_date,
                    amount=-round(this_amount, 2),
                    merchant_name=merchant.name,
                    category=merchant.category,
                    raw_descriptor=render_descriptor(merchant, rng),
                    is_recurring=True,
                    recurring_group_id=group_id,
                    price_hike=hiked,
                )
            )

        current = current + timedelta(days=spec.period_days)

    return events
