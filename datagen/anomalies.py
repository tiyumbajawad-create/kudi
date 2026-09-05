"""Injects labeled anomaly scenarios into an otherwise-normal ledger, so
the (future) anomaly detector's precision@k and recall can be scored
against honest ground truth (§6.4) instead of guessed at.

Scenarios implemented:
  - card_testing_burst: several tiny charges within minutes, then one large
    charge, all at a brand-new merchant.
  - duplicate_charge: the same charge posted twice in a short window.
  - large_out_of_pattern: a purchase far outside the account's normal range.
  - new_merchant_odd_hour: a first-time merchant at an unusual hour.
  - subscription_double_bill: an existing recurring group billed twice in
    one cycle.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import timedelta

from datagen.events import GroundTruthEvent
from datagen.merchants import CATALOG, Merchant, render_descriptor

_NEW_MERCHANT_NAMES = [
    "QuickTech Electronics Kiosk",
    "Global Gadgets Direct",
    "NightOwl Liquor & More",
    "Prestige Jewelers Online",
]


def _pick_new_merchant(rng: random.Random) -> Merchant:
    name = rng.choice(_NEW_MERCHANT_NAMES)
    return Merchant(
        name=name,
        category="Shopping>Electronics",
        templates=["{name} {trunc}", "{name} #{store}"],
        amount_range=(200, 900),
    )


def inject_anomalies(
    events: list[GroundTruthEvent],
    rng: random.Random,
    event_id_fn: Callable[[], str],
    prevalence: float = 0.005,
) -> list[GroundTruthEvent]:
    """Returns a NEW list: original events + injected anomalous ones."""
    non_transfer = [e for e in events if not e.is_transfer]
    n_target = max(3, round(len(non_transfer) * prevalence))
    accounts = sorted({e.account_id for e in events})
    scenario_types = [
        "card_testing_burst",
        "duplicate_charge",
        "large_out_of_pattern",
        "new_merchant_odd_hour",
        "subscription_double_bill",
    ]

    injected: list[GroundTruthEvent] = []
    recurring_events = [e for e in events if e.is_recurring]

    while len(injected) < n_target:
        scenario = rng.choice(scenario_types)
        account = rng.choice(accounts)
        anchor_date = rng.choice(events).txn_date

        if scenario == "card_testing_burst":
            merchant = _pick_new_merchant(rng)
            for amt in (rng.uniform(1, 3), rng.uniform(1, 3), rng.uniform(300, 800)):
                injected.append(
                    GroundTruthEvent(
                        event_id=event_id_fn(),
                        account_id=account,
                        txn_date=anchor_date,
                        amount=-round(amt, 2),
                        merchant_name=merchant.name,
                        category=merchant.category,
                        raw_descriptor=render_descriptor(merchant, rng),
                        is_anomaly=True,
                        anomaly_type="card_testing_burst",
                    )
                )

        elif scenario == "duplicate_charge":
            pool = [e for e in non_transfer if e.account_id == account] or non_transfer
            source = rng.choice(pool)
            dup = GroundTruthEvent(
                event_id=event_id_fn(),
                account_id=source.account_id,
                txn_date=source.txn_date + timedelta(minutes=rng.randint(1, 90)),
                amount=source.amount,
                merchant_name=source.merchant_name,
                category=source.category,
                raw_descriptor=source.raw_descriptor,
                is_anomaly=True,
                anomaly_type="duplicate_charge",
            )
            injected.append(dup)

        elif scenario == "large_out_of_pattern":
            merchant = rng.choice(CATALOG)
            injected.append(
                GroundTruthEvent(
                    event_id=event_id_fn(),
                    account_id=account,
                    txn_date=anchor_date,
                    amount=-round(rng.uniform(1500, 4000), 2),
                    merchant_name=merchant.name,
                    category=merchant.category,
                    raw_descriptor=render_descriptor(merchant, rng),
                    is_anomaly=True,
                    anomaly_type="large_out_of_pattern",
                )
            )

        elif scenario == "new_merchant_odd_hour":
            merchant = _pick_new_merchant(rng)
            injected.append(
                GroundTruthEvent(
                    event_id=event_id_fn(),
                    account_id=account,
                    txn_date=anchor_date,
                    amount=-round(rng.uniform(150, 500), 2),
                    merchant_name=merchant.name,
                    category=merchant.category,
                    raw_descriptor=render_descriptor(merchant, rng) + " 0347AM",
                    is_anomaly=True,
                    anomaly_type="new_merchant_odd_hour",
                )
            )

        elif scenario == "subscription_double_bill" and recurring_events:
            source = rng.choice(recurring_events)
            injected.append(
                GroundTruthEvent(
                    event_id=event_id_fn(),
                    account_id=source.account_id,
                    txn_date=source.txn_date + timedelta(hours=rng.randint(1, 20)),
                    amount=source.amount,
                    merchant_name=source.merchant_name,
                    category=source.category,
                    raw_descriptor=source.raw_descriptor,
                    is_recurring=True,
                    recurring_group_id=source.recurring_group_id,
                    is_anomaly=True,
                    anomaly_type="subscription_double_bill",
                )
            )

    return events + injected
