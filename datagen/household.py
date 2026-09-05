"""Simulates a household's transaction history across a checking account
and a credit-card account over N months (design doc §8).

Deliberately simple, readable simulation logic -- the value here is the
*noise* (merchant descriptor variety, jitter, seasonality) layered on top
of an otherwise plausible spend pattern, not a sophisticated economic
model.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

from datagen.events import GroundTruthEvent
from datagen.merchants import (
    CATALOG,
    RECURRING_MERCHANTS,
    Merchant,
    by_category_root,
    render_descriptor,
)
from datagen.recurring import generate_recurring_series

CHECKING = "chase-checking"
CREDIT_CARD = "chase-credit-card"


@dataclass
class HouseholdProfile:
    monthly_income: float = 5200.0
    months: int = 12
    seed: int = 42
    start_date: date = date(2025, 1, 1)


def _make_event_id_factory(rng: random.Random) -> Callable[[], str]:
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"evt-{counter['n']:06d}-{rng.getrandbits(24):06x}"

    return factory


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    return date(year, month, 1)


def simulate_household(profile: HouseholdProfile) -> list[GroundTruthEvent]:
    rng = random.Random(profile.seed)
    end_date = _add_months(profile.start_date, profile.months)
    event_id = _make_event_id_factory(rng)
    events: list[GroundTruthEvent] = []

    # --- Recurring bills and subscriptions (rent, utilities, streaming) ---
    for merchant in RECURRING_MERCHANTS:
        account = (
            CHECKING
            if merchant.category.startswith("Housing") or merchant.category.startswith("Utilities")
            else CREDIT_CARD
        )
        events.extend(
            generate_recurring_series(
                merchant, account, profile.start_date, end_date, rng, event_id
            )
        )

    # --- Salary: biweekly direct deposit into checking ---
    payroll = next(m for m in CATALOG if m.name == "Acme Corp Payroll")
    interest = next(m for m in CATALOG if m.name == "Ally Bank Interest")
    pay_date = profile.start_date + timedelta(days=3)
    per_paycheck = round(profile.monthly_income / 2, 2)
    while pay_date < end_date:
        events.append(
            GroundTruthEvent(
                event_id=event_id(),
                account_id=CHECKING,
                txn_date=pay_date,
                amount=per_paycheck,
                merchant_name=payroll.name,
                category=payroll.category,
                raw_descriptor=render_descriptor(payroll, rng),
            )
        )
        pay_date += timedelta(days=14)

    # --- Small monthly interest credit ---
    d = profile.start_date + timedelta(days=27)
    while d < end_date:
        events.append(
            GroundTruthEvent(
                event_id=event_id(),
                account_id=CHECKING,
                txn_date=d,
                amount=round(rng.uniform(*interest.amount_range), 2),
                merchant_name=interest.name,
                category=interest.category,
                raw_descriptor=render_descriptor(interest, rng),
            )
        )
        d = _add_months(d, 1) + timedelta(days=27 - 1)

    # --- Day-by-day discretionary spend: groceries, coffee, dining, gas, shopping ---
    day = profile.start_date
    coffee_merchants = by_category_root("Food & Drink")
    transport_merchants = by_category_root("Transport")
    shopping_merchants = by_category_root("Shopping")
    health_merchants = by_category_root("Health")

    while day < end_date:
        dow = day.weekday()  # 0=Mon
        is_holiday_season = day.month in (11, 12)

        # coffee habit: ~4x/week
        if rng.random() < 4 / 7:
            m = rng.choice([mm for mm in coffee_merchants if mm.category.endswith("Coffee")])
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        # weekly grocery run, heavier on weekends
        if dow in (5, 6) and rng.random() < 0.7:
            m = rng.choice([mm for mm in coffee_merchants if mm.category.endswith("Groceries")])
            events.append(_oneoff(event_id, CHECKING, day, m, rng))

        # dining out / bars a few times a week, more on weekends
        dine_prob = 0.5 if dow in (4, 5, 6) else 0.15
        if rng.random() < dine_prob:
            pool = [
                mm
                for mm in coffee_merchants
                if mm.category.split(">")[1] in ("Restaurants", "Bars", "Fast Food")
            ]
            m = rng.choice(pool)
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        # gas roughly weekly
        if dow == 2 and rng.random() < 0.8:
            m = rng.choice([mm for mm in transport_merchants if mm.category.endswith("Gas")])
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        # rideshare / transit sporadically
        if rng.random() < 0.15:
            m = rng.choice([mm for mm in transport_merchants if not mm.category.endswith("Gas")])
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        # shopping: baseline low probability, elevated in holiday season
        shop_prob = 0.35 if is_holiday_season else 0.1
        if rng.random() < shop_prob:
            m = rng.choice(shopping_merchants)
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        # occasional health/pharmacy spend
        if rng.random() < 0.05:
            m = rng.choice(health_merchants)
            events.append(_oneoff(event_id, CREDIT_CARD, day, m, rng))

        day += timedelta(days=1)

    # --- Monthly credit-card payment: transfer pair between accounts ---
    cc_charges_by_month: dict[tuple[int, int], float] = {}
    for e in events:
        if e.account_id == CREDIT_CARD and e.amount < 0:
            key = (e.txn_date.year, e.txn_date.month)
            cc_charges_by_month[key] = cc_charges_by_month.get(key, 0.0) + (-e.amount)

    for (year, month), total in sorted(cc_charges_by_month.items()):
        pay_on = date(year, month, 28) if month != 12 else date(year, 12, 28)
        if pay_on >= end_date:
            continue
        checking_leg = event_id()
        cc_leg = event_id()
        events.append(
            GroundTruthEvent(
                event_id=checking_leg,
                account_id=CHECKING,
                txn_date=pay_on,
                amount=-round(total, 2),
                merchant_name="Chase Credit Card Payment",
                category="Transfers>Credit Card Payment",
                raw_descriptor="ONLINE PMT TO CREDIT CARD - THANK YOU",
                is_transfer=True,
                transfer_pair_event_id=cc_leg,
            )
        )
        events.append(
            GroundTruthEvent(
                event_id=cc_leg,
                account_id=CREDIT_CARD,
                txn_date=pay_on,
                amount=round(total, 2),
                merchant_name="Chase Credit Card Payment",
                category="Transfers>Credit Card Payment",
                raw_descriptor="PAYMENT THANK YOU",
                is_transfer=True,
                transfer_pair_event_id=checking_leg,
            )
        )

    events.sort(key=lambda e: (e.account_id, e.txn_date))
    return events


def _oneoff(
    event_id_fn: Callable[[], str],
    account_id: str,
    day: date,
    merchant: Merchant,
    rng: random.Random,
) -> GroundTruthEvent:
    amount = round(rng.uniform(*merchant.amount_range), 2)
    return GroundTruthEvent(
        event_id=event_id_fn(),
        account_id=account_id,
        txn_date=day,
        amount=-amount,
        merchant_name=merchant.name,
        category=merchant.category,
        raw_descriptor=render_descriptor(merchant, rng),
    )
