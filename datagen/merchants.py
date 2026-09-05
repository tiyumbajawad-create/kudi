"""Merchant catalog for the synthetic data generator.

Each merchant has one canonical name, a category-taxonomy leaf, and several
noisy raw-descriptor *templates* that reproduce the mess real bank exports
contain: processor prefixes, store numbers, phone tails, truncation, geo
suffixes (design doc §5.1, §8). The same merchant renders differently across
transactions -- that noise is exactly what makes categorization hard, and
exactly what merchant normalization (a later milestone) has to undo.

This ships a curated ~60-merchant starter catalog covering every category
leaf at least once, not the full ~300 merchants sketched in the design doc.
The `Merchant` schema and `CATALOG` list are the extension point: growing
coverage is additive (append entries), never a code change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RecurringSpec:
    """Describes a merchant that bills on a schedule (design doc §7)."""

    period_days: int  # target inter-arrival period
    jitter_days: int  # +/- wobble around the period
    base_amount: float
    amount_cv: float  # coefficient of variation; ~0 = fixed price
    hike_after_months: int | None = None  # None = never hikes in-window
    hike_pct: float = 0.0
    skip_prob: float = 0.0  # probability a given cycle is simply missed


@dataclass(frozen=True)
class Merchant:
    name: str  # canonical name
    category: str  # "Root>Leaf"
    templates: list[str]  # raw-descriptor templates, see render_descriptor
    amount_range: tuple[float, float] = (5.0, 50.0)
    sign: int = -1  # -1 outflow (default), +1 inflow
    recurring: RecurringSpec | None = None


_CITIES = ["ATLANTA GA", "AUSTIN TX", "DENVER CO", "SEATTLE WA", "BOISE ID", "TAMPA FL"]
_STATES = ["GA", "TX", "CO", "WA", "ID", "FL", "NY", "CA"]


def _phone(rng: random.Random) -> str:
    return f"{rng.randint(200, 999)}555{rng.randint(1000, 9999)}"


def _store(rng: random.Random) -> str:
    return str(rng.randint(1, 9999))


def render_descriptor(merchant: Merchant, rng: random.Random) -> str:
    """Fill one randomly-chosen template with randomized noise fields."""
    template = rng.choice(merchant.templates)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    trunc = f"*{rng.choice(letters)}{rng.randint(2, 9)}{rng.choice(letters)}"
    return template.format(
        name=merchant.name,
        phone=_phone(rng),
        store=_store(rng),
        city=rng.choice(_CITIES),
        state=rng.choice(_STATES),
        trunc=trunc,
    )


def _m(name: str, category: str, templates: list[str], **kw: object) -> Merchant:
    return Merchant(name=name, category=category, templates=templates, **kw)  # type: ignore[arg-type]


CATALOG: list[Merchant] = [
    # --- Income ---
    _m(
        "Acme Corp Payroll",
        "Income>Salary",
        ["{name} DIRECT DEP", "PAYROLL {name}"],
        amount_range=(2200, 3800),
        sign=1,
    ),
    _m(
        "Ally Bank Interest",
        "Income>Interest",
        ["INTEREST PAYMENT", "{name} INT"],
        amount_range=(0.5, 6.0),
        sign=1,
    ),
    # --- Housing ---
    _m(
        "Meridian Properties",
        "Housing>Rent",
        ["{name} RENT ACH", "{name} #{store}"],
        recurring=RecurringSpec(30, 2, 1450.0, 0.0),
    ),
    # --- Utilities ---
    _m(
        "Georgia Power",
        "Utilities>Electricity",
        ["{name} ELEC PYMT", "{name} UTIL {phone}"],
        recurring=RecurringSpec(30, 3, 110.0, 0.35, hike_after_months=9, hike_pct=0.08),
    ),
    _m(
        "City Water Dept",
        "Utilities>Water",
        ["{name} WATER BILL", "{name} #{store}"],
        recurring=RecurringSpec(30, 4, 45.0, 0.25),
    ),
    _m(
        "Xfinity Internet",
        "Utilities>Internet",
        ["COMCAST {name}", "{name} {phone}"],
        recurring=RecurringSpec(30, 1, 79.99, 0.02, hike_after_months=6, hike_pct=0.1),
    ),
    _m(
        "Verizon Wireless",
        "Utilities>Phone",
        ["VZW {name}", "{name} *{store}"],
        recurring=RecurringSpec(30, 1, 85.0, 0.05),
    ),
    # --- Food & Drink ---
    _m(
        "Kroger",
        "Food & Drink>Groceries",
        ["KROGER #{store}", "{name} {city}"],
        amount_range=(35, 180),
    ),
    _m(
        "Whole Foods Market",
        "Food & Drink>Groceries",
        ["WHOLEFDS {city}", "{name} #{store}"],
        amount_range=(40, 160),
    ),
    _m(
        "Blue Bottle Coffee",
        "Food & Drink>Coffee",
        ["SQ *{name} {phone} {state}", "SQ *BLUE BOTTLE COF {phone} {state}"],
        amount_range=(3.5, 7.5),
    ),
    _m(
        "Starbucks",
        "Food & Drink>Coffee",
        ["STARBUCKS #{store}", "SBUX {city}"],
        amount_range=(4, 9),
    ),
    _m(
        "Chipotle",
        "Food & Drink>Fast Food",
        ["CHIPOTLE {store}", "{name} {city}"],
        amount_range=(9, 16),
    ),
    _m(
        "Chick-fil-A",
        "Food & Drink>Fast Food",
        ["{name} #{store}", "CHICKFILA {city}"],
        amount_range=(7, 14),
    ),
    _m(
        "The Local Tavern",
        "Food & Drink>Bars",
        ["TST* {name}", "{name} {city}"],
        amount_range=(15, 60),
    ),
    _m(
        "Olive Branch Bistro",
        "Food & Drink>Restaurants",
        ["TST* {name}", "{name} {city} {state}"],
        amount_range=(25, 90),
    ),
    _m(
        "DoorDash",
        "Food & Drink>Restaurants",
        ["DOORDASH*{trunc}", "{name} {city}"],
        amount_range=(18, 45),
    ),
    # --- Transport ---
    _m(
        "Shell Oil",
        "Transport>Gas",
        ["SHELL OIL {store}", "{name} #{store} {state}"],
        amount_range=(30, 65),
    ),
    _m("Chevron", "Transport>Gas", ["CHEVRON {store}", "{name} {city}"], amount_range=(28, 60)),
    _m(
        "Uber",
        "Transport>Rideshare",
        ["UBER *TRIP {trunc}", "UBER *EATS {trunc}"],
        amount_range=(8, 35),
    ),
    _m(
        "Lyft", "Transport>Rideshare", ["LYFT *RIDE {trunc}", "{name} {city}"], amount_range=(8, 30)
    ),
    _m(
        "MARTA",
        "Transport>Public Transit",
        ["{name} FARE", "{name} #{store}"],
        amount_range=(2.5, 5),
    ),
    _m(
        "ParkMobile",
        "Transport>Parking",
        ["PARKMOBILE {store}", "{name} {city}"],
        amount_range=(3, 20),
    ),
    # --- Shopping ---
    _m(
        "Amazon",
        "Shopping>General Merchandise",
        ["AMZN Mktp US{trunc}", "AMAZON.COM*{trunc}"],
        amount_range=(8, 200),
    ),
    _m(
        "Target",
        "Shopping>General Merchandise",
        ["TARGET #{store}", "{name} {city}"],
        amount_range=(15, 150),
    ),
    _m(
        "Best Buy",
        "Shopping>Electronics",
        ["BESTBUY.COM{trunc}", "BEST BUY #{store}"],
        amount_range=(30, 900),
    ),
    _m(
        "Apple Store",
        "Shopping>Electronics",
        ["APPLE.COM/BILL", "APPLE STORE #{store}"],
        amount_range=(9.99, 1200),
    ),
    _m("H&M", "Shopping>Clothing", ["H&M {city}", "{name} US #{store}"], amount_range=(20, 120)),
    _m("IKEA", "Shopping>Home Goods", ["IKEA {city}", "{name} #{store}"], amount_range=(25, 500)),
    # --- Health ---
    _m(
        "CVS Pharmacy",
        "Health>Pharmacy",
        ["CVS/PHARM #{store}", "{name} {city}"],
        amount_range=(8, 60),
    ),
    _m(
        "Piedmont Medical",
        "Health>Medical",
        ["{name} CLINIC", "{name} BILLING {phone}"],
        amount_range=(20, 300),
    ),
    _m(
        "Anytime Fitness",
        "Health>Fitness",
        ["{name} #{store}", "{name} MEMBER DUES"],
        recurring=RecurringSpec(30, 1, 39.99, 0.0),
    ),
    # --- Entertainment ---
    _m(
        "Netflix",
        "Entertainment>Streaming",
        ["NETFLIX.COM", "{name} *{trunc}"],
        recurring=RecurringSpec(30, 0, 15.49, 0.0, hike_after_months=10, hike_pct=0.13),
    ),
    _m(
        "Spotify",
        "Entertainment>Streaming",
        ["SPOTIFY USA", "{name} *{trunc}"],
        recurring=RecurringSpec(30, 0, 11.99, 0.0),
    ),
    _m(
        "AMC Theatres",
        "Entertainment>Movies",
        ["AMC {city}", "{name} #{store}"],
        amount_range=(12, 45),
    ),
    _m("Steam", "Entertainment>Games", ["STEAMGAMES.COM", "{name} *{trunc}"], amount_range=(5, 70)),
    # --- Travel ---
    _m(
        "Delta Air Lines",
        "Travel>Flights",
        ["DELTA AIR {trunc}", "{name} {phone}"],
        amount_range=(150, 650),
    ),
    _m(
        "Marriott", "Travel>Hotels", ["MARRIOTT {city}", "{name} #{store}"], amount_range=(120, 400)
    ),
    _m(
        "Hertz",
        "Travel>Rental Car",
        ["HERTZ RENT {city}", "{name} #{store}"],
        amount_range=(60, 300),
    ),
    # --- Fees & Interest ---
    _m(
        "Chase Bank Fee",
        "Fees & Interest>Bank Fees",
        ["MONTHLY SERVICE FEE", "OVERDRAFT FEE"],
        amount_range=(10, 35),
    ),
    _m(
        "Chase Interest Charge",
        "Fees & Interest>Interest Charges",
        ["INTEREST CHARGE ON PURCHASES"],
        amount_range=(5, 60),
    ),
]


# Recurring merchants, split out for the household simulator and eval harness.
RECURRING_MERCHANTS = [m for m in CATALOG if m.recurring is not None]
NON_RECURRING_MERCHANTS = [m for m in CATALOG if m.recurring is None]


def by_category_root(root: str) -> list[Merchant]:
    return [m for m in CATALOG if m.category.split(">")[0] == root]
