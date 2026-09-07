"""Tests for the merchant normalization cascade (§5.1)."""

import random

import pytest

from datagen.merchants import CATALOG, render_descriptor
from kudi.enrich.normalize import clean_descriptor, normalize_merchant

# Descriptors with no merchant-identifying text at all (pure system-
# generated fee/interest lines) are an honest, expected limitation --
# there's nothing for the normalizer to resolve.
_GENUINELY_UNRESOLVABLE = {
    "INTEREST PAYMENT",
    "OVERDRAFT FEE",
    "MONTHLY SERVICE FEE",
    "INTEREST CHARGE ON PURCHASES",
}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SQ *BLUE BOTTLE COF 4155551234 CA", "Blue Bottle Coffee"),
        ("KROGER #7212", "Kroger"),
        ("TST* The Local Tavern", "The Local Tavern"),
        ("AMZN Mktp US*2K4", "Amazon"),
        ("STEAMGAMES.COM", "Steam"),
        ("SBUX SEATTLE WA", "Starbucks"),
        ("WHOLEFDS AUSTIN TX", "Whole Foods Market"),
        ("UBER *TRIP *E3Z", "Uber"),
        ("LYFT *RIDE *G9G", "Lyft"),
        ("DOORDASH**X6W", "DoorDash"),
        ("CVS/PHARM #4457", "CVS Pharmacy"),
        ("NETFLIX.COM", "Netflix"),
    ],
)
def test_normalize_known_descriptors(raw: str, expected: str) -> None:
    assert normalize_merchant(raw) == expected


def test_normalize_across_full_catalog_with_noise() -> None:
    """Regression test at scale: every merchant in the catalog, across
    many randomly-noised renderings, must resolve back to its canonical
    name (except the genuinely unresolvable generic descriptors)."""
    rng = random.Random(123)
    failures = []
    for merchant in CATALOG:
        for _ in range(10):
            raw = render_descriptor(merchant, rng)
            norm = normalize_merchant(raw)
            if norm != merchant.name and norm not in _GENUINELY_UNRESOLVABLE:
                failures.append((merchant.name, raw, norm))

    assert not failures, f"{len(failures)} unexpected normalization failures: {failures[:10]}"


def test_clean_descriptor_strips_phone_and_store_id() -> None:
    assert "4155551234" not in clean_descriptor("SQ *BLUE BOTTLE COF 4155551234 CA")
    assert "#7212" not in clean_descriptor("KROGER #7212")


def test_normalize_empty_string_is_safe() -> None:
    assert normalize_merchant("") == ""


def test_normalize_is_deterministic() -> None:
    raw = "SQ *BLUE BOTTLE COF 4155551234 CA"
    assert normalize_merchant(raw) == normalize_merchant(raw)
