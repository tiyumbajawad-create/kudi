"""Tests for the layered categorization cascade (§5.2): corrections >
rules > classifier > honest fallback."""

from datetime import date
from decimal import Decimal

from kudi.enrich.categorize import FALLBACK_CATEGORY, Categorizer
from kudi.enrich.rules import RuleTable


def test_rule_hit_when_no_correction() -> None:
    cat = Categorizer(rules=RuleTable.load(), model=None, confidence_threshold=0.5)
    result = cat.categorize_one("KROGER #123", Decimal("-45.00"), date(2026, 1, 15))
    assert result.category == "Food & Drink>Groceries"
    assert result.category_source == "rule"
    assert result.category_confidence == 1.0


def test_user_correction_wins_over_rule() -> None:
    rules = RuleTable.load()
    corrections = {"Kroger": "Shopping>General Merchandise"}  # deliberately "wrong" override
    cat = Categorizer(rules=rules, model=None, confidence_threshold=0.5, corrections=corrections)
    result = cat.categorize_one("KROGER #123", Decimal("-45.00"), date(2026, 1, 15))
    assert result.category == "Shopping>General Merchandise"
    assert result.category_source == "user"


def test_no_model_and_no_rule_hit_falls_back_honestly() -> None:
    cat = Categorizer(rules=RuleTable.load(), model=None, confidence_threshold=0.5)
    result = cat.categorize_one(
        "Some Totally Novel Merchant Ltd", Decimal("-10.00"), date(2026, 1, 1)
    )
    assert result.category == FALLBACK_CATEGORY
    assert result.category_source is None
    assert result.category_confidence is None


def test_merchant_norm_is_always_populated() -> None:
    cat = Categorizer(rules=RuleTable.load(), model=None, confidence_threshold=0.5)
    result = cat.categorize_one("KROGER #123", Decimal("-45.00"), date(2026, 1, 15))
    assert result.merchant_norm == "Kroger"


def test_keyword_rule_fires_for_generic_bank_descriptor() -> None:
    cat = Categorizer(rules=RuleTable.load(), model=None, confidence_threshold=0.5)
    result = cat.categorize_one("OVERDRAFT FEE", Decimal("-35.00"), date(2026, 1, 1))
    assert result.category == "Fees & Interest>Bank Fees"
    assert result.category_source == "rule"
