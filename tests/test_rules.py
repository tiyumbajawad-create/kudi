"""Tests for the rule table (§5.2 layer 2)."""

from kudi.enrich.rules import RuleTable


def test_loads_default_rules() -> None:
    rules = RuleTable.load()
    assert rules.covers("Kroger")
    assert rules.match("Kroger", "KROGER #123") == "Food & Drink>Groceries"


def test_no_match_returns_none() -> None:
    rules = RuleTable.load()
    assert rules.match("Some Totally Unknown Merchant", "random text") is None


def test_keyword_fallback_for_generic_descriptors() -> None:
    """Descriptors with no merchant-identifying text (bank-generated fee
    lines) are matched by raw-text keyword instead of merchant_norm."""
    rules = RuleTable.load()
    assert rules.match("OVERDRAFT FEE", "OVERDRAFT FEE") == "Fees & Interest>Bank Fees"
    assert rules.match("unrelated", "some OVERDRAFT FEE charge") == "Fees & Interest>Bank Fees"


def test_merchant_norm_rule_takes_priority_over_keyword() -> None:
    rules = RuleTable.load()
    # "Kroger" has a merchant rule; ensure it doesn't accidentally also
    # trip a keyword rule and that merchant rule wins if both could apply
    result = rules.match("Kroger", "KROGER #123")
    assert result == "Food & Drink>Groceries"


def test_covers_reflects_merchant_norm_table_only() -> None:
    rules = RuleTable.load()
    assert rules.covers("Kroger") is True
    assert rules.covers("Uber") is False  # Uber is classifier-zone by design
