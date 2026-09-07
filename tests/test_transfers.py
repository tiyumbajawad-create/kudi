"""Tests for transfer-pair detection (§5.4)."""

from datetime import date
from decimal import Decimal

from kudi.enrich.transfers import TransferCandidate, detect_transfer_pairs


def test_detects_matched_pair_across_accounts() -> None:
    cands = [
        TransferCandidate(
            "t1",
            "checking",
            date(2026, 1, 28),
            Decimal("-500.00"),
            "ONLINE PMT TO CREDIT CARD - THANK YOU",
        ),
        TransferCandidate(
            "t2", "credit-card", date(2026, 1, 28), Decimal("500.00"), "PAYMENT THANK YOU"
        ),
    ]
    assert detect_transfer_pairs(cands) == [("t1", "t2")]


def test_ignores_non_transfer_keywords() -> None:
    cands = [
        TransferCandidate("t1", "checking", date(2026, 1, 15), Decimal("-50.00"), "KROGER GROCERY"),
        TransferCandidate("t2", "credit-card", date(2026, 1, 15), Decimal("50.00"), "SOME REFUND"),
    ]
    assert detect_transfer_pairs(cands) == []


def test_same_account_pairs_are_not_transfers() -> None:
    cands = [
        TransferCandidate("t1", "checking", date(2026, 1, 28), Decimal("-500.00"), "TRANSFER OUT"),
        TransferCandidate("t2", "checking", date(2026, 1, 28), Decimal("500.00"), "TRANSFER IN"),
    ]
    assert detect_transfer_pairs(cands) == []


def test_amounts_must_be_exactly_opposite() -> None:
    cands = [
        TransferCandidate("t1", "checking", date(2026, 1, 28), Decimal("-500.00"), "TRANSFER"),
        TransferCandidate("t2", "credit-card", date(2026, 1, 28), Decimal("499.00"), "TRANSFER"),
    ]
    assert detect_transfer_pairs(cands) == []


def test_gap_beyond_window_is_not_a_match() -> None:
    cands = [
        TransferCandidate("t1", "checking", date(2026, 1, 1), Decimal("-500.00"), "TRANSFER"),
        TransferCandidate("t2", "credit-card", date(2026, 1, 10), Decimal("500.00"), "TRANSFER"),
    ]
    assert detect_transfer_pairs(cands) == []


def test_each_transaction_used_in_at_most_one_pair() -> None:
    cands = [
        TransferCandidate("t1", "checking", date(2026, 1, 28), Decimal("-500.00"), "TRANSFER"),
        TransferCandidate("t2", "credit-card", date(2026, 1, 28), Decimal("500.00"), "TRANSFER"),
        TransferCandidate("t3", "savings", date(2026, 1, 28), Decimal("500.00"), "TRANSFER"),
    ]
    pairs = detect_transfer_pairs(cands)
    assert len(pairs) == 1
    used = {tid for pair in pairs for tid in pair}
    assert len(used) == 2
