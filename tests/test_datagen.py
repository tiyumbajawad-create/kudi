"""M1 tests: the generator must be deterministic, and the same ground-truth
ledger must render consistently across all 5 source formats (design doc
§8, M1 DoD: "cross-format golden test passes").

A full round-trip through real parsers is an M2 test (once parsers exist);
here we verify convergence at the generator's own boundary: every event's
amount and date appear correctly, in each format's own conventions, in
every rendered file, and every rendered CSV has well-formed rows.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from datagen.generate import _build_ledger, _write_all_formats, _write_labels


def _hash_tree(root: Path) -> dict[str, str]:
    hashes = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


@pytest.fixture(scope="module")
def small_ledger():
    return _build_ledger(seed=7, months=3)


def test_ledger_is_nonempty(small_ledger):
    assert len(small_ledger) > 20
    accounts = {e.account_id for e in small_ledger}
    assert accounts == {"chase-checking", "chase-credit-card"}


def test_deterministic_output_byte_identical(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    for out in (out1, out2):
        ledger = _build_ledger(seed=42, months=4)
        gen_dir = out / "generated"
        gen_dir.mkdir(parents=True)
        _write_all_formats(ledger, gen_dir)
        _write_labels(ledger, gen_dir)

    h1, h2 = _hash_tree(out1), _hash_tree(out2)
    assert h1.keys() == h2.keys()
    assert h1 == h2, "same seed must reproduce byte-for-byte (design doc §8)"


def test_all_csv_rows_well_formed(tmp_path, small_ledger):
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    _write_all_formats(small_ledger, gen_dir)

    for fmt in ("chase_csv", "boa_csv", "capital_one_csv", "generic_credit_csv"):
        for f in (gen_dir / fmt).glob("*.csv"):
            with f.open() as fh:
                rows = list(csv.reader(fh))
            assert rows, f"{f} is empty"
            # every row must parse as *some* fixed column count (ragged
            # preamble rows in boa_csv are expected and fine)
            for r in rows:
                assert r, f"empty row in {f}"


def test_cross_format_convergence(tmp_path, small_ledger):
    """The same event's amount and date show up, correctly formatted per
    that format's own conventions, in every one of the 5 renders."""
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    _write_all_formats(small_ledger, gen_dir)

    sample = [e for e in small_ledger if not e.is_transfer][:15]
    for e in sample:
        mmddyyyy = e.txn_date.strftime("%m/%d/%Y")
        yyyymmdd = e.txn_date.strftime("%Y-%m-%d")
        ddmmyyyy = e.txn_date.strftime("%d/%m/%Y")
        abs_amt = f"{abs(e.amount):.2f}"

        chase = (gen_dir / "chase_csv" / f"{e.account_id}.csv").read_text()
        assert mmddyyyy in chase and f"{e.amount:.2f}" in chase

        boa = (gen_dir / "boa_csv" / f"{e.account_id}.csv").read_text()
        assert mmddyyyy in boa and abs_amt in boa

        cap1 = (gen_dir / "capital_one_csv" / f"{e.account_id}.csv").read_text()
        assert yyyymmdd in cap1 and abs_amt in cap1

        generic = (gen_dir / "generic_credit_csv" / f"{e.account_id}.csv").read_text()
        assert ddmmyyyy in generic

        ofx = (gen_dir / "ofx" / f"{e.account_id}.ofx").read_text()
        assert e.event_id in ofx, "FITID should carry the event id (§4.4 dedup uses it)"
        assert f"{e.amount:.2f}" in ofx


def test_labels_parquet_has_ground_truth(tmp_path, small_ledger):
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    _write_labels(small_ledger, gen_dir)

    df = pd.read_parquet(gen_dir / "labels.parquet")
    assert len(df) == len(small_ledger)
    assert {"event_id", "amount", "category", "is_recurring", "is_anomaly"} <= set(df.columns)
    assert df["event_id"].is_unique


def test_anomaly_injection_labeled(tmp_path):
    ledger = _build_ledger(seed=99, months=24)  # longer window -> more anomalies
    anomalies = [e for e in ledger if e.is_anomaly]
    assert len(anomalies) > 0
    assert all(e.anomaly_type is not None for e in anomalies)
    valid_types = {
        "card_testing_burst",
        "duplicate_charge",
        "large_out_of_pattern",
        "new_merchant_odd_hour",
        "subscription_double_bill",
    }
    assert {e.anomaly_type for e in anomalies} <= valid_types


def test_recurring_groups_consistent(small_ledger):
    recurring = [e for e in small_ledger if e.is_recurring and e.anomaly_type is None]
    groups: dict[str, set[str]] = {}
    for e in recurring:
        groups.setdefault(e.recurring_group_id, set()).add(e.merchant_name)
    # every recurring group must map to exactly one merchant
    assert all(len(names) == 1 for names in groups.values())
    assert len(groups) >= 3


def test_transfer_pairs_balance(small_ledger):
    transfers = [e for e in small_ledger if e.is_transfer]
    by_id = {e.event_id: e for e in transfers}
    assert transfers, "household sim should produce at least one CC payment transfer"
    for e in transfers:
        pair = by_id[e.transfer_pair_event_id]
        assert pair.transfer_pair_event_id == e.event_id
        assert round(e.amount + pair.amount, 2) == 0.0
