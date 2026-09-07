"""Orchestrates one file through the full ingestion pipeline (§4.5):
detect -> parse -> normalize -> dedupe -> persist. Every stage's counts
land in the returned IngestReport so ingestion is never a black box.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from kudi.ingest.dedupe import compute_txn_id
from kudi.ingest.detect import detect_format
from kudi.ingest.parsers import boa, capital_one, chase, generic_credit, ofx
from kudi.ingest.parsers.base import ParseResult
from kudi.ingest.parsers.common import decode_bytes
from kudi.schema import IngestReport, Transaction
from kudi.store.repository import upsert_transactions

PARSERS: dict[str, Callable[[str], ParseResult]] = {
    "chase_csv": chase.parse,
    "boa_csv": boa.parse,
    "capital_one_csv": capital_one.parse,
    "generic_credit_csv": generic_credit.parse,
    "ofx": ofx.parse,
}


def ingest_file(
    path: Path,
    session: Session,
    account_id: str | None = None,
    format_override: str | None = None,
) -> IngestReport:
    """Ingest one file. `account_id` is required for every CSV format
    (they don't self-identify their account); OFX carries its own ACCTID
    and `account_id` there is only used to override it."""
    report = IngestReport(source_file=str(path))

    text = decode_bytes(path.read_bytes())

    if format_override:
        fmt = format_override
    else:
        detection = detect_format(text)
        if detection.format_name is None:
            report.rows_failed += 1
            report.failures.append(
                "could not confidently detect a known format "
                f"(best confidence {detection.confidence:.2f}); "
                f"headers seen: {detection.headers_seen}"
            )
            return report
        fmt = detection.format_name

    parser = PARSERS.get(fmt)
    if parser is None:
        report.rows_failed += 1
        report.failures.append(f"no parser registered for format {fmt!r}")
        return report

    parsed = parser(text)
    report.rows_skipped += parsed.skipped
    report.rows_failed += len(parsed.failures)
    report.failures.extend(parsed.failures)
    report.rows_parsed = len(parsed.records)

    occurrence_counts: dict[tuple[str, str, str, str], int] = {}
    transactions: list[Transaction] = []
    now = datetime.now(UTC)

    for i, rec in enumerate(parsed.records):
        acct = rec.account_id or account_id
        if acct is None:
            report.rows_failed += 1
            report.rows_parsed -= 1
            report.failures.append(
                f"record {i}: no account_id -- {fmt} does not self-identify its "
                "account; pass account_id explicitly"
            )
            continue

        key = (acct, rec.posted_date.isoformat(), str(rec.amount), rec.raw_description)
        occurrence = occurrence_counts.get(key, 0)
        occurrence_counts[key] = occurrence + 1

        txn_id = compute_txn_id(
            acct,
            rec.posted_date,
            rec.amount,
            rec.raw_description,
            fitid=rec.fitid,
            occurrence=occurrence,
        )
        transactions.append(
            Transaction(
                txn_id=txn_id,
                account_id=acct,
                posted_date=rec.posted_date,
                amount=rec.amount,
                raw_description=rec.raw_description,
                source_format=fmt,
                ingested_at=now,
            )
        )

    _inserted, already_present = upsert_transactions(session, transactions)
    report.rows_deduped = already_present
    return report
