"""Capital One-style CSV: YYYY-MM-DD dates, and a bank-supplied
'Category' column that's deliberately wrong some of the time (design doc
§4.1). We keep it as a hint on the record but never treat it as ground
truth -- that's the categorizer's job in M3, and it starts from user
corrections and rules, not the bank's own guess."""

from __future__ import annotations

from kudi.ingest.parsers.base import ParseResult, RawRecord
from kudi.ingest.parsers.common import (
    AmountParseError,
    CsvStructureError,
    DateParseError,
    parse_date_yyyymmdd,
    parse_decimal,
    read_csv_rows,
)

FORMAT_NAME = "capital_one_csv"
HEADER = ["Transaction Date", "Posted Date", "Description", "Category", "Debit", "Credit"]


def parse(text: str) -> ParseResult:
    result = ParseResult()
    try:
        rows = read_csv_rows(text)
    except CsvStructureError as exc:
        result.failures.append(f"malformed CSV: {exc}")
        return result
    if not rows:
        result.failures.append("empty file")
        return result

    header, data_rows = rows[0], rows[1:]
    if [h.strip() for h in header] != HEADER:
        result.failures.append(f"unexpected header: {header}")
        return result

    for i, row in enumerate(data_rows, start=2):
        if not row or all(not cell.strip() for cell in row):
            result.skipped += 1
            continue
        if len(row) != len(HEADER):
            result.failures.append(f"row {i}: expected {len(HEADER)} columns, got {len(row)}")
            continue
        txn_date, _posted_date, description, category, debit, credit = row
        try:
            posted_date = parse_date_yyyymmdd(txn_date)
            if debit.strip():
                amount = -parse_decimal(debit)
            elif credit.strip():
                amount = parse_decimal(credit)
            else:
                raise AmountParseError("both Debit and Credit are empty")
        except (DateParseError, AmountParseError) as exc:
            result.failures.append(f"row {i}: {exc}")
            continue
        result.records.append(
            RawRecord(
                posted_date=posted_date,
                amount=amount,
                raw_description=description,
                category_hint=category.strip() or None,
            )
        )

    return result
