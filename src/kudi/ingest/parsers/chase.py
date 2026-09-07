"""Chase-style CSV: 'Details,Posting Date,Description,Amount,Type'
header, MM/DD/YYYY dates, one signed Amount column. The cleanest of the
five formats -- no preamble, no summary rows."""

from __future__ import annotations

from kudi.ingest.parsers.base import ParseResult, RawRecord
from kudi.ingest.parsers.common import (
    AmountParseError,
    CsvStructureError,
    DateParseError,
    parse_date_mmddyyyy,
    parse_decimal,
    read_csv_rows,
)

FORMAT_NAME = "chase_csv"
HEADER = ["Details", "Posting Date", "Description", "Amount", "Type"]


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
        _details, posting_date, description, amount, _txn_type = row
        try:
            record = RawRecord(
                posted_date=parse_date_mmddyyyy(posting_date),
                amount=parse_decimal(amount),
                raw_description=description,
            )
        except (DateParseError, AmountParseError) as exc:
            result.failures.append(f"row {i}: {exc}")
            continue
        result.records.append(record)

    return result
