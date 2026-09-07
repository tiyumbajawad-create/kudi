"""Generic credit-card CSV: DD/MM/YYYY (the classic date-ambiguity trap
-- if this were parsed as MM/DD it would silently misfile a 13th-of-the-
month cross with a US-format file), thousands separators in amounts, and
a trailing summary row that isn't a transaction at all (design doc
§4.1)."""

from __future__ import annotations

from kudi.ingest.parsers.base import ParseResult, RawRecord
from kudi.ingest.parsers.common import (
    AmountParseError,
    CsvStructureError,
    DateParseError,
    parse_date_ddmmyyyy,
    parse_decimal,
    read_csv_rows,
)

FORMAT_NAME = "generic_credit_csv"
HEADER = ["Date", "Description", "Amount", "Type"]


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
        if row[0].strip().upper() == "TOTAL":
            result.skipped += 1  # trailing summary row, not a transaction
            continue
        if len(row) != len(HEADER):
            result.failures.append(f"row {i}: expected {len(HEADER)} columns, got {len(row)}")
            continue
        posted, description, amount_str, kind = row
        try:
            posted_date = parse_date_ddmmyyyy(posted)
            magnitude = parse_decimal(amount_str)
            amount = -magnitude if kind.strip().upper() == "DEBIT" else magnitude
        except (DateParseError, AmountParseError) as exc:
            result.failures.append(f"row {i}: {exc}")
            continue
        result.records.append(
            RawRecord(posted_date=posted_date, amount=amount, raw_description=description)
        )

    return result
