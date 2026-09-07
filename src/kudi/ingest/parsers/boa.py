"""Bank of America-style CSV: several preamble lines before the real
header, then separate Debit/Credit columns instead of one signed Amount.
Skipping the preamble (never blindly assuming row 0 is the header) is
the whole point of this parser."""

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

FORMAT_NAME = "boa_csv"
HEADER = ["Date", "Description", "Debit", "Credit"]
_MAX_PREAMBLE_SCAN = 15


def parse(text: str) -> ParseResult:
    result = ParseResult()
    try:
        rows = read_csv_rows(text)
    except CsvStructureError as exc:
        result.failures.append(f"malformed CSV: {exc}")
        return result

    header_idx = None
    for i, row in enumerate(rows[:_MAX_PREAMBLE_SCAN]):
        if [c.strip() for c in row] == HEADER:
            header_idx = i
            break
    if header_idx is None:
        result.failures.append(
            f"could not find header {HEADER} in first {_MAX_PREAMBLE_SCAN} lines"
        )
        return result

    result.skipped += header_idx  # preamble lines before the header

    for i, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
        if not row or all(not cell.strip() for cell in row):
            result.skipped += 1
            continue
        if len(row) != len(HEADER):
            result.failures.append(f"row {i}: expected {len(HEADER)} columns, got {len(row)}")
            continue
        posted, description, debit, credit = row
        try:
            posted_date = parse_date_mmddyyyy(posted)
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
            RawRecord(posted_date=posted_date, amount=amount, raw_description=description)
        )

    return result
