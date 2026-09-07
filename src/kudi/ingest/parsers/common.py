"""Shared helpers for format parsers. Dates are parsed with explicit,
format-specific strptime calls -- never dateutil guessing (design doc
§4.3), because a guessed date on money data is worse than a loud crash."""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


class DateParseError(ValueError):
    pass


class AmountParseError(ValueError):
    pass


class CsvStructureError(ValueError):
    """The file isn't even well-formed CSV (e.g. a stray bare \\r).
    Distinct from a single bad row: this means the whole file is
    unreadable as CSV, so there's nothing to salvage row-by-row."""


def decode_bytes(data: bytes) -> str:
    """UTF-8 first, falling back to cp1252 for older bank exports."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def read_csv_rows(text: str) -> list[list[str]]:
    """csv.reader can raise on structurally broken input (stray bare
    \\r, etc.) -- treat that as a loud, reportable failure rather than
    letting a crash take down the whole ingest run."""
    try:
        return list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise CsvStructureError(str(exc)) from exc


def parse_date_mmddyyyy(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").date()
    except ValueError as exc:
        raise DateParseError(f"expected MM/DD/YYYY, got {s!r}") from exc


def parse_date_ddmmyyyy(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y").date()
    except ValueError as exc:
        raise DateParseError(f"expected DD/MM/YYYY, got {s!r}") from exc


def parse_date_yyyymmdd(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise DateParseError(f"expected YYYY-MM-DD, got {s!r}") from exc


def parse_decimal(s: str) -> Decimal:
    """Strips thousands separators and whitespace. Never uses float."""
    cleaned = s.strip().replace(",", "").replace("$", "")
    if not cleaned:
        raise AmountParseError(f"empty amount field: {s!r}")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise AmountParseError(f"could not parse amount: {s!r}") from exc
