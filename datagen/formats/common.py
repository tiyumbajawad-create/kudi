"""Shared helpers for per-format renderers."""

from __future__ import annotations

from datetime import date


def csv_quote(value: str) -> str:
    if '"' in value or "," in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def thousands(amount: float) -> str:
    """1234.5 -> '1,234.50'"""
    return f"{amount:,.2f}"


def mmddyyyy(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def yyyymmdd_dash(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def ddmmyyyy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def ofx_datetime(d: date, tz: str = "-5:EST") -> str:
    return f"{d.strftime('%Y%m%d')}120000[{tz}]"
