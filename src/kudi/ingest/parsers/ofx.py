"""OFX/QFX: SGML-ish tag soup rather than well-formed XML -- tags aren't
always closed, so we extract with regex rather than an XML parser. Two
things make this format easier than the CSVs: it self-identifies its
account (ACCTID), and it carries a stable FITID that the dedup logic
prefers over the computed hash (design doc §4.4)."""

from __future__ import annotations

import re
from datetime import date, datetime

from kudi.ingest.parsers.base import ParseResult, RawRecord
from kudi.ingest.parsers.common import AmountParseError, DateParseError, parse_decimal

FORMAT_NAME = "ofx"

_ACCTID_RE = re.compile(r"<ACCTID>([^<\s]+)")
_STMTTRN_RE = re.compile(r"<STMTTRN>(.*?)</STMTTRN>", re.DOTALL)
_TAG_RE = re.compile(r"<(\w+)>([^<\r\n]*)")


def _extract_tags(block: str) -> dict[str, str]:
    return {tag: value.strip() for tag, value in _TAG_RE.findall(block)}


def _parse_ofx_date(s: str) -> date:
    digits = s.split("[")[0]  # strip the [-5:EST] timezone suffix
    return datetime.strptime(digits[:8], "%Y%m%d").date()


def parse(text: str) -> ParseResult:
    result = ParseResult()

    if "<OFX>" not in text:
        result.failures.append("no <OFX> tag found")
        return result

    acct_match = _ACCTID_RE.search(text)
    account_id = acct_match.group(1) if acct_match else None

    blocks = _STMTTRN_RE.findall(text)
    if not blocks:
        result.failures.append("no <STMTTRN> blocks found")
        return result

    for i, block in enumerate(blocks, start=1):
        tags = _extract_tags(block)
        missing = [t for t in ("DTPOSTED", "TRNAMT", "NAME") if t not in tags]
        if missing:
            result.failures.append(f"transaction {i}: missing tags {missing}")
            continue
        try:
            posted_date = _parse_ofx_date(tags["DTPOSTED"])
            amount = parse_decimal(tags["TRNAMT"])
        except (DateParseError, AmountParseError, ValueError) as exc:
            result.failures.append(f"transaction {i}: {exc}")
            continue
        result.records.append(
            RawRecord(
                posted_date=posted_date,
                amount=amount,
                raw_description=tags["NAME"],
                account_id=account_id,
                fitid=tags.get("FITID") or None,
            )
        )

    return result
