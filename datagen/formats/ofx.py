"""OFX/QFX: SGML-ish tag soup, a stable FITID per transaction (a gift to
the dedup logic -- §4.4 uses it when present), and timezone-suffixed
datetimes (design doc §4.1)."""

from __future__ import annotations

from datagen.events import GroundTruthEvent
from datagen.formats.common import ofx_datetime

EXTENSION = "ofx"
FORMAT_NAME = "ofx"


def render(events: list[GroundTruthEvent], account_id: str) -> str:
    header = (
        "OFXHEADER:100\nDATA:OFXSGML\nVERSION:102\nSECURITY:NONE\n"
        "ENCODING:USASCII\nCHARSET:1252\nCOMPRESSION:NONE\nOLDFILEUID:NONE\n"
        "NEWFILEUID:NONE\n\n"
    )
    parts = [
        header,
        "<OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS>",
        f"<BANKACCTFROM><ACCTID>{account_id}</ACCTID></BANKACCTFROM>",
        "<BANKTRANLIST>",
    ]
    for e in events:
        trntype = "DEBIT" if e.amount < 0 else "CREDIT"
        parts.append(
            "<STMTTRN>"
            f"<TRNTYPE>{trntype}</TRNTYPE>"
            f"<DTPOSTED>{ofx_datetime(e.txn_date)}</DTPOSTED>"
            f"<TRNAMT>{e.amount:.2f}</TRNAMT>"
            f"<FITID>{e.event_id}</FITID>"
            f"<NAME>{e.raw_descriptor}</NAME>"
            "</STMTTRN>"
        )
    parts.append("</BANKTRANLIST></STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>")
    return "\n".join(parts) + "\n"
