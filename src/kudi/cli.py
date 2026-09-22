"""Kudi CLI: ingest|report|anomalies|subscriptions, driving the same
service layer as the FastAPI app (design doc §9) so logic is never
duplicated between the two."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from sqlalchemy.orm import Session

from kudi import service
from kudi.store.db import create_all, get_engine, get_session_factory
from kudi.store.repository import get_anomalies, get_transactions

app = typer.Typer(
    help="Kudi: transaction categorization, anomaly detection, and recurring-charge inference."
)


def _get_session(db: Path) -> Session:
    engine = get_engine(str(db))
    create_all(engine)
    return get_session_factory(engine)()


@app.command()
def version() -> None:
    """Print the installed Kudi version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("kudi"))


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="One or more transaction export files."),
    db: Path = typer.Option(Path("kudi.db"), help="SQLite database path."),
) -> None:
    """Ingest one or more files, then run the full enrichment pipeline
    (categorize -> detect transfers -> detect recurring -> detect
    anomalies) across every account in the store."""
    session = _get_session(db)

    missing = [p for p in paths if not p.exists()]
    if missing:
        typer.echo(f"File(s) not found: {missing}", err=True)
        raise typer.Exit(code=1)

    reports = service.ingest_paths(session, paths)
    for r in reports:
        status = "OK" if not r.failures else f"{len(r.failures)} issue(s)"
        typer.echo(
            f"{Path(r.source_file).name}: parsed={r.rows_parsed} "
            f"skipped={r.rows_skipped} deduped={r.rows_deduped} "
            f"failed={r.rows_failed} [{status}]"
        )
        for f in r.failures[:5]:
            typer.echo(f"  - {f}")

    summary = service.enrich_all(session)
    typer.echo(
        f"\nEnriched {summary['transactions']} transactions across "
        f"{summary['accounts']} account(s): {summary['transfers']} transfer "
        f"legs, {summary['recurring_groups']} recurring group(s) detected."
    )


@app.command()
def report(
    account_id: str = typer.Option(..., help="Account to summarize."),
    db: Path = typer.Option(Path("kudi.db"), help="SQLite database path."),
) -> None:
    """Monthly spend summary by category for one account."""
    session = _get_session(db)
    if not get_transactions(session, account_id=account_id):
        typer.echo(f"No transactions found for account {account_id!r}.")
        raise typer.Exit(code=0)

    totals = service.monthly_summary(session, account_id)

    for month in sorted(totals):
        typer.echo(f"\n{month}")
        for category, amount in sorted(totals[month].items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {category:<35} ${amount:>10,.2f}")


@app.command()
def anomalies(
    account_id: str = typer.Option(..., help="Account to check."),
    min_score: float = typer.Option(0.9, help="Minimum anomaly score (0-1) to show."),
    limit: int = typer.Option(20, help="Max alerts to show."),
    db: Path = typer.Option(Path("kudi.db"), help="SQLite database path."),
) -> None:
    """Ranked anomaly alerts, highest score first."""
    session = _get_session(db)
    rows = get_anomalies(session, account_id=account_id, min_score=min_score, limit=limit)
    if not rows:
        typer.echo("No anomalies at or above this score threshold.")
        raise typer.Exit(code=0)

    for r in rows:
        reasons = json.loads(r.anomaly_reasons_json)
        typer.echo(
            f"\n[{r.anomaly_score:.2f}] {r.posted_date} {r.merchant_norm} "
            f"${abs(float(r.amount)):.2f}"
        )
        for reason in reasons:
            typer.echo(f"    - {reason}")


@app.command()
def subscriptions(
    account_id: str = typer.Option(..., help="Account to audit."),
    db: Path = typer.Option(Path("kudi.db"), help="SQLite database path."),
) -> None:
    """Subscription audit: recurring charges, plus price-hike, missed-
    charge, and duplicate-billing alerts."""
    session = _get_session(db)
    alerts = service.get_recurring_alerts(session, account_id)

    audit: list[Any] = alerts["audit"]
    if not audit:
        typer.echo(f"No recurring charges detected for account {account_id!r}.")
    else:
        typer.echo("Recurring charges:")
        total_monthly = 0.0
        for row in audit:
            typer.echo(
                f"  {row.merchant_norm:<25} {row.period_name:<10} "
                f"${row.monthly_equivalent:>8.2f}/mo  (${row.annualized_cost:,.2f}/yr, "
                f"confidence {row.confidence:.2f})"
            )
            total_monthly += row.monthly_equivalent
        typer.echo(f"\n  Total: ${total_monthly:,.2f}/mo (${total_monthly * 12:,.2f}/yr)")

    if alerts["price_hikes"]:
        typer.echo("\nPrice hikes:")
        for h in alerts["price_hikes"]:
            typer.echo(
                f"  {h.merchant_norm}: ${h.old_amount:.2f} -> ${h.new_amount:.2f} "
                f"(+{h.pct_increase:.0%}) on {h.detected_date}"
            )

    if alerts["missed_charges"]:
        typer.echo("\nMissed charges:")
        for m in alerts["missed_charges"]:
            typer.echo(
                f"  {m.merchant_norm}: expected {m.expected_date}, {m.days_overdue} day(s) overdue"
            )

    if alerts["duplicate_billing"]:
        typer.echo("\nPossible duplicate billing:")
        for d in alerts["duplicate_billing"]:
            typer.echo(
                f"  {d.merchant_norm}: ${d.amount:.2f} on both {d.first_date} and {d.second_date}"
            )


if __name__ == "__main__":
    app()
