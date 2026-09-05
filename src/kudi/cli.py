"""Kudi CLI. Real subcommands (ingest, report, anomalies, subscriptions)
land in M2-M5; this stub exists so the package is importable/runnable
from M0 onward."""

import typer

app = typer.Typer(
    help="Kudi: transaction categorization, anomaly detection, and recurring-charge inference."
)


@app.command()
def version() -> None:
    """Print the installed Kudi version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("kudi"))


if __name__ == "__main__":
    app()
