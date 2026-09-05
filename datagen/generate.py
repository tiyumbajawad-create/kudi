"""Entry point for `make data`: generate a deterministic synthetic ledger
and render it into all 5 source formats from design doc §4.1, plus a
labels.parquet carrying category/recurring/anomaly ground truth (§8).

    python -m datagen.generate --seed 42 --out data/
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import typer

from datagen.anomalies import inject_anomalies
from datagen.events import GroundTruthEvent
from datagen.formats import boa, capital_one, chase, generic_credit, ofx
from datagen.household import HouseholdProfile, simulate_household

app = typer.Typer(add_completion=False)


def _build_ledger(seed: int, months: int) -> list[GroundTruthEvent]:
    profile = HouseholdProfile(seed=seed, months=months)
    events = simulate_household(profile)

    rng = random.Random(seed ^ 0xA11CE)
    counter = {"n": len(events)}

    def event_id() -> str:
        counter["n"] += 1
        return f"anom-{counter['n']:06d}-{rng.getrandbits(24):06x}"

    events = inject_anomalies(events, rng, event_id)
    events.sort(key=lambda e: (e.account_id, e.txn_date, e.event_id))
    return events


def _write_all_formats(events: list[GroundTruthEvent], out: Path) -> None:
    by_account: dict[str, list[GroundTruthEvent]] = {}
    for e in events:
        by_account.setdefault(e.account_id, []).append(e)

    for fmt_dir, _ext in [
        ("chase_csv", "csv"),
        ("boa_csv", "csv"),
        ("capital_one_csv", "csv"),
        ("generic_credit_csv", "csv"),
        ("ofx", "ofx"),
    ]:
        (out / fmt_dir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(1)  # only used for capital_one's wrong-category-hint noise
    for account_id, acct_events in by_account.items():
        (out / "chase_csv" / f"{account_id}.csv").write_text(chase.render(acct_events))
        (out / "boa_csv" / f"{account_id}.csv").write_text(boa.render(acct_events, account_id))
        (out / "capital_one_csv" / f"{account_id}.csv").write_text(
            capital_one.render(acct_events, rng)
        )
        (out / "generic_credit_csv" / f"{account_id}.csv").write_text(
            generic_credit.render(acct_events)
        )
        (out / "ofx" / f"{account_id}.ofx").write_text(ofx.render(acct_events, account_id))


def _write_labels(events: list[GroundTruthEvent], out: Path) -> None:
    df = pd.DataFrame([e.to_label_row() for e in events])
    df.to_parquet(out / "labels.parquet", index=False)


@app.command()
def main(
    seed: int = typer.Option(42, help="RNG seed; same seed -> byte-identical output."),
    out: Path = typer.Option(Path("data"), help="Output directory root."),
    months: int = typer.Option(12, help="Months of history to simulate."),
) -> None:
    events = _build_ledger(seed=seed, months=months)
    out_dir = out / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_all_formats(events, out_dir)
    _write_labels(events, out_dir)

    n_anomaly = sum(1 for e in events if e.is_anomaly)
    n_recurring = sum(1 for e in events if e.is_recurring)
    typer.echo(
        f"Generated {len(events)} events across {len({e.account_id for e in events})} "
        f"accounts ({n_recurring} recurring, {n_anomaly} injected anomalies) -> {out_dir}"
    )


if __name__ == "__main__":
    app()
