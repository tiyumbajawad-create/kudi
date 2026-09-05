# Kudi

> Categorizes transactions, flags fraud-like anomalies, and detects
> subscriptions from bank exports — local-first, zero-cost demo.

**Status:** M1 — synthetic data generator. Design doc lives at `DESIGN.md`.

## What this is

Kudi ingests bank transaction exports in inconsistent formats, normalizes
them into a canonical schema, and runs three ML capabilities over them:

1. **Categorization** of messy merchant strings into a category taxonomy.
2. **Anomaly detection** — flags transactions unusual for *this* account's
   own history (a proxy for fraud detection).
3. **Recurring-charge inference** — detects subscriptions and periodic
   bills, including price hikes and missed/duplicate charges.

Everything runs locally on synthetic data with zero paid dependencies. An
appendix in `DESIGN.md` maps every component onto AWS services for a cloud
deployment.

## Quickstart

```bash
make install   # install package + dev deps
make data      # generate synthetic household data (seeded, reproducible)
make demo      # ingest -> categorize -> detect anomalies -> report
make test      # run the test suite
make serve     # start the FastAPI serving layer
```

## Synthetic data generator

`make data` (or `python -m datagen.generate --seed 42 --out data/`) simulates
a household's transaction history across a checking and a credit-card
account, then renders the *same* ground-truth ledger into all 5 source
formats the ingestion pipeline is designed to handle:

- Chase-style CSV
- Bank of America-style CSV (header preamble, separate debit/credit columns)
- Capital One-style CSV (partially-wrong bank-supplied category hints)
- Generic credit-card CSV (DD/MM/YYYY dates, thousands separators, a trailing
  summary row)
- OFX/QFX (tag soup, stable FITID, timezone-suffixed dates)

It also injects labeled anomaly scenarios (card-testing bursts, duplicate
charges, out-of-pattern purchases, subscription double-bills) and labeled
recurring-charge series with jitter, skipped cycles, and price hikes —
all recorded in `labels.parquet` so later milestones have honest ground
truth to evaluate against. Generation is seeded and fully deterministic:
`make data SEED=42` reproduces the dataset byte-for-byte.

## Status / roadmap

See `DESIGN.md` §12 for the full milestone breakdown (M0–M6). This repo is
being built incrementally and in the open — commit history reflects real
build order, not a single dump.

## Design

Full design rationale, ML methodology, evaluation targets, and the AWS
deployment mapping live in [`DESIGN.md`](./DESIGN.md).
