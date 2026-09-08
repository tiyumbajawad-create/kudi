# Kudi

> Categorizes transactions, flags fraud-like anomalies, and detects
> subscriptions from bank exports — local-first, zero-cost demo.

**Status:** M4 — anomaly + recurring detection. Design doc lives at `DESIGN.md`.

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

## Ingestion pipeline

Each of the 5 source formats has its own parser under `src/kudi/ingest/parsers/`,
converging on the canonical `Transaction` schema. The pipeline
(`src/kudi/ingest/pipeline.py`) runs: **detect → parse → dedupe → persist**.

- **Detection** (`detect.py`) sniffs an OFX tag probe first, then CSV header
  fingerprinting — exact match, falling back to per-field fuzzy matching with
  a column-count penalty. Below a confidence threshold it refuses to guess
  and reports the headers it saw, rather than silently mis-parsing money data.
- **Dedup** (`dedupe.py`) is idempotent by design: re-ingesting the same file,
  or an overlapping export window, is a no-op. FITID (when a format provides
  one, like OFX) wins over the computed content hash; a per-file occurrence
  counter keeps legitimately identical same-day transactions from colliding.
- **Store** (`src/kudi/store/`) is SQLite via SQLAlchemy, upserting on
  `txn_id` so nothing is ever double-inserted.

## Categorization

Layered, cheapest-first (`src/kudi/enrich/`): user corrections override
everything; a curated rule table (`config/category_rules.yaml`) handles
~20 high-frequency merchants deterministically; everything else falls
to a calibrated char n-gram TF-IDF classifier; below a confidence
threshold, transactions get an honest `Other>Uncategorized` rather than
a wrong confident guess.

The trained artifact and its model card live in `models/` (small enough
to commit — `make demo` needs no training step). Retrain with:

```bash
python -m scripts.train_categorizer
```

The evaluation is grouped by merchant, not by row (§5.3) — held-out test
merchants are never seen in training — and the model card documents
this honestly: on this demo's small classifier-zone catalog, the
headline macro-F1 doesn't hit the design doc's target, and the card
explains exactly why (some categories only have one same-category
example to generalize from) rather than papering over it. A deliberately
*wrong* naive row-split baseline is included side-by-side to make the
train/test leakage effect visible, not just asserted.

## Anomaly + recurring detection

`src/kudi/detect/` implements both detectors from §6-7:

- **Anomaly detection** — two layers, combined as `max(rule_score,
  forest_score)`: interpretable per-signal scores (amount outlier,
  new merchant, velocity burst, duplicate charge, geo mismatch,
  day-of-week deviation) plus an Isolation Forest over engineered
  features. Fit fresh per account at analysis time — this is
  *personal* anomaly detection, so there's no single global model to
  ship. Cold-start accounts (<30 transactions or <30 days of history)
  get a conservative subset of high-precision rules only.
- **Recurring-charge inference** — deliberately statistical, not ML:
  period-template matching (weekly/biweekly/monthly/quarterly/annual)
  combined with amount-consistency scoring. Derived alerts cover price
  hikes, missed charges, duplicate billing, and a subscription audit
  report.

Evaluated against the synthetic generator's own injected ground truth
across 5 seeds (`python -m scripts.evaluate_detectors` →
`docs/m4_eval_report.md`). Recurring detection clears every §7.3
target (precision 0.97, recall 0.88, next-date MAE 0.15 days).
Anomaly detection lands close to but just under its precision@10
target (0.57 vs. 0.6) — the eval report documents two real bugs found
and fixed while measuring this (not by inspection), and is upfront
about where the number still falls short and why, rather than only
reporting the parts that look good.

## Status / roadmap

See `DESIGN.md` §12 for the full milestone breakdown (M0–M6). This repo is
being built incrementally and in the open — commit history reflects real
build order, not a single dump.

## Design

Full design rationale, ML methodology, evaluation targets, and the AWS
deployment mapping live in [`DESIGN.md`](./DESIGN.md).
