# Kudi — Design Document

**Personal finance transaction categorizer + anomaly detector**
Version 1.0 · August 2026 · Status: awaiting approval before build

> Working name "Kudi" — easy to rename before the repo goes public.

---

## 1. Overview

Kudi ingests bank transaction exports in inconsistent formats, normalizes them into a canonical schema, and runs three ML capabilities over them:

1. **Categorization** — NLP classification of messy merchant strings (`SQ *BLUE BOTTLE COF 4155551234 CA`) into a category taxonomy (`Food & Drink > Coffee`).
2. **Anomaly detection** — flagging transactions that are unusual for *this* account's history (amount, merchant, timing, velocity), a proxy for fraud detection.
3. **Recurring-charge inference** — detecting subscriptions and periodic bills, including ones with drifting amounts and imperfect intervals, and flagging price hikes and missed/duplicate charges.

Results are exposed through a FastAPI serving layer and a small CLI. Everything runs locally with zero paid dependencies; an appendix maps every component onto AWS services for a cloud deployment.

### 1.1 Goals

- A **portfolio-grade public GitHub repo**: anyone can `git clone`, `make demo`, and see the full pipeline run on generated data in under two minutes, with no accounts, keys, or costs.
- Demonstrate **ML depth**: proper train/eval methodology, baselines before fancy models, honest metrics, error analysis.
- Demonstrate **SWE depth**: clean architecture, typed code, tests, CI, idempotent pipelines, security thinking.
- Remain **cloud-ready**: every component behind an interface so the AWS deployment (Appendix A) is a mapping exercise, not a rewrite.

### 1.2 Non-goals (v1)

- No bank API integrations (Plaid etc.) — file ingestion only. Keeps the demo self-contained and avoids credentials entirely.
- No web frontend (v1 is API + CLI; a small dashboard is a stretch milestone M6).
- No multi-user auth/tenancy — single-owner tool; the serving layer is designed so auth can be added at the edge.
- No real-money advice; this is analysis, not a financial product.

---

## 2. Repository layout and stack

```
kudi/
├── README.md                  # pitch, demo GIF, architecture diagram, quickstart
├── DESIGN.md                  # this document
├── Makefile                   # make demo | test | lint | data | serve
├── pyproject.toml             # single source of deps + tool config
├── src/kudi/
│   ├── schema.py              # canonical models (pydantic)
│   ├── ingest/
│   │   ├── detect.py          # format sniffing
│   │   ├── parsers/           # one module per source format
│   │   └── pipeline.py        # parse → normalize → dedupe → load
│   ├── enrich/
│   │   ├── normalize.py       # merchant-string cleaning
│   │   └── categorize.py      # rules + ML classifier
│   ├── detect/
│   │   ├── anomaly.py         # anomaly scoring
│   │   └── recurring.py       # recurring-charge inference
│   ├── store/
│   │   ├── db.py              # SQLite via SQLAlchemy; Postgres-compatible
│   │   └── repository.py      # all queries live here
│   ├── api/                   # FastAPI app, routers, response models
│   └── cli.py                 # typer CLI: ingest, categorize, scan, report
├── datagen/                   # synthetic data generator (own package)
├── models/                    # trained artifacts (versioned, small) + cards
├── tests/                     # unit + integration + golden-file tests
├── notebooks/                 # EDA, error analysis (outputs stripped by CI)
└── .github/workflows/ci.yml
```

**Stack:** Python 3.12, pydantic v2, SQLAlchemy 2 + SQLite (file-based, swap-in Postgres), scikit-learn, pandas, rapidfuzz, FastAPI, typer, pytest + hypothesis, ruff + mypy (strict), GitHub Actions. No service that costs money anywhere in the default path.

Why this stack: every piece is boring, standard, and legible to a reviewer skimming the repo. The ML story carries the novelty; the engineering should look effortless.

---

## 3. Canonical data model

All parsers converge on one schema. Every downstream component consumes only this.

```python
class Transaction(BaseModel):
    txn_id: str            # deterministic hash — see §4.4
    account_id: str        # opaque label, e.g. "chase-checking"
    posted_date: date
    amount: Decimal        # signed; negative = outflow. Never float.
    currency: str          # ISO 4217, default "USD"
    raw_description: str   # untouched source string
    merchant_norm: str | None   # cleaned merchant name (§5.1)
    category: str | None        # taxonomy leaf (§5.2)
    category_source: Literal["rule", "model", "user"] | None
    category_confidence: float | None
    is_recurring: bool = False
    recurring_group_id: str | None
    anomaly_score: float | None   # 0–1, calibrated (§6)
    anomaly_reasons: list[str]    # human-readable, possibly empty
    source_format: str            # which parser produced it
    ingested_at: datetime
```

**Category taxonomy:** two-level, ~40 leaves under 12 roots (Income, Housing, Utilities, Food & Drink, Transport, Shopping, Health, Entertainment, Travel, Fees & Interest, Transfers, Other). Stored as a versioned YAML file so users can extend it; the classifier trains against the leaves. `Transfers` is critical — transfer detection (§5.4) prevents double-counting and false anomalies.

**User feedback is first-class:** a `corrections` table stores user re-categorizations keyed by `merchant_norm`. Corrections override model output on future ingests and become training data (`category_source="user"` never gets clobbered).

---

## 4. Ingestion pipeline

### 4.1 Supported source formats (v1)

Five formats, chosen to cover the real spread of messiness:

| Format | Traits it exercises |
|---|---|
| Chase-style CSV | `MM/DD/YYYY`, single signed Amount column, quoted descriptions |
| Bank of America-style CSV | separate Debit/Credit columns, header preamble lines before data |
| Capital One-style CSV | `YYYY-MM-DD`, category hints included (partially wrong on purpose) |
| Generic credit-card CSV | `DD/MM/YYYY` (!), thousands separators, trailing summary rows |
| OFX/QFX | SGML-ish tag soup, FITID present, timezone-suffixed dates |

Formats are described by name (“Chase-style”) because they mimic the *shape* of real exports without claiming to be official bank formats.

### 4.2 Format detection

`detect.py` sniffs: extension → OFX tag probe → CSV header fingerprinting (exact header-set match, then fuzzy column-name match with confidence score). Below a confidence threshold the pipeline refuses to guess and reports the headers it saw — **failing loudly beats silently mis-parsing money data**. Detection is data-driven (a registry of `FormatSpec`s), so adding format #6 is one spec + one fixture file, no code changes to the detector.

### 4.3 Parse → normalize

Each parser is a pure function `bytes → list[RawRecord]` returning source-faithful records; a shared normalizer maps `RawRecord → Transaction` handling: date parsing (format-specific, explicit — never `dateutil` guessing), amount parsing (Decimal, sign conventions per format, separator stripping), encoding fallback (UTF-8 → cp1252), and preamble/summary-row skipping. Every parser has golden-file tests: fixture in, expected canonical JSON out.

### 4.4 Idempotency and dedup

`txn_id = sha256(account_id | posted_date | amount | normalized_raw_description)[:16]`, or the FITID when the format provides one. Re-ingesting the same file, or overlapping exports (a 30-day and a 90-day export of the same account), is a no-op for existing rows — upsert on `txn_id`, never blind insert. A subtle case handled explicitly: two *legitimately identical* transactions same day (two identical coffees) collide under the hash, so the hash includes an occurrence counter computed per-file. Tests cover all three cases: exact re-ingest, overlapping window, same-day duplicates.

### 4.5 Pipeline shape

`ingest(path) → detect → parse → normalize → dedupe → persist → enrich (categorize) → detect (anomaly, recurring)`. Each stage logs counts in/out and writes a per-file `IngestReport` (rows parsed, skipped, deduped, failed — with row numbers and reasons). The report is returned by the API/CLI so ingestion is never a black box.

---

## 5. ML component 1 — merchant normalization and categorization

### 5.1 Merchant string normalization

Raw descriptors are noisy: processor prefixes (`SQ *`, `TST*`, `PAYPAL *`, `AplPay`), store numbers, phone numbers, city/state tails, truncation (`AMZN Mktp US*2K4`). Normalization is a deterministic, unit-tested cascade: uppercase → strip processor prefixes (curated list) → strip phone numbers/store IDs/trailing geo via regex → collapse whitespace → **alias resolution** against a curated merchant alias table (rapidfuzz token-set ratio ≥ 92 → canonical name). Output: `merchant_norm`. This stage alone dramatically improves everything downstream and is showcased in the README with before/after examples.

### 5.2 Categorization — layered, cheapest first

1. **User corrections** (exact `merchant_norm` match) — always win.
2. **Rule table** — curated `merchant_norm`/pattern → category for high-frequency merchants (~200 entries). Deterministic, explainable, covers a large share of real volume.
3. **ML classifier** — for everything else: TF-IDF over **character n-grams (2–5)** of the *raw* description + normalized merchant, plus lightweight non-text features (amount bucket, sign, day-of-week, is-weekend), fed to a linear model (logistic regression, calibrated with isotonic regression). Character n-grams are the right call for this domain: they are robust to truncation and store-number noise that word tokens choke on.
4. **Fallback** — below confidence threshold τ (tuned on validation, target precision ≥ 0.9 for auto-assignment): category `Other/Uncategorized`, flagged for review. **A wrong confident label is worse than an honest "unknown".**

### 5.3 Evaluation

- Split **by merchant, not by row** — random row splits leak (the same merchant appears in train and test and the metric lies). Grouped split on `merchant_norm`.
- Metrics: macro-F1 (headline, honest under class imbalance), per-class precision/recall table, coverage-vs-precision curve for the threshold choice, confusion matrix in the model card.
- Baselines reported alongside: majority-class, rules-only, word-token TF-IDF — the README shows the progression, which is the ML-maturity signal reviewers look for.
- Targets on synthetic held-out: macro-F1 ≥ 0.85; auto-assign precision ≥ 0.9 at ≥ 0.8 coverage.
- `models/` ships the trained artifact + a **model card** (data, date, metrics, limitations) — small enough to commit, so `make demo` needs no training step.

### 5.4 Transfer detection

Matched-pair heuristic: opposite-sign amounts of equal magnitude across accounts within ±3 days with transfer-ish keywords (`TRANSFER`, `ZELLE`, `PAYMENT THANK YOU`) → both legs categorized `Transfers` and excluded from spend analytics and anomaly baselines.

---

## 6. ML component 2 — anomaly detection

### 6.1 Framing

Unsupervised, per-account, cold-start-aware. This is *personal* anomaly detection: $900 rent is normal, a $900 electronics charge at 3am may not be. No labeled fraud exists at this scale, so the design is unsupervised scoring + injected-anomaly evaluation.

### 6.2 Two-layer detector

**Layer 1 — interpretable per-signal scores** (each produces a score and a reason string):

- *Amount outlier*: robust z-score (median/MAD) of |amount| within the transaction's category for this account; MAD floor prevents zero-variance blowups on constant-amount categories.
- *New merchant*: first-ever `merchant_norm`, weighted by amount (a new $4 coffee is not interesting; a new $600 charge is).
- *Velocity*: transactions per rolling 24h/7d vs account baseline (Poisson tail probability).
- *Timing*: hour-of-day/day-of-week deviation, only for merchants with enough history to have a pattern.
- *Duplicate charge*: same merchant + amount within a short window, not explained by a recurring group.
- *Geography (soft)*: mismatch of parsed geo tail vs account's dominant region — descriptor geo is unreliable, so capped weight.

**Layer 2 — Isolation Forest** over engineered features (log|amount|, category frequency rank, merchant novelty, hour, gap-since-last-txn, rolling stats) to catch *combinations* no single rule sees.

**Combination:** final score = max(rule-layer score, calibrated IF score); reasons list carries every signal that fired, plus `"unusual combination of factors"` when only the forest fires. Scores mapped to percentile-of-history so `0.98` means "more unusual than 98% of this account's transactions" — a claim a user can trust.

### 6.3 Cold start and drift

< 30 transactions or < 30 days of history for an account → conservative mode: only high-precision rules (duplicate charge, extreme absolute amount), clearly reported as "baseline still forming". Baselines computed over a rolling 12-month window so life changes (a move, a raise) age out.

### 6.4 Evaluation

The data generator injects labeled anomalies (§8) at ~0.5% prevalence. Metrics: **precision@k** (k = 10 per account per month — models the real "review a short list" workflow), PR-AUC as a secondary number, plus a breakdown by anomaly type showing which injected classes are caught and which are missed. Targets: precision@10 ≥ 0.6, recall of injected anomalies ≥ 0.7 at the alert threshold. False-positive experience matters: every alert carries its reasons; an `ack` endpoint suppresses re-alerting for an acknowledged pattern.

---

## 7. ML component 3 — recurring-charge inference

### 7.1 Algorithm

Per `(account_id, merchant_norm)` group with ≥ 3 transactions:

1. Compute inter-arrival gaps (days).
2. Match gap distribution against period templates — weekly 7±2, biweekly 14±3, monthly 28–32 (tolerating month-length wobble and "same date each month" vs "every 30 days" conventions), quarterly 91±7, annual 365±14 — scoring by fraction of gaps within tolerance.
3. Amount consistency: coefficient of variation of amounts ≤ 0.15 → fixed-price subscription; ≤ 0.5 → variable recurring bill (utilities); else not recurring on amount grounds unless period fit is very strong.
4. Combined confidence = period score × amount-consistency score, weighted by observation count; groups above threshold get a `recurring_group_id` and a predicted next date/amount range.

Deliberately statistical rather than ML — periodicity is a signal-processing problem, and the deterministic method is explainable and testable. (A comparison against an autocorrelation/FFT variant lives in a notebook as an extension.)

### 7.2 Derived alerts

Recurring groups power the highest-value outputs: **price-hike detection** (new amount exceeds trailing median by >10% and >$2), **missed charge** (predicted date passed with no transaction, useful for spotting failed payments), **duplicate subscription billing**, and a **subscription audit report** (CLI: all recurring charges, monthly total, per-service annualized cost).

### 7.3 Evaluation

Generator labels ground-truth recurring series (with jitter, price changes, skipped months). Metrics: series-level precision/recall of recurring detection (target ≥ 0.9 / ≥ 0.85), MAE of next-date prediction (target ≤ 2 days), detection rate of injected price hikes.

---

## 8. Synthetic data generator (`datagen/`)

The generator is a **first-class deliverable**, not a fixture factory — it is what makes the repo runnable by anyone and the evaluations meaningful.

- **Household simulation:** configurable profile (income, rent, habits) simulates 6–24 months across N accounts: salary deposits, rent, groceries with realistic week cycles, coffee habits, seasonal spend (holidays, summer travel), inter-account transfers.
- **Merchant realism:** ~300-merchant catalog where each merchant has multiple raw-descriptor templates reproducing real-world noise (processor prefixes, store numbers, phone tails, truncation, geo suffixes) — the same merchant renders differently across transactions, which is exactly what makes categorization hard.
- **Recurring series:** subscriptions and bills with date jitter, occasional skips, and scheduled price hikes — all recorded as ground truth.
- **Anomaly injection:** labeled scenarios at configurable prevalence — card-testing bursts (several small charges in minutes then a large one), duplicate charges, out-of-pattern large purchases, new-merchant + odd-hour combos, subscription double-bills.
- **Output:** renders the *same* ground-truth ledger into all five source formats of §4.1 (which simultaneously tests that parsers converge to identical canonical rows — a golden cross-format test), plus a `labels.parquet` with category, recurring, and anomaly ground truth.
- **Deterministic:** seeded; `make data SEED=42` reproduces the dataset byte-for-byte. Ground-truth labels are never visible to the models except through eval harnesses.

Honest limitation, stated in the README: models trained on synthetic data reflect the generator's noise model; the architecture (and the user-correction loop) is what transfers to real data. This candor is itself a portfolio signal.

---

## 9. Serving layer

FastAPI app (`make serve`), OpenAPI docs auto-generated:

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | multipart file upload → `IngestReport` |
| `GET /transactions` | filter by account/date/category/min-anomaly-score; paginated |
| `POST /transactions/{id}/category` | user correction (recorded, future-proof) |
| `GET /anomalies` | ranked alerts with reasons; `POST /anomalies/{id}/ack` |
| `GET /recurring` | subscription audit incl. price-hike/missed-charge alerts |
| `GET /insights/summary` | monthly totals by category, trends |
| `GET /healthz`, `GET /version` | ops hygiene |

Design notes: response models are separate pydantic classes (no ORM leakage); category corrections and acks are the two write paths besides ingest; the model registry loads versioned artifacts at startup and `/version` reports model versions — small touches that read as production maturity. The CLI (`kudi ingest|report|anomalies|subscriptions`) drives the same service layer, so logic is never duplicated.

---

## 10. Privacy and security design

Even though the demo runs on synthetic data, the repo documents and enforces real-data hygiene — this section is a differentiator, written up in `SECURITY.md`:

- **Local-first:** no telemetry, no external calls at runtime. The default path never sends a byte off-machine.
- **Data/code separation:** all data lives under `./data/` (gitignored); a pre-commit hook + CI check greps staged files for account-number and CSV-export patterns to make committing real data hard. Notebook outputs stripped by pre-commit (`nbstripout`).
- **At-rest option:** `LEDGERLENS_DB_KEY` env var enables SQLCipher for the SQLite store; documented, off by default for demo simplicity.
- **PII minimization:** account numbers, if present in source files, are masked to last-4 at parse time *before* persistence; raw source files are read, reported on, and never copied into the store.
- **API posture:** binds to `127.0.0.1` by default; upload size limits; content-type validation; parsers treat input as hostile (no eval of anything, bounded row counts, decompression limits n/a since archives are refused).
- **Dependency hygiene:** pinned lockfile, `pip-audit` in CI, Dependabot on.
- **Threat model section** in SECURITY.md: what's protected against (accidental data leakage to the repo, malicious input files, dependency CVEs), what's out of scope (a compromised host).

---

## 11. Testing and CI

- **Unit tests** per module; **property-based tests** (hypothesis) for parsers (random valid amounts/dates/encodings never crash and round-trip correctly) and for the dedup hash.
- **Golden-file tests**: each format fixture → expected canonical JSON; cross-format convergence test from §8.
- **Integration test**: generate data → ingest all formats → run all three ML components → assert eval metrics above floor thresholds (catches silent model regressions in CI, not just crashes).
- **API tests** via httpx against the app in-process.
- **CI (GitHub Actions)**: ruff (lint+format) → mypy --strict → pytest with coverage gate ≥ 90% on `src/` → pip-audit → the integration metric floor. Badges in README.
- **Pre-commit**: ruff, mypy, nbstripout, the real-data grep from §10.

---

## 12. Milestones

| # | Milestone | Contents | Definition of done |
|---|---|---|---|
| M0 | Skeleton | repo layout, pyproject, CI green on hello-world test, pre-commit | `make test` green in CI |
| M1 | Data generator | household sim, merchant noise, 5 output formats, ground-truth labels, seeded | cross-format golden test passes |
| M2 | Ingestion | detection, 5 parsers, normalization, dedup, SQLite store, IngestReport | overlapping re-ingest is a no-op; property tests pass |
| M3 | Categorization | normalization cascade, rules, classifier, calibration, model card | metric targets of §5.3 met on held-out |
| M4 | Anomaly + recurring | both detectors, injected-anomaly eval, subscription audit | metric targets of §6.4 and §7.3 met |
| M5 | Serving + polish | FastAPI, CLI, README with GIF + architecture diagram, SECURITY.md, model cards | `make demo` end-to-end < 2 min on a clean clone |
| M6 | Stretch | tiny dashboard (htmx or single-page), AWS deploy of Appendix A, DistilBERT categorizer comparison notebook | optional |

Each milestone lands as a reviewed PR on GitHub (even solo — self-reviewed PRs with real descriptions are themselves portfolio evidence of process).

---

## 13. Quality bar ("flawless" operationalized)

- Zero mypy --strict errors; zero ruff violations; ≥ 90% coverage on `src/`.
- Every number claimed in the README is reproducible by a make target.
- Every model ships with a model card stating its limitations.
- No path in `make demo` touches the network or costs money.
- A stranger can go clone → demo → understand the architecture from the README diagram in under five minutes.
- No silent failure anywhere in ingestion: every dropped row is accounted for in an IngestReport.

---

## Appendix A — AWS deployment mapping (cert-relevant)

Documented (and optionally deployed as M6) as `docs/aws.md` with an architecture diagram:

| Local component | AWS service | Notes |
|---|---|---|
| File drop / ingestion trigger | S3 + EventBridge → Lambda | S3 event kicks off ingest; SSE-KMS encryption at rest |
| Ingestion pipeline | Lambda (or Step Functions if stages split) | container image Lambda reusing the same package |
| Store | Aurora Serverless v2 Postgres (or DynamoDB variant discussion) | SQLAlchemy makes Postgres a connection-string change |
| Model artifacts | S3 + versioning | model registry reads from S3 |
| Batch scoring | Lambda / SageMaker Processing for retraining | scikit-learn models are Lambda-sized |
| API | API Gateway + Lambda (Mangum) or ECS Fargate | trade-off discussion included |
| Secrets | Secrets Manager + IAM least-privilege roles | maps to §10 |
| Observability | CloudWatch logs/metrics/alarms, structured logging | IngestReport metrics as custom metrics |
| IaC | Terraform (or CDK) in `infra/` | cert-relevant either way |

The doc discusses cost (fits in free tier for demo volumes), VPC posture, and why Lambda over SageMaker endpoints at this scale — the *reasoning* is the cert-prep value.

---

## Appendix B — Key design decisions (summary for the README)

1. Character n-gram TF-IDF + linear model over a transformer for v1: robust to descriptor noise, milliseconds to score, trivially deployable; transformer comparison is a notebook, not a dependency.
2. Rules before ML, corrections before rules: cheapest sufficient layer wins; everything explainable is explained.
3. Grouped-by-merchant eval splits: the single most common leaked-metric mistake in this problem domain, avoided and documented.
4. Unsupervised anomaly detection with injected-anomaly evaluation: honest framing for a domain with no labels.
5. Statistical recurring detection over ML: right tool, explainable, testable.
6. Synthetic-data-first: makes the repo runnable by anyone, evals meaningful, and privacy risk zero — with limitations stated plainly.
7. SQLite → Postgres and local → AWS as documented seams, not rewrites.
