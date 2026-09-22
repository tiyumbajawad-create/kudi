# Security

Kudi runs entirely on synthetic data in its default configuration, but
this document describes and enforces the hygiene a real-data deployment
would need (design doc §10). Most of this is enforced by tooling, not
just policy.

## Threat model

**In scope — what this repo protects against:**
- Accidental commission of real financial data into the repository or
  its history.
- Malicious or malformed input files during ingestion (a hostile CSV
  or OFX file crashing the pipeline, or being silently mis-parsed).
- Known dependency vulnerabilities (via `pip-audit` in CI and
  Dependabot).

**Out of scope:**
- A compromised host. If the machine running Kudi is compromised, no
  application-level control here protects the data on it.
- Multi-user authentication/authorization. Kudi is a single-owner
  tool; if exposed beyond localhost, auth should be added at the edge
  (a reverse proxy, an API gateway) rather than inside the app.
- Anything about the bank's own systems, APIs, or export mechanisms —
  Kudi only ever reads files the user already has.

## Local-first by default

- No telemetry, no external network calls at runtime. `make demo`
  never touches the network or costs money (design doc §13).
- The FastAPI server binds to `127.0.0.1` by default
  (`uvicorn kudi.api.app:app`), not `0.0.0.0` — it is not
  internet-facing out of the box.

## Data / code separation

- Everything under `./data/` is gitignored (`.gitignore`); nothing
  generated or ingested is meant to enter version control.
- A pre-commit hook (`.pre-commit-config.yaml`, hook id
  `no-real-data`) greps staged files for patterns that look like real
  account numbers or CSV export headers, and refuses the commit if it
  finds one. This is a heuristic, not a guarantee — it catches obvious
  mistakes, not determined misuse.
- Notebook outputs are stripped before commit (`nbstripout`), since
  notebook cell outputs are an easy, easy-to-miss place for real data
  to leak into git history.

## Ingestion treats input as hostile

- Parsers never call `eval` or execute anything from the input file.
- CSV structural errors (malformed rows, stray control characters)
  are caught and reported per-file rather than crashing the whole
  ingest run or corrupting state — see `kudi.ingest.parsers.common.
  CsvStructureError` and the property-based tests in
  `tests/test_parsers_properties.py` that specifically probe this.
- Row-level parse failures (a bad date, a bad amount) are recorded in
  the `IngestReport` with a reason, never silently dropped or guessed
  at — "failing loudly beats silently mis-parsing money data" (§4.2).
- Format detection has a confidence threshold; below it, the pipeline
  refuses to guess a format rather than risk mis-parsing an
  unrecognized file as something it isn't (§4.2).

## PII minimization

- Kudi does not read or store full account numbers, card numbers, SSNs,
  or other government-issued IDs from source files. The canonical
  schema (`kudi.schema.Transaction`) has no field for any of these —
  there is nowhere for them to end up even if a source file contained
  them.
- Account identity is a user-supplied label (`account_id`, e.g.
  `"chase-checking"`), never derived from anything in the file itself
  except for OFX's own `<ACCTID>` tag, which banks already populate
  with a masked or internal identifier, not a raw account number.

## At-rest option

- The SQLite store (`kudi.store.db`) does not encrypt data at rest by
  default, matching the demo's synthetic-data-only posture. A real
  deployment should either run SQLCipher (a drop-in encrypted SQLite
  variant) or move to a Postgres deployment with disk-level encryption
  (Appendix A's Aurora Serverless mapping already assumes SSE-KMS at
  rest).

## API posture

- No upload size limit is currently enforced at the application layer
  in the FastAPI `/ingest` endpoint; a production deployment behind a
  reverse proxy should enforce one there (e.g. nginx `client_max_body_size`).
  This is a known gap, listed here rather than silently left
  unaddressed.
- Content-type is not currently validated on upload beyond what
  `python-multipart` itself enforces; format detection (§4.2) is the
  actual safety net — an unrecognized or malicious file fails format
  detection and is refused with a report, not silently processed.

## Dependency hygiene

- `pyproject.toml` pins minimum versions; CI runs `pip-audit` on every
  push (currently informational — see `.github/workflows/ci.yml` — it
  does not yet fail the build on a finding, which should change once a
  lockfile pins exact versions).
- Dependabot should be enabled on the GitHub repository settings
  (not committed here, since that's a repo setting, not a file) to
  get automated PRs for vulnerable dependencies.

## Reporting

This is a personal portfolio project, not a maintained production
service. If you find a real security issue in the approach here (not
just "you should add X"), open a GitHub issue.
