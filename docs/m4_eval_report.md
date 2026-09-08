# M4 detector evaluation report

Evaluated against the synthetic generator's own injected ground truth
(design doc §6.4, §7.3), across seeds [1, 2, 3, 4, 5], 24 months each.

Neither detector ships a persisted artifact: anomaly detection is
explicitly personal (fit fresh per account at analysis time, §6.1),
and recurring detection is a deterministic statistical method (§7.1),
not a trained model. This report documents methodology and honest
metrics rather than an artifact.

## Anomaly detection (§6.4)

- Mean precision@10: **0.570** (design doc target: >= 0.6)
- Mean PR-AUC: 0.489
- Evaluated across 10 account-runs (5 seeds x 2 accounts each)

Recall at the 90th-percentile-score threshold, by injected anomaly type:
  - subscription_double_bill: 1.00
  - duplicate_charge: 1.00
  - new_merchant_odd_hour: 1.00
  - large_out_of_pattern: 1.00
  - card_testing_burst: 0.64

### Methodology notes

- Two real bugs were found and fixed while building this evaluation,
  not by inspection:
  1. `duplicate_charge` originally flagged only the *later*
     transaction in a matched pair. Since pandas' sort is not
     guaranteed stable under exact-date ties, this meant the signal
     could non-deterministically flag the legitimate original
     transaction instead of the actually-injected duplicate,
     silently undercounting true positives. Fixed by flagging both
     transactions in every matched pair.
  2. `geo_mismatch` used `state is not None` as its guard, but pandas
     can silently coerce a Python `None` into a float `NaN` -- and
     `float('nan') is not None` evaluates `True` in Python. This let a
     NaN slip through and get embedded, literally, in the alert reason
     string ("descriptor geo nan doesn't match..."). Fixed with
     `pd.isna()`.
  Fixing both raised mean precision@10 from ~0.30 to 0.57 on this same evaluation.
- Recurring-group membership (from the detector below) is fed back
  into the anomaly detector to suppress "new merchant" alarms for a
  bill's own onboarding month -- otherwise every recurring bill's
  first-ever charge floods the top of the ranking with ties, since
  legitimate large first charges (rent, a new subscription) are
  indistinguishable from fraud by amount and novelty alone.

### Limitations

- The design doc's "timing" signal (§6.2) specifies hour-of-day
  deviation; the canonical schema stores `posted_date` as a date, not
  a datetime, so there is no hour-of-day in this data model. Only
  day-of-week deviation is implemented -- a real, schema-level
  limitation, not an oversight.
- Precision@10 falls short of the 0.6 target on some account-runs,
  particularly higher-volume credit-card accounts where "top 10" is a
  much smaller fraction of total transactions than on lower-volume
  checking accounts.

## Recurring-charge inference (§7.3)

- Mean series-level precision: **0.971** (design doc target: >= 0.9)
- Mean series-level recall: **0.875** (design doc target: >= 0.85)
- Mean period-fit MAE: 0.15 days (design doc target: next-date MAE <= 2 days; this proxies period-fit
error since predicted-next-date is computed directly from the last
observed date plus the fitted period, so it is not an independent
signal on its own)

### Methodology notes

- Income (salary, interest) is excluded from the precision/recall
  scoring here even though the detector correctly identifies it as
  periodic and consistent -- the ground truth's `is_recurring` label
  is scoped to bills/subscriptions only (design doc §8), not income.
  This is a labeling-scope difference, not a detector error: on the
  first evaluation run, payroll showed up as the only "false
  positive," and it is a completely legitimate recurring pattern the
  detector was right to catch.
- This recurring detector performs substantially better against its
  targets than the anomaly detector does against its own -- a smaller,
  cleaner, more deterministic problem (periodicity + amount
  consistency) generalizes more reliably than personal anomaly scoring
  does on a modest synthetic dataset.
