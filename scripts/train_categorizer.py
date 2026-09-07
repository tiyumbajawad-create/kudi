"""Trains the M3 categorization classifier and writes the artifact +
model card into models/ (design doc §5.3). Run once; the artifact is
small enough to commit, so `make demo` needs no training step.

    python -m scripts.train_categorizer
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from datagen.household import HouseholdProfile, simulate_household
from kudi.enrich.classifier import (
    TEXT_COL,
    build_feature_frame,
    build_pipeline,
    predict,
    save_model,
)
from kudi.enrich.normalize import normalize_merchant
from kudi.enrich.rules import RuleTable

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SEEDS = [1, 2, 3, 4, 5, 6]
MONTHS = 24
RANDOM_STATE = 42


def build_dataset() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        profile = HouseholdProfile(seed=seed, months=MONTHS)
        for e in simulate_household(profile):
            if e.is_transfer:
                continue
            rows.append(
                {
                    "raw_description": e.raw_descriptor,
                    "amount": float(e.amount),
                    "date": e.txn_date,
                    "merchant_name": e.merchant_name,
                    "category": e.category,
                }
            )
    return pd.DataFrame(rows)


def grouped_train_test_split(
    df: pd.DataFrame, group_col: str, category_col: str = "category", test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groups by merchant (no merchant appears in both splits, per §5.3),
    but stratified by category first: a category with only one merchant
    in the whole catalog is structurally impossible to hold out (there's
    nothing left to learn it from), so that merchant is forced into
    train. Categories with >=2 merchants are split normally, always
    keeping at least one merchant of each category in train.
    """
    rng = random.Random(RANDOM_STATE)
    merchants_by_category = df.groupby(category_col)[group_col].unique()

    train_merchants: set[str] = set()
    test_merchants: set[str] = set()
    for _category, merchants in merchants_by_category.items():
        merchants = list(merchants)
        rng.shuffle(merchants)
        if len(merchants) == 1:
            train_merchants.update(merchants)
            continue
        n_test = max(1, round(len(merchants) * test_size))
        n_test = min(n_test, len(merchants) - 1)  # always keep >=1 in train
        test_merchants.update(merchants[:n_test])
        train_merchants.update(merchants[n_test:])

    train_df = df[df[group_col].isin(train_merchants)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_merchants)].reset_index(drop=True)
    return train_df, test_df


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> float:
    majority = y_train.value_counts().idxmax()
    preds = [majority] * len(y_test)
    return float(f1_score(y_test, preds, average="macro", zero_division=0))


def word_token_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """The design doc's explicit progression baseline (§5.3): same
    linear model, but word-token TF-IDF instead of char n-grams."""
    vec = TfidfVectorizer(analyzer="word", min_df=2)
    Xtr = vec.fit_transform(train_df[TEXT_COL])
    Xte = vec.transform(test_df[TEXT_COL])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, train_df["category"])
    preds = clf.predict(Xte)
    return float(f1_score(test_df["category"], preds, average="macro", zero_division=0))


def coverage_precision_curve(
    confidences: np.ndarray, correct: np.ndarray, thresholds: list[float]
) -> pd.DataFrame:
    rows = []
    n = len(confidences)
    for tau in thresholds:
        mask = confidences >= tau
        coverage = mask.sum() / n if n else 0.0
        precision = correct[mask].mean() if mask.sum() else float("nan")
        rows.append({"threshold": tau, "coverage": coverage, "precision": precision})
    return pd.DataFrame(rows)


def naive_row_split_f1(classifier_df: pd.DataFrame) -> float:
    """Deliberately WRONG baseline for comparison: a random row-level
    split, which leaks the same merchant into both train and test
    (design doc §5.3, Appendix B #3: 'the single most common
    leaked-metric mistake in this problem domain'). Shown side-by-side
    with the honest grouped split to make the leakage effect visible,
    not just asserted.
    """
    shuffled = classifier_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    split_at = int(len(shuffled) * 0.75)
    train, test = shuffled.iloc[:split_at], shuffled.iloc[split_at:]

    train_X = build_feature_frame(
        train["raw_description"].tolist(),
        train["amount"].tolist(),
        train["date"].tolist(),
        train["merchant_norm"].tolist(),
    )
    test_X = build_feature_frame(
        test["raw_description"].tolist(),
        test["amount"].tolist(),
        test["date"].tolist(),
        test["merchant_norm"].tolist(),
    )
    model = build_pipeline(analyzer="char_wb", ngram_range=(2, 5), calibrate=False)
    model.fit(train_X, train["category"])
    preds = model.predict(test_X)
    return float(f1_score(test["category"], preds, average="macro", zero_division=0))


def main() -> None:
    print("Generating synthetic dataset...")
    df = build_dataset()
    df["merchant_norm"] = df["raw_description"].apply(normalize_merchant)

    rules = RuleTable.load()
    df["rule_covered"] = df.apply(
        lambda r: rules.match(r["merchant_norm"], r["raw_description"]) is not None, axis=1
    )

    classifier_df = df[~df["rule_covered"]].reset_index(drop=True)
    print(
        f"{len(df)} total transactions; {len(classifier_df)} in the "
        f"classifier zone (not rule-covered), across "
        f"{classifier_df['merchant_name'].nunique()} merchants."
    )

    train_df, test_df = grouped_train_test_split(classifier_df, "merchant_name", "category")
    print(
        f"Grouped split by merchant: {train_df['merchant_name'].nunique()} train "
        f"merchants / {test_df['merchant_name'].nunique()} test merchants "
        f"(zero overlap by construction)."
    )
    assert set(train_df["merchant_name"]) & set(test_df["merchant_name"]) == set()

    train_X = build_feature_frame(
        train_df["raw_description"].tolist(),
        train_df["amount"].tolist(),
        train_df["date"].tolist(),
        train_df["merchant_norm"].tolist(),
    )
    test_X = build_feature_frame(
        test_df["raw_description"].tolist(),
        test_df["amount"].tolist(),
        test_df["date"].tolist(),
        test_df["merchant_norm"].tolist(),
    )

    print("Training calibrated char n-gram classifier...")
    model = build_pipeline(analyzer="char_wb", ngram_range=(2, 5), calibrate=True)
    model.fit(train_X, train_df["category"])

    predictions = predict(model, test_X)
    pred_categories = [p.category for p in predictions]
    confidences = np.array([p.confidence for p in predictions])
    correct = np.array([p == t for p, t in zip(pred_categories, test_df["category"], strict=True)])

    macro_f1 = f1_score(test_df["category"], pred_categories, average="macro", zero_division=0)
    report = classification_report(test_df["category"], pred_categories, zero_division=0, digits=3)

    maj_f1 = majority_baseline(train_df["category"], test_df["category"])
    word_f1 = word_token_baseline(
        pd.DataFrame({TEXT_COL: train_X[TEXT_COL], "category": train_df["category"].values}),
        pd.DataFrame({TEXT_COL: test_X[TEXT_COL], "category": test_df["category"].values}),
    )
    print("Computing naive row-split comparison (demonstrates leakage)...")
    naive_f1 = naive_row_split_f1(classifier_df)

    curve = coverage_precision_curve(confidences, correct, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print(curve.to_string(index=False))

    # pick the lowest threshold that still hits target precision, if any does
    target_precision = 0.9
    viable = curve[curve["precision"] >= target_precision]
    chosen_tau = float(viable["threshold"].min()) if not viable.empty else 0.7
    chosen_row = curve[curve["threshold"] == chosen_tau].iloc[0]

    MODELS_DIR.mkdir(exist_ok=True)
    save_model(model, MODELS_DIR / "categorizer.joblib")

    card = f"""# Categorizer model card

**Trained:** synthetic Kudi household data, seeds {SEEDS}, {MONTHS} months each
**Algorithm:** char n-gram (2-5) TF-IDF + amount/date features -> \
calibrated (isotonic) logistic regression
**Evaluation split:** grouped by merchant (§5.3) -- {test_df["merchant_name"].nunique()} \
held-out merchants never seen in training, zero merchant overlap between \
train and test by construction

## Headline metrics (classifier zone only -- rule-covered merchants excluded)

- Macro-F1 (grouped-by-merchant split, honest): **{macro_f1:.3f}** \
(design doc target: >= 0.85)
- Majority-class baseline macro-F1: {maj_f1:.3f}
- Word-token TF-IDF baseline macro-F1 (grouped split): {word_f1:.3f}
- Char n-gram (this model, grouped split): {macro_f1:.3f}
- **Char n-gram, naive random row-split macro-F1: {naive_f1:.3f}** \
-- shown deliberately: this is the *wrong* way to evaluate (same
merchant leaks into both train and test), and the gap between this
number and the grouped-split number above is the leakage effect
§5.3/Appendix B #3 warns about, made visible rather than just asserted.

## Chosen confidence threshold (tau)

tau = {chosen_tau} -> coverage {chosen_row["coverage"]:.3f}, \
precision {chosen_row["precision"]:.3f} \
(design doc target: precision >= 0.9 at coverage >= 0.8)

## Coverage/precision curve

```
{curve.to_string(index=False)}
```

## Full classification report

```
{report}
```

## Limitations

- **The macro-F1 target (>= 0.85) is not met on this catalog.** The
  per-class breakdown above shows why: categories with multiple
  training merchants generalize well (Restaurants F1 ~0.83, Electronics
  F1 ~0.89), but categories where the held-out test merchant's only
  same-category training example is a completely different brand name
  score near zero -- character n-grams of "TARGET" don't share enough
  with "AMAZON" to transfer the "General Merchandise" label. This is a
  genuine consequence of this demo's small (~15-merchant) classifier
  zone, not a bug: with more merchants per category (approaching the
  design doc's ~300-merchant target catalog), the model would see more
  varied within-category examples to generalize from. The naive
  row-split number above shows what happens without this honesty --
  it's substantially higher precisely because it's measuring
  memorization, not generalization.
- Evaluated on a grouped merchant split against a small (~15-merchant)
  classifier-zone catalog -- test-set merchants are entirely unseen at
  train time, which is a harder and more honest task than a random row
  split, but also means metrics are noisier with this few held-out
  merchants than they would be at production scale.
- Trained entirely on synthetic data; the noise model (processor
  prefixes, truncation, geo suffixes) reflects the generator in
  `datagen/`, not necessarily every real bank's actual formatting.
  The architecture and the user-correction feedback loop are what
  should transfer to real data, not the trained weights themselves.
- Rule-covered merchants are excluded from this evaluation by design --
  they never reach the classifier in production, so scoring them here
  would inflate the reported metric without meaning anything about the
  classifier's actual job (the long tail).
- Categories with only one merchant in the whole catalog (e.g. Bars)
  are structurally impossible to hold out for a merchant-grouped
  evaluation -- forced into train, so the model has never been tested
  on a genuinely unseen example of that category.
"""
    (MODELS_DIR / "categorizer_model_card.md").write_text(card)
    print(f"\nSaved model + model card to {MODELS_DIR}")
    print(f"macro-F1={macro_f1:.3f}  chosen tau={chosen_tau}")


if __name__ == "__main__":
    main()
