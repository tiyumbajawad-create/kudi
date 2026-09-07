"""Layer 3 of categorization (design doc §5.2): for everything the
rules table doesn't cover, TF-IDF over character n-grams (2-5) of the
raw description + normalized merchant, plus a few cheap non-text
features, fed to a calibrated logistic regression.

Character n-grams are the right call here (over word tokens): they're
robust to truncation and store-number noise that word tokenization
chokes on -- "CHIPOTLE 60" and "CHIPOTLE 7058" share almost no *words*
in common once the numbers vary, but share plenty of character
n-grams.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kudi.enrich.features import NUMERIC_FEATURE_NAMES, numeric_features
from kudi.enrich.normalize import clean_descriptor, normalize_merchant

TEXT_COL = "text"


def build_text_feature(raw_description: str, merchant_norm: str) -> str:
    return f"{clean_descriptor(raw_description)} {merchant_norm}".strip()


def build_feature_frame(
    raw_descriptions: list[str],
    amounts: list[Decimal | float],
    dates: list[date],
    merchant_norms: list[str] | None = None,
) -> pd.DataFrame:
    if merchant_norms is None:
        merchant_norms = [normalize_merchant(r) for r in raw_descriptions]

    rows: list[dict[str, str | float]] = []
    for raw, norm, amount, d in zip(raw_descriptions, merchant_norms, amounts, dates, strict=True):
        row: dict[str, str | float] = {TEXT_COL: build_text_feature(raw, norm)}
        row.update(dict(zip(NUMERIC_FEATURE_NAMES, numeric_features(amount, d), strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_preprocessor(
    analyzer: str = "char_wb", ngram_range: tuple[int, int] = (2, 5)
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2),
                TEXT_COL,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURE_NAMES),
        ]
    )


def build_pipeline(
    analyzer: str = "char_wb", ngram_range: tuple[int, int] = (2, 5), calibrate: bool = True
) -> Pipeline | CalibratedClassifierCV:
    pipeline = Pipeline(
        [
            ("features", _make_preprocessor(analyzer, ngram_range)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    if not calibrate:
        return pipeline
    return CalibratedClassifierCV(pipeline, method="isotonic", cv=3)


@dataclass
class PredictionResult:
    category: str
    confidence: float


def predict(model: Any, X: pd.DataFrame) -> list[PredictionResult]:
    proba = model.predict_proba(X)
    classes = model.classes_
    best_idx = np.argmax(proba, axis=1)
    return [
        PredictionResult(category=str(classes[i]), confidence=float(proba[row, i]))
        for row, i in enumerate(best_idx)
    ]


def save_model(model: Any, path: Path) -> None:
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)
