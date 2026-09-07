"""Tests for the classifier module, plus a check that the committed
artifact in models/ actually loads and works (design doc §13: 'every
number claimed... is reproducible')."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from kudi.enrich.categorize import Categorizer
from kudi.enrich.classifier import build_feature_frame, build_pipeline, load_model, predict
from kudi.enrich.rules import RuleTable

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_pipeline_trains_and_predicts_on_toy_data() -> None:
    raws = ["DoorDash DENVER CO"] * 6 + ["Uber TAMPA FL"] * 6
    amounts = [Decimal("-20.00")] * 12
    dates = [date(2026, 1, 1)] * 12
    labels = ["Food & Drink>Restaurants"] * 6 + ["Transport>Rideshare"] * 6

    X = build_feature_frame(raws, amounts, dates)
    model = build_pipeline(calibrate=False)
    model.fit(X, labels)

    preds = predict(model, X)
    assert len(preds) == 12
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
    # should trivially separate two obviously distinct merchants
    n_correct = sum(p.category == label for p, label in zip(preds, labels, strict=True))
    accuracy = n_correct / len(labels)
    assert accuracy == 1.0


def test_build_feature_frame_shape() -> None:
    X = build_feature_frame(
        ["KROGER #123", "SHELL OIL 4690"],
        [Decimal("-45.00"), Decimal("-30.00")],
        [date(2026, 1, 1), date(2026, 1, 2)],
    )
    assert len(X) == 2
    assert "text" in X.columns
    assert "amount_bucket" in X.columns


@pytest.mark.skipif(
    not (MODELS_DIR / "categorizer.joblib").exists(),
    reason="trained artifact not present (run scripts/train_categorizer.py)",
)
def test_committed_model_artifact_loads_and_predicts() -> None:
    model = load_model(MODELS_DIR / "categorizer.joblib")
    X = build_feature_frame(
        ["Some Novel Restaurant Chain #4821"],
        [Decimal("-25.00")],
        [date(2026, 3, 15)],
    )
    preds = predict(model, X)
    assert len(preds) == 1
    assert 0.0 <= preds[0].confidence <= 1.0
    assert isinstance(preds[0].category, str)


@pytest.mark.skipif(
    not (MODELS_DIR / "categorizer.joblib").exists(),
    reason="trained artifact not present (run scripts/train_categorizer.py)",
)
def test_full_categorizer_with_committed_model() -> None:
    model = load_model(MODELS_DIR / "categorizer.joblib")
    cat = Categorizer(rules=RuleTable.load(), model=model, confidence_threshold=0.5)

    # a rule-covered merchant should never touch the model
    rule_result = cat.categorize_one("KROGER #123", Decimal("-45.00"), date(2026, 1, 15))
    assert rule_result.category_source == "rule"

    # a classifier-zone merchant should get a model verdict (possibly
    # the honest fallback, but always source="model")
    model_result = cat.categorize_one("Amazon SEATTLE WA", Decimal("-60.00"), date(2026, 3, 1))
    assert model_result.category_source == "model"
