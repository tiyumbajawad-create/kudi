"""Orchestrates the full layered categorization cascade (design doc
§5.2): user corrections always win, then the rule table, then the ML
classifier, then an honest "uncategorized" fallback below the
confidence threshold. A wrong confident label is worse than an honest
"unknown" -- so the fallback is a first-class outcome, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from kudi.enrich.classifier import build_feature_frame, predict
from kudi.enrich.normalize import normalize_merchant
from kudi.enrich.rules import RuleTable

FALLBACK_CATEGORY = "Other>Uncategorized"


@dataclass
class CategorizationResult:
    category: str
    category_source: Literal["user", "rule", "model"] | None
    category_confidence: float | None
    merchant_norm: str


class Categorizer:
    def __init__(
        self,
        rules: RuleTable,
        model: Any | None,
        confidence_threshold: float,
        corrections: dict[str, str] | None = None,
    ):
        self.rules = rules
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.corrections = corrections or {}

    def categorize_one(
        self, raw_description: str, amount: Decimal | float, posted_date: date
    ) -> CategorizationResult:
        merchant_norm = normalize_merchant(raw_description)

        if merchant_norm in self.corrections:
            return CategorizationResult(
                category=self.corrections[merchant_norm],
                category_source="user",
                category_confidence=1.0,
                merchant_norm=merchant_norm,
            )

        rule_hit = self.rules.match(merchant_norm, raw_description)
        if rule_hit is not None:
            return CategorizationResult(
                category=rule_hit,
                category_source="rule",
                category_confidence=1.0,
                merchant_norm=merchant_norm,
            )

        if self.model is not None:
            X = build_feature_frame([raw_description], [amount], [posted_date], [merchant_norm])
            result = predict(self.model, X)[0]
            if result.confidence >= self.confidence_threshold:
                return CategorizationResult(
                    category=result.category,
                    category_source="model",
                    category_confidence=result.confidence,
                    merchant_norm=merchant_norm,
                )
            return CategorizationResult(
                category=FALLBACK_CATEGORY,
                category_source="model",
                category_confidence=result.confidence,
                merchant_norm=merchant_norm,
            )

        return CategorizationResult(
            category=FALLBACK_CATEGORY,
            category_source=None,
            category_confidence=None,
            merchant_norm=merchant_norm,
        )
