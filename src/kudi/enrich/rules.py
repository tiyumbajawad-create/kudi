"""The rule table: curated merchant_norm/pattern -> category mappings
for high-frequency merchants (design doc §5.2, layer 2). Deterministic
and explainable, checked before the ML classifier ever runs."""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "category_rules.yaml"


class RuleTable:
    def __init__(self, by_merchant_norm: dict[str, str], by_raw_keyword: dict[str, str]):
        self.by_merchant_norm = by_merchant_norm
        self.by_raw_keyword = {k.upper(): v for k, v in by_raw_keyword.items()}

    @classmethod
    def load(cls, path: Path | None = None) -> RuleTable:
        path = path or _DEFAULT_RULES_PATH
        data = yaml.safe_load(path.read_text())
        return cls(
            by_merchant_norm=data.get("by_merchant_norm", {}),
            by_raw_keyword=data.get("by_raw_keyword", {}),
        )

    def match(self, merchant_norm: str, raw_description: str) -> str | None:
        """Returns a category, or None if no rule fires."""
        if merchant_norm in self.by_merchant_norm:
            return self.by_merchant_norm[merchant_norm]

        raw_upper = raw_description.upper()
        for keyword, category in self.by_raw_keyword.items():
            if keyword in raw_upper:
                return category

        return None

    def covers(self, merchant_norm: str) -> bool:
        return merchant_norm in self.by_merchant_norm
