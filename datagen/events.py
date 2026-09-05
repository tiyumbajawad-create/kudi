"""The generator's internal ground-truth record.

This is deliberately *not* `kudi.schema.Transaction` -- that schema models
what a parser produces from a real file (txn_id, source_format,
ingested_at). A GroundTruthEvent models what actually happened in the
simulated household, before it's been rendered (noisily, per-format) into
any file at all. The same event gets rendered once per output format
(design doc §8), which is what makes the cross-format convergence test
meaningful: it's the same event_id underneath five different messy strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class GroundTruthEvent:
    event_id: str
    account_id: str
    txn_date: date
    amount: float  # signed; negative = outflow
    merchant_name: str
    category: str  # "Root>Leaf"
    raw_descriptor: str  # noisy string, shared across all format renders

    is_recurring: bool = False
    recurring_group_id: str | None = None
    price_hike: bool = False

    is_transfer: bool = False
    transfer_pair_event_id: str | None = None

    is_anomaly: bool = False
    anomaly_type: str | None = None

    def to_label_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "date": self.txn_date.isoformat(),
            "amount": self.amount,
            "merchant_name": self.merchant_name,
            "category": self.category,
            "is_recurring": self.is_recurring,
            "recurring_group_id": self.recurring_group_id,
            "price_hike": self.price_hike,
            "is_transfer": self.is_transfer,
            "transfer_pair_event_id": self.transfer_pair_event_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        }
