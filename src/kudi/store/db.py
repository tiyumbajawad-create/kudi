"""SQLite via SQLAlchemy 2.0. The ORM model mirrors `kudi.schema.Transaction`
field-for-field; swapping to Postgres later is a connection-string change,
not a rewrite (design doc §2, Appendix B #7)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class TransactionORM(Base):
    __tablename__ = "transactions"

    txn_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(String, index=True)
    posted_date: Mapped[date]
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String, default="USD")
    raw_description: Mapped[str] = mapped_column(String)

    merchant_norm: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    category_source: Mapped[str | None] = mapped_column(String, nullable=True)
    category_confidence: Mapped[float | None] = mapped_column(nullable=True)

    is_recurring: Mapped[bool] = mapped_column(default=False)
    recurring_group_id: Mapped[str | None] = mapped_column(String, nullable=True)

    anomaly_score: Mapped[float | None] = mapped_column(nullable=True)
    anomaly_reasons_json: Mapped[str] = mapped_column(String, default="[]")

    source_format: Mapped[str] = mapped_column(String)
    ingested_at: Mapped[datetime] = mapped_column(DateTime)


def get_engine(db_path: str = "kudi.db") -> Engine:
    return create_engine(f"sqlite:///{db_path}")


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
