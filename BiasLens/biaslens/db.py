from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class TestItem(Base):
    """A single prompt to send to a model. Variants of one problem share `pair_id`."""
    __tablename__ = "test_items"
    __test__ = False  # stop pytest from trying to collect this ORM class as a test

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bias_type: Mapped[str] = mapped_column(String(32), index=True)
    template_id: Mapped[str] = mapped_column(String(64), index=True)
    pair_id: Mapped[str] = mapped_column(String(80), index=True)
    variant: Mapped[str] = mapped_column(String(32))
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    item_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    responses: Mapped[list["Response"]] = relationship(back_populates="item")

class Run(Base):
    """One execution of the harness for one model."""
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(80), index=True)
    label: Mapped[str] = mapped_column(String(80))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    adapter: Mapped[str] = mapped_column(String(32), default="groq")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    responses: Mapped[list["Response"]] = relationship(back_populates="run")

class Response(Base):
    """A raw model response plus its parsed fields. Everything traces back to here."""
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("test_items.id"), index=True)
    rep: Mapped[int] = mapped_column(Integer, default=0)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_s: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="responses")
    item: Mapped[TestItem] = relationship(back_populates="responses")

class Score(Base):
    """A computed bias metric for one model/run."""
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    model: Mapped[str] = mapped_column(String(80), index=True)
    bias_type: Mapped[str] = mapped_column(String(32), index=True)
    metric: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    n: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

def make_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, future=True, connect_args=connect_args)

def init_db(db_url: str):
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_sessionmaker(db_url: str):
    engine = init_db(db_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
