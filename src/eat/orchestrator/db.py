"""SQLAlchemy models for runs and jobs.

Single SQLite DB by default (`./eat.sqlite`); override via EAT_DB_URL. We keep
the schema tiny — most rich data lives in the `params`/`metrics`/`artifacts`
JSON blobs so we don't have to migrate every time we add a field.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from eat.config import get_settings


class Base(DeclarativeBase):
    pass


class RunModel(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project: Mapped[str] = mapped_column(String(128), index=True)
    recipe: Mapped[str] = mapped_column(String(128))
    targets_json: Mapped[str] = mapped_column(String, default="[]")
    compute: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), index=True)
    notes: Mapped[str | None] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    jobs: Mapped[list["JobModel"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class JobModel(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    project: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    compute: Mapped[str] = mapped_column(String(32))
    queue: Mapped[str | None] = mapped_column(String(64), default=None)
    rq_job_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    deps_json: Mapped[str] = mapped_column(String, default="[]")
    params_json: Mapped[str] = mapped_column(String, default="{}")
    artifacts_json: Mapped[str] = mapped_column(String, default="{}")
    metrics_json: Mapped[str] = mapped_column(String, default="[]")
    logs_path: Mapped[str | None] = mapped_column(String, default=None)
    error: Mapped[str | None] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    run: Mapped[RunModel] = relationship(back_populates="jobs")


# ---------------------------------------------------------------------------
# session helpers
# ---------------------------------------------------------------------------

_engine = None
_Session: sessionmaker | None = None


def _get_engine():
    global _engine, _Session
    if _engine is None:
        s = get_settings()
        connect_args: dict[str, Any] = {}
        if s.db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(s.db_url, future=True, connect_args=connect_args)
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


def init_db() -> None:
    _get_engine()


@contextmanager
def session() -> Iterator[Session]:
    _get_engine()
    assert _Session is not None
    s: Session = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------------------------------------------------------------------------
# small json helpers (kept in one place so callers stay tidy)
# ---------------------------------------------------------------------------


def loads(s: str | None, default):
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


def dumps(obj) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
