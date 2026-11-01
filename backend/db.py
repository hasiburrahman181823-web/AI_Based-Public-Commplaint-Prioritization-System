"""Synchronous SQLAlchemy sessions for worker-thread FastAPI endpoints.

Applications must run ``alembic upgrade head`` before serving. The factory never
creates domain tables implicitly, avoiding competing startup DDL in workers.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def make_engine(url: str, **overrides) -> Engine:
    database_url = make_url(url)
    if database_url.drivername == "postgresql":
        database_url = database_url.set(drivername="postgresql+psycopg")
    sqlite = database_url.get_backend_name() == "sqlite"
    memory = sqlite and database_url.database in (None, "", ":memory:")
    options = {"pool_pre_ping": True}
    if sqlite:
        options["connect_args"] = {"check_same_thread": False, "timeout": 5}
        if memory:
            options["poolclass"] = StaticPool
        elif database_url.database and not database_url.database.startswith("file:"):
            Path(database_url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    options.update(overrides)
    engine = create_engine(database_url, **options)
    if sqlite:
        @event.listens_for(engine, "connect")
        def sqlite_settings(connection, _):
            cursor = connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("PRAGMA busy_timeout = 5000")
                if not memory:
                    cursor.execute("PRAGMA journal_mode = WAL")
            finally:
                cursor.close()
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]):
    session = factory()
    try:
        with session.begin():
            yield session
    finally:
        session.close()
